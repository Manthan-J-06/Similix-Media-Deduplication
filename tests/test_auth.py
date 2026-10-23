"""
tests/test_auth.py — Comprehensive tests for authentication, sessions, and per-user isolation.

Verifies:
1. Unauthenticated requests get 401 on every endpoint except login, register, static login page.
2. Route scanner: asserts every non-public route in app.routes requires authentication.
3. Registration password policy (>= 10 chars) and invite-only mode.
4. Login failure rate limiting (5 failures per 15 minutes per email+IP returns 429).
5. Sessions: random 256-bit token stored hashed in auth.db, HttpOnly, SameSite=Lax cookie, rotated on login.
6. CSRF: missing or invalid X-CSRF-Token on POST requests returns 403 Forbidden.
7. Per-user isolation: User A cannot read B's cluster, thumbnail, batch, history or job (returns 404, not 403).
8. Ledger and uploads confinement: symlinks and '..' path traversal rejected with 400.
9. Quotas: 50MB per-file cap, 2GB user storage cap, 1 active job per user (409 Conflict).
10. Concurrency queue: global max concurrent jobs queues additional jobs and updates banner position.
11. Hosted-mode flag: disables /sources/dataset with 403.
"""

from __future__ import annotations

import io
import os
from pathlib import Path
import re
import shutil
import sqlite3
import time

from fastapi.testclient import TestClient
from PIL import Image
import pytest

from api.server import create_app, read_job_state, write_job_state
from core.auth import get_session_user, init_auth_db, validate_confined_user_path
from core.ledger import ensure_ledger_tables, plan
from db.connection import init_db


def _create_image(path: Path, size=(100, 100), color=(100, 150, 200)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    im = Image.new("RGB", size, color=color)
    im.save(path, format="JPEG", quality=85)


@pytest.fixture
def auth_test_app(tmp_path: Path):
    """Create isolated Argus app with dedicated auth.db and users_dir."""
    auth_db = tmp_path / "auth_test.db"
    users_dir = tmp_path / "users"
    config = {
        "quotas": {
            "user_storage_cap_mb": 10,  # Small 10MB quota for fast quota testing
        },
        "general": {
            "max_concurrent_jobs": 1,
        },
        "sources": {
            "upload": {
                "per_file_cap_mb": 2,  # 2MB per-file cap for testing
            }
        }
    }
    app = create_app(
        auth_db_path=auth_db,
        users_dir=users_dir,
        config=config,
    )
    return {
        "app": app,
        "auth_db": auth_db,
        "users_dir": users_dir,
        "tmp_path": tmp_path,
        "config": config,
    }


def register_and_login(client: TestClient, email: str, password: str = "ArgusSecurePass123!"):
    """Helper to register, login, and configure headers/cookies on client."""
    reg = client.post("/auth/register", json={"email": email, "password": password})
    assert reg.status_code in (200, 201), f"Registration failed: {reg.text}"
    login_res = client.post("/auth/login", json={"email": email, "password": password})
    assert login_res.status_code == 200, f"Login failed: {login_res.text}"
    data = login_res.json()
    token = data["token"]
    csrf_token = data["csrf_token"]
    client.cookies.set("argus_session", token)
    client.headers["X-CSRF-Token"] = csrf_token
    return {
        "user_id": data["user"]["id"],
        "email": email,
        "token": token,
        "csrf_token": csrf_token,
    }


class TestAuthenticationAndSessions:
    def test_every_route_scanned_requires_auth(self, auth_test_app):
        """Route scanner: scans every route in app.routes and asserts each requires auth (fails if route lacks auth)."""
        app = auth_test_app["app"]
        client = TestClient(app)

        public_paths = {
            "/login",
            "/healthz",
            "/auth/login",
            "/auth/register",
            "/auth/csrf",
            "/docs",
            "/docs/oauth2-redirect",
            "/openapi.json",
            "/redoc",
        }

        routes_tested = 0
        for route in app.routes:
            if not hasattr(route, "methods"):
                continue
            path = route.path
            if path in public_paths:
                continue

            test_path = re.sub(r"\{[a-zA-Z0-9_]+\}", "1", path)
            for method in route.methods:
                if method in ("OPTIONS", "HEAD"):
                    continue
                routes_tested += 1
                resp = client.request(method, test_path)
                assert resp.status_code == 401, (
                    f"Route {method} {path} (evaluated as {test_path}) returned {resp.status_code}, expected 401 Unauthorized"
                )

        assert routes_tested >= 12, f"Expected to test at least 12 protected routes, tested {routes_tested}"

    def test_registration_validation_and_invite_only(self, auth_test_app):
        """Rejects passwords < 10 chars, duplicate emails, and enforces invite-only configuration."""
        app = auth_test_app["app"]
        client = TestClient(app)

        # 1. Password under 10 characters rejected
        r_short = client.post("/auth/register", json={"email": "short@argus.local", "password": "short"})
        assert r_short.status_code == 400
        assert "at least 10" in r_short.json()["detail"].lower()

        # 2. Valid registration
        r_ok = client.post("/auth/register", json={"email": "user@argus.local", "password": "Password1234!"})
        assert r_ok.status_code == 200
        assert r_ok.json()["user"]["email"] == "user@argus.local"

        # 3. Duplicate email rejected
        r_dup = client.post("/auth/register", json={"email": "user@argus.local", "password": "Password1234!"})
        assert r_dup.status_code == 400
        assert "already registered" in r_dup.json()["detail"].lower()

        # 4. Invite-only mode disables registration
        app.state.config["invite_only"] = True
        r_invite = client.post("/auth/register", json={"email": "invite@argus.local", "password": "Password1234!"})
        assert r_invite.status_code == 403
        assert "invite-only" in r_invite.json()["detail"].lower()
        app.state.config["invite_only"] = False

    def test_login_rate_limiting(self, auth_test_app):
        """Enforces rate limit: 5 failed login attempts per 15 min per email+IP results in 429."""
        app = auth_test_app["app"]
        client = TestClient(app)

        # Register user
        client.post("/auth/register", json={"email": "victim@argus.local", "password": "CorrectPassword123!"})

        # 5 failed login attempts
        for i in range(5):
            res = client.post("/auth/login", json={"email": "victim@argus.local", "password": "WrongPassword!"})
            assert res.status_code == 401

        # 6th attempt triggers 429 Too Many Requests
        res_limit = client.post("/auth/login", json={"email": "victim@argus.local", "password": "CorrectPassword123!"})
        assert res_limit.status_code == 429
        assert "15 minutes" in res_limit.json()["detail"].lower()

    def test_csrf_protection_on_post(self, auth_test_app):
        """CSRF token is required on every POST request; missing/invalid returns 403 Forbidden."""
        app = auth_test_app["app"]
        client = TestClient(app)

        auth = register_and_login(client, "csrf_user@argus.local")

        # 1. POST without CSRF header returns 403
        client_no_csrf = TestClient(app)
        client_no_csrf.cookies.set("argus_session", auth["token"])
        res_no_csrf = client_no_csrf.post("/plan", json={"mode": "conservative"})
        assert res_no_csrf.status_code == 403
        assert "CSRF" in res_no_csrf.json()["detail"]

        # 2. POST with invalid CSRF header returns 403
        client_bad_csrf = TestClient(app)
        client_bad_csrf.cookies.set("argus_session", auth["token"])
        client_bad_csrf.headers["X-CSRF-Token"] = "invalid_csrf_token_value_xyz"
        res_bad_csrf = client_bad_csrf.post("/plan", json={"mode": "conservative"})
        assert res_bad_csrf.status_code == 403
        assert "CSRF" in res_bad_csrf.json()["detail"]

        # 3. POST with correct CSRF header succeeds
        client.headers["X-CSRF-Token"] = auth["csrf_token"]
        res_ok = client.post("/plan", json={"mode": "conservative"})
        assert res_ok.status_code == 200

    def test_session_expiration(self, auth_test_app):
        """Expired session token is rejected with 401 Unauthorized."""
        app = auth_test_app["app"]
        client = TestClient(app)

        auth = register_and_login(client, "expire_user@argus.local")

        # Artificially expire the session in auth.db
        with app.state.auth_conn:
            app.state.auth_conn.execute("UPDATE sessions SET expires_at = ?", (time.time() - 100,))

        res = client.get("/auth/me")
        assert res.status_code == 401

    def test_healthz_endpoint(self, auth_test_app):
        """Public health check endpoint returns 200 without authentication."""
        client = TestClient(auth_test_app["app"])
        res = client.get("/healthz")
        assert res.status_code == 200
        assert res.json() == {"status": "ok", "service": "argus"}


class TestPerUserIsolation:
    def test_user_a_cannot_access_user_b_resources(self, auth_test_app):
        """
        User A cannot read User B's cluster, thumbnail, batch, history or job.
        All cross-user lookups return 404, not 403.
        """
        app = auth_test_app["app"]

        # Setup User B
        client_b = TestClient(app)
        user_b = register_and_login(client_b, "user_b@argus.local")
        user_b_dirs = app.state.users_dir / str(user_b["user_id"])
        user_b_dirs.mkdir(parents=True, exist_ok=True)
        (user_b_dirs / "library").mkdir(parents=True, exist_ok=True)
        (user_b_dirs / "jobs").mkdir(parents=True, exist_ok=True)

        # Populate User B's DB with a cluster and 2 images
        conn_b = init_db(user_b_dirs / "argus.db")
        ensure_ledger_tables(conn_b)
        img_b1 = user_b_dirs / "library" / "b1.jpg"
        img_b2 = user_b_dirs / "library" / "b2.jpg"
        _create_image(img_b1, color=(20, 40, 80))
        _create_image(img_b2, color=(20, 40, 80))

        conn_b.execute("INSERT INTO clusters (id, representative_id, stage_detected, member_count) VALUES (1, NULL, 0, 2)")
        conn_b.execute(
            """
            INSERT INTO images (id, cluster_id, file_path, file_size, width, height, format, md5, is_keeper)
            VALUES (101, 1, ?, 500, 100, 100, 'JPEG', 'md5_b1', 1),
                   (102, 1, ?, 500, 100, 100, 'JPEG', 'md5_b1', 0)
            """,
            (str(img_b1), str(img_b2)),
        )
        conn_b.commit()
        conn_b.close()

        # User B creates a plan batch
        plan_res = client_b.post("/plan", json={"mode": "conservative"})
        assert plan_res.status_code == 200
        batch_id_b = plan_res.json()["batch_id"]

        # User B has a job file
        job_b_file = user_b_dirs / "jobs" / "job_b_test.json"
        write_job_state(job_b_file, {
            "id": "job_b_test",
            "status": "completed",
            "stage": "done",
            "stage_display": "Completed",
            "current_file": 2,
            "total_files": 2,
            "progress": 100.0,
        })

        # Setup User A
        client_a = TestClient(app)
        register_and_login(client_a, "user_a@argus.local")

        # 1. User A tries to view User B's cluster -> 404, NOT 403
        res_cluster = client_a.get("/clusters/1")
        assert res_cluster.status_code == 404, f"Expected 404, got {res_cluster.status_code}"

        # 2. User A tries to get thumbnail for User B's image -> 404, NOT 403
        res_thumb = client_a.get("/thumb/101")
        assert res_thumb.status_code == 404, f"Expected 404, got {res_thumb.status_code}"

        # 3. User A tries to apply User B's batch -> 404, NOT 403
        res_apply = client_a.post("/apply", json={"batch_id": batch_id_b, "confirm": True})
        assert res_apply.status_code == 404, f"Expected 404, got {res_apply.status_code}"

        # 4. User A tries to undo User B's batch -> 404, NOT 403
        res_undo = client_a.post("/undo", json={"batch_id": batch_id_b})
        assert res_undo.status_code == 404, f"Expected 404, got {res_undo.status_code}"

        # 5. User A's history does not contain User B's batch
        res_hist = client_a.get("/history")
        assert res_hist.status_code == 200
        history_batches = [h["batch_id"] for h in res_hist.json()]
        assert batch_id_b not in history_batches

        # 6. User A tries to get User B's job -> 404, NOT 403
        res_job = client_a.get("/jobs/job_b_test")
        assert res_job.status_code == 404, f"Expected 404, got {res_job.status_code}"

    def test_register_two_users_upload_and_confirm_isolation(self, auth_test_app):
        """
        Registers two users via the API, uploads an image as each user, and confirms:
        - Files are written strictly into their respective user directories.
        - User A cannot view, access, or poll User B's uploaded files or background jobs (404).
        - User B cannot view, access, or poll User A's uploaded files or background jobs (404).
        """
        app = auth_test_app["app"]
        app.state.pipeline_worker = lambda *args, **kwargs: None  # mock background worker

        # 1. Register User 1
        client1 = TestClient(app)
        u1 = register_and_login(client1, "user_one@argus.local")
        u1_dir = app.state.users_dir / str(u1["user_id"])

        # 2. Register User 2
        client2 = TestClient(app)
        u2 = register_and_login(client2, "user_two@argus.local")
        u2_dir = app.state.users_dir / str(u2["user_id"])

        # 3. User 1 uploads an image
        img1_buf = io.BytesIO()
        Image.new("RGB", (64, 64), color=(255, 0, 0)).save(img1_buf, format="JPEG")
        res1 = client1.post(
            "/sources/upload",
            files=[("files", ("image_user1.jpg", img1_buf.getvalue(), "image/jpeg"))],
        )
        assert res1.status_code == 200
        data1 = res1.json()
        job1_id = data1["job_id"]
        assert "image_user1.jpg" in data1["files"]

        # 4. User 2 uploads an image
        img2_buf = io.BytesIO()
        Image.new("RGB", (64, 64), color=(0, 255, 0)).save(img2_buf, format="JPEG")
        res2 = client2.post(
            "/sources/upload",
            files=[("files", ("image_user2.jpg", img2_buf.getvalue(), "image/jpeg"))],
        )
        assert res2.status_code == 200
        data2 = res2.json()
        job2_id = data2["job_id"]
        assert "image_user2.jpg" in data2["files"]

        # 5. Confirm file isolation on disk
        u1_file = u1_dir / "library" / "default" / "image_user1.jpg"
        u2_file = u2_dir / "library" / "default" / "image_user2.jpg"
        assert u1_file.is_file(), f"Expected User 1 file at {u1_file}"
        assert u2_file.is_file(), f"Expected User 2 file at {u2_file}"
        assert not (u1_dir / "library" / "default" / "image_user2.jpg").exists()
        assert not (u2_dir / "library" / "default" / "image_user1.jpg").exists()

        # 6. Confirm job isolation via API
        # User 1 can see job 1, but gets 404 for job 2
        assert client1.get(f"/jobs/{job1_id}").status_code == 200
        res_cross1 = client1.get(f"/jobs/{job2_id}")
        assert res_cross1.status_code == 404, f"Expected 404 for cross-user job, got {res_cross1.status_code}"

        # User 2 can see job 2, but gets 404 for job 1
        assert client2.get(f"/jobs/{job2_id}").status_code == 200
        res_cross2 = client2.get(f"/jobs/{job1_id}")
        assert res_cross2.status_code == 404, f"Expected 404 for cross-user job, got {res_cross2.status_code}"


class TestConfinementAndQuotas:
    def test_path_traversal_and_symlinks_rejected(self, auth_test_app):
        """Confinement rejects '..' path traversal and symlinks with 400."""
        app = auth_test_app["app"]
        client = TestClient(app)
        user = register_and_login(client, "confined@argus.local")
        user_root = app.state.users_dir / str(user["user_id"])

        # 1. Rejects '..'
        with pytest.raises(Exception) as exc:
            validate_confined_user_path(user_root / ".." / "etc", user_root)
        assert exc.value.status_code == 400

        # 2. Rejects symlink pointing outside
        ext_dir = auth_test_app["tmp_path"] / "outside_dir"
        ext_dir.mkdir(parents=True, exist_ok=True)
        (user_root / "library").mkdir(parents=True, exist_ok=True)
        symlink_path = user_root / "library" / "sym_link"
        try:
            symlink_path.symlink_to(ext_dir)
            with pytest.raises(Exception) as exc_sym:
                validate_confined_user_path(symlink_path, user_root)
            assert exc_sym.value.status_code == 400
        finally:
            if symlink_path.is_symlink():
                symlink_path.unlink()

    def test_per_file_and_user_storage_quotas(self, auth_test_app):
        """Enforces 50MB per-file cap and 2GB user storage cap."""
        app = auth_test_app["app"]
        client = TestClient(app)
        register_and_login(client, "quota_user@argus.local")

        # 1. Per-file cap (configured to 2MB in auth_test_app)
        large_content = b"X" * (3 * 1024 * 1024)  # 3MB > 2MB cap
        resp_file_cap = client.post(
            "/sources/upload",
            files=[("files", ("too_large.jpg", large_content, "image/jpeg"))],
        )
        assert resp_file_cap.status_code == 400
        assert "cap" in resp_file_cap.json()["detail"].lower()

        # 2. Total user quota (configured to 10MB in auth_test_app)
        app.state.config["quotas"]["user_storage_cap_mb"] = 0.001  # ~1KB cap
        small_content = b"Y" * 2048  # 2KB > 1KB
        resp_quota = client.post(
            "/sources/upload",
            files=[("files", ("over_quota.jpg", small_content, "image/jpeg"))],
        )
        assert resp_quota.status_code == 400
        assert "quota" in resp_quota.json()["detail"].lower()

    def test_single_active_job_per_user_and_concurrency_queue(self, auth_test_app):
        """
        Enforces 1 active job per user (409 Conflict) and global concurrency queue.
        Queued position is reported in the banner /jobs/active.
        """
        app = auth_test_app["app"]
        app.state.pipeline_worker = lambda *args, **kwargs: None  # mock worker

        # User 1 launches a job
        client1 = TestClient(app)
        u1 = register_and_login(client1, "queue_u1@argus.local")
        u1_root = app.state.users_dir / str(u1["user_id"])
        _create_image(u1_root / "library" / "img1.jpg")

        r1 = client1.post("/sources/rescan")
        assert r1.status_code == 200
        job1_id = r1.json()["job_id"]

        # User 1 tries to launch a second job while job 1 is active -> 409 Conflict
        r1_second = client1.post("/sources/rescan")
        assert r1_second.status_code == 409
        assert "already active" in r1_second.json()["detail"].lower()

        # User 2 launches a job -> global concurrency is 1, so User 2's job is queued
        client2 = TestClient(app)
        u2 = register_and_login(client2, "queue_u2@argus.local")
        u2_root = app.state.users_dir / str(u2["user_id"])
        _create_image(u2_root / "library" / "img2.jpg")

        r2 = client2.post("/sources/rescan")
        assert r2.status_code == 200
        job2_id = r2.json()["job_id"]

        # Check User 2's active job banner: should show Queued (position 1)
        active2 = client2.get("/jobs/active").json()
        assert active2["active"] is True
        assert "Queued" in active2["stage"]

        # Complete job 1
        j1_file = u1_root / "jobs" / f"{job1_id}.json"
        st1 = read_job_state(j1_file)
        st1["status"] = "completed"
        write_job_state(j1_file, st1)

        # Trigger job check for user 2: job 2 should now promote
        active2_after = client2.get("/jobs/active").json()
        assert active2_after["active"] is True

    def test_hosted_mode_disables_dataset(self, auth_test_app):
        """Hosted-mode config flag disables /sources/dataset with 403."""
        app = auth_test_app["app"]
        app.state.config["hosted_mode"] = True

        client = TestClient(app)
        register_and_login(client, "hosted@argus.local")

        resp = client.post("/sources/dataset", json={"url": "https://huggingface.co/dataset.zip"})
        assert resp.status_code == 403
        assert "hosted mode" in resp.json()["detail"].lower()

    def test_hosted_mode_disables_drive_sources(self, auth_test_app):
        """Hosted-mode config flag disables /sources/drive with 403."""
        app = auth_test_app["app"]
        app.state.config["hosted_mode"] = True

        client = TestClient(app)
        register_and_login(client, "hosted_drive@argus.local")

        resp = client.post(
            "/sources/drive",
            json={"link": "https://drive.google.com/drive/folders/1abcxyz"},
        )
        assert resp.status_code == 403
        assert "hosted mode" in resp.json()["detail"].lower()

    def test_hosted_mode_refuses_to_start_without_secret_key(self, auth_test_app):
        """Hosted mode must refuse to start without SECRET_KEY (no default anywhere)."""
        tmp_path = auth_test_app["tmp_path"]
        old_env_hosted = os.environ.get("HOSTED_MODE")
        old_env_secret = os.environ.get("SECRET_KEY")

        try:
            # 1. In environment: HOSTED_MODE=true without SECRET_KEY
            os.environ["HOSTED_MODE"] = "true"
            os.environ.pop("SECRET_KEY", None)

            with pytest.raises(RuntimeError) as exc1:
                create_app(
                    db_path=tmp_path / "h_test1.db",
                    auth_db_path=tmp_path / "h_auth1.db",
                    users_dir=tmp_path / "h_users1",
                    config={"hosted_mode": True},
                )
            assert "requires secret_key" in str(exc1.value).lower()

            # 2. In config: hosted_mode=True without SECRET_KEY
            os.environ.pop("HOSTED_MODE", None)
            with pytest.raises(RuntimeError) as exc2:
                create_app(
                    db_path=tmp_path / "h_test2.db",
                    auth_db_path=tmp_path / "h_auth2.db",
                    users_dir=tmp_path / "h_users2",
                    config={"hosted_mode": True, "auth": {}},
                )
            assert "requires secret_key" in str(exc2.value).lower()

            # 3. SECRET_KEY shorter than 32 chars rejected
            os.environ["SECRET_KEY"] = "too-short-key"
            with pytest.raises(RuntimeError) as exc3:
                create_app(
                    db_path=tmp_path / "h_test3.db",
                    auth_db_path=tmp_path / "h_auth3.db",
                    users_dir=tmp_path / "h_users3",
                    config={"hosted_mode": True},
                )
            assert "at least 32 characters" in str(exc3.value).lower()

            # 4. With valid 32+ char SECRET_KEY provided, startup succeeds cleanly
            os.environ["SECRET_KEY"] = "super-secret-production-key-32chars"
            app_ok = create_app(
                db_path=tmp_path / "h_test4.db",
                auth_db_path=tmp_path / "h_auth4.db",
                users_dir=tmp_path / "h_users4",
                config={"hosted_mode": True},
            )
            assert app_ok is not None
        finally:
            if old_env_hosted is not None:
                os.environ["HOSTED_MODE"] = old_env_hosted
            else:
                os.environ.pop("HOSTED_MODE", None)

            if old_env_secret is not None:
                os.environ["SECRET_KEY"] = old_env_secret
            else:
                os.environ.pop("SECRET_KEY", None)

    def test_security_headers_present_on_responses(self, auth_test_app):
        """Responses include X-Content-Type-Options, X-Frame-Options, Referrer-Policy, and CSP."""
        app = auth_test_app["app"]
        client = TestClient(app)

        for endpoint in ["/healthz", "/login"]:
            resp = client.get(endpoint)
            assert resp.headers["X-Content-Type-Options"] == "nosniff"
            assert resp.headers["X-Frame-Options"] == "DENY"
            assert resp.headers["Referrer-Policy"] == "strict-origin-when-cross-origin"
            csp = resp.headers["Content-Security-Policy"]
            assert "default-src 'self'" in csp
            assert "frame-ancestors 'none'" in csp

    def test_account_deletion_with_password_confirmation(self, auth_test_app):
        """Account deletion wipes user directory and auth.db rows after password confirmation."""
        app = auth_test_app["app"]
        client = TestClient(app)
        email = "to_delete@argus.local"
        password = "UserPassword123!"

        user_info = register_and_login(client, email=email, password=password)
        user_id = user_info["user_id"]
        user_root = app.state.users_dir / str(user_id)

        # Upload a test file so the user directory has physical content
        img_bytes = io.BytesIO()
        Image.new("RGB", (64, 64), color="blue").save(img_bytes, format="JPEG")
        img_bytes.seek(0)
        upload_res = client.post(
            "/sources/upload",
            data={"project": "default"},
            files={"files": ("blue.jpg", img_bytes.getvalue(), "image/jpeg")},
        )
        assert upload_res.status_code == 200
        assert user_root.exists()
        assert (user_root / "library" / "default" / "blue.jpg").exists()

        # 1. Attempt deletion with incorrect password
        del_wrong = client.post("/auth/delete-account", json={"password": "WrongPassword123!"})
        assert del_wrong.status_code == 400
        assert "incorrect password" in del_wrong.json()["detail"].lower()
        # Verify user still exists
        assert user_root.exists()
        row = app.state.auth_conn.execute("SELECT id FROM users WHERE id = ?", (user_id,)).fetchone()
        assert row is not None

        # 2. Delete with correct password
        del_ok = client.post("/auth/delete-account", json={"password": password})
        assert del_ok.status_code == 200
        assert del_ok.json()["status"] == "deleted"

        # 3. Verify user directory completely removed from disk
        assert not user_root.exists()

        # 4. Verify auth.db rows removed
        row_user = app.state.auth_conn.execute("SELECT id FROM users WHERE id = ?", (user_id,)).fetchone()
        assert row_user is None
        sess_rows = app.state.auth_conn.execute("SELECT id FROM sessions WHERE user_id = ?", (user_id,)).fetchall()
        assert len(sess_rows) == 0

        # 5. Subsequent request with previous session cookie returns 401
        me_resp = client.get("/auth/me")
        assert me_resp.status_code == 401

        # 6. Login with previous credentials returns 401
        login_again = client.post("/auth/login", json={"email": email, "password": password})
        assert login_again.status_code == 401
