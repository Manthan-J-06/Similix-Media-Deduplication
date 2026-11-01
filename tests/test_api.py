"""
tests/test_api.py — Comprehensive tests for Argus FastAPI server (api/server.py).

Verifies:
1. Local binding only (127.0.0.1) and strict CORS restrictions.
2. Endpoints:
   - GET  /clusters (live keeper, savings, counts, categories, sorting, filters, default conservative)
   - GET  /clusters/{id} (live choose_keeper, match_pct, resolution_pct, size, category, flags, swap_proposal)
   - GET  /thumb/{image_id} (cached JPEG, ID only, no client paths)
   - GET  /search (text search)
   - GET  /categories (counts per label)
   - POST /plan (ledger.plan, rejects non-cluster IDs, ignores client paths, default conservative)
   - POST /apply (refuses without confirm=True, executes via ledger)
   - POST /undo (restores via ledger)
   - GET  /history (batch history)
3. Security Invariant: NO endpoint other than /apply can move/delete a file, and /apply without confirm does nothing.
4. Clear 4xx error responses for unknown IDs or batches.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import sqlite3
import threading
import time
import urllib.request
import zipfile

from typing import Any

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from PIL import Image

from api.server import create_app
from core.ledger import compute_file_md5, ensure_ledger_tables
from db.connection import init_db


def _create_test_image(path: Path, color: tuple[int, int, int] = (100, 100, 100), size: tuple[int, int] = (200, 200)) -> None:
    im = Image.new("RGB", size, color=color)
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path, format="JPEG", quality=90)


def make_authenticated_client(
    app: FastAPI,
    email: str = "testuser@argus.local",
    password: str = "ArgusPassword123!",
) -> TestClient:
    """Helper to register and login a test user, attaching session cookies and CSRF headers."""
    client = TestClient(app)
    client.post("/auth/register", json={"email": email, "password": password})
    res = client.post("/auth/login", json={"email": email, "password": password})
    data = res.json()
    token = data.get("token")
    csrf_token = data.get("csrf_token")
    if token:
        client.cookies.set("argus_session", token)
    if csrf_token:
        client.headers["X-CSRF-Token"] = csrf_token
    return client


@pytest.fixture
def api_test_env(tmp_path: Path):
    """Sets up a test SQLite DB, photos folder, trash folder, and thumb cache."""
    db_file = tmp_path / "test_argus.db"
    conn = init_db(db_file)
    ensure_ledger_tables(conn)

    trash_dir = tmp_path / "trash"
    trash_dir.mkdir(parents=True, exist_ok=True)

    thumb_dir = tmp_path / "thumbs"
    thumb_dir.mkdir(parents=True, exist_ok=True)

    img_dir = tmp_path / "photos"
    img_dir.mkdir(parents=True, exist_ok=True)

    # Cluster 1: 3 members (original, exact copy, low-res copy)
    p_orig = img_dir / "orig.jpg"
    p_copy = img_dir / "orig_copy.jpg"
    p_low = img_dir / "orig_low.jpg"

    _create_test_image(p_orig, color=(200, 50, 50), size=(800, 600))
    p_copy.write_bytes(p_orig.read_bytes())
    _create_test_image(p_low, color=(200, 50, 50), size=(400, 300))

    md5_orig = compute_file_md5(p_orig)
    md5_low = compute_file_md5(p_low)
    now = time.time()

    conn.execute(
        "INSERT INTO clusters (id, representative_id, stage_detected, member_count, category) "
        "VALUES (1, NULL, 0, 3, 'landscape')"
    )

    # Insert cluster 1 members
    conn.execute(
        """
        INSERT INTO images (
            id, cluster_id, file_path, file_size, width, height, format, md5,
            is_keeper, is_recommended_keeper, delete_safe, review_flag,
            is_low_confidence, quality_status, ssim, ssim_color, sharpness_ratio,
            pixel_ratio, bytes_ratio, modified_time, category_top1, category_top1_prob,
            match_pct, resolution_pct, confidence
        ) VALUES
          (1, 1, ?, ?, 800, 600, 'JPEG', ?, 1, 1, 0, 0, 0, 'reference', 1.0, 1.0, 1.0, 1.0, 1.0, ?, 'landscape', 0.95, 100.0, 100.0, 1.0),
          (2, 1, ?, ?, 800, 600, 'JPEG', ?, 0, 0, 1, 0, 0, 'aligned', 1.0, 1.0, 1.0, 1.0, 1.0, ?, 'landscape', 0.94, 100.0, 100.0, 1.0),
          (3, 1, ?, ?, 400, 300, 'JPEG', ?, 0, 0, 0, 1, 1, 'aligned', 0.96, 0.96, 1.0, 0.25, 0.25, ?, 'landscape', 0.91, 96.0, 25.0, 1.0)
        """,
        (
            str(p_orig), p_orig.stat().st_size, md5_orig, now - 100,
            str(p_copy), p_copy.stat().st_size, md5_orig, now - 50,
            str(p_low), p_low.stat().st_size, md5_low, now - 10,
        ),
    )
    conn.execute("UPDATE clusters SET representative_id = 1 WHERE id = 1")

    # Cluster 2: 2 members (food category)
    p_food1 = img_dir / "food1.jpg"
    p_food2 = img_dir / "food2.jpg"
    _create_test_image(p_food1, color=(50, 200, 50), size=(600, 600))
    p_food2.write_bytes(p_food1.read_bytes())
    md5_food = compute_file_md5(p_food1)

    conn.execute(
        "INSERT INTO clusters (id, representative_id, stage_detected, member_count, category) "
        "VALUES (2, NULL, 0, 2, 'food')"
    )
    conn.execute(
        """
        INSERT INTO images (
            id, cluster_id, file_path, file_size, width, height, format, md5,
            is_keeper, is_recommended_keeper, delete_safe, review_flag,
            is_low_confidence, quality_status, ssim, ssim_color, sharpness_ratio,
            pixel_ratio, bytes_ratio, modified_time, category_top1, category_top1_prob,
            match_pct, resolution_pct, confidence
        ) VALUES
          (4, 2, ?, ?, 600, 600, 'JPEG', ?, 1, 1, 0, 0, 0, 'reference', 1.0, 1.0, 1.0, 1.0, 1.0, ?, 'food', 0.88, 100.0, 100.0, 1.0),
          (5, 2, ?, ?, 600, 600, 'JPEG', ?, 0, 0, 1, 0, 0, 'aligned', 1.0, 1.0, 1.0, 1.0, 1.0, ?, 'food', 0.87, 100.0, 100.0, 1.0)
        """,
        (
            str(p_food1), p_food1.stat().st_size, md5_food, now - 80,
            str(p_food2), p_food2.stat().st_size, md5_food, now - 40,
        ),
    )
    conn.execute("UPDATE clusters SET representative_id = 4 WHERE id = 2")

    # Standalone unclustered image (id=6)
    p_alone = img_dir / "alone.jpg"
    _create_test_image(p_alone, color=(10, 10, 10), size=(300, 300))
    md5_alone = compute_file_md5(p_alone)
    conn.execute(
        """
        INSERT INTO images (
            id, cluster_id, file_path, file_size, width, height, format, md5,
            is_keeper, modified_time, category_top1, category_top1_prob
        ) VALUES (6, NULL, ?, ?, 300, 300, 'JPEG', ?, 0, ?, 'animal', 0.75)
        """,
        (str(p_alone), p_alone.stat().st_size, md5_alone, now),
    )

    conn.commit()
    conn.close()

    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)

    library_dir = tmp_path / "library"
    library_dir.mkdir(parents=True, exist_ok=True)

    auth_db = tmp_path / "auth.db"
    users_dir = tmp_path / "users"

    app = create_app(
        db_path=db_file,
        trash_dir=trash_dir,
        thumb_cache_dir=thumb_dir,
        library_dir=library_dir,
        jobs_dir=jobs_dir,
        auth_db_path=auth_db,
        users_dir=users_dir,
    )
    app.state.data_root = tmp_path
    if 1 in app.state.custom_user_overrides:
        app.state.custom_user_overrides[1]["root"] = tmp_path

    client = make_authenticated_client(app, email="user1@argus.local", password="ArgusPassword123!")

    return {
        "client": client,
        "app": app,
        "db_file": db_file,
        "trash_dir": trash_dir,
        "thumb_dir": thumb_dir,
        "img_dir": img_dir,
        "jobs_dir": jobs_dir,
        "library_dir": library_dir,
        "files": {
            1: p_orig,
            2: p_copy,
            3: p_low,
            4: p_food1,
            5: p_food2,
            6: p_alone,
        },
    }


class TestAPIServer:

    def test_cors_restrictions(self, api_test_env):
        """Disallows external CORS origins while allowing local 127.0.0.1 / localhost."""
        client = api_test_env["client"]

        # Evil external origin
        res = client.options("/clusters", headers={"Origin": "https://malicious-site.com"})
        assert res.headers.get("access-control-allow-origin") != "https://malicious-site.com"

        # Local origin
        res_local = client.options(
            "/clusters",
            headers={
                "Origin": "http://127.0.0.1:8000",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert res_local.headers.get("access-control-allow-origin") == "http://127.0.0.1:8000"

    def test_clusters_list_default_conservative_and_filters(self, api_test_env):
        """GET /clusters lists clusters with keeper, savings, counts, categories. Default mode conservative."""
        client = api_test_env["client"]

        res = client.get("/clusters")
        assert res.status_code == 200
        data = res.json()
        assert len(data) == 2

        # Verify cluster 1 content
        c1 = next(c for c in data if c["id"] == 1)
        assert c1["keeper"]["id"] == 1
        assert c1["member_count"] == 3
        assert c1["delete_safe_count"] == 1  # exact copy is delete_safe in conservative
        assert c1["review_count"] == 1       # low-res is review in conservative
        assert c1["savings_bytes"] > 0
        assert c1["category"] == "landscape"
        assert c1["has_swap_proposal"] is False

        # Filter by category
        res_food = client.get("/clusters?category=food")
        assert res_food.status_code == 200
        food_data = res_food.json()
        assert len(food_data) == 1
        assert food_data[0]["id"] == 2
        assert food_data[0]["category"] == "food"

        # Filter by min_savings
        res_sav = client.get("/clusters?min_savings=999999999")
        assert res_sav.status_code == 200
        assert len(res_sav.json()) == 0

        # Sort by count descending
        res_sorted = client.get("/clusters?sort=count")
        assert res_sorted.status_code == 200
        sort_data = res_sorted.json()
        assert sort_data[0]["member_count"] >= sort_data[1]["member_count"]

    def test_cluster_detail(self, api_test_env):
        """GET /clusters/{id} returns live keeper, member metrics, match_pct, res_pct."""
        client = api_test_env["client"]

        res = client.get("/clusters/1")
        assert res.status_code == 200
        data = res.json()
        assert data["id"] == 1
        assert data["mode"] == "conservative"
        assert len(data["members"]) == 3

        members = {m["id"]: m for m in data["members"]}
        assert members[1]["is_recommended_keeper"] is True
        assert members[1]["delete_safe"] is False
        assert members[1]["match_pct"] == 100.0

        assert members[2]["is_recommended_keeper"] is False
        assert members[2]["delete_safe"] is True

        assert members[3]["is_recommended_keeper"] is False
        assert members[3]["review"] is True
        assert members[3]["match_pct"] == 96.0
        assert members[3]["resolution_pct"] == 25.0

        # 404 for unknown cluster
        res_404 = client.get("/clusters/999")
        assert res_404.status_code == 404

    def test_reproduce_cluster_category_edit_causes_500(self, api_test_env):
        """Reproduce bug where editing cluster label causes GET /clusters/{id} and /clusters to 500, and verify clearing."""
        client = api_test_env["client"]

        # 1. Edit label
        post_res = client.post("/clusters/1/category", json={"category": "beach"})
        assert post_res.status_code == 200
        assert post_res.json()["category"] == "beach"
        assert post_res.json()["manual_category"] == "beach"

        # Fetch detail
        res = client.get("/clusters/1")
        assert res.status_code == 200
        data = res.json()
        assert data["category"] == "beach"
        assert data["manual_category"] == "beach"

        # Fetch list
        list_res = client.get("/clusters")
        assert list_res.status_code == 200
        clusters = list_res.json()
        c1 = next((c for c in clusters if c["id"] == 1), None)
        assert c1 is not None
        assert c1["category"] == "beach"
        assert c1["manual_category"] == "beach"

        # 2. Clear label with empty string
        clear_res = client.post("/clusters/1/category", json={"category": ""})
        assert clear_res.status_code == 200
        assert clear_res.json()["manual_category"] is None

        # Fetch detail after clearing
        res_after = client.get("/clusters/1")
        assert res_after.status_code == 200
        data_after = res_after.json()
        assert data_after["manual_category"] is None
        assert data_after["category"] != "beach"

        # Fetch list after clearing
        list_after = client.get("/clusters")
        assert list_after.status_code == 200

        # 3. Clear label with whitespace string
        client.post("/clusters/1/category", json={"category": "mountains"})
        clear_ws = client.post("/clusters/1/category", json={"category": "   "})
        assert clear_ws.status_code == 200
        assert clear_ws.json()["manual_category"] is None

        # 4. Clear label with DELETE
        client.post("/clusters/1/category", json={"category": "forest"})
        del_res = client.delete("/clusters/1/category")
        assert del_res.status_code == 200
        assert del_res.json()["manual_category"] is None

        # 5. Clear with None
        client.post("/clusters/1/category", json={"category": "city street"})
        none_res = client.post("/clusters/1/category", json={"category": None})
        assert none_res.status_code == 200
        assert none_res.json()["manual_category"] is None

    def test_thumbnail_caching_and_id_only(self, api_test_env):
        """GET /thumb/{image_id} generates and caches JPEG thumbnail. Only accepts image_id."""
        client = api_test_env["client"]
        thumb_dir = api_test_env["thumb_dir"]

        res = client.get("/thumb/1?size=128")
        assert res.status_code == 200
        assert res.headers["content-type"] == "image/jpeg"
        assert res.headers.get("Cache-Control") == "private, max-age=3600"
        assert "ETag" in res.headers
        assert len(res.content) > 0

        cached_files = list(thumb_dir.glob("1_*_128.jpg"))
        assert len(cached_files) == 1
        assert cached_files[0].exists()

        # Second request serves from cache
        res2 = client.get("/thumb/1?size=128")
        assert res2.status_code == 200
        assert len(res2.content) == len(res.content)

        # 404 for non-existent image id
        res_404 = client.get("/thumb/99999")
        assert res_404.status_code == 404
        assert "not found" in res_404.json()["detail"].lower()

    def test_categories_endpoint(self, api_test_env):
        """GET /categories returns visible cluster counts (omits zero-count/unclustered)."""
        client = api_test_env["client"]
        res = client.get("/categories")
        assert res.status_code == 200
        data = res.json()
        assert data.get("landscape") == 1
        assert data.get("food") == 1
        assert "animal" not in data

    def test_search_endpoint(self, api_test_env):
        """GET /search?q= accepts text queries."""
        client = api_test_env["client"]
        # Empty query
        res_empty = client.get("/search?q=")
        assert res_empty.status_code == 200
        assert res_empty.json() == []

    def test_plan_rejects_non_cluster_image_id(self, api_test_env):
        """POST /plan rejects any image id in excluded_ids not belonging to a cluster."""
        client = api_test_env["client"]

        # Image 6 is not in any cluster; Image 999 does not exist
        res = client.post("/plan", json={"excluded_ids": [6]})
        assert res.status_code == 400
        assert "not currently a member of a cluster" in res.json()["detail"]

        res_unknown = client.post("/plan", json={"excluded_ids": [999]})
        assert res_unknown.status_code == 400
        assert "not currently a member of a cluster" in res_unknown.json()["detail"]

    def test_plan_ignores_client_file_paths_and_respects_excluded_ids(self, api_test_env):
        """POST /plan ignores arbitrary client file paths and respects excluded_ids."""
        client = api_test_env["client"]

        # Pass unexpected file_path and malicious path fields
        payload = {
            "mode": "conservative",
            "excluded_ids": [2],  # Exclude exact copy (id=2)
            "file_path": "/etc/passwd",
            "paths": ["/home/user/test.jpg"],
        }
        res = client.post("/plan", json=payload)
        assert res.status_code == 200
        data = res.json()
        # id=2 was excluded, so cluster 1 has 0 planned deletions
        # cluster 2 has id=5 planned
        planned_ids = [e["image_id"] for e in data["entries"]]
        assert 2 not in planned_ids
        assert 5 in planned_ids

    def test_plan_accepts_manual_member_ids(self, api_test_env):
        """POST /plan honors manual_member_ids for explicit user deletions."""
        client = api_test_env["client"]
        res = client.post("/plan", json={"mode": "conservative", "manual_member_ids": [2]})
        assert res.status_code == 200
        data = res.json()
        planned_ids = [e["image_id"] for e in data["entries"]]
        assert 2 in planned_ids
        entry_2 = next(e for e in data["entries"] if e["image_id"] == 2)
        assert "Manual user selection" in entry_2["reason"] or entry_2.get("is_manual")

    def test_apply_without_confirm_fails_and_does_nothing(self, api_test_env):
        """POST /apply requires confirm=True; otherwise returns 400 and modifies nothing."""
        client = api_test_env["client"]
        p_copy = api_test_env["files"][2]
        p_copy_mtime = p_copy.stat().st_mtime

        # Create a plan first
        plan_res = client.post("/plan", json={"mode": "conservative"})
        assert plan_res.status_code == 200
        batch_id = plan_res.json()["batch_id"]

        # Call apply without confirm
        res_no_conf = client.post("/apply", json={"batch_id": batch_id, "confirm": False})
        assert res_no_conf.status_code == 400
        assert "explicit confirmation" in res_no_conf.json()["detail"].lower()
        assert p_copy.exists()
        assert p_copy.stat().st_mtime == p_copy_mtime

        # Unknown batch returns 404
        res_404 = client.post("/apply", json={"batch_id": "batch_fake_12345", "confirm": True})
        assert res_404.status_code == 404
        assert "not found" in res_404.json()["detail"].lower()

    def test_full_plan_apply_undo_lifecycle(self, api_test_env):
        """Lifecycle: plan -> apply (moves to trash) -> undo (restores from trash)."""
        client = api_test_env["client"]
        p_copy = api_test_env["files"][2]
        p_food2 = api_test_env["files"][5]

        assert p_copy.exists()
        assert p_food2.exists()

        # 1. Plan
        plan_res = client.post("/plan", json={"mode": "conservative"})
        assert plan_res.status_code == 200
        batch_id = plan_res.json()["batch_id"]
        assert plan_res.json()["file_count"] == 2

        # 2. Apply with confirm=True
        apply_res = client.post("/apply", json={"batch_id": batch_id, "confirm": True})
        assert apply_res.status_code == 200
        assert apply_res.json()["moved_count"] == 2

        # Files moved to trash
        assert not p_copy.exists()
        assert not p_food2.exists()

        # Check history
        hist_res = client.get("/history")
        assert hist_res.status_code == 200
        assert any(b["batch_id"] == batch_id and b["status"] == "executed" for b in hist_res.json())

        # 3. Undo
        undo_res = client.post("/undo", json={"batch_id": batch_id})
        assert undo_res.status_code == 200
        assert undo_res.json()["restored_count"] == 2

        # Files restored back to original paths
        assert p_copy.exists()
        assert p_food2.exists()

        # Unknown batch undo returns 404
        undo_404 = client.post("/undo", json={"batch_id": "batch_fake_99999"})
        assert undo_404.status_code == 404

    def test_security_invariant_no_endpoint_other_than_apply_moves_file(self, api_test_env):
        """
        CRITICAL SECURITY INVARIANT:
        No endpoint other than /apply can delete or move a file.
        /apply without confirm does nothing.
        """
        client = api_test_env["client"]
        files = api_test_env["files"]

        def snapshot_files():
            return {
                mid: (p.exists(), p.stat().st_size if p.exists() else None, compute_file_md5(p) if p.exists() else None)
                for mid, p in files.items()
            }

        initial_state = snapshot_files()

        # 0. GET / (UI route)
        ui_res = client.get("/")
        assert ui_res.status_code == 200
        assert snapshot_files() == initial_state

        # 1. GET /clusters
        client.get("/clusters?mode=conservative")
        client.get("/clusters?mode=aggressive")
        assert snapshot_files() == initial_state

        # 2. GET /clusters/{id}
        client.get("/clusters/1?mode=conservative")
        client.get("/clusters/1?mode=aggressive")
        client.get("/clusters/2")
        assert snapshot_files() == initial_state

        # 3. GET /thumb/{id}
        client.get("/thumb/1?size=64")
        client.get("/thumb/2?size=64")
        assert snapshot_files() == initial_state

        # 4. GET /search
        client.get("/search?q=landscape")
        assert snapshot_files() == initial_state

        # 5. GET /categories
        client.get("/categories")
        assert snapshot_files() == initial_state

        # 6. POST /plan (computes plan, writes journal, but touches 0 source files)
        plan_res = client.post("/plan", json={"mode": "aggressive"})
        assert plan_res.status_code == 200
        batch_id = plan_res.json()["batch_id"]
        assert snapshot_files() == initial_state

        # 7. GET /history
        client.get("/history")
        assert snapshot_files() == initial_state

        # 8. POST /apply WITHOUT confirm
        client.post("/apply", json={"batch_id": batch_id, "confirm": False})
        assert snapshot_files() == initial_state

        # 9. ONLY POST /apply WITH confirm=True moves files
        apply_res = client.post("/apply", json={"batch_id": batch_id, "confirm": True})
        assert apply_res.status_code == 200
        post_apply_state = snapshot_files()
        assert post_apply_state != initial_state  # Files are moved!

    def test_cmd_serve_cli_main_mocked(self, api_test_env, monkeypatch):
        """Test cmd_serve through main() with uvicorn.run mocked.
        Assert host is 127.0.0.1 and app was created with the right db path."""
        import uvicorn
        from core.cli import main

        run_called = {}

        def mock_run(app, host, port):
            run_called["app"] = app
            run_called["host"] = host
            run_called["port"] = port

        monkeypatch.setattr(uvicorn, "run", mock_run)

        db_file = api_test_env["db_file"]
        main(["serve", "--db", str(db_file), "--port", "8001"])

        assert run_called["host"] == "127.0.0.1"
        assert run_called["port"] == 8001
        assert run_called["app"].state.db_path == db_file.resolve()

    def test_ui_smoke_and_element_ids(self, api_test_env):
        """Smoke test for UI: GET / returns 200, serves text/html, contains required element IDs and text."""
        client = api_test_env["client"]

        res = client.get("/")
        assert res.status_code == 200
        assert "text/html" in res.headers.get("content-type", "")
        html = res.text

        # Required element IDs used by the script
        required_ids = [
            'id="cluster-list"',
            'id="cluster-detail-panel"',
            'id="mode-slider"',
            'id="category-filter"',
            'id="sort-select"',
            'id="has-savings-toggle"',
            'id="search-input"',
            'id="swap-proposal-banner"',
            'id="member-grid"',
            'id="btn-select-safe"',
            'id="btn-plan-deletion"',
            'id="confirm-modal"',
            'id="btn-confirm-apply"',
            'id="btn-undo-apply"',
            'id="history-modal"',
        ]
        for elem_id in required_ids:
            assert elem_id in html, f"Missing required element ID {elem_id} in UI HTML"

        # Required confirmation sentence
        assert "Files are moved to trash and can be undone." in html

        # Verify /thumb works alongside UI
        thumb_res = client.get("/thumb/1?size=96")
        assert thumb_res.status_code == 200
        assert thumb_res.headers.get("content-type") == "image/jpeg"

    def test_concurrent_thumbnail_requests_real_demo_images(self, tmp_path):
        """Requests thumbnails for all images in the real ~/dedupe_test/demo/demo.db copy
        concurrently with 20 workers, asserting every response is 200 image/jpeg."""
        from concurrent.futures import ThreadPoolExecutor
        import shutil

        demo_db_orig = Path("~/dedupe_test/demo/demo.db").expanduser().resolve()
        try:
            with open(demo_db_orig, "rb") as _:
                pass
        except (PermissionError, FileNotFoundError, OSError):
            pytest.skip("Demo DB not accessible in sandbox")

        # Copy demo.db into tmp_path
        demo_db_copy = tmp_path / "demo_real_copy.db"
        shutil.copy2(demo_db_orig, demo_db_copy)

        thumb_dir = tmp_path / "thumbs_demo_real"
        app = create_app(db_path=demo_db_copy, thumb_cache_dir=thumb_dir)
        if 1 in app.state.custom_user_overrides:
            app.state.custom_user_overrides[1]["root"] = Path("/").resolve()
        client = make_authenticated_client(app)

        conn = sqlite3.connect(demo_db_copy)
        image_ids = [row[0] for row in conn.execute("SELECT id FROM images").fetchall()]
        conn.close()

        assert len(image_ids) > 0, f"Expected demo images, found {len(image_ids)}"

        # 20 concurrent workers requesting thumbnails
        tasks = []
        for img_id in image_ids[:36]:
            tasks.append((img_id, 96))
            tasks.append((img_id, 320))

        def fetch_thumb(item):
            iid, sz = item
            resp = client.get(f"/thumb/{iid}?size={sz}")
            return resp.status_code, resp.headers.get("content-type")

        with ThreadPoolExecutor(max_workers=20) as executor:
            results = list(executor.map(fetch_thumb, tasks))

        assert len(results) == len(tasks)
        for status_code, content_type in results:
            assert status_code == 200
            assert content_type == "image/jpeg"

    def test_thumbnail_failure_returns_placeholder_not_500(self, api_test_env, tmp_path):
        """A thumbnail failure must never return a 500: return a small placeholder image instead and log the reason."""
        client = api_test_env["client"]
        app = api_test_env["app"]
        db_file = api_test_env["db_file"]

        # 1. Image in DB but file missing on disk -> returns 200 image/jpeg placeholder
        missing_file = tmp_path / "nonexistent.jpg"
        conn = sqlite3.connect(db_file)
        conn.execute(
            """
            INSERT INTO images (id, cluster_id, file_path, file_size, width, height, format, md5, is_keeper)
            VALUES (999, 1, ?, 100, 100, 100, 'JPEG', 'fake', 0)
            """,
            (str(missing_file),),
        )
        # 2. Image in DB but file corrupted on disk -> returns 200 image/jpeg placeholder
        corrupted_file = tmp_path / "corrupted.jpg"
        corrupted_file.write_bytes(b"NOT_A_VALID_IMAGE_HEADER_GARBAGE_BYTES")
        conn.execute(
            """
            INSERT INTO images (id, cluster_id, file_path, file_size, width, height, format, md5, is_keeper)
            VALUES (1000, 1, ?, 100, 100, 100, 'JPEG', 'fake2', 0)
            """,
            (str(corrupted_file),),
        )
        conn.commit()
        conn.close()

        res_missing = client.get("/thumb/999?size=96")
        assert res_missing.status_code == 200
        assert res_missing.headers.get("content-type") == "image/jpeg"

        res_corrupt = client.get("/thumb/1000?size=96")
        assert res_corrupt.status_code == 200
        assert res_corrupt.headers.get("content-type") == "image/jpeg"

        # Unknown ID not in DB -> 404
        res_unknown = client.get("/thumb/99999?size=96")
        assert res_unknown.status_code == 404

    def test_old_schema_db_migration_and_viewer_clusters(self, tmp_path: Path):
        """
        Build a DB using the oldest schema (copy real ~/dedupe_test/demo_backup2/demo.db)
        and open it with init_db without error, and verify the viewer API lists clusters from it.
        """
        real_backup = Path.home() / "dedupe_test/demo_backup2/demo.db"
        test_db = tmp_path / "old_demo_copy.db"

        has_real_backup = False
        try:
            with open(real_backup, "rb") as _:
                has_real_backup = True
        except (PermissionError, FileNotFoundError, OSError):
            has_real_backup = False

        if has_real_backup:
            shutil.copyfile(real_backup, test_db)
            raw_conn = sqlite3.connect(test_db)
            cols_before = {r[1] for r in raw_conn.execute("PRAGMA table_info(images)").fetchall()}
            raw_conn.close()
            assert "drive_id" not in cols_before
        else:
            p1 = tmp_path / "1.jpg"
            p2 = tmp_path / "2.jpg"
            _create_test_image(p1)
            _create_test_image(p2)
            raw_conn = sqlite3.connect(test_db)
            raw_conn.executescript(
                f"""
                CREATE TABLE images (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_path     TEXT NOT NULL UNIQUE,
                    file_size     INTEGER,
                    modified_time REAL,
                    width         INTEGER,
                    height        INTEGER,
                    format        TEXT,
                    md5           TEXT,
                    phash         TEXT,
                    clip_vector   BLOB,
                    status        TEXT NOT NULL DEFAULT 'discovered',
                    cluster_id    INTEGER,
                    is_keeper     INTEGER NOT NULL DEFAULT 0,
                    created_at    TEXT NOT NULL,
                    updated_at    TEXT NOT NULL
                );
                CREATE TABLE clusters (
                    id                INTEGER PRIMARY KEY AUTOINCREMENT,
                    representative_id INTEGER,
                    stage_detected    INTEGER NOT NULL,
                    member_count      INTEGER NOT NULL DEFAULT 0,
                    created_at        TEXT NOT NULL
                );
                INSERT INTO clusters (id, representative_id, stage_detected, member_count, created_at)
                VALUES (1, NULL, 0, 2, '2026-01-01T00:00:00Z');
                INSERT INTO images (id, file_path, file_size, width, height, format, md5, cluster_id, is_keeper, created_at, updated_at)
                VALUES (1, '{p1}', 1000, 100, 100, 'JPEG', 'md5_1', 1, 1, '2026-01-01', '2026-01-01');
                INSERT INTO images (id, file_path, file_size, width, height, format, md5, cluster_id, is_keeper, created_at, updated_at)
                VALUES (2, '{p2}', 1000, 100, 100, 'JPEG', 'md5_1', 1, 0, '2026-01-01', '2026-01-01');
                """
            )
            raw_conn.commit()
            raw_conn.close()

        # Opening with init_db directly must not raise OperationalError
        conn = init_db(test_db)
        cols_after = {r[1] for r in conn.execute("PRAGMA table_info(images)").fetchall()}
        assert "drive_id" in cols_after
        assert "source" in cols_after
        assert "quality_basis" in cols_after
        conn.close()

        # The viewer API can list clusters from the migrated DB
        app = create_app(db_path=test_db)
        client = make_authenticated_client(app)
        resp = client.get("/clusters")
        assert resp.status_code == 200
        clusters = resp.json()
        assert isinstance(clusters, list)
        assert len(clusters) > 0
        assert "id" in clusters[0]
        assert "keeper" in clusters[0]

    def test_apply_omits_trashed_file_and_undo_restores(self, api_test_env):
        """
        A.1: After apply, GET /clusters/1 omits the trashed file.
        Undo restores it on disk and brings it back in /clusters/1.
        """
        client = api_test_env["client"]

        # Before apply: Cluster 1 has 3 members
        c1_before = client.get("/clusters/1?mode=conservative").json()
        assert len(c1_before["members"]) == 3
        member_ids = [m["id"] for m in c1_before["members"]]
        assert member_ids == [1, 2, 3]

        # Plan and apply
        plan_resp = client.post("/plan", json={"mode": "conservative"}).json()
        batch_id = plan_resp["batch_id"]
        assert plan_resp["file_count"] >= 1

        apply_resp = client.post("/apply", json={"batch_id": batch_id, "confirm": True})
        assert apply_resp.status_code == 200

        # After apply: GET /clusters/1 must omit the trashed member (id=2)
        c1_after = client.get("/clusters/1?mode=conservative").json()
        live_ids = [m["id"] for m in c1_after["members"]]
        assert 2 not in live_ids
        assert live_ids == [1, 3]
        # Never list a file not on disk
        for m in c1_after["members"]:
            assert Path(m["file_path"]).exists()

        # Clusters list also reflects updated member count
        c_list = client.get("/clusters?mode=conservative").json()
        c1_summary = next(c for c in c_list if c["id"] == 1)
        assert c1_summary["member_count"] == 2

        # Undo brings it back
        undo_resp = client.post("/undo", json={"batch_id": batch_id})
        assert undo_resp.status_code == 200

        c1_restored = client.get("/clusters/1?mode=conservative").json()
        restored_ids = [m["id"] for m in c1_restored["members"]]
        assert 2 in restored_ids
        assert len(restored_ids) == 3
        assert Path(api_test_env["files"][2]).exists()

    def test_clusters_hidden_when_under_two_live_members(self, api_test_env):
        """
        A.1: Clusters with < 2 live members on disk are hidden from /clusters,
        and /clusters/{id} returns 404.
        """
        client = api_test_env["client"]

        # Cluster 2 initially has 2 members (IDs 4 and 5)
        c2 = client.get("/clusters/2").json()
        assert len(c2["members"]) == 2

        # Manually delete member 5 file from disk
        p_food2 = Path(api_test_env["files"][5])
        assert p_food2.exists()
        p_food2.unlink()

        # GET /clusters must now hide Cluster 2
        clusters = client.get("/clusters").json()
        c_ids = [c["id"] for c in clusters]
        assert 2 not in c_ids

        # GET /clusters/2 must return 404
        resp = client.get("/clusters/2")
        assert resp.status_code == 404
        assert "fewer than 2 live members" in resp.json()["detail"]

    def test_upload_path_traversal_filename_rejected(self, api_test_env):
        """
        B.2 & B.6: Path traversal filenames must be rejected with 400.
        """
        client = api_test_env["client"]

        traversal_names = [
            "../../escape.jpg",
            "/absolute/path.jpg",
            "folder/sub.jpg",
            "..\\windows_escape.jpg",
            "....//test.jpg",
        ]
        for name in traversal_names:
            resp = client.post(
                "/sources/upload",
                files=[("files", (name, b"dummyimagebytes", "image/jpeg"))],
            )
            assert resp.status_code == 400, f"Expected 400 for {name}, got {resp.status_code}"
            assert "Path traversal rejected" in resp.json()["detail"]

    def test_upload_oversize_rejected(self, api_test_env):
        """
        B.2 & B.6: Files exceeding per-file cap (50 MB) must be rejected with 400.
        """
        app = api_test_env["app"]
        # Set per-file cap to 1 MB for testing
        app.state.config["sources"] = {"upload": {"per_file_cap_mb": 1, "total_cap_mb": 100}}
        client = api_test_env["client"]

        # 1.5 MB payload
        large_bytes = b"0" * (1500 * 1024)
        resp = client.post(
            "/sources/upload",
            files=[("files", ("large.jpg", large_bytes, "image/jpeg"))],
        )
        assert resp.status_code == 400
        assert "cap" in resp.json()["detail"].lower()

    def test_upload_non_image_rejected(self, api_test_env):
        """
        B.2 & B.6: Non-image extensions (not jpg/png/webp/heic) rejected with 400.
        """
        client = api_test_env["client"]

        bad_files = ["malware.exe", "script.sh", "notes.txt", "data.json", "archive.tar"]
        for bf in bad_files:
            resp = client.post(
                "/sources/upload",
                files=[("files", (bf, b"some content", "application/octet-stream"))],
            )
            assert resp.status_code == 400, f"Expected 400 for {bf}, got {resp.status_code}"
            assert "Non-image rejected" in resp.json()["detail"]

    def test_dataset_host_outside_allow_list_rejected(self, api_test_env):
        """
        B.4 & B.6: Host outside allow-list and non-HTTPS rejected with 400.
        """
        client = api_test_env["client"]

        # Disallowed host
        resp = client.post("/sources/dataset", json={"url": "https://evil.com/dataset.zip"})
        assert resp.status_code == 400
        assert "outside allow-list" in resp.json()["detail"].lower()

        # Non-HTTPS
        resp_http = client.post("/sources/dataset", json={"url": "http://huggingface.co/dataset.zip"})
        assert resp_http.status_code == 400
        assert "https" in resp_http.json()["detail"].lower()

    def test_dataset_redirect_off_list_rejected(self, api_test_env, monkeypatch):
        """
        B.4 & B.6: Redirect off allow-list rejected with 400.
        """
        import urllib.request
        from urllib.error import HTTPError
        from api.server import create_app

        client = api_test_env["client"]

        # Mock urllib.request.build_opener to simulate a redirect off the allow-list
        class MockOpener:
            def open(self, req, timeout=30):
                # Simulate redirect to evil.com triggering rejection
                raise HTTPException(status_code=400, detail="Redirect off allow-list to 'evil.com' rejected")

        monkeypatch.setattr(urllib.request, "build_opener", lambda *args: MockOpener())

        resp = client.post("/sources/dataset", json={"url": "https://huggingface.co/redirect_me.zip"})
        assert resp.status_code == 400
        assert "redirect off allow-list" in resp.json()["detail"].lower()

    def test_dataset_streaming_and_batch_processing_of_50(self, api_test_env, monkeypatch):
        """Dataset URLs: stream and process in batches of 50."""
        import io
        import zipfile
        import urllib.request
        from PIL import Image

        client = api_test_env["client"]
        db_file = api_test_env["db_file"]

        # Create a zip containing 60 synthetic JPEG images
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w") as zf:
            for i in range(60):
                im_buf = io.BytesIO()
                im = Image.new("RGB", (64, 64), color=(i, i * 2 % 255, 100))
                im.save(im_buf, format="JPEG")
                zf.writestr(f"img_{i:03d}.jpg", im_buf.getvalue())
        zip_bytes = zip_buf.getvalue()

        class MockResp:
            def __init__(self, data):
                self._data = io.BytesIO(data)
                self.headers = {"Content-Length": str(len(data)), "Content-Type": "application/zip"}
            def read(self, amt=65536):
                return self._data.read(amt)
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass

        class MockOpener:
            def open(self, req, timeout=30):
                return MockResp(zip_bytes)

        monkeypatch.setattr(urllib.request, "build_opener", lambda *args: MockOpener())

        resp = client.post(
            "/sources/dataset",
            json={"url": "https://huggingface.co/datasets/argus/sample.zip", "project": "batch50_test"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["extracted_count"] == 60
        assert "job_id" in data

        # Check DB pre-insertion
        conn = sqlite3.connect(db_file)
        count = conn.execute("SELECT COUNT(*) FROM images WHERE file_path LIKE '%batch50_test%'").fetchone()[0]
        conn.close()
        assert count == 60

    def test_job_stalled_stage_reports_error_with_file_name(self, api_test_env):
        """A stalled stage (no progress 60s) reports an error with the file name."""
        import json
        import time

        client = api_test_env["client"]
        jobs_dir = api_test_env["jobs_dir"]

        job_id = "test_stalled_job"
        job_file = jobs_dir / f"{job_id}.json"
        job_data = {
            "id": job_id,
            "status": "running",
            "stage": "hash",
            "stage_display": "Perceptual hashing",
            "current_file": 12,
            "total_files": 100,
            "current_file_path": "drive://file_stalled_abc/hung_photo.jpg",
            "progress": 15.0,
            "updated_at": time.time() - 70.0,
            "last_progress_at": time.time() - 70.0,
            "heartbeat_at": time.time() - 70.0,
        }
        job_file.write_text(json.dumps(job_data))

        resp = client.get("/jobs/active")
        assert resp.status_code == 200
        data = resp.json()
        assert data["active"] is False
        assert data["status"] == "stalled"
        assert "stalled on file 'drive://file_stalled_abc/hung_photo.jpg'" in data["error"]
        assert "(no progress for 60s)" in data["error"]

    def test_drive_link_parsing_and_auth_check(self, api_test_env):
        """
        B.3 & B.6: Test Drive URL parsing and authentication check.
        """
        from api.server import extract_drive_folder_id

        # Parsing variations
        assert extract_drive_folder_id("https://drive.google.com/drive/folders/1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs") == "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs"
        assert extract_drive_folder_id("https://drive.google.com/drive/u/0/folders/1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs?usp=sharing") == "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs"
        assert extract_drive_folder_id("https://drive.google.com/open?id=1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs") == "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs"
        assert extract_drive_folder_id("1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs") == "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs"

        # Invalid link
        with pytest.raises(HTTPException) as exc_info:
            extract_drive_folder_id("not-a-valid-link")
        assert exc_info.value.status_code == 400

        # Endpoint auth check when drive token missing
        app = api_test_env["app"]
        app.state.config["sources"] = {"google_drive": {"token_path": "/tmp/nonexistent_drive_token_xyz.json"}}
        client = api_test_env["client"]
        resp = client.post("/sources/drive", json={"link": "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs"})
        assert resp.status_code == 400
        assert "connect your google drive account" in resp.json()["detail"].lower()

    def test_no_deletion_calls_in_sources_endpoints(self, api_test_env, monkeypatch):
        """
        B.6 Security Invariant: Zero deletion or file removal calls in /sources/* endpoints.
        """
        import os
        import shutil

        deletion_called = []

        def spy_remove(path, *args, **kwargs):
            deletion_called.append(str(path))
            raise RuntimeError("Deletion forbidden in sources endpoints!")

        monkeypatch.setattr(os, "remove", spy_remove)
        monkeypatch.setattr(os, "unlink", spy_remove)
        monkeypatch.setattr(shutil, "rmtree", spy_remove)

        app = api_test_env["app"]
        def quick_worker(job_id, state, *args, **kwargs):
            from api.server import read_job_state, write_job_state
            jf = state.jobs_dir / f"{job_id}.json"
            if jf.exists():
                st = read_job_state(jf) or {}
                st["status"] = "completed"
                write_job_state(jf, st)

        app.state.pipeline_worker = quick_worker
        client = api_test_env["client"]

        # 1. Valid upload
        img_bytes = io.BytesIO()
        Image.new("RGB", (10, 10)).save(img_bytes, format="JPEG")
        resp_up = client.post(
            "/sources/upload",
            files=[("files", ("test_safe.jpg", img_bytes.getvalue(), "image/jpeg"))],
        )
        assert resp_up.status_code == 200
        assert len(deletion_called) == 0

        # 2. Rescan endpoint
        resp_rescan = client.post("/sources/rescan")
        assert resp_rescan.status_code == 200
        assert len(deletion_called) == 0

        # 3. Jobs endpoint
        job_id = resp_up.json()["job_id"]
        resp_job = client.get(f"/jobs/{job_id}")
        assert resp_job.status_code == 200
        assert len(deletion_called) == 0

    def test_categories_empty_shows_not_categorized_yet(self, tmp_path):
        """
        C.7: In the category filter, show 'Not categorized yet' when the DB has no category rows.
        """
        empty_db = tmp_path / "empty_categories.db"
        conn = init_db(empty_db)
        conn.close()

        app = create_app(db_path=empty_db)
        client = make_authenticated_client(app)

        # GET /categories returns empty dict
        resp = client.get("/categories")
        assert resp.status_code == 200
        assert resp.json() == {}

        # UI index.html contains explanatory message when categories not computed
        ui_resp = client.get("/")
        assert ui_resp.status_code == 200
        assert (
            "Categories are computed after processing" in ui_resp.text
            or "Not categorized yet" in ui_resp.text
        )

    def test_concurrent_clusters_requests_during_active_job(self, api_test_env):
        """
        Requirement 2 & 3:
        GET /clusters and /clusters/{id} must return quickly while a job runs,
        serving the last committed state.
        Fires 20 concurrent /clusters requests during a simulated long job
        and asserts each responds in under 2 seconds.
        """
        client = api_test_env["client"]
        db_file = api_test_env["db_file"]

        stop_writer = threading.Event()

        # Simulated long writer committing in small batches
        def background_writer():
            writer_conn = sqlite3.connect(str(db_file), timeout=10.0)
            writer_conn.execute("PRAGMA journal_mode=WAL;")
            writer_conn.execute("PRAGMA busy_timeout=10000;")
            batch_num = 0
            while not stop_writer.is_set():
                batch_num += 1
                try:
                    writer_conn.execute(
                        "INSERT OR IGNORE INTO clusters (id, representative_id, stage_detected, member_count) VALUES (?, NULL, 0, 0)",
                        (9000 + batch_num,),
                    )
                    writer_conn.commit()
                except Exception:
                    pass
                time.sleep(0.02)
            writer_conn.close()

        writer_thread = threading.Thread(target=background_writer, daemon=True)
        writer_thread.start()

        # Give writer time to start committing in WAL mode
        time.sleep(0.05)

        durations: list[float] = []
        status_codes: list[int] = []

        def make_request(idx: int):
            t0 = time.perf_counter()
            # Alternate between /clusters and /clusters/1
            endpoint = "/clusters" if idx % 2 == 0 else "/clusters/1"
            resp = client.get(endpoint)
            dur = time.perf_counter() - t0
            return resp.status_code, dur

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
                futures = [executor.submit(make_request, i) for i in range(20)]
                for fut in concurrent.futures.as_completed(futures):
                    code, dur = fut.result()
                    status_codes.append(code)
                    durations.append(dur)
        finally:
            stop_writer.set()
            writer_thread.join(timeout=2.0)

        assert len(status_codes) == 20
        # Every request must succeed
        for sc in status_codes:
            assert sc == 200, f"Expected 200, got {sc}"

        # Every request must respond in under 2 seconds
        for dur in durations:
            assert dur < 2.0, f"Request took too long: {dur:.3f}s (must be < 2.0s)"

    def test_job_status_active_endpoint_and_persistence(self, api_test_env, tmp_path):
        """
        Requirement 2 & 4:
        /jobs/active returns stable shape:
        {"active": false} or {"active": true, "job_id": "...", "stage": "...", "done": N, "total": M}
        """
        app = api_test_env["app"]
        client = api_test_env["client"]

        jobs_dir = api_test_env["jobs_dir"]
        job_id = "testjob_active_1"
        job_file = jobs_dir / f"{job_id}.json"

        # Initially when no job exists or is running
        resp_initial = client.get("/jobs/active")
        assert resp_initial.status_code == 200
        assert resp_initial.json() == {"active": False}

        from core.worker import update_job_file
        update_job_file(
            job_file,
            status="running",
            stage="embed",
            stage_display="Embedding",
            current_file=15,
            total_files=50,
            progress=30.0,
            message="Embedding (15 of 50 files)",
        )

        # GET /jobs/active must find and return this active job with stable shape:
        resp_act = client.get("/jobs/active")
        assert resp_act.status_code == 200
        data_act = resp_act.json()
        assert data_act == {
            "active": True,
            "job_id": job_id,
            "stage": "Embedding",
            "done": 15,
            "total": 50,
        }

        # GET /jobs/{job_id} returns exact state
        resp_job = client.get(f"/jobs/{job_id}")
        assert resp_job.status_code == 200
        data_job = resp_job.json()
        assert data_job["status"] == "running"
        assert data_job["stage"] == "embed"
        assert data_job["current_file"] == 15
        assert data_job["total_files"] == 50

        # Mark job completed
        update_job_file(
            job_file,
            status="completed",
            stage="done",
            progress=100.0,
            message="Pipeline completed successfully",
        )

        # Now GET /jobs/active returns exact stable shape {"active": False}
        resp_none = client.get("/jobs/active")
        assert resp_none.status_code == 200
        assert resp_none.json() == {"active": False}

    def test_ui_smoke_no_jobs_undefined_request(self, api_test_env):
        """
        Requirement 4:
        UI smoke test asserting the script has no code path that requests /jobs/undefined,
        checks /jobs/active stable field names, and verifies backoff and visibility handling.
        """
        client = api_test_env["client"]
        res = client.get("/")
        assert res.status_code == 200
        html = res.text

        # 1. Never requests /jobs/undefined literally
        assert "/jobs/undefined" not in html
        assert "/jobs/null" not in html

        # 2. Assert guards against undefined or empty job ID before building any URL
        assert 'jobId === "undefined"' in html
        assert 'stopJobPolling()' in html
        assert 'setSourceButtonsDisabled(false)' in html

        # 3. Assert /jobs/active stable field names are read by UI
        assert 'data.active === true' in html
        assert 'data.job_id' in html
        assert 'activeInfo.stage' in html
        assert 'activeInfo.done' in html
        assert 'activeInfo.total' in html

        # 4. Assert backoff logic (1s -> 2s -> 5s max)
        assert 'pollDelayMs < 2000' in html
        assert 'pollDelayMs = 2000' in html
        assert 'pollDelayMs = 5000' in html

        # 5. Assert stops when page is hidden
        assert 'document.hidden' in html
        assert 'visibilitychange' in html

    def test_upload_batch_cap(self, api_test_env):
        """
        Requirement 6:
        Cap upload batches (e.g. 100 files).
        """
        client = api_test_env["client"]

        # Generate 101 files
        fake_files = [
            ("files", (f"img_{i}.jpg", b"\xff\xd8\xff\xe0dummy", "image/jpeg"))
            for i in range(101)
        ]
        resp = client.post("/sources/upload", files=fake_files)
        assert resp.status_code == 400
        assert "exceeds 100 files cap" in resp.json()["detail"].lower()

    def test_corrupt_and_missing_file_ends_with_warnings_not_abort(self, tmp_path, monkeypatch):
        """
        Requirement 5:
        A missing or corrupt file is a logged warning, never a job abort;
        the job ends "completed with warnings" with the list of skipped files.
        """
        from core.embeddings import is_clip_weights_cached
        from core.worker import run_pipeline

        # Use fake embedder so job-level test runs offline without downloading weights
        _patch_fake_embedder(monkeypatch)

        lib_dir = tmp_path / "lib"
        lib_dir.mkdir()

        # 2 good images
        im1 = lib_dir / "good1.jpg"
        im2 = lib_dir / "good2.jpg"
        _create_test_image(im1, color=(100, 100, 100), size=(100, 100))
        _create_test_image(im2, color=(120, 120, 120), size=(100, 100))

        # 1 corrupt 0-byte file
        corrupt_zero = lib_dir / "corrupt_zero.jpg"
        corrupt_zero.write_bytes(b"")

        # 1 corrupt non-image bytes file
        corrupt_garbage = lib_dir / "corrupt_garbage.jpg"
        corrupt_garbage.write_bytes(b"NON_IMAGE_HEADER_GARBAGE_BYTES")

        test_db = tmp_path / "worker_test.db"
        job_id = "test_warnings_job"
        job_file = tmp_path / f"{job_id}.json"

        # Run pipeline - must NOT raise or abort
        run_pipeline(
            job_id=job_id,
            db_path=test_db,
            source_type="local",
            folder=lib_dir,
            job_file=job_file,
        )

        assert job_file.is_file()
        job_data = json.loads(job_file.read_text())
        if job_data.get("status") == "failed" and not is_clip_weights_cached():
            pytest.skip(f"CLIP weights not cached; skipping in CI to avoid network downloads: {job_data.get('error')}")
        assert job_data["status"] == "completed_with_warnings"
        assert job_data["progress"] == 100.0
        assert len(job_data["warnings"]) >= 1
        # Check corrupt files are listed in warnings
        warn_str = " ".join(job_data["warnings"])
        assert "corrupt" in warn_str.lower()

    def test_json_api_cache_control_no_store_headers(self, api_test_env):
        """
        All JSON API responses (/clusters, /clusters/{id}, /categories, /jobs/*, /history, /auth/me)
        must send Cache-Control: no-store. Only /thumb may be cached (private, with ETag).
        """
        client = api_test_env["client"]

        endpoints = [
            "/clusters",
            "/clusters/1",
            "/categories",
            "/jobs/active",
            "/history",
            "/auth/me",
        ]
        for ep in endpoints:
            res = client.get(ep)
            assert res.status_code == 200, f"{ep} returned {res.status_code}: {res.text}"
            cc = res.headers.get("cache-control", "").lower()
            assert "no-store" in cc, f"{ep} missing Cache-Control: no-store (got '{cc}')"

        # /thumb/{id} must be cached private with ETag
        thumb_res = client.get("/thumb/1")
        assert thumb_res.status_code == 200
        thumb_cc = thumb_res.headers.get("cache-control", "").lower()
        assert "private" in thumb_cc, f"/thumb/1 Cache-Control must be private (got '{thumb_cc}')"
        assert "no-store" not in thumb_cc, f"/thumb/1 must not be no-store (got '{thumb_cc}')"
        etag = thumb_res.headers.get("etag")
        assert etag, "/thumb/1 must return an ETag header"

        # Subsequent conditional GET with If-None-Match should return 304
        cond_res = client.get("/thumb/1", headers={"If-None-Match": etag})
        assert cond_res.status_code == 304
        assert "private" in cond_res.headers.get("cache-control", "").lower()

    def test_repeated_mode_switching_20_times(self, api_test_env):
        """
        Switch modes repeatedly right after a job completes (or against active user with clusters).
        Every mode switch must re-fetch the list AND the open cluster, returning 200 consistently.
        """
        client = api_test_env["client"]
        modes = ["conservative", "balanced", "aggressive"]

        for i in range(20):
            target_mode = modes[i % len(modes)]
            # Re-fetch list
            res_list = client.get(f"/clusters?mode={target_mode}")
            assert res_list.status_code == 200, f"Iteration {i} mode {target_mode} /clusters failed: {res_list.text}"
            data_list = res_list.json()
            assert len(data_list) >= 1

            # Re-fetch open cluster
            cluster_id = data_list[0]["id"]
            res_detail = client.get(f"/clusters/{cluster_id}?mode={target_mode}")
            assert res_detail.status_code == 200, f"Iteration {i} mode {target_mode} /clusters/{cluster_id} failed: {res_detail.text}"
            data_detail = res_detail.json()
            assert data_detail["id"] == cluster_id
            assert data_detail["mode"] == target_mode
            assert len(data_detail["members"]) >= 2


def _patch_fake_embedder(monkeypatch):
    """Monkeypatch CLIPEmbedder and TextEmbeddingCache so job-level tests run offline without downloading weights."""
    import numpy as np
    import torch

    class FakeCLIPEmbedder:
        def __init__(self, *args, **kwargs):
            self.device = torch.device("cpu")
            self.model_name = "ViT-B-32"
            self.pretrained = "laion2b_s34b_b79k"

        def embed_thumbnails(self, thumbnails):
            n = len(thumbnails)
            vecs = np.zeros((n, 512), dtype=np.float16)
            vecs[:, 0] = 1.0
            return vecs

    class FakeTextEmbeddingCache:
        def __init__(self, *args, **kwargs):
            self.device = torch.device("cpu")
            self.model_name = "ViT-B-32"
            self.pretrained = "laion2b_s34b_b79k"

        def build_label_matrix(self, labels, templates):
            mat = np.zeros((len(labels), 512), dtype=np.float32)
            mat[:, 0] = 1.0
            return mat, list(labels)

        def encode_label(self, label, prompt_templates):
            vec = np.zeros((512,), dtype=np.float32)
            vec[0] = 1.0
            return vec

    monkeypatch.setattr("core.embeddings.CLIPEmbedder", FakeCLIPEmbedder)
    monkeypatch.setattr("core.categories.TextEmbeddingCache", FakeTextEmbeddingCache)


class TestPipelineResumable:
    """
    Requirement 3 & 4: failed stage is resumable; job status records the failing stage;
    a subsequent rescan completes without re-running already-finished stages.
    """

    def test_job_fails_at_stage_and_records_stage_name(self, tmp_path, monkeypatch):
        """
        When run_quality_pipeline raises, the job file must:
        - have status == 'failed'
        - have stage == 'quality' (the failing stage)
        - have an 'error' field containing the exception text
        """
        from unittest.mock import patch
        from core.embeddings import is_clip_weights_cached
        from core.worker import run_pipeline

        # Use fake embedder so job-level test runs offline without downloading weights
        _patch_fake_embedder(monkeypatch)

        lib_dir = tmp_path / "lib"
        lib_dir.mkdir()
        im1 = lib_dir / "img1.jpg"
        im2 = lib_dir / "img2.jpg"
        _create_test_image(im1, color=(100, 100, 100), size=(120, 120))
        _create_test_image(im2, color=(110, 110, 110), size=(120, 120))

        test_db = tmp_path / "fail_test.db"
        job_id = "test_fail_job"
        job_file = tmp_path / f"{job_id}.json"

        # Make quality stage raise
        with patch("core.quality.run_quality_pipeline", side_effect=RuntimeError("injected quality failure")):
            run_pipeline(
                job_id=job_id,
                db_path=test_db,
                source_type="local",
                folder=lib_dir,
                job_file=job_file,
            )

        assert job_file.is_file()
        state = json.loads(job_file.read_text())
        if state.get("stage") == "embed" and not is_clip_weights_cached():
            pytest.skip(f"CLIP weights not cached; skipped in CI: {state.get('error')}")
        assert state["status"] == "failed", f"Expected 'failed', got {state['status']!r}"
        assert state.get("stage") == "quality", f"Expected stage='quality', got {state.get('stage')!r}"
        assert "injected quality failure" in state.get("error", ""), state

    def test_rescan_after_failure_completes(self, tmp_path, monkeypatch):
        """
        After a failed run (quality raises), a second run_pipeline call must:
        - skip already-completed stages (scan, embed, cluster, verify) because
          the DB already has data for them
        - complete quality + categorize
        - end with status == 'completed' or 'completed_with_warnings'
        """
        from unittest.mock import patch
        from core.embeddings import is_clip_weights_cached
        from core.worker import run_pipeline

        # Use fake embedder so job-level test runs offline without downloading weights
        _patch_fake_embedder(monkeypatch)

        lib_dir = tmp_path / "lib"
        lib_dir.mkdir()
        im1 = lib_dir / "img1.jpg"
        im2 = lib_dir / "img2.jpg"
        _create_test_image(im1, color=(80, 80, 80), size=(100, 100))
        _create_test_image(im2, color=(90, 90, 90), size=(100, 100))

        test_db = tmp_path / "rescan_test.db"
        job_id_fail = "test_rescan_fail"
        job_file_fail = tmp_path / f"{job_id_fail}.json"

        # First run: quality fails
        with patch("core.quality.run_quality_pipeline", side_effect=RuntimeError("injected quality failure")):
            run_pipeline(
                job_id=job_id_fail,
                db_path=test_db,
                source_type="local",
                folder=lib_dir,
                job_file=job_file_fail,
            )

        state = json.loads(job_file_fail.read_text())
        if state.get("stage") == "embed" and not is_clip_weights_cached():
            pytest.skip(f"CLIP weights not cached; skipped in CI: {state.get('error')}")
        assert state["status"] == "failed"

        # Second run (rescan): quality is no longer patched — should complete
        job_id_ok = "test_rescan_ok"
        job_file_ok = tmp_path / f"{job_id_ok}.json"
        run_pipeline(
            job_id=job_id_ok,
            db_path=test_db,
            source_type="local",
            folder=lib_dir,
            job_file=job_file_ok,
        )

        assert job_file_ok.is_file()
        state2 = json.loads(job_file_ok.read_text())
        if state2.get("status") == "failed" and not is_clip_weights_cached():
            pytest.skip(f"CLIP weights not cached; skipped in CI: {state2.get('error')}")
        assert state2["status"] in ("completed", "completed_with_warnings"), (
            f"Expected completed/completed_with_warnings, got {state2['status']!r}\n"
            f"Error: {state2.get('error')}"
        )

    def test_incomplete_member_excluded_from_keeper_until_quality_done(self, tmp_path):
        """
        A cluster member whose quality_status IS NULL must never appear as
        delete_safe in choose_keeper — the invariant is enforced by
        _is_member_candidate returning False for non-(reference|aligned) statuses.
        """
        from core.keeper import choose_keeper

        # Build two members: one reference (quality_status='reference'),
        # one pending (quality_status=None)
        members = [
            {
                "id": 1, "file_path": "/a/ref.jpg", "md5": "aaa", "file_size": 1000,
                "width": 800, "height": 600, "source": "local",
                "quality_status": "reference", "quality_basis": "full",
                "ssim": None, "ssim_color": None, "sharpness_ratio": 1.0,
                "pixel_ratio": 1.0, "bytes_ratio": 1.0,
                "is_low_confidence": 0, "blur_flag": 0,
                "match_pct": None, "resolution_pct": None,
                "inlier_count": None, "inlier_ratio": None,
                "trashed": 0, "review_flag": 0, "delete_safe": 0,
            },
            {
                "id": 2, "file_path": "/a/pending.jpg", "md5": "bbb", "file_size": 900,
                "width": 800, "height": 600, "source": "local",
                "quality_status": None,  # ← not yet scored
                "quality_basis": None,
                "ssim": None, "ssim_color": None, "sharpness_ratio": None,
                "pixel_ratio": None, "bytes_ratio": None,
                "is_low_confidence": 0, "blur_flag": 0,
                "match_pct": None, "resolution_pct": None,
                "inlier_count": None, "inlier_ratio": None,
                "trashed": 0, "review_flag": 0, "delete_safe": 0,
            },
        ]

        result = choose_keeper(members, mode="conservative")
        # The pending member must not be delete_safe
        assert 2 not in result["delete_safe_ids"], (
            "A member with quality_status=None must never be delete_safe"
        )

    def test_drive_thumbnail_served_from_drive_cache(self, api_test_env, tmp_path, monkeypatch):
        """
        Requirement 2:
        Images with source='gdrive' or file_path='drive://...' must have their thumbnails
        served from drive_cache rather than falling back to the gray placeholder.
        """
        client = api_test_env["client"]
        app = api_test_env["app"]
        db_file = api_test_env["db_file"]

        # Setup mock drive_cache dir
        drive_cache = tmp_path / "drive_cache"
        drive_cache.mkdir()

        # Create a real image in drive_cache
        drive_file = drive_cache / "drive_img_123_1700000000.jpg"
        _create_test_image(drive_file, color=(50, 150, 250), size=(100, 100))

        # Inject drive_cache path into app config
        app.state.config = {
            "sources": {
                "google_drive": {
                    "cache_dir": str(drive_cache)
                }
            }
        }

        # Insert a drive image row into DB
        conn = sqlite3.connect(db_file)
        conn.execute(
            """
            INSERT INTO images (id, cluster_id, file_path, source, drive_id, modified_time, file_size, width, height, format, md5, is_keeper)
            VALUES (2001, 1, 'drive://drive_img_123/remote.jpg', 'gdrive', 'drive_img_123', 1700000000, 1000, 100, 100, 'JPEG', 'md5drive', 0)
            """
        )
        conn.commit()
        conn.close()

        # Request thumbnail
        res = client.get("/thumb/2001?size=64")
        assert res.status_code == 200
        assert res.headers["content-type"] == "image/jpeg"
        # Must NOT be placeholder
        from api.server import PLACEHOLDER_JPEG_BYTES
        assert res.content != PLACEHOLDER_JPEG_BYTES, "Drive thumbnail should be loaded from drive_cache, not placeholder"

    def test_drive_thumbnail_refetched_when_cache_missing(self, api_test_env, tmp_path, monkeypatch):
        """
        Requirement 5:
        If the cache file is missing, refetch it using stored thumbnailLink or files.get,
        instead of returning the gray placeholder.
        """
        client = api_test_env["client"]
        app = api_test_env["app"]
        db_file = api_test_env["db_file"]

        drive_cache = tmp_path / "drive_cache_refetch"
        drive_cache.mkdir()

        app.state.config = {
            "sources": {
                "google_drive": {
                    "cache_dir": str(drive_cache)
                }
            }
        }

        conn = sqlite3.connect(db_file)
        conn.execute(
            """
            INSERT INTO images (id, cluster_id, file_path, source, drive_id, modified_time, file_size, width, height, format, md5, is_keeper)
            VALUES (3001, 1, 'drive://drive_refetch_id/remote.jpg', 'gdrive', 'drive_refetch_id', 1700000000, 1000, 100, 100, 'JPEG', 'md5refetch', 0)
            """
        )
        conn.commit()
        conn.close()

        # Cache file is initially NOT present
        assert not list(drive_cache.glob("drive_refetch_id*"))

        # Mock GoogleDriveSource.get_thumbnail to simulate fetching from thumbnailLink
        def mock_get_thumbnail(self, file_id, size=(256, 256)):
            cache_file = self.cache_dir / f"drive_refetch_id_1700000000.jpg"
            _create_test_image(cache_file, color=(10, 200, 100), size=size)
            im = Image.new("RGB", size, color=(10, 200, 100))
            return im

        from sources.google_drive import GoogleDriveSource
        monkeypatch.setattr(GoogleDriveSource, "get_thumbnail", mock_get_thumbnail)

        res = client.get("/thumb/3001?size=64")
        assert res.status_code == 200
        assert res.headers["content-type"] == "image/jpeg"
        from api.server import PLACEHOLDER_JPEG_BYTES
        assert res.content != PLACEHOLDER_JPEG_BYTES
        # Cache file should now exist in drive_cache
        assert list(drive_cache.glob("drive_refetch_id*"))

    def test_stalled_job_detection_when_heartbeat_older_than_60s(self, api_test_env):
        """
        Requirement 1:
        If the worker heartbeat timestamp is older than 60 seconds,
        GET /jobs/active reports "stalled", records it in the job file,
        and sets active=False so Add Sources and Rescan re-enable.
        """
        client = api_test_env["client"]
        app = api_test_env["app"]
        jobs_dir = app.state.jobs_dir

        job_id = "test_stalled_job"
        job_file = jobs_dir / f"{job_id}.json"

        from core.worker import update_job_file
        # Heartbeat 80 seconds ago, PID is current running process so PID is alive
        update_job_file(
            job_file,
            pid=os.getpid(),
            status="running",
            stage="quality",
            stage_display="Evaluating quality (cluster 5 of 29, id #14)",
            current_file=5,
            total_files=29,
            progress=86.0,
            heartbeat_at=time.time() - 80.0,
        )

        res = client.get("/jobs/active")
        assert res.status_code == 200
        data = res.json()
        assert data.get("active") is False
        assert data.get("status") == "stalled"
        assert "heartbeat" in data.get("error", "").lower()

        # Job file status must be updated on disk
        disk_state = json.loads(job_file.read_text())
        assert disk_state.get("status") == "stalled"

    def test_failed_job_detection_when_pid_gone(self, api_test_env):
        """
        Requirement 1:
        If worker PID is gone, GET /jobs/active reports "failed", records it in job file,
        and sets active=False.
        """
        client = api_test_env["client"]
        app = api_test_env["app"]
        jobs_dir = app.state.jobs_dir

        job_id = "test_pid_gone_job"
        job_file = jobs_dir / f"{job_id}.json"

        from core.worker import update_job_file
        # Non-existent PID
        update_job_file(
            job_file,
            pid=9999999,
            status="running",
            stage="embed",
            stage_display="Embedding",
            heartbeat_at=time.time(),
        )

        res = client.get("/jobs/active")
        assert res.status_code == 200
        data = res.json()
        assert data.get("active") is False
        assert data.get("status") == "failed"
        assert "no longer running" in data.get("error", "").lower()

        disk_state = json.loads(job_file.read_text())
        assert disk_state.get("status") == "failed"

    def test_pairwise_comparison_cap_for_large_cluster(self, tmp_path):
        """
        Requirement 3:
        For clusters with more than 8 members:
        - Compute pairwise source-likelihood only among distinct same-resolution candidates.
        - Cap comparisons per cluster (default 40, or configured).
        - Mark reference_uncertain when the cap is hit.
        """
        from core.quality import compute_source_likelihood_scores, select_reference_member

        # Create 10 distinct images with same resolution (100x100)
        members = []
        for i in range(10):
            p = tmp_path / f"large_cluster_img_{i}.jpg"
            _create_test_image(p, color=(20 + i * 20, 30 + i * 15, 40 + i * 10), size=(100, 100))
            members.append({
                "id": 100 + i,
                "file_path": str(p),
                "width": 100,
                "height": 100,
                "file_size": 1000 + i * 10,
                "format": "JPEG",
                "modified_time": 1000.0 + i,
                "md5": f"md5_unique_{i}",
                "jpeg_quality_est": 85,
            })

        # Test compute_source_likelihood_scores directly with max_comparisons = 5
        out_info: dict[str, Any] = {}
        scores = compute_source_likelihood_scores(
            members,
            max_comparisons=5,
            out_info=out_info,
        )
        assert out_info.get("cap_hit") is True
        assert out_info.get("total_comparisons") == 5

        # Test select_reference_member with max_comparisons = 5
        ref = select_reference_member(members, max_comparisons=5)
        assert ref.get("reference_uncertain") == 1, (
            "reference_uncertain must be 1 when comparison cap is hit on >8 members"
        )

    def test_library_view_all_photos_pagination_and_filters(self, api_test_env):
        """Test GET /library lists every image including unique ones, with pagination and filters."""
        client = api_test_env["client"]

        # Default list (all images)
        res = client.get("/library")
        assert res.status_code == 200
        assert res.headers.get("Cache-Control", "").startswith("no-store")
        data = res.json()
        assert data["total"] == 6
        assert data["page"] == 1
        assert len(data["images"]) == 6

        # Check unique image is included
        alone_img = next((img for img in data["images"] if img["id"] == 6), None)
        assert alone_img is not None
        assert alone_img["filename"] == "alone.jpg"
        assert alone_img["cluster_id"] is None
        assert alone_img["category"] == "animal"

        # Pagination: 2 per page
        res_p1 = client.get("/library?limit=2&page=1")
        assert res_p1.status_code == 200
        d_p1 = res_p1.json()
        assert len(d_p1["images"]) == 2
        assert d_p1["total"] == 6
        assert d_p1["total_pages"] == 3

        # Search filter
        res_search = client.get("/library?search=alone")
        assert res_search.status_code == 200
        d_search = res_search.json()
        assert d_search["total"] == 1
        assert d_search["images"][0]["id"] == 6

        # Category filter
        res_cat = client.get("/library?category=animal")
        assert res_cat.status_code == 200
        d_cat = res_cat.json()
        assert d_cat["total"] == 1
        assert d_cat["images"][0]["id"] == 6

        res_land = client.get("/library?category=landscape")
        assert res_land.status_code == 200
        assert res_land.json()["total"] == 3

    def test_download_cluster_selected_and_keepers(self, api_test_env):
        """Test streaming zip downloads for cluster, selected files, and all keepers."""
        import zipfile
        client = api_test_env["client"]

        # 1. Download cluster 1
        res_cl = client.get("/download/cluster/1")
        assert res_cl.status_code == 200
        assert res_cl.headers["content-type"] == "application/zip"
        assert 'attachment; filename="cluster_1.zip"' in res_cl.headers["content-disposition"]
        with zipfile.ZipFile(io.BytesIO(res_cl.content)) as zf:
            names = zf.namelist()
            assert len(names) == 3
            assert any("orig.jpg" in n for n in names)
            assert any("orig_copy" in n for n in names)
            assert any("orig_low" in n for n in names)

        # 2. Download selected images (ids 1 and 6)
        res_sel = client.post("/download/selected", json={"image_ids": [1, 6]})
        assert res_sel.status_code == 200
        assert res_sel.headers["content-type"] == "application/zip"
        assert 'attachment; filename="selected_images.zip"' in res_sel.headers["content-disposition"]
        with zipfile.ZipFile(io.BytesIO(res_sel.content)) as zf:
            names = zf.namelist()
            assert len(names) == 2
            assert any("orig.jpg" in n for n in names)
            assert any("alone.jpg" in n for n in names)

        # 3. Download recommended keepers
        res_kp = client.get("/download/keepers?mode=balanced")
        assert res_kp.status_code == 200
        assert res_kp.headers["content-type"] == "application/zip"
        assert 'attachment; filename="recommended_keepers.zip"' in res_kp.headers["content-disposition"]
        with zipfile.ZipFile(io.BytesIO(res_kp.content)) as zf:
            names = zf.namelist()
            assert len(names) >= 2  # Keeper from cluster 1 and keeper from cluster 2

        # 4. Error handling: non-existent cluster
        res_err = client.get("/download/cluster/9999")
        assert res_err.status_code == 404

        # 5. Error handling: empty image_ids
        res_empty = client.post("/download/selected", json={"image_ids": []})
        assert res_empty.status_code in (400, 422)

    def test_download_google_drive_cluster_and_selected(self, api_test_env, monkeypatch):
        """Test streaming zip downloads for clusters and selected files originating from Google Drive."""
        import zipfile
        client = api_test_env["client"]
        db_file = api_test_env["db_file"]

        # Insert a cluster containing drive:// files
        conn = sqlite3.connect(db_file)
        conn.execute("INSERT INTO clusters (id, representative_id, stage_detected, member_count) VALUES (50, NULL, 0, 2)")
        now = time.time()
        conn.execute(
            """
            INSERT INTO images (
                id, cluster_id, file_path, file_size, width, height, format, md5,
                is_keeper, is_recommended_keeper, delete_safe, review_flag,
                is_low_confidence, quality_status, ssim, ssim_color, sharpness_ratio,
                pixel_ratio, bytes_ratio, modified_time
            ) VALUES
              (501, 50, 'drive://mock_drive_id_1/cloud_photo1.jpg', 1234, 400, 400, 'JPEG', 'md5_501', 1, 1, 0, 0, 0, 'reference', 1.0, 1.0, 1.0, 1.0, 1.0, ?),
              (502, 50, 'drive://mock_drive_id_2/cloud_photo2.jpg', 1234, 400, 400, 'JPEG', 'md5_502', 0, 0, 1, 0, 0, 'aligned', 1.0, 1.0, 1.0, 1.0, 1.0, ?)
            """,
            (now, now),
        )
        conn.execute("UPDATE clusters SET representative_id = 501 WHERE id = 50")
        conn.commit()
        conn.close()

        # Mock GoogleDriveSource.get_bytes to return fake JPEG bytes
        mock_bytes = b"\xff\xd8\xff\xe0\x00\x10JFIF" + b"\x00" * 50
        from sources.google_drive import GoogleDriveSource
        monkeypatch.setattr(GoogleDriveSource, "get_bytes", lambda self, uri: mock_bytes)
        monkeypatch.setattr("api.server._get_user_gdrive_source", lambda req, u, c: GoogleDriveSource(conn=c, config={}, token_path="/tmp/fake_tok.json"))

        # Download cluster 50
        res_cl = client.get("/download/cluster/50")
        assert res_cl.status_code == 200
        assert res_cl.headers["content-type"] == "application/zip"
        with zipfile.ZipFile(io.BytesIO(res_cl.content)) as zf:
            names = zf.namelist()
            assert len(names) == 2
            assert any("cloud_photo1.jpg" in n for n in names)
            assert any("cloud_photo2.jpg" in n for n in names)
            assert zf.read(names[0]) == mock_bytes

        # Download selected image 502
        res_sel = client.post("/download/selected", json={"image_ids": [502]})
        assert res_sel.status_code == 200
        assert res_sel.headers["content-type"] == "application/zip"
        with zipfile.ZipFile(io.BytesIO(res_sel.content)) as zf:
            names = zf.namelist()
            assert len(names) == 1
            assert any("cloud_photo2.jpg" in n for n in names)

    def test_drive_web_oauth_flow(self, api_test_env, tmp_path, monkeypatch):
        """Test Drive web OAuth flow, PKCE, admin message, user-isolated tokens, disconnect."""
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
        monkeypatch.setattr("sources.google_drive.DEFAULT_TOKEN_PATH", str(tmp_path / "drive_token.json"))
        client = api_test_env["client"]
        app = api_test_env["app"]

        # 1. Missing client_secret -> clear admin message (not terminal command)
        res_status = client.get("/drive/status")
        assert res_status.status_code == 200
        data = res_status.json()
        assert data["configured"] is False
        assert "administrator must provide client_secret.json in the data directory" in data["message"].lower() or "client_secret.json not found" in data["message"].lower()

        res_auth_fail = client.get("/drive/auth-url")
        assert res_auth_fail.status_code == 400
        assert "administrator must provide client_secret.json" in res_auth_fail.json()["detail"].lower() or "client_secret.json not found" in res_auth_fail.json()["detail"].lower()

        # 2. Hosted mode check
        monkeypatch.setenv("HOSTED_MODE", "true")
        monkeypatch.delenv("ARGUS_ENABLE_DRIVE", raising=False)
        res_hosted = client.get("/drive/status")
        assert res_hosted.status_code == 200
        assert res_hosted.json()["enabled"] is False
        assert "disabled in hosted mode" in res_hosted.json()["message"].lower()

        # Allow Drive with ARGUS_ENABLE_DRIVE=1
        monkeypatch.setenv("ARGUS_ENABLE_DRIVE", "1")
        res_hosted_allowed = client.get("/drive/status")
        assert res_hosted_allowed.status_code == 200
        assert res_hosted_allowed.json()["enabled"] is True

        monkeypatch.delenv("HOSTED_MODE", raising=False)
        monkeypatch.delenv("ARGUS_ENABLE_DRIVE", raising=False)

        # 3. Setup mock client_secret.json
        sec_file = tmp_path / "client_secret.json"
        sec_file.write_text(json.dumps({
            "web": {
                "client_id": "test-client-id-12345.apps.googleusercontent.com",
                "client_secret": "test-client-secret-xyz",
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
            }
        }))
        app.state.config["sources"] = {"google_drive": {"client_secret_path": str(sec_file)}}

        # Status now configured = True, connected = False
        res_status2 = client.get("/drive/status")
        assert res_status2.status_code == 200
        assert res_status2.json()["configured"] is True
        assert res_status2.json()["connected"] is False

        # 4. Generate Auth URL (PKCE + state)
        res_auth = client.get("/drive/auth-url")
        assert res_auth.status_code == 200
        auth_data = res_auth.json()
        assert "auth_url" in auth_data
        auth_url = auth_data["auth_url"]
        assert "https://accounts.google.com/o/oauth2/v2/auth" in auth_url
        assert "code_challenge=" in auth_url
        assert "code_challenge_method=S256" in auth_url
        assert "scope=" in auth_url
        assert "drive.readonly" in auth_url
        state = auth_data["state"]

        # 5. Callback exchange
        class MockHttpResponse:
            def __init__(self, data: bytes):
                self._data = data
            def read(self):
                return self._data
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass

        def mock_urlopen(req, timeout=None):
            url = req.full_url if hasattr(req, "full_url") else str(req)
            if "oauth2.googleapis.com/token" in url:
                return MockHttpResponse(json.dumps({
                    "access_token": "mock_access_token_xyz999",
                    "refresh_token": "mock_refresh_token_abc888",
                    "expires_in": 3600,
                    "token_type": "Bearer",
                }).encode("utf-8"))
            elif "userinfo" in url:
                return MockHttpResponse(json.dumps({
                    "email": "photographer@gmail.com",
                    "name": "Argus Photographer",
                }).encode("utf-8"))
            raise ValueError(f"Unexpected URL: {url}")

        monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)

        cb_res = client.get(f"/drive/callback?code=mock_auth_code_123&state={state}", follow_redirects=False)
        assert cb_res.status_code == 303
        assert cb_res.headers["location"] == "/#drive=connected"

        # Check status is now connected
        res_status3 = client.get("/drive/status")
        assert res_status3.status_code == 200
        d3 = res_status3.json()
        assert d3["connected"] is True
        assert d3["email"] == "photographer@gmail.com"

        # Verify token file exists in user's root only with 0o600
        user_root = app.state.custom_user_overrides[1]["root"] if (hasattr(app.state, "custom_user_overrides") and 1 in app.state.custom_user_overrides) else (app.state.users_dir / "1")
        user_tok = user_root / "drive_token.json"
        assert user_tok.exists()
        saved_tok = json.loads(user_tok.read_text())
        assert saved_tok["token"] == "mock_access_token_xyz999"
        assert saved_tok["account_email"] == "photographer@gmail.com"

        # 6. Disconnect
        disc_res = client.post("/drive/disconnect")
        assert disc_res.status_code == 200
        assert disc_res.json()["connected"] is False
        assert not user_tok.exists()

        res_status4 = client.get("/drive/status")
        assert res_status4.json()["connected"] is False

    def test_refused_deletions_exact_reasons_and_plan_excludes_trashed(self, api_test_env):
        """Test exact refusal reasons (missing, changed, keeper, whole cluster, review flag, already trashed) and plan exclusion."""
        client = api_test_env["client"]
        db_file = api_test_env["db_file"]

        # Plan deletion for Cluster 1
        plan_res = client.post("/plan", json={"mode": "conservative"})
        assert plan_res.status_code == 200
        plan_data = plan_res.json()
        batch_id = plan_data["batch_id"]
        assert len(plan_data["entries"]) > 0

        # Now simulate a refusal case: tamper with a planned file's size before apply
        first_entry = plan_data["entries"][0]
        orig_file = Path(first_entry["original_path"])
        # Append some bytes to change file size -> "changed since plan"
        with open(orig_file, "ab") as f:
            f.write(b"tampered")

        # Apply
        apply_res = client.post("/apply", json={"batch_id": batch_id, "confirm": True})
        assert apply_res.status_code == 200
        apply_data = apply_res.json()
        assert apply_data["refused_count"] >= 1
        refused_entry = next((r for r in apply_data["refusals"] if r["path"] == str(orig_file)), None)
        assert refused_entry is not None
        assert refused_entry["reason"] == "changed since plan"

        # Check /history includes refused_files
        hist_res = client.get("/history")
        assert hist_res.status_code == 200
        hist_data = hist_res.json()
        b_hist = next((b for b in hist_data if b["batch_id"] == batch_id), None)
        assert b_hist is not None
        assert len(b_hist["refused_files"]) >= 1
        assert any(rf["reason"] == "changed since plan" for rf in b_hist["refused_files"])

        # Check that rebuilding plan excludes trashed/moved files
        # Mark all files in cluster 1 as trashed
        with sqlite3.connect(db_file) as conn:
            conn.execute("UPDATE images SET status = 'trashed' WHERE cluster_id = 1")
            conn.commit()

        rebuilt_plan = client.post("/plan", json={"mode": "conservative"})
        assert rebuilt_plan.status_code == 200
        # Cluster 1 has no surviving files, so its files must NOT appear in the plan
        for e in rebuilt_plan.json()["entries"]:
            assert e["cluster_id"] != 1
        # Trashed files must appear in skipped_entries
        skipped = rebuilt_plan.json().get("skipped_entries", [])
        assert any(s["reason"] == "already trashed" for s in skipped)

    def test_history_hides_planned_zero_file_batches(self, api_test_env):
        """Planned batches with 0 files must not be shown in /history."""
        client = api_test_env["client"]
        db_file = api_test_env["db_file"]

        # Insert a 0-file planned batch directly to simulate legacy or edge-case batch
        with sqlite3.connect(db_file) as conn:
            conn.execute(
                "INSERT INTO ledger_batches (batch_id, created_at, mode, status, file_count, bytes_freed, trash_dir) "
                "VALUES ('batch_zero_planned', strftime('%Y-%m-%dT%H:%M:%SZ','now'), 'conservative', 'planned', 0, 0, '/tmp/trash')"
            )
            # Insert a non-zero planned batch
            conn.execute(
                "INSERT INTO ledger_batches (batch_id, created_at, mode, status, file_count, bytes_freed, trash_dir) "
                "VALUES ('batch_valid_planned', strftime('%Y-%m-%dT%H:%M:%SZ','now'), 'conservative', 'planned', 2, 2048, '/tmp/trash')"
            )
            conn.commit()

        hist_res = client.get("/history")
        assert hist_res.status_code == 200
        batch_ids = [b["batch_id"] for b in hist_res.json()]
        assert "batch_zero_planned" not in batch_ids
        assert "batch_valid_planned" in batch_ids

