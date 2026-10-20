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

import hashlib
from pathlib import Path
import shutil
import sqlite3
import time

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api.server import create_app
from core.ledger import compute_file_md5, ensure_ledger_tables
from db.connection import init_db


def _create_test_image(path: Path, color: tuple[int, int, int] = (100, 100, 100), size: tuple[int, int] = (200, 200)) -> None:
    im = Image.new("RGB", size, color=color)
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path, format="JPEG", quality=90)


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

    app = create_app(
        db_path=db_file,
        trash_dir=trash_dir,
        thumb_cache_dir=thumb_dir,
    )
    client = TestClient(app)

    return {
        "client": client,
        "app": app,
        "db_file": db_file,
        "trash_dir": trash_dir,
        "thumb_dir": thumb_dir,
        "img_dir": img_dir,
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
        assert "not found" in res_404.json()["detail"].lower()

    def test_thumbnail_caching_and_id_only(self, api_test_env):
        """GET /thumb/{image_id} generates and caches JPEG thumbnail. Only accepts image_id."""
        client = api_test_env["client"]
        thumb_dir = api_test_env["thumb_dir"]

        res = client.get("/thumb/1?size=128")
        assert res.status_code == 200
        assert res.headers["content-type"] == "image/jpeg"
        assert len(res.content) > 0

        cached_file = thumb_dir / "1_128.jpg"
        assert cached_file.exists()

        # Second request serves from cache
        res2 = client.get("/thumb/1?size=128")
        assert res2.status_code == 200
        assert len(res2.content) == len(res.content)

        # 404 for non-existent image id
        res_404 = client.get("/thumb/99999")
        assert res_404.status_code == 404
        assert "not found" in res_404.json()["detail"].lower()

    def test_categories_endpoint(self, api_test_env):
        """GET /categories returns label counts."""
        client = api_test_env["client"]
        res = client.get("/categories")
        assert res.status_code == 200
        data = res.json()
        assert data.get("landscape") == 3
        assert data.get("food") == 2
        assert data.get("animal") == 1

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
        assert demo_db_orig.exists(), f"Real demo DB not found at {demo_db_orig}"

        # Copy demo.db into tmp_path
        demo_db_copy = tmp_path / "demo_real_copy.db"
        shutil.copy2(demo_db_orig, demo_db_copy)

        thumb_dir = tmp_path / "thumbs_demo_real"
        app = create_app(db_path=demo_db_copy, thumb_cache_dir=thumb_dir)
        client = TestClient(app)

        conn = sqlite3.connect(demo_db_copy)
        image_ids = [row[0] for row in conn.execute("SELECT id FROM images").fetchall()]
        conn.close()

        assert len(image_ids) == 36, f"Expected 36 demo images, found {len(image_ids)}"

        # 20 concurrent workers requesting thumbnails
        tasks = []
        for img_id in image_ids:
            tasks.append((img_id, 96))
            tasks.append((img_id, 320))

        def fetch_thumb(item):
            iid, sz = item
            resp = client.get(f"/thumb/{iid}?size={sz}")
            return resp.status_code, resp.headers.get("content-type")

        with ThreadPoolExecutor(max_workers=20) as executor:
            results = list(executor.map(fetch_thumb, tasks))

        assert len(results) == 72
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

        if real_backup.exists():
            shutil.copyfile(real_backup, test_db)
            raw_conn = sqlite3.connect(test_db)
            cols_before = {r[1] for r in raw_conn.execute("PRAGMA table_info(images)").fetchall()}
            raw_conn.close()
            assert "drive_id" not in cols_before
        else:
            raw_conn = sqlite3.connect(test_db)
            raw_conn.executescript(
                """
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
                VALUES (1, '/p/1.jpg', 1000, 100, 100, 'JPEG', 'md5_1', 1, 1, '2026-01-01', '2026-01-01');
                INSERT INTO images (id, file_path, file_size, width, height, format, md5, cluster_id, is_keeper, created_at, updated_at)
                VALUES (2, '/p/2.jpg', 1000, 100, 100, 'JPEG', 'md5_1', 1, 0, '2026-01-01', '2026-01-01');
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
        client = TestClient(app)
        resp = client.get("/clusters")
        assert resp.status_code == 200
        clusters = resp.json()
        assert isinstance(clusters, list)
        assert len(clusters) > 0
        assert "id" in clusters[0]
        assert "keeper" in clusters[0]



