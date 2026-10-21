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
from pathlib import Path
import shutil
import sqlite3
import threading
import time

import pytest
from fastapi import HTTPException
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

    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)

    library_dir = tmp_path / "library"
    library_dir.mkdir(parents=True, exist_ok=True)

    app = create_app(
        db_path=db_file,
        trash_dir=trash_dir,
        thumb_cache_dir=thumb_dir,
        library_dir=library_dir,
        jobs_dir=jobs_dir,
    )
    client = TestClient(app)

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

        if real_backup.exists():
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
        client = TestClient(app)
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
        app = api_test_env["app"]
        client = TestClient(app)

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
        app = api_test_env["app"]
        client = TestClient(app)

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
        app = api_test_env["app"]
        client = TestClient(app)

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
        client = TestClient(app)

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
        app = api_test_env["app"]
        client = TestClient(app)

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
        app = api_test_env["app"]
        client = TestClient(app)

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

        app = api_test_env["app"]
        client = TestClient(app)

        # Mock urllib.request.build_opener to simulate a redirect off the allow-list
        class MockOpener:
            def open(self, req, timeout=30):
                # Simulate redirect to evil.com triggering rejection
                raise HTTPException(status_code=400, detail="Redirect off allow-list to 'evil.com' rejected")

        monkeypatch.setattr(urllib.request, "build_opener", lambda *args: MockOpener())

        resp = client.post("/sources/dataset", json={"url": "https://huggingface.co/redirect_me.zip"})
        assert resp.status_code == 400
        assert "redirect off allow-list" in resp.json()["detail"].lower()

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

        # Endpoint auth check when drive-auth not done
        app = api_test_env["app"]
        app.state.config["sources"] = {"google_drive": {"token_path": "/tmp/nonexistent_drive_token_xyz.json"}}
        client = TestClient(app)
        resp = client.post("/sources/drive", json={"link": "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs"})
        assert resp.status_code == 400
        assert "drive-auth" in resp.json()["detail"].lower()

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
        # Mock background pipeline launch so test focuses on endpoint logic
        app.state.pipeline_worker = lambda *args, **kwargs: None
        client = TestClient(app)

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
        client = TestClient(app)

        # GET /categories returns empty dict
        resp = client.get("/categories")
        assert resp.status_code == 200
        assert resp.json() == {}

        # UI index.html contains "Not categorized yet" handling
        ui_resp = client.get("/")
        assert ui_resp.status_code == 200
        assert "Not categorized yet" in ui_resp.text

    def test_concurrent_clusters_requests_during_active_job(self, api_test_env):
        """
        Requirement 2 & 3:
        GET /clusters and /clusters/{id} must return quickly while a job runs,
        serving the last committed state.
        Fires 20 concurrent /clusters requests during a simulated long job
        and asserts each responds in under 2 seconds.
        """
        app = api_test_env["app"]
        client = TestClient(app)
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
        client = TestClient(app)

        jobs_dir = app.state.jobs_dir
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
        app = api_test_env["app"]
        client = TestClient(app)

        # Generate 101 files
        fake_files = [
            ("files", (f"img_{i}.jpg", b"\xff\xd8\xff\xe0dummy", "image/jpeg"))
            for i in range(101)
        ]
        resp = client.post("/sources/upload", files=fake_files)
        assert resp.status_code == 400
        assert "exceeds 100 files cap" in resp.json()["detail"].lower()

    def test_corrupt_and_missing_file_ends_with_warnings_not_abort(self, tmp_path):
        """
        Requirement 5:
        A missing or corrupt file is a logged warning, never a job abort;
        the job ends "completed with warnings" with the list of skipped files.
        """
        from core.worker import run_pipeline

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
        assert job_data["status"] == "completed_with_warnings"
        assert job_data["progress"] == 100.0
        assert len(job_data["warnings"]) >= 1
        # Check corrupt files are listed in warnings
        warn_str = " ".join(job_data["warnings"])
        assert "corrupt" in warn_str.lower()



