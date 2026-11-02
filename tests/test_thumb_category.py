"""
tests/test_thumb_category.py — Tests for thumbnail isolation and category counts.

1. Two users with the same image id see their own thumbnails (different bytes).
2. Deleting and recreating an account never serves an old (stale) cached thumbnail.
3. /categories count matches the number of clusters returned by /clusters?category=<cat>.
4. A cluster with mixed-category members shows "mixed" in the cluster response.
"""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api.server import create_app
from core.ledger import compute_file_md5, ensure_ledger_tables
from db.connection import init_db


def _make_jpg(path: Path, color: tuple[int, int, int], size: tuple[int, int] = (100, 100)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=color).save(path, format="JPEG", quality=85)


def _make_app(tmp_path: Path) -> tuple:
    """Create a fresh app with auth. Returns (app, users_dir, auth_db)."""
    auth_db = tmp_path / "auth.db"
    users_dir = tmp_path / "users"
    users_dir.mkdir(exist_ok=True)
    app = create_app(
        auth_db_path=auth_db,
        users_dir=users_dir,
    )
    return app, users_dir, auth_db


def _user_db(user_dir: Path) -> Path:
    similix_db = user_dir / "similix.db"
    return similix_db if similix_db.exists() else (user_dir / "argus.db")


def _register_login(client: TestClient, email: str, password: str = "ArgusPass1234!") -> str:
    client.post("/auth/register", json={"email": email, "password": password})
    res = client.post("/auth/login", json={"email": email, "password": password})
    data = res.json()
    csrf = data.get("csrf_token", "")
    tok = data.get("token", "")
    if tok:
        client.cookies.set("argus_session", tok)
    if csrf:
        client.headers["X-CSRF-Token"] = csrf
    return csrf


class TestThumbnailIsolation:

    def test_different_users_same_image_id_get_different_thumbnails(self, tmp_path: Path):
        """Two users uploading their own images that get id=1 must receive distinct thumbnails."""
        app, users_dir, auth_db = _make_app(tmp_path)

        # --- User A ---
        client_a = TestClient(app, raise_server_exceptions=True)
        _register_login(client_a, "a@argus.test")
        client_a.get("/auth/me")

        user_a_dir = users_dir / "1"
        img_a = user_a_dir / "library" / "red.jpg"
        _make_jpg(img_a, color=(200, 0, 0))
        md5_a = compute_file_md5(img_a)

        conn_a = init_db(_user_db(user_a_dir))
        conn_a.execute(
            "INSERT INTO images (id, file_path, file_size, modified_time, md5) VALUES (1, ?, ?, ?, ?)",
            (str(img_a), img_a.stat().st_size, img_a.stat().st_mtime, md5_a),
        )
        conn_a.commit()
        conn_a.close()

        thumb_res_a = client_a.get("/thumb/1?size=64")
        assert thumb_res_a.status_code == 200
        assert thumb_res_a.headers.get("Cache-Control") == "private, max-age=3600"
        assert thumb_res_a.headers.get("ETag") == f'"{md5_a}"'
        bytes_a = thumb_res_a.content

        # --- User B ---
        client_b = TestClient(app, raise_server_exceptions=True)
        _register_login(client_b, "b@argus.test")
        client_b.get("/auth/me")

        user_b_dir = users_dir / "2"
        img_b = user_b_dir / "library" / "blue.jpg"
        _make_jpg(img_b, color=(0, 0, 200))
        md5_b = compute_file_md5(img_b)

        conn_b = init_db(_user_db(user_b_dir))
        conn_b.execute(
            "INSERT INTO images (id, file_path, file_size, modified_time, md5) VALUES (1, ?, ?, ?, ?)",
            (str(img_b), img_b.stat().st_size, img_b.stat().st_mtime, md5_b),
        )
        conn_b.commit()
        conn_b.close()

        thumb_res_b = client_b.get("/thumb/1?size=64")
        assert thumb_res_b.status_code == 200
        assert thumb_res_b.headers.get("Cache-Control") == "private, max-age=3600"
        assert thumb_res_b.headers.get("ETag") == f'"{md5_b}"'
        bytes_b = thumb_res_b.content

        # They must differ — different source images
        assert bytes_a != bytes_b, "Users A and B got identical thumbnail bytes for their own id=1 images"

    def test_stale_cache_not_reused_after_account_delete_recreate(self, tmp_path: Path):
        """After deleting user A and registering again, old cached thumbnail must not appear."""
        app, users_dir, auth_db = _make_app(tmp_path)
        password = "ArgusPass1234!"
        email = "cycle@argus.test"

        # Register + create red image
        client = TestClient(app, raise_server_exceptions=True)
        _register_login(client, email, password)
        client.get("/auth/me")

        user_dir = users_dir / "1"
        img_red = user_dir / "library" / "red.jpg"
        _make_jpg(img_red, color=(255, 0, 0))
        md5_red = compute_file_md5(img_red)

        conn = init_db(_user_db(user_dir))
        conn.execute(
            "INSERT INTO images (id, file_path, file_size, modified_time, md5) VALUES (1, ?, ?, ?, ?)",
            (str(img_red), img_red.stat().st_size, img_red.stat().st_mtime, md5_red),
        )
        conn.commit()
        conn.close()

        # Fetch thumbnail to populate cache
        t1 = client.get("/thumb/1?size=64")
        assert t1.status_code == 200
        bytes_before = t1.content
        assert (user_dir / "thumbs").exists()
        assert len(list((user_dir / "thumbs").glob("*.jpg"))) > 0

        # Delete account
        del_res = client.post(
            "/auth/delete-account",
            json={"password": password},
        )
        assert del_res.status_code == 200
        assert not (user_dir / "thumbs").exists()

        # Re-register and create a BLUE image (different content)
        client2 = TestClient(app, raise_server_exceptions=True)
        _register_login(client2, email, password)
        me_res = client2.get("/auth/me")
        new_uid = me_res.json()["id"]
        new_user_dir = users_dir / str(new_uid)

        img_blue = new_user_dir / "library" / "blue.jpg"
        _make_jpg(img_blue, color=(0, 0, 255))
        md5_blue = compute_file_md5(img_blue)

        conn2 = init_db(_user_db(new_user_dir))
        conn2.execute(
            "INSERT INTO images (id, file_path, file_size, modified_time, md5) VALUES (1, ?, ?, ?, ?)",
            (str(img_blue), img_blue.stat().st_size, img_blue.stat().st_mtime, md5_blue),
        )
        conn2.commit()
        conn2.close()

        t2 = client2.get("/thumb/1?size=64")
        assert t2.status_code == 200
        bytes_after = t2.content

        # Must be a different image (blue, not red)
        assert bytes_after != bytes_before, "Stale cached thumbnail was reused after account delete/recreate"

    def test_stale_cache_deleted_on_db_reset_or_image_change(self, tmp_path: Path):
        """When DB is reset or an image id is replaced with a new file, the old cache is evicted."""
        app, users_dir, auth_db = _make_app(tmp_path)
        password = "ArgusPass1234!"
        email = "dbreset@argus.test"

        client = TestClient(app, raise_server_exceptions=True)
        _register_login(client, email, password)
        client.get("/auth/me")

        user_dir = users_dir / "1"
        img1 = user_dir / "library" / "first.jpg"
        _make_jpg(img1, color=(10, 200, 10))
        md5_1 = compute_file_md5(img1)

        conn = init_db(_user_db(user_dir))
        conn.execute(
            "INSERT INTO images (id, file_path, file_size, modified_time, md5) VALUES (1, ?, ?, ?, ?)",
            (str(img1), img1.stat().st_size, img1.stat().st_mtime, md5_1),
        )
        conn.commit()

        t1 = client.get("/thumb/1?size=64")
        assert t1.status_code == 200
        bytes_1 = t1.content

        # DB reset / image change: wipe images and insert a completely different image with same ID 1
        img2 = user_dir / "library" / "second.jpg"
        _make_jpg(img2, color=(200, 10, 200))
        md5_2 = compute_file_md5(img2)

        conn.execute("DELETE FROM images")
        conn.execute(
            "INSERT INTO images (id, file_path, file_size, modified_time, md5) VALUES (1, ?, ?, ?, ?)",
            (str(img2), img2.stat().st_size, img2.stat().st_mtime, md5_2),
        )
        conn.commit()
        conn.close()

        # Old thumbnail file was for md5_1
        t2 = client.get("/thumb/1?size=64")
        assert t2.status_code == 200
        bytes_2 = t2.content
        assert bytes_2 != bytes_1

        # Check thumbs directory: only the new key exists, old one was evicted!
        thumbs = list((user_dir / "thumbs").glob("1_*_64.jpg"))
        assert len(thumbs) == 1
        assert md5_2[:12] in thumbs[0].name
        assert md5_1[:12] not in thumbs[0].name


class TestCategoryCountsMatchClusters:

    def _make_cluster_app(self, tmp_path: Path):
        """Build a minimal DB with 2 clusters and return authenticated client."""
        db_file = tmp_path / "c.db"
        conn = init_db(db_file)
        ensure_ledger_tables(conn)

        img_dir = tmp_path / "photos"
        img_dir.mkdir()

        now = time.time()

        # Cluster 1: landscape x2 images
        p1 = img_dir / "land1.jpg"; _make_jpg(p1, (10, 100, 10))
        p2 = img_dir / "land2.jpg"; _make_jpg(p2, (10, 110, 10))
        md5_1 = compute_file_md5(p1)
        md5_2 = compute_file_md5(p2)

        conn.execute("INSERT INTO clusters (id, representative_id, stage_detected, member_count, category) VALUES (1, NULL, 0, 2, 'landscape')")
        conn.execute("INSERT INTO images (id, cluster_id, file_path, file_size, width, height, format, md5, is_keeper, is_recommended_keeper, delete_safe, quality_status, ssim, ssim_color, sharpness_ratio, pixel_ratio, bytes_ratio, modified_time, category_top1, category_top1_prob, match_pct, resolution_pct, confidence) VALUES (1, 1, ?, ?, 100, 100, 'JPEG', ?, 1, 1, 0, 'reference', 1.0, 1.0, 1.0, 1.0, 1.0, ?, 'landscape', 0.9, 100.0, 100.0, 1.0)", (str(p1), p1.stat().st_size, md5_1, now))
        conn.execute("INSERT INTO images (id, cluster_id, file_path, file_size, width, height, format, md5, is_keeper, is_recommended_keeper, delete_safe, quality_status, ssim, ssim_color, sharpness_ratio, pixel_ratio, bytes_ratio, modified_time, category_top1, category_top1_prob, match_pct, resolution_pct, confidence) VALUES (2, 1, ?, ?, 100, 100, 'JPEG', ?, 0, 0, 1, 'aligned', 1.0, 1.0, 1.0, 1.0, 1.0, ?, 'landscape', 0.85, 99.0, 100.0, 0.98)", (str(p2), p2.stat().st_size, md5_2, now - 1))
        conn.execute("UPDATE clusters SET representative_id = 1 WHERE id = 1")

        # Cluster 2: food x2 images
        p3 = img_dir / "food1.jpg"; _make_jpg(p3, (200, 50, 50))
        p4 = img_dir / "food2.jpg"; _make_jpg(p4, (210, 60, 60))
        md5_3 = compute_file_md5(p3)
        md5_4 = compute_file_md5(p4)

        conn.execute("INSERT INTO clusters (id, representative_id, stage_detected, member_count, category) VALUES (2, NULL, 0, 2, 'food')")
        conn.execute("INSERT INTO images (id, cluster_id, file_path, file_size, width, height, format, md5, is_keeper, is_recommended_keeper, delete_safe, quality_status, ssim, ssim_color, sharpness_ratio, pixel_ratio, bytes_ratio, modified_time, category_top1, category_top1_prob, match_pct, resolution_pct, confidence) VALUES (3, 2, ?, ?, 100, 100, 'JPEG', ?, 1, 1, 0, 'reference', 1.0, 1.0, 1.0, 1.0, 1.0, ?, 'food', 0.9, 100.0, 100.0, 1.0)", (str(p3), p3.stat().st_size, md5_3, now))
        conn.execute("INSERT INTO images (id, cluster_id, file_path, file_size, width, height, format, md5, is_keeper, is_recommended_keeper, delete_safe, quality_status, ssim, ssim_color, sharpness_ratio, pixel_ratio, bytes_ratio, modified_time, category_top1, category_top1_prob, match_pct, resolution_pct, confidence) VALUES (4, 2, ?, ?, 100, 100, 'JPEG', ?, 0, 0, 1, 'aligned', 1.0, 1.0, 1.0, 1.0, 1.0, ?, 'food', 0.88, 99.0, 100.0, 0.98)", (str(p4), p4.stat().st_size, md5_4, now - 1))
        conn.execute("UPDATE clusters SET representative_id = 3 WHERE id = 2")

        conn.commit(); conn.close()

        auth_db = tmp_path / "auth.db"
        users_dir = tmp_path / "users"
        users_dir.mkdir(exist_ok=True)
        app = create_app(
            db_path=db_file,
            trash_dir=tmp_path / "trash",
            thumb_cache_dir=tmp_path / "thumbs",
            library_dir=tmp_path / "library",
            jobs_dir=tmp_path / "jobs",
            auth_db_path=auth_db,
            users_dir=users_dir,
        )
        if 1 in app.state.custom_user_overrides:
            app.state.custom_user_overrides[1]["root"] = tmp_path

        client = TestClient(app, raise_server_exceptions=True)
        _register_login(client, "cat@argus.test")
        return client

    def test_category_count_matches_cluster_list_length(self, tmp_path: Path):
        """Count from /categories must equal len(/clusters?category=<cat>)."""
        client = self._make_cluster_app(tmp_path)

        cats_res = client.get("/categories")
        assert cats_res.status_code == 200
        cats = cats_res.json()

        # There must be at least landscape and food
        assert "landscape" in cats, f"landscape missing from /categories: {cats}"
        assert "food" in cats, f"food missing from /categories: {cats}"
        assert cats["landscape"] == 1
        assert cats["food"] == 1

        for cat, count in cats.items():
            clusters_res = client.get(f"/clusters?category={cat}")
            assert clusters_res.status_code == 200
            cluster_list = clusters_res.json()
            assert len(cluster_list) == count, (
                f"Category '{cat}': /categories says {count} but /clusters returned {len(cluster_list)}"
            )

    def test_zero_count_categories_not_in_response(self, tmp_path: Path):
        """Categories with 0 matching clusters must not appear in /categories."""
        client = self._make_cluster_app(tmp_path)

        cats_res = client.get("/categories")
        assert cats_res.status_code == 200
        cats = cats_res.json()

        for cat, cnt in cats.items():
            assert cnt > 0, f"Category '{cat}' has count 0 but should have been hidden"


class TestMixedCategoryCluster:

    def test_mixed_category_cluster_shows_mixed(self, tmp_path: Path):
        """A cluster whose members have equally split categories shows 'mixed'."""
        db_file = tmp_path / "mix.db"
        conn = init_db(db_file)
        ensure_ledger_tables(conn)

        img_dir = tmp_path / "photos"
        img_dir.mkdir()
        now = time.time()

        # Two images: one 'landscape', one 'food' — equal split → mixed
        p1 = img_dir / "land.jpg"; _make_jpg(p1, (10, 100, 10))
        p2 = img_dir / "food.jpg"; _make_jpg(p2, (200, 50, 50))
        md5_1 = compute_file_md5(p1)
        md5_2 = compute_file_md5(p2)

        conn.execute("INSERT INTO clusters (id, representative_id, stage_detected, member_count, category) VALUES (1, NULL, 0, 2, 'landscape')")
        conn.execute("INSERT INTO images (id, cluster_id, file_path, file_size, width, height, format, md5, is_keeper, is_recommended_keeper, delete_safe, quality_status, ssim, ssim_color, sharpness_ratio, pixel_ratio, bytes_ratio, modified_time, category_top1, category_top1_prob, match_pct, resolution_pct, confidence) VALUES (1, 1, ?, ?, 100, 100, 'JPEG', ?, 1, 1, 0, 'reference', 1.0, 1.0, 1.0, 1.0, 1.0, ?, 'landscape', 0.9, 100.0, 100.0, 1.0)", (str(p1), p1.stat().st_size, md5_1, now))
        conn.execute("INSERT INTO images (id, cluster_id, file_path, file_size, width, height, format, md5, is_keeper, is_recommended_keeper, delete_safe, quality_status, ssim, ssim_color, sharpness_ratio, pixel_ratio, bytes_ratio, modified_time, category_top1, category_top1_prob, match_pct, resolution_pct, confidence) VALUES (2, 1, ?, ?, 100, 100, 'JPEG', ?, 0, 0, 1, 'aligned', 1.0, 1.0, 1.0, 1.0, 1.0, ?, 'food', 0.88, 99.0, 100.0, 0.98)", (str(p2), p2.stat().st_size, md5_2, now - 1))
        conn.execute("UPDATE clusters SET representative_id = 1 WHERE id = 1")
        conn.commit(); conn.close()

        auth_db = tmp_path / "auth.db"
        users_dir = tmp_path / "users"
        users_dir.mkdir(exist_ok=True)
        app = create_app(
            db_path=db_file,
            trash_dir=tmp_path / "trash",
            thumb_cache_dir=tmp_path / "thumbs",
            library_dir=tmp_path / "library",
            jobs_dir=tmp_path / "jobs",
            auth_db_path=auth_db,
            users_dir=users_dir,
        )
        if 1 in app.state.custom_user_overrides:
            app.state.custom_user_overrides[1]["root"] = tmp_path

        client = TestClient(app, raise_server_exceptions=True)
        _register_login(client, "mix@argus.test")

        clusters_res = client.get("/clusters")
        assert clusters_res.status_code == 200
        clusters = clusters_res.json()
        assert len(clusters) == 1
        assert clusters[0]["category"] == "mixed", (
            f"Expected 'mixed' but got '{clusters[0]['category']}'"
        )

    def test_majority_category_used_when_clear(self, tmp_path: Path):
        """A cluster with 3 landscape + 1 food should show 'landscape', not 'mixed'."""
        db_file = tmp_path / "maj.db"
        conn = init_db(db_file)
        ensure_ledger_tables(conn)

        img_dir = tmp_path / "photos"
        img_dir.mkdir()
        now = time.time()

        imgs = []
        for i, (color, cat) in enumerate([
            ((10, 100, 10), "landscape"),
            ((10, 110, 10), "landscape"),
            ((10, 120, 10), "landscape"),
            ((200, 50, 50), "food"),
        ]):
            p = img_dir / f"img{i}.jpg"; _make_jpg(p, color)
            imgs.append((p, cat))

        conn.execute("INSERT INTO clusters (id, representative_id, stage_detected, member_count, category) VALUES (1, NULL, 0, 4, 'landscape')")
        for idx, (p, cat) in enumerate(imgs, start=1):
            md5 = compute_file_md5(p)
            is_keeper = 1 if idx == 1 else 0
            conn.execute(
                "INSERT INTO images (id, cluster_id, file_path, file_size, width, height, format, md5, is_keeper, is_recommended_keeper, delete_safe, quality_status, ssim, ssim_color, sharpness_ratio, pixel_ratio, bytes_ratio, modified_time, category_top1, category_top1_prob, match_pct, resolution_pct, confidence) VALUES (?, 1, ?, ?, 100, 100, 'JPEG', ?, ?, ?, ?, 'aligned', 1.0, 1.0, 1.0, 1.0, 1.0, ?, ?, 0.9, 99.0, 100.0, 0.98)",
                (idx, str(p), p.stat().st_size, md5, is_keeper, is_keeper, 0 if is_keeper else 1, now - idx, cat)
            )
        conn.execute("UPDATE clusters SET representative_id = 1 WHERE id = 1")
        conn.commit(); conn.close()

        auth_db = tmp_path / "auth.db"
        users_dir = tmp_path / "users"
        users_dir.mkdir(exist_ok=True)
        app = create_app(
            db_path=db_file,
            trash_dir=tmp_path / "trash",
            thumb_cache_dir=tmp_path / "thumbs",
            library_dir=tmp_path / "library",
            jobs_dir=tmp_path / "jobs",
            auth_db_path=auth_db,
            users_dir=users_dir,
        )
        if 1 in app.state.custom_user_overrides:
            app.state.custom_user_overrides[1]["root"] = tmp_path

        client = TestClient(app, raise_server_exceptions=True)
        _register_login(client, "maj@argus.test")

        clusters_res = client.get("/clusters")
        assert clusters_res.status_code == 200
        clusters = clusters_res.json()
        assert len(clusters) == 1
        assert clusters[0]["category"] == "landscape", (
            f"Expected 'landscape' majority but got '{clusters[0]['category']}'"
        )
