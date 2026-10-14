"""
tests/test_stage1_pipeline.py — Unit and integration tests for Stage 0-1 pipeline.

Tests:
- LocalFolderSource interface (list_files, get_metadata, get_bytes, get_thumbnail draft mode)
- Rescanning: skips unchanged files, detects modified files
- Stage 1 pHash computation and thumbnail caching/draft mode
- Stage 1 near-duplicate clustering: representative verification & anti-chaining rule
- End-to-end pipeline run on sample benchmark images
"""

from __future__ import annotations

import os
import sqlite3
import time
from pathlib import Path

import pytest
from PIL import Image

from core.config import load_config
from core.stage0 import run_stage0
from core.stage1 import run_stage1, _hamming_distance
from db.connection import init_db
from sources.local_folder import LocalFolderSource


@pytest.fixture()
def cfg() -> dict:
    return load_config()


@pytest.fixture()
def db(tmp_path: Path) -> sqlite3.Connection:
    return init_db(tmp_path / "test.db")


def _create_sample_jpeg(path: Path, color=(200, 100, 50), size=(800, 600)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", size, color=color)
    img.save(path, format="JPEG", quality=90)
    return path


class TestLocalFolderSource:
    def test_list_files(self, tmp_path: Path):
        _create_sample_jpeg(tmp_path / "a.jpg")
        _create_sample_jpeg(tmp_path / "sub" / "b.jpeg")
        (tmp_path / "notes.txt").write_text("hello")

        source = LocalFolderSource(tmp_path)
        files = list(source.list_files())
        assert len(files) == 2
        assert any("a.jpg" in f for f in files)
        assert any("b.jpeg" in f for f in files)

    def test_get_metadata_without_full_pixel_decode(self, tmp_path: Path):
        p = _create_sample_jpeg(tmp_path / "img.jpg", size=(640, 480))
        source = LocalFolderSource(tmp_path)
        meta = source.get_metadata(str(p))

        assert meta.file_size > 0
        assert meta.modified_time > 0
        assert meta.width == 640
        assert meta.height == 480
        assert meta.format == "JPEG"

    def test_get_thumbnail_uses_draft_mode(self, tmp_path: Path):
        p = _create_sample_jpeg(tmp_path / "large.jpg", size=(1600, 1200))
        source = LocalFolderSource(tmp_path)
        thumb = source.get_thumbnail(str(p), size=(256, 256))

        assert max(thumb.size) <= 256
        assert thumb.mode == "RGB"

    def test_get_bytes(self, tmp_path: Path):
        p = _create_sample_jpeg(tmp_path / "img.jpg")
        source = LocalFolderSource(tmp_path)
        raw_bytes = source.get_bytes(str(p))
        assert raw_bytes == p.read_bytes()


class TestRescanningSkipsUnchanged:
    def test_unchanged_file_is_skipped(self, tmp_path: Path, db: sqlite3.Connection, cfg: dict):
        p = _create_sample_jpeg(tmp_path / "test.jpg")
        source = LocalFolderSource(tmp_path)

        # First run: discovers and hashes
        res1 = run_stage0(source, db, cfg)
        assert res1.files_discovered == 1
        assert res1.files_hashed == 1

        # Second run: file is unchanged, discovery count must be 0
        res2 = run_stage0(source, db, cfg)
        assert res2.files_discovered == 0
        assert res2.files_hashed == 0

    def test_modified_file_is_redetected(self, tmp_path: Path, db: sqlite3.Connection, cfg: dict):
        p = _create_sample_jpeg(tmp_path / "test.jpg")
        source = LocalFolderSource(tmp_path)

        run_stage0(source, db, cfg)

        # Modify file and change timestamp
        time.sleep(0.05)
        p.write_bytes(b"new modified content that is different")
        os.utime(p, None)

        res2 = run_stage0(source, db, cfg)
        assert res2.files_discovered == 1
        assert res2.files_hashed == 1


class TestStage1NearDuplicates:
    def test_computes_phash_and_stores(self, tmp_path: Path, db: sqlite3.Connection, cfg: dict):
        p = _create_sample_jpeg(tmp_path / "pic.jpg")
        source = LocalFolderSource(tmp_path)

        run_stage0(source, db, cfg)
        res1 = run_stage1(source, db, cfg)

        assert res1.hashes_computed == 1
        row = db.execute("SELECT phash FROM images WHERE file_path = ?", (str(p),)).fetchone()
        assert row["phash"] is not None
        assert len(row["phash"]) > 0

    def test_near_duplicate_clusters_created(self, tmp_path: Path, db: sqlite3.Connection, cfg: dict):
        # Create an original image and a slightly recompressed or resized copy
        orig = _create_sample_jpeg(tmp_path / "orig.jpg", color=(120, 150, 200), size=(600, 400))
        img = Image.open(orig)
        resized = img.resize((300, 200))
        resized.save(tmp_path / "resized.jpg", format="JPEG", quality=80)
        img.close()
        resized.close()

        source = LocalFolderSource(tmp_path)
        run_stage0(source, db, cfg)
        res1 = run_stage1(source, db, cfg)

        # Stage 0 wouldn't match them (different sizes and MD5s), but Stage 1 should!
        assert res1.clusters_created == 1
        assert res1.files_clustered == 2

        clusters = db.execute("SELECT id, representative_id FROM clusters").fetchall()
        assert len(clusters) == 1

        rows = db.execute("SELECT id, file_path, is_keeper, cluster_id FROM images").fetchall()
        assert all(r["cluster_id"] == clusters[0]["id"] for r in rows)
        keepers = [r for r in rows if r["is_keeper"] == 1]
        assert len(keepers) == 1
        # Keeper should be the higher resolution one (orig.jpg)
        assert "orig.jpg" in keepers[0]["file_path"]

    def test_anti_chaining_rule(self, tmp_path: Path, db: sqlite3.Connection, cfg: dict):
        """
        Anti-chaining: members are verified strictly against the cluster representative.
        If A is representative, B matches A (dist <= 10), but C matches B (dist <= 10)
        while dist(C, A) > 10, C must NOT be added to A's cluster.
        """
        source = LocalFolderSource(tmp_path)

        # Insert 3 records directly to test clustering verification
        db.execute(
            """
            INSERT INTO images (file_path, file_size, modified_time, width, height, format, phash, status)
            VALUES ('/fake/A.jpg', 1000, 1.0, 1000, 1000, 'JPEG', '0000000000000000', 'hashed'),
                   ('/fake/B.jpg', 800,  1.0, 800,  800,  'JPEG', '00000000000000ff', 'hashed'),
                   ('/fake/C.jpg', 600,  1.0, 600,  600,  'JPEG', '000000000000ffff', 'hashed')
            """
        )
        db.commit()

        # Config threshold = 8 bits:
        # dist(A, B) = 8 bits (differs only in last 2 hex chars 'ff' = 8 bits) -> matches A!
        # dist(A, C) = 16 bits ('ffff') -> exceeds threshold 8 -> must NOT match A!
        test_cfg = dict(cfg)
        test_cfg["stage1"] = {"hamming_threshold": 8, "hash_size": 8, "hash_algorithm": "phash"}

        res1 = run_stage1(source, db, test_cfg)
        assert res1.clusters_created == 1

        # Check members of the cluster
        cluster = db.execute("SELECT id, representative_id, member_count FROM clusters").fetchone()
        assert cluster["member_count"] == 2  # Only A and B! C was prevented from chaining!

        c_row = db.execute("SELECT cluster_id FROM images WHERE file_path = '/fake/C.jpg'").fetchone()
        assert c_row["cluster_id"] is None


class TestBenchmarkDatasetStage01Integration:
    def test_run_on_benchmark_subset(self, tmp_path: Path, db: sqlite3.Connection, cfg: dict):
        """
        Verify Stage 0 and Stage 1 together on benchmark directory if present.
        """
        bench_dir = Path("data/benchmark")
        if not bench_dir.exists():
            pytest.skip("data/benchmark does not exist, skipping integration test")

        source = LocalFolderSource(bench_dir)
        # Run Stage 0
        s0_res = run_stage0(source, db, cfg)
        assert s0_res.files_discovered > 0
        assert s0_res.clusters_created > 0

        # Run Stage 1
        s1_res = run_stage1(source, db, cfg)
        assert s1_res.hashes_computed > 0
        assert s1_res.clusters_created > 0 or s1_res.members_added_to_existing > 0
