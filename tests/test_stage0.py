"""
tests/test_stage0.py — Unit tests for Stage 0.

Uses pytest's tmp_path fixture for throwaway directories and in-memory
SQLite databases so tests have zero side-effects on the repo.
"""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import pytest

from db.connection import init_db
from core.config import load_config
from core.stage0 import (
    Stage0Result,
    _image_extensions,
    _iter_image_files,
    _md5,
    _detect_raw_pairs,
    _build_exact_clusters,
    run_stage0,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def cfg() -> dict:
    """Real config loaded from repo-root config.toml."""
    return load_config()


@pytest.fixture()
def db(tmp_path: Path) -> sqlite3.Connection:
    """Fresh in-memory-equivalent DB (file in tmp_path, not :memory:,
    so WAL mode works correctly)."""
    return init_db(tmp_path / "test.db")


def _write(path: Path, content: bytes = b"fake image data") -> Path:
    """Helper: create a file with given content."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


# ---------------------------------------------------------------------------
# _md5
# ---------------------------------------------------------------------------

class TestMd5:
    def test_known_digest(self, tmp_path: Path):
        data = b"hello argus"
        f = _write(tmp_path / "a.jpg", data)
        expected = hashlib.md5(data).hexdigest()
        assert _md5(f) == expected

    def test_different_content_different_digest(self, tmp_path: Path):
        f1 = _write(tmp_path / "a.jpg", b"aaa")
        f2 = _write(tmp_path / "b.jpg", b"bbb")
        assert _md5(f1) != _md5(f2)

    def test_identical_content_same_digest(self, tmp_path: Path):
        data = b"same bytes"
        f1 = _write(tmp_path / "a.jpg", data)
        f2 = _write(tmp_path / "b.jpg", data)
        assert _md5(f1) == _md5(f2)


# ---------------------------------------------------------------------------
# _iter_image_files
# ---------------------------------------------------------------------------

class TestIterImageFiles:
    def test_finds_jpg_and_png(self, tmp_path: Path, cfg: dict):
        exts = _image_extensions(cfg)
        _write(tmp_path / "photo.jpg")
        _write(tmp_path / "icon.png")
        _write(tmp_path / "notes.txt")   # should be ignored
        found = list(_iter_image_files(tmp_path, exts))
        names = {p.name for p in found}
        assert "photo.jpg" in names
        assert "icon.png" in names
        assert "notes.txt" not in names

    def test_recursive(self, tmp_path: Path, cfg: dict):
        exts = _image_extensions(cfg)
        _write(tmp_path / "sub" / "deep" / "img.jpeg")
        found = list(_iter_image_files(tmp_path, exts))
        assert any(p.name == "img.jpeg" for p in found)

    def test_empty_directory(self, tmp_path: Path, cfg: dict):
        exts = _image_extensions(cfg)
        assert list(_iter_image_files(tmp_path, exts)) == []

    def test_raw_extensions_included(self, tmp_path: Path, cfg: dict):
        exts = _image_extensions(cfg)
        _write(tmp_path / "shot.arw")
        found = list(_iter_image_files(tmp_path, exts))
        assert any(p.name == "shot.arw" for p in found)

    def test_case_insensitive_extension(self, tmp_path: Path, cfg: dict):
        exts = _image_extensions(cfg)
        _write(tmp_path / "PHOTO.JPG")
        found = list(_iter_image_files(tmp_path, exts))
        assert any(p.name == "PHOTO.JPG" for p in found)


# ---------------------------------------------------------------------------
# run_stage0 — basic scan and hashing
# ---------------------------------------------------------------------------

class TestRunStage0Basic:
    def test_files_appear_in_db(self, tmp_path: Path, db: sqlite3.Connection, cfg: dict):
        _write(tmp_path / "a.jpg", b"img_a")
        _write(tmp_path / "b.png", b"img_b")
        run_stage0(tmp_path, db, cfg)
        rows = db.execute("SELECT file_path, status, md5 FROM images").fetchall()
        assert len(rows) == 2
        for row in rows:
            assert row["status"] == "stage0_done"
            assert row["md5"] is not None

    def test_file_size_stored(self, tmp_path: Path, db: sqlite3.Connection, cfg: dict):
        content = b"x" * 1234
        _write(tmp_path / "sized.jpg", content)
        run_stage0(tmp_path, db, cfg)
        row = db.execute("SELECT file_size FROM images").fetchone()
        assert row["file_size"] == 1234

    def test_no_images_returns_zero_result(
        self, tmp_path: Path, db: sqlite3.Connection, cfg: dict
    ):
        result = run_stage0(tmp_path, db, cfg)
        assert result.files_discovered == 0
        assert result.files_hashed == 0
        assert result.clusters_created == 0


# ---------------------------------------------------------------------------
# run_stage0 — exact duplicates
# ---------------------------------------------------------------------------

class TestExactDuplicates:
    def test_identical_files_form_cluster(
        self, tmp_path: Path, db: sqlite3.Connection, cfg: dict
    ):
        data = b"duplicate bytes"
        _write(tmp_path / "copy1.jpg", data)
        _write(tmp_path / "copy2.jpg", data)
        result = run_stage0(tmp_path, db, cfg)
        assert result.clusters_created == 1
        assert result.files_clustered == 2

    def test_distinct_files_no_cluster(
        self, tmp_path: Path, db: sqlite3.Connection, cfg: dict
    ):
        _write(tmp_path / "a.jpg", b"aaa")
        _write(tmp_path / "b.jpg", b"bbb")
        result = run_stage0(tmp_path, db, cfg)
        assert result.clusters_created == 0

    def test_keeper_is_shortest_path(
        self, tmp_path: Path, db: sqlite3.Connection, cfg: dict
    ):
        data = b"same"
        # 'a.jpg' path is shorter than 'sub/b.jpg'
        _write(tmp_path / "a.jpg", data)
        _write(tmp_path / "sub" / "b.jpg", data)
        run_stage0(tmp_path, db, cfg)
        keeper = db.execute(
            "SELECT file_path FROM images WHERE is_keeper = 1"
        ).fetchone()
        assert keeper is not None
        assert Path(keeper["file_path"]).name == "a.jpg"

    def test_three_copies_one_cluster(
        self, tmp_path: Path, db: sqlite3.Connection, cfg: dict
    ):
        data = b"triple"
        for name in ("x.jpg", "y.jpg", "z.jpg"):
            _write(tmp_path / name, data)
        result = run_stage0(tmp_path, db, cfg)
        assert result.clusters_created == 1
        assert result.files_clustered == 3

    def test_two_independent_clusters(
        self, tmp_path: Path, db: sqlite3.Connection, cfg: dict
    ):
        _write(tmp_path / "a1.jpg", b"group_a")
        _write(tmp_path / "a2.jpg", b"group_a")
        _write(tmp_path / "b1.jpg", b"group_b")
        _write(tmp_path / "b2.jpg", b"group_b")
        result = run_stage0(tmp_path, db, cfg)
        assert result.clusters_created == 2
        assert result.files_clustered == 4


# ---------------------------------------------------------------------------
# run_stage0 — RAW+JPEG pair detection
# ---------------------------------------------------------------------------

class TestRawPairDetection:
    def test_raw_jpeg_pair_flagged(
        self, tmp_path: Path, db: sqlite3.Connection, cfg: dict
    ):
        _write(tmp_path / "IMG_001.ARW", b"raw data")
        _write(tmp_path / "IMG_001.JPG", b"jpeg data")
        run_stage0(tmp_path, db, cfg)
        rows = db.execute("SELECT file_path, is_raw_pair FROM images").fetchall()
        flagged = {Path(r["file_path"]).name for r in rows if r["is_raw_pair"] == 1}
        assert "IMG_001.ARW" in flagged
        assert "IMG_001.JPG" in flagged

    def test_raw_pair_not_clustered(
        self, tmp_path: Path, db: sqlite3.Connection, cfg: dict
    ):
        """Even if the bytes happen to be identical, RAW pairs must not form clusters."""
        same = b"same bytes"
        _write(tmp_path / "shot.NEF", same)
        _write(tmp_path / "shot.JPG", same)
        result = run_stage0(tmp_path, db, cfg)
        assert result.clusters_created == 0

    def test_lone_raw_not_flagged(
        self, tmp_path: Path, db: sqlite3.Connection, cfg: dict
    ):
        _write(tmp_path / "solo.CR2", b"raw only")
        run_stage0(tmp_path, db, cfg)
        row = db.execute("SELECT is_raw_pair FROM images").fetchone()
        assert row["is_raw_pair"] == 0


# ---------------------------------------------------------------------------
# run_stage0 — resumability
# ---------------------------------------------------------------------------

class TestResumability:
    def test_second_run_does_not_duplicate_rows(
        self, tmp_path: Path, db: sqlite3.Connection, cfg: dict
    ):
        _write(tmp_path / "img.jpg", b"data")
        run_stage0(tmp_path, db, cfg)
        run_stage0(tmp_path, db, cfg)  # run again — should be a no-op
        count = db.execute("SELECT COUNT(*) FROM images").fetchone()[0]
        assert count == 1

    def test_second_run_does_not_duplicate_clusters(
        self, tmp_path: Path, db: sqlite3.Connection, cfg: dict
    ):
        data = b"dup"
        _write(tmp_path / "p.jpg", data)
        _write(tmp_path / "q.jpg", data)
        run_stage0(tmp_path, db, cfg)
        run_stage0(tmp_path, db, cfg)
        count = db.execute("SELECT COUNT(*) FROM clusters").fetchone()[0]
        assert count == 1
