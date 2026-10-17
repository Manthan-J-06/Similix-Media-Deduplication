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


def _make_dummy_image(label: bytes, fmt: str = "JPEG", target_size: int | None = None) -> bytes:
    import io
    from PIL import Image
    buf = io.BytesIO()
    h = hashlib.md5(label).digest()
    img = Image.new("RGB", (16, 16), color=(h[0], h[1], h[2]))
    img.save(buf, format=fmt)
    if target_size is not None and target_size >= len(buf.getvalue()):
        data = buf.getvalue() + b"0" * (target_size - len(buf.getvalue()))
    else:
        data = buf.getvalue() + b"__TAG__:" + label
    return data


def _write(path: Path, content: bytes = b"fake image data", raw: bool = False) -> Path:
    """Helper: create a file with given content."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not raw and path.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tiff", ".tif", ".heic", ".heif"):
        target_len = len(content) if len(content) >= 700 else None
        fmt = "PNG" if path.suffix.lower() == ".png" else "JPEG"
        content = _make_dummy_image(content, fmt=fmt, target_size=target_len)
    path.write_bytes(content)
    return path


# ---------------------------------------------------------------------------
# _md5
# ---------------------------------------------------------------------------

class TestMd5:
    def test_known_digest(self, tmp_path: Path):
        data = b"hello argus"
        f = _write(tmp_path / "a.jpg", data, raw=True)
        expected = hashlib.md5(data).hexdigest()
        assert _md5(f) == expected

    def test_different_content_different_digest(self, tmp_path: Path):
        f1 = _write(tmp_path / "a.jpg", b"aaa", raw=True)
        f2 = _write(tmp_path / "b.jpg", b"bbb", raw=True)
        assert _md5(f1) != _md5(f2)

    def test_identical_content_same_digest(self, tmp_path: Path):
        data = b"same bytes"
        f1 = _write(tmp_path / "a.jpg", data, raw=True)
        f2 = _write(tmp_path / "b.jpg", data, raw=True)
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
            assert row["status"] in ("hashed", "clustered")
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


# ---------------------------------------------------------------------------
# run_stage0 — format support and corrupt/skipped file tracking
# ---------------------------------------------------------------------------

class TestScanRobustnessAndFormats:
    def test_scan_heic_file(self, tmp_path: Path, db: sqlite3.Connection, cfg: dict):
        """Confirm pillow-heif is registered and scanning a HEIC file succeeds."""
        import pillow_heif
        from PIL import Image
        pillow_heif.register_heif_opener()

        heic_path = tmp_path / "sample.heic"
        img = Image.new("RGB", (32, 32), color=(70, 130, 180))
        img.save(heic_path, format="HEIF")

        res = run_stage0(tmp_path, db, cfg)
        assert res.files_discovered == 1
        assert res.files_skipped == 0

        row = db.execute("SELECT file_path, format, width, height, status, md5 FROM images").fetchone()
        assert row is not None
        assert row["status"] in ("hashed", "stage0_done")
        assert row["format"] in ("HEIF", "HEIC")
        assert row["width"] == 32
        assert row["height"] == 32
        assert row["md5"] is not None

    def test_folder_with_mixed_formats_and_one_corrupt(
        self, tmp_path: Path, db: sqlite3.Connection, cfg: dict
    ):
        """
        A folder with .JPG, .jpeg, .PNG, .webp, .heic and one corrupt file
        reports 5 discovered and 1 skipped with a reason.
        """
        import pillow_heif
        from PIL import Image
        pillow_heif.register_heif_opener()

        # 1. .JPG
        img_jpg = Image.new("RGB", (20, 20), color=(100, 20, 30))
        img_jpg.save(tmp_path / "img1.JPG", format="JPEG")

        # 2. .jpeg
        img_jpeg = Image.new("RGB", (20, 20), color=(20, 100, 30))
        img_jpeg.save(tmp_path / "img2.jpeg", format="JPEG")

        # 3. .PNG
        img_png = Image.new("RGB", (20, 20), color=(20, 30, 100))
        img_png.save(tmp_path / "img3.PNG", format="PNG")

        # 4. .webp
        img_webp = Image.new("RGB", (20, 20), color=(50, 50, 50))
        img_webp.save(tmp_path / "img4.webp", format="WEBP")

        # 5. .heic
        img_heic = Image.new("RGB", (20, 20), color=(80, 80, 80))
        img_heic.save(tmp_path / "img5.heic", format="HEIF")

        # 6. corrupt file
        corrupt_path = tmp_path / "corrupt.jpg"
        corrupt_path.write_bytes(b"NOT_A_VALID_IMAGE_HEADER_CORRUPT")

        # Run Stage 0
        res = run_stage0(tmp_path, db, cfg)
        assert res.files_discovered == 5
        assert res.files_skipped == 1
        assert len(res.skipped_reasons) == 1

        skipped_file, reason = res.skipped_reasons[0]
        assert Path(skipped_file).name == "corrupt.jpg"
        assert "cannot identify image file" in reason.lower()

    def test_cli_scan_summary_skipped_and_accepted_extensions(
        self, tmp_path: Path, capsys
    ):
        """CLI scan reports accepted formats and 'Skipped: N' with reasons."""
        from core.cli import main
        import pillow_heif
        from PIL import Image
        pillow_heif.register_heif_opener()

        img_dir = tmp_path / "images"
        img_dir.mkdir()
        db_file = tmp_path / "test.db"

        # Create valid images and one corrupt
        Image.new("RGB", (16, 16), color="red").save(img_dir / "a.JPG", format="JPEG")
        Image.new("RGB", (16, 16), color="green").save(img_dir / "b.jpeg", format="JPEG")
        Image.new("RGB", (16, 16), color="blue").save(img_dir / "c.PNG", format="PNG")
        Image.new("RGB", (16, 16), color="yellow").save(img_dir / "d.webp", format="WEBP")
        Image.new("RGB", (16, 16), color="cyan").save(img_dir / "e.heic", format="HEIF")
        (img_dir / "broken.jpg").write_bytes(b"garbage content")

        main(["scan", str(img_dir), "--db", str(db_file)])
        captured = capsys.readouterr().out

        # 1. Accepted extensions check
        assert "Accepted exts" in captured
        assert "case-insensitive" in captured
        assert ".JPG" in captured
        assert ".HEIC" in captured
        assert ".jpeg" in captured
        assert ".webp" in captured
        assert ".tiff" in captured

        # 2. Discovered and skipped check
        assert "Discovered / updated : 5" in captured
        assert "Skipped: 1" in captured
        assert "broken.jpg" in captured
        assert "cannot identify image file" in captured.lower()
