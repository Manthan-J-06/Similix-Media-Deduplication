"""
tests/test_make_dataset.py — Tests for bench/make_dataset.py.

All tests use synthetic images created with PIL.Image.new() — no real photos
required. Tests are isolated to tmp_path; no files are written to data/.
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import pytest
from PIL import Image

from bench.make_dataset import (
    CSV_FIELDNAMES,
    DatasetStats,
    VariantRecord,
    _make_brightness,
    _make_contrast,
    _make_crops,
    _make_exact_copy,
    _make_format_converts,
    _make_jpeg_recompressed,
    _make_resizes,
    _make_rotations,
    _open_rgb,
    _unique_stem,
    build_dataset,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _solid(tmp_path: Path, name: str = "img.jpg", size: tuple = (800, 600)) -> Path:
    """
    Create a solid-colour JPEG at *tmp_path/name* and return its path.
    Uses a deterministic colour so tests that compare bytes can reason about it.
    """
    img = Image.new("RGB", size, color=(100, 149, 237))  # cornflower blue
    dest = tmp_path / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    img.save(dest, format="JPEG", quality=95)
    img.close()
    return dest


def _solid_img(size: tuple = (800, 600)) -> Image.Image:
    """Return an in-memory RGB image (caller is responsible for closing)."""
    return Image.new("RGB", size, color=(100, 149, 237))


# ---------------------------------------------------------------------------
# _open_rgb
# ---------------------------------------------------------------------------

class TestOpenRgb:
    def test_jpeg_opens_as_rgb(self, tmp_path: Path):
        src = _solid(tmp_path)
        img = _open_rgb(src)
        assert img is not None
        assert img.mode == "RGB"
        img.close()

    def test_png_with_alpha_converts_to_rgb(self, tmp_path: Path):
        rgba = Image.new("RGBA", (100, 100), (0, 255, 0, 128))
        src = tmp_path / "alpha.png"
        rgba.save(src)
        img = _open_rgb(src)
        assert img is not None
        assert img.mode == "RGB"
        img.close()

    def test_raw_extension_returns_none(self, tmp_path: Path):
        # A file with a RAW extension — even if it contains JPEG bytes —
        # should return None because Pillow won't be asked to open it.
        src = tmp_path / "shot.arw"
        src.write_bytes(b"not real raw")
        assert _open_rgb(src) is None

    def test_unreadable_file_returns_none(self, tmp_path: Path):
        src = tmp_path / "broken.jpg"
        src.write_bytes(b"this is not a jpeg")
        assert _open_rgb(src) is None


# ---------------------------------------------------------------------------
# _unique_stem
# ---------------------------------------------------------------------------

class TestUniqueStem:
    def test_first_use_returns_stem(self, tmp_path: Path):
        p = tmp_path / "photo.jpg"
        seen: set[str] = set()
        assert _unique_stem(p, seen) == "photo"

    def test_collision_appends_extension(self, tmp_path: Path):
        seen: set[str] = {"photo"}
        p = tmp_path / "photo.png"
        stem = _unique_stem(p, seen)
        assert stem == "photo_png"

    def test_no_collision_no_suffix(self, tmp_path: Path):
        seen: set[str] = set()
        p = tmp_path / "unique.jpg"
        stem = _unique_stem(p, seen)
        assert "_jpg" not in stem


# ---------------------------------------------------------------------------
# _make_exact_copy
# ---------------------------------------------------------------------------

class TestExactCopy:
    def test_output_exists(self, tmp_path: Path):
        src = _solid(tmp_path, "orig.jpg")
        rec = _make_exact_copy(src, tmp_path, "orig")
        assert (tmp_path / rec.filename).exists()

    def test_bytes_are_identical(self, tmp_path: Path):
        src = _solid(tmp_path, "orig.jpg")
        rec = _make_exact_copy(src, tmp_path, "orig")
        assert src.read_bytes() == (tmp_path / rec.filename).read_bytes()

    def test_md5_matches_original(self, tmp_path: Path):
        src = _solid(tmp_path, "orig.jpg")
        rec = _make_exact_copy(src, tmp_path, "orig")
        orig_md5 = hashlib.md5(src.read_bytes()).hexdigest()
        copy_md5 = hashlib.md5((tmp_path / rec.filename).read_bytes()).hexdigest()
        assert orig_md5 == copy_md5

    def test_transformation_tag(self, tmp_path: Path):
        src = _solid(tmp_path, "orig.jpg")
        rec = _make_exact_copy(src, tmp_path, "orig")
        assert rec.transformation == "exact_copy"

    def test_filename_contains_exact(self, tmp_path: Path):
        src = _solid(tmp_path, "orig.jpg")
        rec = _make_exact_copy(src, tmp_path, "orig")
        assert "__exact" in rec.filename


# ---------------------------------------------------------------------------
# _make_resizes
# ---------------------------------------------------------------------------

class TestResizes:
    def test_produces_correct_number(self, tmp_path: Path):
        src = _solid(tmp_path, "big.jpg", size=(1200, 900))
        img = _open_rgb(src)
        recs = _make_resizes(img, src, tmp_path, "big", sizes=[128, 256, 512])
        img.close()
        assert len(recs) == 3

    def test_output_fits_within_requested_size(self, tmp_path: Path):
        src = _solid(tmp_path, "big.jpg", size=(1200, 900))
        img = _open_rgb(src)
        recs = _make_resizes(img, src, tmp_path, "big", sizes=[256])
        img.close()
        assert recs
        out_img = Image.open(tmp_path / recs[0].filename)
        assert max(out_img.size) <= 256
        out_img.close()

    def test_skips_upscale(self, tmp_path: Path):
        """An original smaller than the requested size must not be upscaled."""
        src = _solid(tmp_path, "small.jpg", size=(100, 100))
        img = _open_rgb(src)
        recs = _make_resizes(img, src, tmp_path, "small", sizes=[256, 512, 1024])
        img.close()
        assert recs == []

    def test_transformation_tag(self, tmp_path: Path):
        src = _solid(tmp_path, "big.jpg", size=(800, 600))
        img = _open_rgb(src)
        recs = _make_resizes(img, src, tmp_path, "big", sizes=[128])
        img.close()
        assert recs[0].transformation == "resize"
        assert "size=128" in recs[0].parameters


# ---------------------------------------------------------------------------
# _make_jpeg_recompressed
# ---------------------------------------------------------------------------

class TestJpegRecompressed:
    def test_produces_one_per_quality(self, tmp_path: Path):
        src = _solid(tmp_path)
        img = _open_rgb(src)
        recs = _make_jpeg_recompressed(img, src, tmp_path, "img", qualities=[85, 75])
        img.close()
        assert len(recs) == 2

    def test_all_files_are_valid_jpegs(self, tmp_path: Path):
        src = _solid(tmp_path)
        img = _open_rgb(src)
        recs = _make_jpeg_recompressed(img, src, tmp_path, "img", qualities=[75])
        img.close()
        out = Image.open(tmp_path / recs[0].filename)
        assert out.format == "JPEG"
        out.close()

    def test_lower_quality_produces_smaller_file(self, tmp_path: Path):
        src = _solid(tmp_path, size=(400, 400))
        img = _open_rgb(src)
        recs = _make_jpeg_recompressed(img, src, tmp_path, "img", qualities=[95, 60])
        img.close()
        size_q95 = (tmp_path / recs[0].filename).stat().st_size
        size_q60 = (tmp_path / recs[1].filename).stat().st_size
        assert size_q60 < size_q95


# ---------------------------------------------------------------------------
# _make_format_converts
# ---------------------------------------------------------------------------

class TestFormatConverts:
    def test_png_output_is_valid(self, tmp_path: Path):
        src = _solid(tmp_path)
        img = _open_rgb(src)
        recs = _make_format_converts(img, src, tmp_path, "img", formats=["PNG"])
        img.close()
        out = Image.open(tmp_path / recs[0].filename)
        assert out.format == "PNG"
        out.close()

    def test_webp_output_is_valid(self, tmp_path: Path):
        src = _solid(tmp_path)
        img = _open_rgb(src)
        recs = _make_format_converts(img, src, tmp_path, "img", formats=["WEBP"])
        img.close()
        out = Image.open(tmp_path / recs[0].filename)
        assert out.format == "WEBP"
        out.close()

    def test_transformation_tags(self, tmp_path: Path):
        src = _solid(tmp_path)
        img = _open_rgb(src)
        recs = _make_format_converts(img, src, tmp_path, "img", formats=["PNG", "WEBP"])
        img.close()
        tags = {r.transformation for r in recs}
        assert "format_png" in tags
        assert "format_webp" in tags


# ---------------------------------------------------------------------------
# _make_crops
# ---------------------------------------------------------------------------

class TestCrops:
    def test_cropped_image_is_smaller(self, tmp_path: Path):
        src = _solid(tmp_path, size=(800, 600))
        img = _open_rgb(src)
        recs = _make_crops(img, src, tmp_path, "img", fractions=[0.05])
        img.close()
        out = Image.open(tmp_path / recs[0].filename)
        assert out.width < 800
        assert out.height < 600
        out.close()

    def test_skips_if_image_too_small(self, tmp_path: Path):
        """A 4×4 image cropped by 50% would produce 0×0 — must be skipped."""
        src = _solid(tmp_path, size=(4, 4))
        img = _open_rgb(src)
        recs = _make_crops(img, src, tmp_path, "img", fractions=[0.5])
        img.close()
        assert recs == []


# ---------------------------------------------------------------------------
# _make_brightness / _make_contrast
# ---------------------------------------------------------------------------

class TestBrightnessContrast:
    def test_brightness_creates_files(self, tmp_path: Path):
        src = _solid(tmp_path)
        img = _open_rgb(src)
        recs = _make_brightness(img, src, tmp_path, "img", factors=[0.85, 1.15])
        img.close()
        assert len(recs) == 2
        assert all((tmp_path / r.filename).exists() for r in recs)

    def test_contrast_creates_files(self, tmp_path: Path):
        src = _solid(tmp_path)
        img = _open_rgb(src)
        recs = _make_contrast(img, src, tmp_path, "img", factors=[0.85, 1.15])
        img.close()
        assert len(recs) == 2
        assert all((tmp_path / r.filename).exists() for r in recs)

    def test_brightness_files_differ_from_original(self, tmp_path: Path):
        src = _solid(tmp_path, size=(200, 200))
        orig_bytes = src.read_bytes()
        img = _open_rgb(src)
        recs = _make_brightness(img, src, tmp_path, "img", factors=[1.15])
        img.close()
        variant_bytes = (tmp_path / recs[0].filename).read_bytes()
        assert variant_bytes != orig_bytes


# ---------------------------------------------------------------------------
# _make_rotations
# ---------------------------------------------------------------------------

class TestRotations:
    def test_90_degree_rotation_swaps_dimensions(self, tmp_path: Path):
        src = _solid(tmp_path, size=(800, 600))  # landscape
        img = _open_rgb(src)
        recs = _make_rotations(img, src, tmp_path, "img", angles=[90])
        img.close()
        out = Image.open(tmp_path / recs[0].filename)
        # After 90° rotation landscape becomes portrait.
        assert out.width < out.height
        out.close()

    def test_transformation_tag(self, tmp_path: Path):
        src = _solid(tmp_path)
        img = _open_rgb(src)
        recs = _make_rotations(img, src, tmp_path, "img", angles=[90])
        img.close()
        assert recs[0].transformation == "rotate90"


# ---------------------------------------------------------------------------
# build_dataset — integration tests
# ---------------------------------------------------------------------------

def _bench_cfg(tmp_path: Path) -> dict:
    """Minimal benchmark config for fast integration tests."""
    return {
        "ground_truth_filename": "ground_truth.csv",
        "resize_sizes": [128],           # 1 size
        "jpeg_qualities": [85],          # 1 quality
        "format_targets": ["PNG"],       # 1 format
        "crop_fractions": [0.05],        # 1 crop
        "brightness_factors": [1.15],    # 1 brightness
        "contrast_factors": [0.85],      # 1 contrast
        "rotate_angles": [90],           # 1 rotation
    }


class TestBuildDataset:
    def test_ground_truth_csv_exists(self, tmp_path: Path):
        originals = tmp_path / "originals"
        output = tmp_path / "benchmark"
        _solid(originals, "a.jpg", size=(400, 300))
        build_dataset(originals, output, _bench_cfg(tmp_path), verbose=False)
        assert (output / "ground_truth.csv").exists()

    def test_csv_has_correct_headers(self, tmp_path: Path):
        originals = tmp_path / "originals"
        output = tmp_path / "benchmark"
        _solid(originals, "a.jpg", size=(400, 300))
        build_dataset(originals, output, _bench_cfg(tmp_path), verbose=False)
        with open(output / "ground_truth.csv", newline="") as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames == CSV_FIELDNAMES

    def test_original_row_present(self, tmp_path: Path):
        originals = tmp_path / "originals"
        output = tmp_path / "benchmark"
        _solid(originals, "img.jpg", size=(400, 300))
        build_dataset(originals, output, _bench_cfg(tmp_path), verbose=False)
        with open(output / "ground_truth.csv", newline="") as f:
            rows = list(csv.DictReader(f))
        originals_rows = [r for r in rows if r["transformation"] == "original"]
        assert len(originals_rows) == 1
        assert originals_rows[0]["original_filename"] == "img.jpg"

    def test_all_transformations_represented(self, tmp_path: Path):
        originals = tmp_path / "originals"
        output = tmp_path / "benchmark"
        _solid(originals, "img.jpg", size=(600, 400))
        build_dataset(originals, output, _bench_cfg(tmp_path), verbose=False)
        with open(output / "ground_truth.csv", newline="") as f:
            transforms = {r["transformation"] for r in csv.DictReader(f)}
        expected = {
            "original", "exact_copy", "resize", "jpeg_recompress",
            "format_png", "crop", "brightness", "contrast", "rotate90",
        }
        assert expected == transforms

    def test_all_csv_files_exist_on_disk(self, tmp_path: Path):
        originals = tmp_path / "originals"
        output = tmp_path / "benchmark"
        _solid(originals, "img.jpg", size=(400, 300))
        build_dataset(originals, output, _bench_cfg(tmp_path), verbose=False)
        with open(output / "ground_truth.csv", newline="") as f:
            rows = list(csv.DictReader(f))
        missing = [r["filename"] for r in rows if not (output / r["filename"]).exists()]
        assert missing == [], f"Missing files: {missing}"

    def test_limit_flag(self, tmp_path: Path):
        originals = tmp_path / "originals"
        output = tmp_path / "benchmark"
        for i in range(5):
            _solid(originals, f"img_{i:02d}.jpg", size=(400, 300))
        stats = build_dataset(
            originals, output, _bench_cfg(tmp_path), limit=2, verbose=False
        )
        assert stats.originals_processed == 2

    def test_two_originals_no_filename_collision(self, tmp_path: Path):
        originals = tmp_path / "originals"
        output = tmp_path / "benchmark"
        _solid(originals, "a.jpg", size=(400, 300))
        _solid(originals, "b.jpg", size=(400, 300))
        build_dataset(originals, output, _bench_cfg(tmp_path), verbose=False)
        with open(output / "ground_truth.csv", newline="") as f:
            filenames = [r["filename"] for r in csv.DictReader(f)]
        assert len(filenames) == len(set(filenames)), "Duplicate filenames in CSV"

    def test_empty_originals_dir_returns_zero_stats(self, tmp_path: Path):
        originals = tmp_path / "originals"
        originals.mkdir()
        output = tmp_path / "benchmark"
        stats = build_dataset(originals, output, _bench_cfg(tmp_path), verbose=False)
        assert stats.originals_processed == 0
        assert stats.variants_created == 0

    def test_second_run_is_idempotent(self, tmp_path: Path):
        """Running build_dataset twice should not duplicate CSV rows."""
        originals = tmp_path / "originals"
        output = tmp_path / "benchmark"
        _solid(originals, "img.jpg", size=(400, 300))
        cfg = _bench_cfg(tmp_path)
        build_dataset(originals, output, cfg, verbose=False)
        build_dataset(originals, output, cfg, verbose=False)
        with open(output / "ground_truth.csv", newline="") as f:
            rows = list(csv.DictReader(f))
        # The CSV is rewritten on each run, so count should be same as single run.
        originals_count = sum(1 for r in rows if r["transformation"] == "original")
        assert originals_count == 1
