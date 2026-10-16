"""
tests/test_quality.py — Unit and integration tests for core/quality.py (Stage 4 prep).

Tests:
1. Reference member selection logic:
   - Primary: highest pixel count
   - Secondary: least compressed (largest bytes per pixel)
   - Tertiary: shortest file path length
2. Synthetic alignment and quality scoring cases:
   - Identical image: ssim ~ 1.0
   - JPEG q30: lower ssim than identical
   - Blurred copy: low sharpness ratio
   - 90-degree rotated copy: aligned ssim stays high
   - Unrelated image: rejected as unalignable
3. Auto-delete safety:
   - Members with quality_status == 'unalignable' are strictly excluded from get_auto_delete_candidates
4. Database pipeline integration:
   - run_quality_pipeline updates images table with all metrics and statuses
"""

from __future__ import annotations

import io
from pathlib import Path
import sqlite3

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import pytest

from core.quality import (
    align_and_compute_metrics,
    compute_laplacian_sharpness,
    run_quality_pipeline,
    select_reference_member,
)
from core.verify import get_auto_delete_candidates
from db.connection import init_db


def _make_rich_test_image(w: int = 800, h: int = 600) -> Image.Image:
    """Generate a synthetic test image with high-contrast geometric features for SIFT."""
    im = Image.new("RGB", (w, h), color=(220, 220, 220))
    draw = ImageDraw.Draw(im)
    for i in range(0, w, 35):
        draw.line([(i, 0), (w - i, h)], fill=((i * 3) % 255, (i * 7) % 255, (i * 11) % 255), width=3)
    for j in range(0, h, 35):
        draw.ellipse([j, j, j + 70, j + 70], outline=(40, 120, 210), width=3)
    draw.rectangle([150, 100, 450, 350], fill=(230, 80, 40), outline=(10, 10, 10), width=4)
    draw.rectangle([480, 150, 720, 500], fill=(40, 180, 90), outline=(10, 10, 10), width=4)
    return im


def _make_unrelated_image(w: int = 800, h: int = 600) -> Image.Image:
    """Generate an unrelated synthetic image with different structure and palette."""
    im = Image.new("RGB", (w, h), color=(15, 20, 25))
    draw = ImageDraw.Draw(im)
    for k in range(0, w, 20):
        draw.line([(k, h), (w, k)], fill=(180, 180, (k * 5) % 255), width=1)
    return im


class TestReferenceMemberSelection:
    def test_highest_pixel_count_selected_first(self):
        members = [
            {"id": 1, "width": 800, "height": 600, "file_size": 100000, "file_path": "/photos/small.jpg"},
            {"id": 2, "width": 1600, "height": 1200, "file_size": 200000, "file_path": "/photos/large.jpg"},
            {"id": 3, "width": 1024, "height": 768, "file_size": 150000, "file_path": "/photos/medium.jpg"},
        ]
        ref = select_reference_member(members)
        assert ref["id"] == 2

    def test_smallest_file_size_preferred_among_equal_pixels(self):
        # Two JPEGs of identical resolution: smaller file size is chosen as reference
        members = [
            {"id": 1, "width": 1024, "height": 768, "file_size": 58000, "file_path": "/photos/original.jpg"},
            {"id": 2, "width": 1024, "height": 768, "file_size": 85000, "file_path": "/photos/q85_resave.jpg"},
        ]
        ref = select_reference_member(members)
        assert ref["id"] == 1

    def test_q85_resave_larger_file_must_not_become_reference(self):
        # A q85 re-save of an original (larger file, same pixels) must NOT become the reference
        members = [
            {"id": 10, "width": 1024, "height": 768, "file_size": 58850, "file_path": "/photos/picsum_0001.jpg", "format": "JPEG"},
            {"id": 11, "width": 1024, "height": 768, "file_size": 75719, "file_path": "/photos/picsum_0001__jpeg_q85.jpg", "format": "JPEG"},
        ]
        ref = select_reference_member(members)
        assert ref["id"] == 10
        assert ref["file_path"] == "/photos/picsum_0001.jpg"

    def test_lossless_preferred_only_if_no_lossy_twin(self):
        # When a lossy twin exists of same pixels, prefer smallest file size (original JPEG over inflated PNG)
        members_with_twin = [
            {"id": 1, "width": 1024, "height": 768, "file_size": 60000, "file_path": "/photos/orig.jpg", "format": "JPEG"},
            {"id": 2, "width": 1024, "height": 768, "file_size": 500000, "file_path": "/photos/conv.png", "format": "PNG"},
        ]
        ref1 = select_reference_member(members_with_twin)
        assert ref1["id"] == 1  # 60KB JPEG wins over 500KB PNG transcode

        # When NO lossy twin exists of same pixels (e.g. only PNGs, or JPEG is lower-res), prefer lossless
        members_no_twin = [
            {"id": 1, "width": 1024, "height": 768, "file_size": 500000, "file_path": "/photos/orig.png", "format": "PNG"},
            {"id": 2, "width": 512, "height": 384, "file_size": 30000, "file_path": "/photos/thumb.jpg", "format": "JPEG"},
        ]
        ref2 = select_reference_member(members_no_twin)
        assert ref2["id"] == 1

    def test_shortest_path_selected_on_tie(self):
        # Both identical dimensions and size. Member 1 has shorter path.
        members = [
            {"id": 1, "width": 1024, "height": 768, "file_size": 100000, "file_path": "/a.jpg"},
            {"id": 2, "width": 1024, "height": 768, "file_size": 100000, "file_path": "/very/long/path/copy.jpg"},
        ]
        ref = select_reference_member(members)
        assert ref["id"] == 1


@pytest.fixture(scope="module")
def base_image() -> Image.Image:
    return _make_rich_test_image(800, 600)


class TestSyntheticQualityCases:

    def test_identical_image(self, base_image: Image.Image):
        m = align_and_compute_metrics(base_image, base_image)
        assert m.quality_status == "aligned"
        assert m.ssim is not None and m.ssim >= 0.99
        assert m.ssim_color is not None and m.ssim_color >= 0.99
        assert pytest.approx(m.sharpness_ratio, rel=1e-2) == 1.0
        assert m.pixel_ratio == 1.0

    def test_jpeg_q30_lower_ssim(self, base_image: Image.Image):
        buf = io.BytesIO()
        base_image.save(buf, format="JPEG", quality=30)
        buf.seek(0)
        q30_im = Image.open(buf)

        m_identical = align_and_compute_metrics(base_image, base_image)
        m_q30 = align_and_compute_metrics(q30_im, base_image)

        assert m_q30.quality_status == "aligned"
        assert m_q30.ssim is not None
        assert m_q30.ssim < m_identical.ssim
        assert m_q30.sharpness_ratio < 1.0

    def test_blurred_copy_low_sharpness_ratio(self, base_image: Image.Image):
        blurred_im = base_image.filter(ImageFilter.GaussianBlur(radius=1.5))
        m = align_and_compute_metrics(blurred_im, base_image)

        assert m.quality_status == "aligned"
        assert m.sharpness_ratio < 0.20  # significantly lower sharpness than reference
        assert m.ssim is not None and m.ssim > 0.80

    def test_90_degree_rotated_copy_aligned_ssim_stays_high(self, base_image: Image.Image):
        rot_im = base_image.rotate(90, expand=True)
        m = align_and_compute_metrics(rot_im, base_image)

        assert m.quality_status == "aligned"
        assert m.ssim is not None and m.ssim >= 0.90
        assert m.ssim_color is not None and m.ssim_color >= 0.90
        assert pytest.approx(m.sharpness_ratio, rel=0.1) == 1.0

    def test_unrelated_image_rejected_as_unalignable(self, base_image: Image.Image):
        unrel_im = _make_unrelated_image(800, 600)
        m = align_and_compute_metrics(unrel_im, base_image)

        assert m.quality_status == "unalignable"
        assert m.ssim is None
        assert m.ssim_color is None


class TestAutoDeleteExclusion:
    def test_unalignable_excluded_from_auto_delete(self, tmp_path: Path):
        db_path = tmp_path / "test.db"
        conn = init_db(db_path)

        # Create cluster
        conn.execute("INSERT INTO clusters (id, stage_detected, member_count) VALUES (1, 0, 3)")
        conn.execute(
            """
            INSERT INTO images (id, file_path, status, cluster_id, is_keeper, is_low_confidence, quality_status)
            VALUES
              (1, '/test/rep.jpg', 'clustered', 1, 1, 0, 'reference'),
              (2, '/test/aligned.jpg', 'clustered', 1, 0, 0, 'aligned'),
              (3, '/test/unalignable.jpg', 'clustered', 1, 0, 0, 'unalignable')
            """
        )
        conn.execute("UPDATE clusters SET representative_id = 1 WHERE id = 1")
        conn.commit()

        candidates = get_auto_delete_candidates(conn)
        cand_ids = [c["id"] for c in candidates]

        # Only the aligned non-keeper should be eligible
        assert cand_ids == [2]
        assert 3 not in cand_ids


class TestQualityPipelineIntegration:
    def test_run_quality_pipeline(self, tmp_path: Path):
        db_path = tmp_path / "test.db"
        conn = init_db(db_path)

        # Create images on disk
        img_dir = tmp_path / "images"
        img_dir.mkdir()

        base_ref = _make_rich_test_image(1200, 900)
        ref_path = img_dir / "ref.jpg"
        base_ref.save(ref_path)

        rot_path = img_dir / "rot90.jpg"
        base_ref.rotate(90, expand=True).save(rot_path)

        unrel_path = img_dir / "unrel.jpg"
        _make_unrelated_image(800, 600).save(unrel_path)

        # Setup database cluster
        conn.execute("INSERT INTO clusters (id, stage_detected, member_count) VALUES (1, 0, 3)")
        conn.execute(
            """
            INSERT INTO images (id, file_path, width, height, file_size, status, cluster_id, is_keeper)
            VALUES
              (1, ?, 1200, 900, ?, 'clustered', 1, 1),
              (2, ?, 900, 1200, ?, 'clustered', 1, 0),
              (3, ?, 800, 600, ?, 'clustered', 1, 0)
            """,
            (
                str(ref_path), ref_path.stat().st_size,
                str(rot_path), rot_path.stat().st_size,
                str(unrel_path), unrel_path.stat().st_size,
            ),
        )
        conn.execute("UPDATE clusters SET representative_id = 1 WHERE id = 1")
        conn.commit()

        res = run_quality_pipeline(conn)

        assert res.clusters_evaluated == 1
        assert res.members_evaluated == 3
        assert res.members_aligned == 1
        assert res.members_unalignable == 1

        # Check DB columns
        rows = {r["id"]: r for r in conn.execute("SELECT * FROM images").fetchall()}
        ref_id = 1 if rows[1]["quality_status"] == "reference" else 2
        other_id = 2 if ref_id == 1 else 1
        assert rows[ref_id]["quality_status"] == "reference"
        assert rows[ref_id]["ssim"] == 1.0

        assert rows[other_id]["quality_status"] == "aligned"
        assert rows[other_id]["ssim"] >= 0.90

        assert rows[3]["quality_status"] == "unalignable"
        assert rows[3]["ssim"] is None
