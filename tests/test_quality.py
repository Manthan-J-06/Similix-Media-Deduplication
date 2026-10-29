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
import os
from pathlib import Path
import sqlite3
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import pytest

from core.keeper import choose_keeper
from core.quality import (
    align_and_compute_metrics,
    compute_laplacian_sharpness,
    compute_source_likelihood_scores,
    load_proxy_rgb,
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

    def test_q85_resave_never_beats_higher_quality_original(self, tmp_path: Path):
        # (a) A q85 re-save with a clean/canonical filename never beats an original saved at higher quality (e.g. q95)
        # even if the q95 original has a longer or copy-like filename.
        img = _make_rich_test_image(800, 600)
        q95_path = tmp_path / "IMG_20261016_copy_raw.jpg"
        q85_path = tmp_path / "IMG_original.jpg"

        img.save(q95_path, format="JPEG", quality=95)
        img.save(q85_path, format="JPEG", quality=85)

        # Using real files so extract_jpeg_quality_estimate runs
        members = [
            {"id": 1, "width": 800, "height": 600, "file_size": q85_path.stat().st_size, "file_path": str(q85_path)},
            {"id": 2, "width": 800, "height": 600, "file_size": q95_path.stat().st_size, "file_path": str(q95_path)},
        ]
        ref = select_reference_member(members)
        assert ref["id"] == 2
        assert ref["file_path"] == str(q95_path)

    def test_cluster_picks_highest_quality_regardless_of_filename(self, tmp_path: Path):
        # (b) A cluster of IMG_0001.jpg, IMG_0001 (1).jpg, and IMG_0001-edited.jpg picks the highest-quality one
        img = _make_rich_test_image(800, 600)
        p_orig = tmp_path / "IMG_0001.jpg"
        p_copy = tmp_path / "IMG_0001 (1).jpg"
        p_edit = tmp_path / "IMG_0001-edited.jpg"

        # Case 1: IMG_0001 (1).jpg has highest quality (q95 vs q75 and q60)
        img.save(p_orig, format="JPEG", quality=75)
        img.save(p_copy, format="JPEG", quality=95)
        img.save(p_edit, format="JPEG", quality=60)

        members = [
            {"id": 1, "width": 800, "height": 600, "file_size": p_orig.stat().st_size, "file_path": str(p_orig)},
            {"id": 2, "width": 800, "height": 600, "file_size": p_copy.stat().st_size, "file_path": str(p_copy)},
            {"id": 3, "width": 800, "height": 600, "file_size": p_edit.stat().st_size, "file_path": str(p_edit)},
        ]
        ref = select_reference_member(members)
        assert ref["id"] == 2
        assert ref["file_path"] == str(p_copy)

        # Case 2: IMG_0001-edited.jpg has highest quality (q95 vs q80 and q70)
        img.save(p_orig, format="JPEG", quality=80)
        img.save(p_copy, format="JPEG", quality=70)
        img.save(p_edit, format="JPEG", quality=95)

        members = [
            {"id": 1, "width": 800, "height": 600, "file_size": p_orig.stat().st_size, "file_path": str(p_orig)},
            {"id": 2, "width": 800, "height": 600, "file_size": p_copy.stat().st_size, "file_path": str(p_copy)},
            {"id": 3, "width": 800, "height": 600, "file_size": p_edit.stat().st_size, "file_path": str(p_edit)},
        ]
        ref2 = select_reference_member(members)
        assert ref2["id"] == 3
        assert ref2["file_path"] == str(p_edit)

    def test_shuffled_filenames_give_same_reference(self, tmp_path: Path):
        # (c) Shuffled filenames give the same reference
        import random
        img = _make_rich_test_image(800, 600)
        files = []
        qualities = [95, 85, 75, 60, 40]
        # Create files with various qualities
        for q in qualities:
            p = tmp_path / f"temp_{q}.jpg"
            img.save(p, format="JPEG", quality=q)
            files.append((q, p))

        # Assign arbitrary names across multiple trials
        for trial in range(5):
            trial_members = []
            shuffled_names = [f"random_name_{trial}_{i}_{random.randint(1000, 9999)}.jpg" for i in range(len(files))]
            random.shuffle(shuffled_names)
            for (q, p), name in zip(files, shuffled_names):
                target = tmp_path / name
                target.write_bytes(p.read_bytes())
                trial_members.append({
                    "id": q,
                    "width": 800,
                    "height": 600,
                    "file_size": target.stat().st_size,
                    "file_path": str(target),
                })
            # Shuffle member list order
            random.shuffle(trial_members)
            ref = select_reference_member(trial_members)
            # The q95 image must ALWAYS be selected as reference regardless of name or list order
            assert ref["id"] == 95

    def test_same_quality_estimate_exact_copy_deterministic(self, tmp_path: Path):
        # (4) Same-quality-estimate case (exact copy and original) remains deterministic
        import random
        img = _make_rich_test_image(800, 600)
        p1 = tmp_path / "photo_a.jpg"
        p2 = tmp_path / "photo_b.jpg"
        img.save(p1, format="JPEG", quality=85)
        p2.write_bytes(p1.read_bytes())  # Exact copy

        # Set identical timestamps
        import os
        now = 1700000000.0
        os.utime(p1, (now, now))
        os.utime(p2, (now, now))

        members = [
            {"id": 1, "width": 800, "height": 600, "file_size": p1.stat().st_size, "file_path": str(p1), "modified_time": now},
            {"id": 2, "width": 800, "height": 600, "file_size": p2.stat().st_size, "file_path": str(p2), "modified_time": now},
        ]

        # Order must be deterministic regardless of input list permutation
        first_pick = select_reference_member(members)["id"]
        for _ in range(10):
            shuffled = list(members)
            random.shuffle(shuffled)
            assert select_reference_member(shuffled)["id"] == first_pick

    def test_original_q75_beats_q85_resave_via_source_likelihood(self, tmp_path: Path):
        # 5(a) original q75 + q85 re-save + q60 re-save + q30: the original is the reference
        im = _make_rich_test_image(800, 600)
        p_orig = tmp_path / "original_q75.jpg"
        p_q85 = tmp_path / "resave_q85.jpg"
        p_q60 = tmp_path / "resave_q60.jpg"
        p_q30 = tmp_path / "resave_q30.jpg"

        # Save true original at q75
        im.save(p_orig, format="JPEG", quality=75)
        with Image.open(p_orig) as orig_loaded:
            # Re-save copies strictly FROM the q75 original!
            orig_loaded.save(p_q85, format="JPEG", quality=85)
            orig_loaded.save(p_q60, format="JPEG", quality=60)
            orig_loaded.save(p_q30, format="JPEG", quality=30)

        members = [
            {"id": 1, "width": 800, "height": 600, "file_size": p_orig.stat().st_size, "file_path": str(p_orig)},
            {"id": 2, "width": 800, "height": 600, "file_size": p_q85.stat().st_size, "file_path": str(p_q85)},
            {"id": 3, "width": 800, "height": 600, "file_size": p_q60.stat().st_size, "file_path": str(p_q60)},
            {"id": 4, "width": 800, "height": 600, "file_size": p_q30.stat().st_size, "file_path": str(p_q30)},
        ]

        ref = select_reference_member(members)
        assert ref["id"] == 1
        assert ref["file_path"] == str(p_orig)

        # Source-likelihood score of original must strictly exceed q85 re-save
        id_to_score = {m["id"]: m["source_likelihood"] for m in members}
        assert id_to_score[1] is not None and id_to_score[2] is not None
        assert id_to_score[1] > id_to_score[2]

    def test_two_copies_tied_within_0002_set_reference_uncertain(self, tmp_path: Path):
        # Two distinct candidates tied within 0.002 set reference_uncertain = 1
        im = _make_rich_test_image(800, 600)
        p1 = tmp_path / "copy_a.jpg"
        p2 = tmp_path / "copy_b.jpg"
        im.save(p1, format="JPEG", quality=85)
        # Distinct candidate with 1 pixel changed (distinct md5, but SSIM difference < 0.002)
        im2 = im.copy()
        im2.putpixel((0, 0), (255, 255, 255))
        im2.save(p2, format="JPEG", quality=85)

        members = [
            {"id": 1, "width": 800, "height": 600, "file_size": p1.stat().st_size, "file_path": str(p1)},
            {"id": 2, "width": 800, "height": 600, "file_size": p2.stat().st_size, "file_path": str(p2)},
        ]

        ref = select_reference_member(members)
        assert ref["reference_uncertain"] == 1

        # In conservative mode, an uncertain cluster gets no delete_safe members; everything goes to review
        res = choose_keeper(members, mode="conservative")
        assert res["reference_uncertain"] == 1
        assert res["delete_safe_ids"] == []
        assert res["review_ids"] == [2] if ref["id"] == 1 else [1]
        assert "uncertain" in res["reasons"][res["review_ids"][0]].lower()

    def test_exact_copy_collapsed_reference_uncertain_zero_and_delete_safe_in_every_mode(self, tmp_path: Path):
        """
        1. Before computing source-likelihood or the uncertainty check, collapse
        byte-identical members (same md5 and size) into one candidate. The
        reference_uncertain flag must compare only distinct candidates.
        A cluster of an original, its exact copy, and q60/q30 re-saves must have
        reference_uncertain = 0, and the exact copy must still be delete_safe in every mode.
        """
        im = _make_rich_test_image(800, 600)
        p_orig = tmp_path / "original.jpg"
        p_exact = tmp_path / "original__exact_copy.jpg"
        p_q60 = tmp_path / "resave_q60.jpg"
        p_q30 = tmp_path / "resave_q30.jpg"

        im.save(p_orig, format="JPEG", quality=75)
        p_exact.write_bytes(p_orig.read_bytes())  # Byte-identical exact copy

        with Image.open(p_orig) as orig_loaded:
            orig_loaded.save(p_q60, format="JPEG", quality=60)
            orig_loaded.save(p_q30, format="JPEG", quality=30)

        import hashlib
        orig_md5 = hashlib.md5(p_orig.read_bytes()).hexdigest()
        q60_md5 = hashlib.md5(p_q60.read_bytes()).hexdigest()
        q30_md5 = hashlib.md5(p_q30.read_bytes()).hexdigest()

        members = [
            {"id": 1, "width": 800, "height": 600, "file_size": p_orig.stat().st_size, "file_path": str(p_orig), "md5": orig_md5, "quality_status": "reference"},
            {"id": 2, "width": 800, "height": 600, "file_size": p_exact.stat().st_size, "file_path": str(p_exact), "md5": orig_md5, "quality_status": "aligned"},
            {"id": 3, "width": 800, "height": 600, "file_size": p_q60.stat().st_size, "file_path": str(p_q60), "md5": q60_md5, "quality_status": "aligned"},
            {"id": 4, "width": 800, "height": 600, "file_size": p_q30.stat().st_size, "file_path": str(p_q30), "md5": q30_md5, "quality_status": "aligned"},
        ]

        ref = select_reference_member(members)
        assert ref["id"] == 1
        # reference_uncertain must be 0 because exact copy was collapsed and original clearly beats q60/q30
        assert ref["reference_uncertain"] == 0

        # Exact copy must still be delete_safe in every mode
        for mode in ["conservative", "balanced", "aggressive"]:
            res = choose_keeper(members, mode=mode)
            assert res["reference_uncertain"] == 0
            assert 2 in res["delete_safe_ids"], f"Exact copy (id=2) was not delete_safe in mode {mode}"
            assert res["keeper_id"] != 2

    def test_shuffled_and_renamed_files_give_same_reference(self, tmp_path: Path):
        # 5(c) shuffled and renamed files give the same result
        import random
        im = _make_rich_test_image(800, 600)
        p_orig = tmp_path / "base_original.jpg"
        im.save(p_orig, format="JPEG", quality=75)

        files = [("orig", p_orig)]
        with Image.open(p_orig) as orig_loaded:
            for q in [85, 60, 30]:
                p = tmp_path / f"temp_resave_q{q}.jpg"
                orig_loaded.save(p, format="JPEG", quality=q)
                files.append((f"q{q}", p))

        # Run across multiple trials with arbitrary names and shuffled order
        for trial in range(5):
            trial_members = []
            shuffled_names = [f"img_{trial}_{i}_{random.randint(1000, 9999)}.jpg" for i in range(len(files))]
            random.shuffle(shuffled_names)
            for (tag, src_path), name in zip(files, shuffled_names):
                target = tmp_path / name
                target.write_bytes(src_path.read_bytes())
                trial_members.append({
                    "id": tag,
                    "width": 800,
                    "height": 600,
                    "file_size": target.stat().st_size,
                    "file_path": str(target),
                })
            random.shuffle(trial_members)
            ref = select_reference_member(trial_members)
            # The true original must ALWAYS be chosen as the reference
            assert ref["id"] == "orig"

    def test_source_likelihood_null_when_no_peers(self, tmp_path: Path):
        """source_likelihood for a member with no same-resolution peers must be NULL (None), never 1.0."""
        im1 = _make_rich_test_image(800, 600)
        im2 = _make_rich_test_image(400, 300)
        p1 = tmp_path / "img_800x600_a.jpg"
        p2 = tmp_path / "img_800x600_b.jpg"
        p3 = tmp_path / "img_400x300_lone.jpg"

        im1.save(p1, format="JPEG", quality=85)
        with Image.open(p1) as loaded:
            loaded.save(p2, format="JPEG", quality=75)
        im2.save(p3, format="JPEG", quality=85)

        members = [
            {"id": 1, "width": 800, "height": 600, "file_size": p1.stat().st_size, "file_path": str(p1)},
            {"id": 2, "width": 800, "height": 600, "file_size": p2.stat().st_size, "file_path": str(p2)},
            {"id": 3, "width": 400, "height": 300, "file_size": p3.stat().st_size, "file_path": str(p3)},
        ]

        scores = compute_source_likelihood_scores(members)
        assert scores[1] is not None
        assert scores[2] is not None
        assert scores[3] is None, f"Expected None for lone member, got {scores[3]}"

        # Ensure select_reference_member handles None without crashing or reading None as score
        ref = select_reference_member(members)
        assert ref["id"] == 1
        assert members[2]["source_likelihood"] is None

    def test_refined_uncertainty_rule_head_to_head_majority(self, tmp_path: Path):
        """
        When derivation test has no winner for the top two candidates (gap < 0.005),
        reference_uncertain = 1. Majority vote still picks the reference (orig id=1),
        but must not clear the flag.
        """
        im = _make_rich_test_image(800, 600)
        p_orig = tmp_path / "orig.jpg"
        p_q85 = tmp_path / "resave_q85.jpg"
        p_q60 = tmp_path / "resave_q60.jpg"
        p_q30 = tmp_path / "resave_q30.jpg"

        im.save(p_orig, format="JPEG", quality=75)
        with Image.open(p_orig) as orig_loaded:
            orig_loaded.save(p_q85, format="JPEG", quality=85)
            orig_loaded.save(p_q60, format="JPEG", quality=60)
            orig_loaded.save(p_q30, format="JPEG", quality=30)

        # Set earlier mtime on original
        now = time.time()
        os.utime(p_orig, (now - 1000, now - 1000))
        os.utime(p_q85, (now - 500, now - 500))

        members = [
            {"id": 1, "width": 800, "height": 600, "file_size": p_orig.stat().st_size, "file_path": str(p_orig)},
            {"id": 2, "width": 800, "height": 600, "file_size": p_q85.stat().st_size, "file_path": str(p_q85)},
            {"id": 3, "width": 800, "height": 600, "file_size": p_q60.stat().st_size, "file_path": str(p_q60)},
            {"id": 4, "width": 800, "height": 600, "file_size": p_q30.stat().st_size, "file_path": str(p_q30)},
        ]

        ref = select_reference_member(members)
        assert ref["id"] == 1
        assert ref["reference_uncertain"] == 1
        assert ref.get("head_to_head_result") is not None

    def test_derivation_original_and_q60_resave_confident(self, tmp_path: Path):
        """Original + its q60 re-save -> confident (reference_uncertain = 0), source = original."""
        im = _make_rich_test_image(800, 600)
        p_orig = tmp_path / "original.jpg"
        p_q60 = tmp_path / "resave_q60.jpg"

        im.save(p_orig, format="JPEG", quality=75)
        with Image.open(p_orig) as orig_loaded:
            orig_loaded.save(p_q60, format="JPEG", quality=60)

        members = [
            {"id": 1, "width": 800, "height": 600, "file_size": p_orig.stat().st_size, "file_path": str(p_orig)},
            {"id": 2, "width": 800, "height": 600, "file_size": p_q60.stat().st_size, "file_path": str(p_q60)},
        ]

        ref = select_reference_member(members)
        assert ref["id"] == 1
        assert ref["reference_uncertain"] == 0
        assert ref.get("derivation_winner") == "A"
        assert ref.get("derivation_gap") is not None and ref["derivation_gap"] >= 0.005

    def test_derivation_two_independently_saved_q75_files_uncertain(self, tmp_path: Path):
        """Two independently saved q75 files of the same raw pixels -> uncertain (reference_uncertain = 1)."""
        im = _make_rich_test_image(800, 600)
        p1 = tmp_path / "indep1.jpg"
        p2 = tmp_path / "indep2.jpg"

        im.save(p1, format="JPEG", quality=75)
        im2 = im.copy()
        im2.putpixel((0, 0), (255, 255, 255))
        im2.save(p2, format="JPEG", quality=75)

        now = time.time()
        os.utime(p1, (now - 100, now - 100))
        os.utime(p2, (now - 50, now - 50))

        members = [
            {"id": 1, "width": 800, "height": 600, "file_size": p1.stat().st_size, "file_path": str(p1), "modified_time": now - 100},
            {"id": 2, "width": 800, "height": 600, "file_size": p2.stat().st_size, "file_path": str(p2), "modified_time": now - 50},
        ]

        ref = select_reference_member(members)
        assert ref["reference_uncertain"] == 1
        assert ref.get("derivation_winner") is None

    def test_derivation_q85_resave_of_q75_original_picks_original_as_source(self, tmp_path: Path):
        """
        A q85 re-save of a q75 original with default mtimes: derivation gap is < 0.005
        so cluster is uncertain (reference_uncertain = 1), but reference is still the original.
        """
        im = _make_rich_test_image(800, 600)
        p_q75 = tmp_path / "original_q75.jpg"
        p_q85 = tmp_path / "resave_q85.jpg"

        im.save(p_q75, format="JPEG", quality=75)
        with Image.open(p_q75) as orig_loaded:
            orig_loaded.save(p_q85, format="JPEG", quality=85)

        members = [
            {"id": "orig_q75", "width": 800, "height": 600, "file_size": p_q75.stat().st_size, "file_path": str(p_q75)},
            {"id": "resave_q85", "width": 800, "height": 600, "file_size": p_q85.stat().st_size, "file_path": str(p_q85)},
        ]

        ref = select_reference_member(members)
        assert ref["id"] == "orig_q75", f"Expected orig_q75 to be source, got {ref['id']}"
        assert ref["reference_uncertain"] == 1




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

    def test_load_proxy_rgb_prioritizes_local_disk_path_over_source(self, tmp_path: Path):
        """When an image exists on local disk, load_proxy_rgb must open it directly without delegating to source."""
        p = tmp_path / "local.jpg"
        im = Image.new("RGB", (100, 100), color=(10, 20, 30))
        im.save(p)

        class FailingSource:
            def get_thumbnail(self, *args, **kwargs):
                raise RuntimeError("Source should not be called for local disk paths!")

        arr, scale, dims = load_proxy_rgb(str(p), proxy_size=512, source=FailingSource())
        assert dims == (100, 100)
        assert arr.shape == (100, 100, 3)

    def test_match_pct_computed_from_real_pixels_q70_q30_blur(self, base_image: Image.Image, tmp_path: Path):
        """Verify match_pct is computed from real pixels (q70 is not falsely 100%, q30 is lower, blur has lower sharpness)."""
        ref_path = tmp_path / "ref.jpg"
        base_image.save(ref_path, format="JPEG", quality=95)

        q70_path = tmp_path / "copy_q70.jpg"
        base_image.save(q70_path, format="JPEG", quality=70)

        q30_path = tmp_path / "copy_q30.jpg"
        base_image.save(q30_path, format="JPEG", quality=30)

        blur_path = tmp_path / "copy_blur.jpg"
        base_image.filter(ImageFilter.GaussianBlur(radius=2.0)).save(blur_path, format="JPEG", quality=95)

        m_exact = align_and_compute_metrics(ref_path, ref_path)
        m_q70 = align_and_compute_metrics(q70_path, ref_path)
        m_q30 = align_and_compute_metrics(q30_path, ref_path)
        m_blur = align_and_compute_metrics(blur_path, ref_path)

        assert m_exact.quality_status == "aligned"
        assert m_q70.quality_status == "aligned"
        assert m_q30.quality_status == "aligned"
        assert m_blur.quality_status == "aligned"

        # Exact match SSIM color is 1.0 (100%)
        assert m_exact.ssim_color == 1.0

        # q70 must be real pixel SSIM, strictly less than 1.0 (never false 100%)
        assert m_q70.ssim_color is not None
        assert 0.90 <= m_q70.ssim_color < 1.0
        q70_pct = round(m_q70.ssim_color * 100.0, 1)
        assert q70_pct < 100.0

        # q30 must have lower SSIM than q70
        assert m_q30.ssim_color is not None
        assert m_q30.ssim_color < m_q70.ssim_color

        # Blur must trigger sharpness reduction
        assert m_blur.sharpness_ratio < 0.50
        assert m_blur.blur_flag is True

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
        assert rows[ref_id]["match_pct"] == 100.0
        assert rows[ref_id]["resolution_pct"] == 100.0

        assert rows[other_id]["quality_status"] == "aligned"
        assert rows[other_id]["ssim"] >= 0.90
        assert rows[other_id]["match_pct"] is not None
        assert rows[other_id]["resolution_pct"] == 100.0

        assert rows[3]["quality_status"] == "unalignable"
        assert rows[3]["ssim"] is None
        assert rows[3]["match_pct"] is None


class TestMatchAndResolutionPct:
    def test_reference_exact_copy_and_rescaled_pct(self, tmp_path: Path):
        """Verify match_pct and resolution_pct values for reference, exact copy, and rescaled/lossy member."""
        from core.quality import run_quality_pipeline

        db_path = tmp_path / "test_pct.db"
        conn = init_db(db_path)

        img_dir = tmp_path / "images"
        img_dir.mkdir()

        base = _make_rich_test_image(800, 600)
        p_ref = img_dir / "ref.jpg"
        base.save(p_ref, format="JPEG", quality=95)

        # Exact byte copy
        p_exact = img_dir / "exact.jpg"
        p_exact.write_bytes(p_ref.read_bytes())

        # Scaled down 50%
        p_scale = img_dir / "scale.jpg"
        base.resize((400, 300)).save(p_scale, format="JPEG", quality=85)

        conn.execute("INSERT INTO clusters (id, stage_detected, member_count) VALUES (1, 0, 3)")
        conn.execute(
            """
            INSERT INTO images (id, file_path, md5, width, height, file_size, status, cluster_id, is_keeper)
            VALUES
              (1, ?, 'md5_ref', 800, 600, ?, 'clustered', 1, 1),
              (2, ?, 'md5_ref', 800, 600, ?, 'clustered', 1, 0),
              (3, ?, 'md5_scale', 400, 300, ?, 'clustered', 1, 0)
            """,
            (
                str(p_ref), p_ref.stat().st_size,
                str(p_exact), p_exact.stat().st_size,
                str(p_scale), p_scale.stat().st_size,
            ),
        )
        conn.execute("UPDATE clusters SET representative_id = 1 WHERE id = 1")
        conn.commit()

        run_quality_pipeline(conn)

        rows = {r["id"]: r for r in conn.execute("SELECT * FROM images").fetchall()}

        # Reference
        assert rows[1]["match_pct"] == 100.0
        assert rows[1]["resolution_pct"] == 100.0

        # Exact copy
        assert rows[2]["match_pct"] == 100.0
        assert rows[2]["resolution_pct"] == 100.0

        # Scaled copy: resolution_pct is pixel_ratio * 100 = (400*300)/(800*600)*100 = 25.0%
        assert rows[3]["resolution_pct"] == 25.0
        assert rows[3]["match_pct"] is not None
        assert 80.0 <= rows[3]["match_pct"] <= 100.0
