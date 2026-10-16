"""
tests/test_keeper.py — Tests for core/keeper.py.

Covers:
1. Exact-copy cluster
2. Rotated copy
3. Tiny thumbnail rejected by resolution floor
4. Unalignable member never delete_safe
5. Ties are deterministic
6. All three modes return different results on a crafted cluster
7. Safety invariant: never mark every member as delete_safe
8. Fused duplicate confidence scoring
"""

from __future__ import annotations

import sqlite3
import pytest

from core.keeper import (
    choose_keeper,
    predict_duplicate_confidence,
    run_keeper_pipeline,
    ensure_keeper_columns,
)


class TestExactCopyCluster:
    def test_exact_copy_cluster(self):
        members = [
            {
                "id": 1,
                "file_path": "/photos/original.jpg",
                "file_size": 100_000,
                "width": 1024,
                "height": 768,
                "md5": "abc123md5",
                "quality_status": "reference",
                "ssim": 1.0,
                "ssim_color": 1.0,
                "pixel_ratio": 1.0,
                "sharpness_ratio": 1.0,
                "is_low_confidence": 0,
            },
            {
                "id": 2,
                "file_path": "/photos/copy.jpg",
                "file_size": 100_000,
                "width": 1024,
                "height": 768,
                "md5": "abc123md5",
                "quality_status": "aligned",
                "ssim": 1.0,
                "ssim_color": 1.0,
                "pixel_ratio": 1.0,
                "sharpness_ratio": 1.0,
                "is_low_confidence": 0,
            },
        ]
        res = choose_keeper(members, mode="balanced")
        # Shorter path "/photos/copy.jpg" vs "/photos/original.jpg" (16 vs 20 chars) -> id 2 keeper
        assert res["keeper_id"] in (1, 2)
        other_id = 1 if res["keeper_id"] == 2 else 2
        assert other_id in res["delete_safe_ids"]
        assert len(res["review_ids"]) == 0
        assert res["total_savings_bytes"] == 100_000
        assert res["quality_retained"][other_id] == 100.0


class TestRotatedCopy:
    def test_rotated_copy_qualifies_and_delete_safe(self):
        members = [
            {
                "id": 10,
                "file_path": "/photos/scene_ref.jpg",
                "file_size": 200_000,
                "width": 1024,
                "height": 768,
                "md5": "md5_ref",
                "quality_status": "reference",
                "ssim": 1.0,
                "ssim_color": 1.0,
                "pixel_ratio": 1.0,
                "sharpness_ratio": 1.0,
                "is_low_confidence": 0,
            },
            {
                "id": 11,
                "file_path": "/photos/scene_rot90.jpg",
                "file_size": 180_000,
                "width": 768,
                "height": 1024,
                "md5": "md5_rot",
                "quality_status": "aligned",
                "ssim": 0.998,
                "ssim_color": 0.995,
                "pixel_ratio": 1.0,
                "sharpness_ratio": 0.99,
                "is_low_confidence": 0,
            },
        ]
        # Rotated copy is smaller (180KB < 200KB) and meets conservative limits (ssim 0.998 >= 0.98)
        res = choose_keeper(members, mode="conservative")
        assert res["keeper_id"] == 11
        assert 10 in res["delete_safe_ids"]
        assert res["total_savings_bytes"] == 200_000
        assert res["quality_retained"][11] == 99.8


class TestTinyThumbnailRejectedByFloor:
    def test_tiny_thumbnail_rejected_by_resolution_floor(self):
        members = [
            {
                "id": 1,
                "file_path": "/photos/original.jpg",
                "file_size": 1_000_000,
                "width": 1024,
                "height": 768,
                "md5": "md5_orig",
                "quality_status": "reference",
                "ssim": 1.0,
                "ssim_color": 1.0,
                "pixel_ratio": 1.0,
                "sharpness_ratio": 1.0,
                "is_low_confidence": 0,
            },
            {
                "id": 2,
                "file_path": "/photos/tiny_thumb.jpg",
                "file_size": 8_000,  # Much smaller file!
                "width": 128,
                "height": 96,
                "md5": "md5_thumb",
                "quality_status": "aligned",
                "ssim": 0.85,
                "ssim_color": 0.82,
                "pixel_ratio": 0.0156,  # 1.5% resolution floor
                "sharpness_ratio": 0.90,
                "is_low_confidence": 0,
            },
        ]
        # In all modes, pixel_ratio 0.0156 is well below pixel_ratio_floor (0.90, 0.60, 0.25)
        for mode in ["conservative", "balanced", "aggressive"]:
            res = choose_keeper(members, mode=mode)
            # Despite being 8KB vs 1MB, thumbnail MUST NOT be chosen as keeper!
            assert res["keeper_id"] == 1
            # Thumbnail is verified and aligned, so it is safe to delete
            assert 2 in res["delete_safe_ids"]


class TestUnalignableMemberNeverDeleteSafe:
    def test_unalignable_member_goes_to_review_never_delete_safe(self):
        members = [
            {
                "id": 1,
                "file_path": "/photos/good_ref.jpg",
                "file_size": 500_000,
                "width": 1024,
                "height": 768,
                "md5": "md5_good",
                "quality_status": "reference",
                "ssim": 1.0,
                "ssim_color": 1.0,
                "pixel_ratio": 1.0,
                "is_low_confidence": 0,
            },
            {
                "id": 2,
                "file_path": "/photos/unrelated_impostor.jpg",
                "file_size": 200_000,
                "width": 1024,
                "height": 768,
                "md5": "md5_impostor",
                "quality_status": "unalignable",  # Alignment failed!
                "ssim": None,
                "ssim_color": None,
                "pixel_ratio": 1.0,
                "is_low_confidence": 0,
            },
            {
                "id": 3,
                "file_path": "/photos/low_keypoints.jpg",
                "file_size": 150_000,
                "width": 1024,
                "height": 768,
                "md5": "md5_lowkp",
                "quality_status": "aligned",
                "ssim": 0.99,
                "ssim_color": 0.99,
                "pixel_ratio": 1.0,
                "is_low_confidence": 1,  # Low confidence / inconclusive!
            },
        ]
        for mode in ["conservative", "balanced", "aggressive"]:
            res = choose_keeper(members, mode=mode)
            assert res["keeper_id"] == 1
            # Neither unalignable nor low_confidence can EVER be in delete_safe_ids
            assert 2 not in res["delete_safe_ids"]
            assert 3 not in res["delete_safe_ids"]
            # Must be in review_ids
            assert 2 in res["review_ids"]
            assert 3 in res["review_ids"]


class TestDeterministicTies:
    def test_ties_are_deterministic(self):
        members = [
            {
                "id": 2,
                "file_path": "/photos/b_file.jpg",
                "file_size": 100_000,
                "width": 1000,
                "height": 1000,
                "md5": "md5_b",
                "quality_status": "aligned",
                "ssim": 0.99,
                "ssim_color": 0.99,
                "pixel_ratio": 1.0,
                "sharpness_ratio": 1.0,
                "is_low_confidence": 0,
            },
            {
                "id": 1,
                "file_path": "/photos/a_file.jpg",
                "file_size": 100_000,
                "width": 1000,
                "height": 1000,
                "md5": "md5_a",
                "quality_status": "reference",
                "ssim": 1.0,
                "ssim_color": 1.0,
                "pixel_ratio": 1.0,
                "sharpness_ratio": 1.0,
                "is_low_confidence": 0,
            },
        ]
        # Same size, same pixels, same path length: lexicographical tie-breaker ("a_file.jpg" < "b_file.jpg")
        for _ in range(10):
            res = choose_keeper(members, mode="balanced")
            assert res["keeper_id"] == 1


class TestCraftedClusterThreeModesDifferentResults:
    def test_all_three_modes_return_different_results(self):
        """
        Craft a cluster where:
        - Conservative mode selects Member 1 (reference).
        - Balanced mode selects Member 2 (smaller file, moderate scale/quality).
        - Aggressive mode selects Member 3 (smallest file, lower resolution/quality, blur flag allowed).
        """
        members = [
            # Member 1: Highest quality uncompressed reference
            {
                "id": 101,
                "file_path": "/photos/scene_reference.png",
                "file_size": 800_000,
                "width": 1200,
                "height": 900,
                "md5": "md5_1",
                "quality_status": "reference",
                "ssim": 1.0,
                "ssim_color": 1.0,
                "pixel_ratio": 1.0,
                "sharpness_ratio": 1.0,
                "blur_flag": False,
                "is_low_confidence": 0,
            },
            # Member 2: High quality JPEG, pixel_ratio = 0.70 (fails conservative floor 0.90, passes balanced 0.60)
            {
                "id": 102,
                "file_path": "/photos/scene_medium.jpg",
                "file_size": 350_000,
                "width": 1000,
                "height": 756,
                "md5": "md5_2",
                "quality_status": "aligned",
                "ssim": 0.96,
                "ssim_color": 0.95,
                "pixel_ratio": 0.70,
                "sharpness_ratio": 0.95,
                "blur_flag": False,
                "is_low_confidence": 0,
            },
            # Member 3: Lower quality downscale, pixel_ratio = 0.35, blur_flag = True
            # Fails conservative (ssim 0.93 < 0.98, px 0.35 < 0.90)
            # Fails balanced (ssim 0.93 < 0.95, px 0.35 < 0.60, blur_flag not allowed)
            # Passes aggressive (ssim 0.93 >= 0.92, px 0.35 >= 0.25, allow_blur = True)
            {
                "id": 103,
                "file_path": "/photos/scene_small.jpg",
                "file_size": 120_000,
                "width": 710,
                "height": 532,
                "md5": "md5_3",
                "quality_status": "aligned",
                "ssim": 0.93,
                "ssim_color": 0.91,
                "pixel_ratio": 0.35,
                "sharpness_ratio": 0.62,
                "blur_flag": True,
                "is_low_confidence": 0,
            },
        ]

        res_cons = choose_keeper(members, mode="conservative")
        res_bal = choose_keeper(members, mode="balanced")
        res_agg = choose_keeper(members, mode="aggressive")

        # 1. Conservative chooses reference (id 101)
        assert res_cons["keeper_id"] == 101
        assert res_cons["total_savings_bytes"] == 350_000 + 120_000

        # 2. Balanced chooses medium copy (id 102)
        assert res_bal["keeper_id"] == 102
        assert res_bal["total_savings_bytes"] == 800_000 + 120_000

        # 3. Aggressive chooses smallest copy (id 103)
        assert res_agg["keeper_id"] == 103
        assert res_agg["total_savings_bytes"] == 800_000 + 350_000

        # Verify all three modes picked distinct keepers!
        assert res_cons["keeper_id"] != res_bal["keeper_id"]
        assert res_bal["keeper_id"] != res_agg["keeper_id"]
        assert res_cons["keeper_id"] != res_agg["keeper_id"]


class TestSafetyInvariants:
    def test_never_mark_every_member_as_delete_safe(self):
        members = [
            {
                "id": 1,
                "file_path": "/photos/single.jpg",
                "file_size": 50_000,
                "width": 800,
                "height": 600,
                "quality_status": "reference",
                "ssim": 1.0,
                "ssim_color": 1.0,
                "pixel_ratio": 1.0,
                "is_low_confidence": 0,
            }
        ]
        res = choose_keeper(members, mode="balanced")
        assert res["keeper_id"] == 1
        assert len(res["delete_safe_ids"]) == 0
        assert len(res["delete_safe_ids"]) < len(members)


class TestFusedDuplicateConfidence:
    def test_md5_fixed_at_100(self):
        conf = predict_duplicate_confidence(
            phash_distance=30,
            best_clip_cosine=0.5,
            sift_inlier_ratio=0.1,
            inlier_count=5,
            md5_match=True,
        )
        assert conf == 100.0

    def test_high_similarity_high_confidence(self):
        conf = predict_duplicate_confidence(
            phash_distance=2,
            best_clip_cosine=0.96,
            sift_inlier_ratio=0.92,
            inlier_count=150,
            md5_match=False,
        )
        assert conf >= 99.0

    def test_low_similarity_low_confidence(self):
        conf = predict_duplicate_confidence(
            phash_distance=28,
            best_clip_cosine=0.65,
            sift_inlier_ratio=0.0,
            inlier_count=0,
            md5_match=False,
        )
        assert conf <= 5.0
