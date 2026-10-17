"""
tests/test_keeper.py — Tests for core/keeper.py.

Covers:
1. Exact-copy cluster
2. Rotated copy
3. Tiny thumbnail rejected by resolution floor
4. Unalignable member never delete_safe
5. Ties are deterministic
6. Safety invariant: never mark every member as delete_safe
7. (a) Aggressive mode never deletes the best copy for a saving under 30%
8. (b) A member better than the keeper goes to review
9. (c) Crafted 4-member cluster (original, q60, q30, 50% scale) gives 3 different keepers across modes
10. Conservative mode: non-identical duplicate goes to review when keeper is not reference
11. Fused duplicate confidence scoring
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
        # Both are 100KB, 1024x768. Shorter path "/photos/copy.jpg" wins tie-breaker -> id 2
        assert res["keeper_id"] in (1, 2)
        other_id = 1 if res["keeper_id"] == 2 else 2
        assert other_id in res["delete_safe_ids"]
        assert len(res["review_ids"]) == 0
        assert res["total_savings_bytes"] == 100_000
        assert res["quality_retained"][other_id] == 100.0


class TestRotatedCopy:
    def test_rotated_copy_qualifies_when_reference_is_keeper(self):
        members = [
            {
                "id": 10,
                "file_path": "/photos/scene_ref.jpg",
                "file_size": 150_000,
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
        # Reference is smaller (150KB < 180KB) and keeper. Rotated copy is delete_safe
        res = choose_keeper(members, mode="balanced")
        assert res["keeper_id"] == 10
        assert 11 in res["delete_safe_ids"]
        assert res["total_savings_bytes"] == 180_000
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
            # Thumbnail is verified and aligned, keeper is strictly better, so it is delete_safe
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


class TestAggressiveModeMinSavingRatio:
    def test_aggressive_mode_never_deletes_best_copy_for_saving_under_30_pct(self):
        """(a) aggressive mode never deletes the best copy for a saving under 30%."""
        members = [
            {
                "id": 1,
                "file_path": "/photos/best_copy.jpg",
                "file_size": 100_000,
                "width": 1024,
                "height": 768,
                "md5": "md5_best",
                "quality_status": "reference",
                "ssim": 1.0,
                "ssim_color": 1.0,
                "pixel_ratio": 1.0,
                "is_low_confidence": 0,
            },
            {
                "id": 2,
                "file_path": "/photos/small_saving.jpg",
                "file_size": 80_000,  # 20% saving (under 30%)
                "width": 512,
                "height": 384,        # Lower resolution copy (passes size invariant)
                "md5": "md5_cand",
                "quality_status": "aligned",
                "ssim": 0.96,
                "ssim_color": 0.95,
                "pixel_ratio": 0.25,
                "is_low_confidence": 0,
            },
        ]
        # In aggressive mode, min_saving_ratio = 0.30 -> requires file_size <= 70,000
        res = choose_keeper(members, mode="aggressive")
        # Reference (best copy) STAYS keeper because 20% saving is under 30%!
        assert res["keeper_id"] == 1
        assert 2 in res["delete_safe_ids"]


class TestMemberBetterThanKeeperGoesToReview:
    def test_member_better_than_keeper_goes_to_review(self):
        """(b) a member that is better than the keeper in resolution or quality goes to review."""
        members = [
            {
                "id": 1,
                "file_path": "/photos/high_res_ref.jpg",
                "file_size": 100_000,
                "width": 1024,
                "height": 768,
                "md5": "md5_high",
                "quality_status": "reference",
                "ssim": 1.0,
                "ssim_color": 1.0,
                "pixel_ratio": 1.0,
                "is_low_confidence": 0,
            },
            {
                "id": 2,
                "file_path": "/photos/low_res_cand.jpg",
                "file_size": 50_000,  # 50% saving
                "width": 600,
                "height": 450,       # Lower resolution!
                "md5": "md5_low",
                "quality_status": "aligned",
                "ssim": 0.96,        # Meets aggressive ssim >= 0.95
                "ssim_color": 0.95,
                "pixel_ratio": 0.3433,
                "is_low_confidence": 0,
            },
        ]
        # In aggressive mode with allow_resolution_loss=True, candidate 2 qualifies and becomes keeper
        res = choose_keeper(members, mode="aggressive", allow_resolution_loss=True)
        assert res["keeper_id"] == 2
        # Member 1 is better than keeper in resolution (1024x768 > 600x450) and quality (1.0 > 0.96)
        assert 1 in res["review_ids"]
        assert 1 not in res["delete_safe_ids"]
        assert res["reasons"][1] == "better than recommended keeper"


class TestCraftedFourMemberClusterThreeModes:
    def test_crafted_4_member_cluster_gives_three_different_keepers(self):
        """
        (c) a crafted 4-member cluster (original, q60, q30, 50% scale)
        gives three different keepers across the three modes under Prompt 10 limits:
        - Conservative: keeper = reference (original)
        - Balanced: ssim >= 0.98, ssim_color >= 0.97, pixel_ratio >= 0.60, min_saving >= 0.10
        - Aggressive (with allow_resolution_loss=True): ssim >= 0.95, ssim_color >= 0.94, pixel_ratio >= 0.25, min_saving >= 0.30
        """
        members = [
            # 1. Original (Reference)
            {
                "id": 1,
                "file_path": "/photos/original.jpg",
                "file_size": 500_000,
                "width": 1000,
                "height": 1000,
                "md5": "md5_orig",
                "quality_status": "reference",
                "ssim": 1.0,
                "ssim_color": 1.0,
                "pixel_ratio": 1.0,
                "blur_flag": False,
                "is_low_confidence": 0,
            },
            # 2. q60: 420KB (16% saving >= 10% balanced, ssim 0.985 >= 0.98; fails aggressive 30% saving)
            {
                "id": 2,
                "file_path": "/photos/q60.jpg",
                "file_size": 420_000,
                "width": 1000,
                "height": 1000,
                "md5": "md5_q60",
                "quality_status": "aligned",
                "ssim": 0.985,
                "ssim_color": 0.975,
                "pixel_ratio": 1.0,
                "blur_flag": False,
                "is_low_confidence": 0,
            },
            # 3. 50% scale: 200KB (60% saving >= 30% aggressive, ssim 0.955 >= 0.95, px 0.25 >= 0.25)
            # Fails conservative (ssim 0.955 < 0.98, px 0.25 < 0.90)
            # Fails balanced (px 0.25 < 0.60)
            {
                "id": 3,
                "file_path": "/photos/scale50.jpg",
                "file_size": 200_000,
                "width": 500,
                "height": 500,
                "md5": "md5_scale50",
                "quality_status": "aligned",
                "ssim": 0.955,
                "ssim_color": 0.945,
                "pixel_ratio": 0.25,
                "blur_flag": False,
                "is_low_confidence": 0,
            },
            # 4. q30: 300KB, ssim 0.89 (fails all modes ssim thresholds 0.98, 0.98, 0.95)
            {
                "id": 4,
                "file_path": "/photos/q30.jpg",
                "file_size": 300_000,
                "width": 1000,
                "height": 1000,
                "md5": "md5_q30",
                "quality_status": "aligned",
                "ssim": 0.89,
                "ssim_color": 0.87,
                "pixel_ratio": 1.0,
                "blur_flag": True,
                "is_low_confidence": 0,
            },
        ]

        res_cons = choose_keeper(members, mode="conservative")
        res_bal = choose_keeper(members, mode="balanced")
        res_agg = choose_keeper(members, mode="aggressive", allow_resolution_loss=True)

        # Conservative chooses original (id 1)
        assert res_cons["keeper_id"] == 1

        # Balanced chooses q60 (id 2)
        assert res_bal["keeper_id"] == 2

        # Aggressive chooses 50% scale (id 3) when allow_resolution_loss=True
        assert res_agg["keeper_id"] == 3

        # All three modes return distinct keepers!
        assert res_cons["keeper_id"] != res_bal["keeper_id"]
        assert res_bal["keeper_id"] != res_agg["keeper_id"]
        assert res_cons["keeper_id"] != res_agg["keeper_id"]

        # When allow_resolution_loss is False (default), aggressive keeps full-res file (id 1)
        res_agg_default = choose_keeper(members, mode="aggressive")
        assert res_agg_default["keeper_id"] == 1


class TestAllowResolutionLoss:
    """
    2. Add [keeper.modes] allow_resolution_loss (conservative false, balanced
       false, aggressive false by default, plus a user override flag).
       A keeper with lower pixel count than the reference is allowed only when
       this is true. With it false, aggressive mode keeps the best-resolution
       copy that meets its limits.
       Tests: aggressive mode with the flag off keeps a full-resolution file;
       with it on, the scale_50pct copy can become keeper.
    """
    def test_aggressive_mode_with_flag_off_keeps_full_resolution_file(self):
        members = [
            {
                "id": 1,
                "file_path": "/photos/original.jpg",
                "file_size": 200_000,
                "width": 1000,
                "height": 1000,
                "md5": "md5_orig",
                "quality_status": "reference",
                "ssim": 1.0,
                "ssim_color": 1.0,
                "pixel_ratio": 1.0,
                "is_low_confidence": 0,
            },
            {
                "id": 2,
                "file_path": "/photos/scale_50pct.jpg",
                "file_size": 80_000,
                "width": 500,
                "height": 500,
                "md5": "md5_scale",
                "quality_status": "aligned",
                "ssim": 0.96,
                "ssim_color": 0.95,
                "pixel_ratio": 0.25,
                "is_low_confidence": 0,
            },
        ]
        # Flag off by default in aggressive mode: keeps full-resolution copy (id 1)
        res_default = choose_keeper(members, mode="aggressive")
        assert res_default["keeper_id"] == 1
        assert res_default["allow_resolution_loss"] is False

        # Flag off via explicit user override
        res_explicit_false = choose_keeper(members, mode="aggressive", allow_resolution_loss=False)
        assert res_explicit_false["keeper_id"] == 1
        assert res_explicit_false["allow_resolution_loss"] is False

    def test_aggressive_mode_with_flag_on_scale_50pct_can_become_keeper(self):
        members = [
            {
                "id": 1,
                "file_path": "/photos/original.jpg",
                "file_size": 200_000,
                "width": 1000,
                "height": 1000,
                "md5": "md5_orig",
                "quality_status": "reference",
                "ssim": 1.0,
                "ssim_color": 1.0,
                "pixel_ratio": 1.0,
                "is_low_confidence": 0,
            },
            {
                "id": 2,
                "file_path": "/photos/scale_50pct.jpg",
                "file_size": 80_000,
                "width": 500,
                "height": 500,
                "md5": "md5_scale",
                "quality_status": "aligned",
                "ssim": 0.96,
                "ssim_color": 0.95,
                "pixel_ratio": 0.25,
                "is_low_confidence": 0,
            },
        ]
        # Flag on via user override: scale_50pct copy becomes keeper
        res_override = choose_keeper(members, mode="aggressive", allow_resolution_loss=True)
        assert res_override["keeper_id"] == 2
        assert res_override["allow_resolution_loss"] is True


class TestConservativeModeRules:
    def test_conservative_mode_keeper_is_reference_and_deletes_exact_plus_worse(self):
        """
        Conservative mode: keeper = the reference.
        delete_safe = byte-identical copies of keeper plus members strictly worse
        (lower resolution or SSIM below keeper by > 0.005) and verified.
        Nothing else is deleted automatically.
        """
        members = [
            {
                "id": 1,
                "file_path": "/photos/ref.jpg",
                "file_size": 150_000,
                "width": 1000,
                "height": 1000,
                "md5": "md5_ref",
                "quality_status": "reference",
                "ssim": 1.0,
                "ssim_color": 1.0,
                "pixel_ratio": 1.0,
                "is_low_confidence": 0,
            },
            # Byte-identical copy of keeper
            {
                "id": 2,
                "file_path": "/photos/ref_copy.jpg",
                "file_size": 150_000,
                "width": 1000,
                "height": 1000,
                "md5": "md5_ref",
                "quality_status": "aligned",
                "ssim": 1.0,
                "ssim_color": 1.0,
                "pixel_ratio": 1.0,
                "is_low_confidence": 0,
            },
            # Strictly worse member: lower resolution (500x500 < 1000x1000)
            {
                "id": 3,
                "file_path": "/photos/thumb.jpg",
                "file_size": 40_000,
                "width": 500,
                "height": 500,
                "md5": "md5_thumb",
                "quality_status": "aligned",
                "ssim": 0.95,
                "ssim_color": 0.94,
                "pixel_ratio": 0.25,
                "is_low_confidence": 0,
            },
            # Strictly worse member: same resolution, larger file size, but lower SSIM (0.985 < 1.0 - 0.005)
            {
                "id": 4,
                "file_path": "/photos/q85_larger_file.jpg",
                "file_size": 180_000,
                "width": 1000,
                "height": 1000,
                "md5": "md5_q85",
                "quality_status": "aligned",
                "ssim": 0.985,
                "ssim_color": 0.980,
                "pixel_ratio": 1.0,
                "is_low_confidence": 0,
            },
            # Same fidelity member: same resolution and SSIM within 0.005 (0.998 >= 1.0 - 0.005) -> review!
            {
                "id": 5,
                "file_path": "/photos/near_identical.jpg",
                "file_size": 150_000,
                "width": 1000,
                "height": 1000,
                "md5": "md5_near",
                "quality_status": "aligned",
                "ssim": 0.998,
                "ssim_color": 0.997,
                "pixel_ratio": 1.0,
                "is_low_confidence": 0,
            },
        ]
        res = choose_keeper(members, mode="conservative")
        # Keeper is the reference
        assert res["keeper_id"] == 1
        # Byte-identical copy (id 2) is delete_safe
        assert 2 in res["delete_safe_ids"]
        # Strictly worse lower-resolution thumbnail (id 3) is delete_safe
        assert 3 in res["delete_safe_ids"]
        # Strictly worse same-resolution member (id 4) is delete_safe (keeper 150K <= member 180K)
        assert 4 in res["delete_safe_ids"]
        # Same-fidelity member (id 5) is NOT strictly worse -> requires review
        assert 5 in res["review_ids"]
        assert 5 not in res["delete_safe_ids"]


class TestKeeperNeverLargerThanSameResolutionMemberItReplaces:
    def test_keeper_never_larger_than_same_resolution_member(self):
        """
        Invariant: the keeper's file size must never exceed any delete_safe member's
        size by design unless that member is a lower-resolution copy.
        Test: recommended keeper is never larger than the smallest same-resolution member it replaces.
        """
        members = [
            # Candidate 1: High quality candidate, 1000x1000, 200KB
            {
                "id": 1,
                "file_path": "/photos/original.jpg",
                "file_size": 250_000,
                "width": 1000,
                "height": 1000,
                "md5": "md5_orig",
                "quality_status": "reference",
                "ssim": 1.0,
                "ssim_color": 1.0,
                "pixel_ratio": 1.0,
                "is_low_confidence": 0,
            },
            # Candidate 2: Chosen keeper in balanced mode, 1000x1000, 200KB
            {
                "id": 2,
                "file_path": "/photos/keeper_cand.jpg",
                "file_size": 200_000,
                "width": 1000,
                "height": 1000,
                "md5": "md5_keeper",
                "quality_status": "aligned",
                "ssim": 0.985,
                "ssim_color": 0.980,
                "pixel_ratio": 1.0,
                "is_low_confidence": 0,
            },
            # Member 3: Same resolution (1000x1000), but smaller file size (120KB) and lower SSIM (0.90)
            # Fails balanced mode candidate threshold, but keeper (200KB) is LARGER than member 3 (120KB)!
            {
                "id": 3,
                "file_path": "/photos/smaller_same_res.jpg",
                "file_size": 120_000,
                "width": 1000,
                "height": 1000,
                "md5": "md5_smaller",
                "quality_status": "aligned",
                "ssim": 0.90,
                "ssim_color": 0.89,
                "pixel_ratio": 1.0,
                "is_low_confidence": 0,
            },
            # Member 4: Lower resolution (500x500), smaller file size (50KB)
            {
                "id": 4,
                "file_path": "/photos/thumb_smaller.jpg",
                "file_size": 50_000,
                "width": 500,
                "height": 500,
                "md5": "md5_thumb",
                "quality_status": "aligned",
                "ssim": 0.96,
                "ssim_color": 0.95,
                "pixel_ratio": 0.25,
                "is_low_confidence": 0,
            },
        ]
        res = choose_keeper(members, mode="balanced")
        assert res["keeper_id"] == 2
        keeper_sz = 200_000

        # Member 3 is same resolution and smaller than keeper -> MUST NOT be in delete_safe!
        assert 3 not in res["delete_safe_ids"]
        assert 3 in res["review_ids"]

        # Member 4 is lower resolution -> allowed in delete_safe even though keeper is larger
        assert 4 in res["delete_safe_ids"]

        # Invariant verification across all delete_safe members:
        id_to_m = {m["id"]: m for m in members}
        for ds_id in res["delete_safe_ids"]:
            m = id_to_m[ds_id]
            px = m["width"] * m["height"]
            if px >= 1000 * 1000:
                assert keeper_sz <= m["file_size"]


class TestByteIdenticalCopiesNeverStrandedInReview:
    def test_byte_identical_copies_of_non_keeper_not_stranded_in_review(self):
        """
        Byte-identical copies of the keeper are always delete_safe.
        A byte-identical copy of a non-keeper goes through that member's own rule,
        so exact duplicate copies are never stranded in review when the keeper is a different file.
        """
        members = [
            # High-res reference original (500KB)
            {
                "id": 1,
                "file_path": "/photos/original.jpg",
                "file_size": 500_000,
                "width": 1000,
                "height": 1000,
                "md5": "md5_original",
                "quality_status": "reference",
                "ssim": 1.0,
                "ssim_color": 1.0,
                "pixel_ratio": 1.0,
                "is_low_confidence": 0,
            },
            # Exact duplicate of the original (also 500KB)
            {
                "id": 2,
                "file_path": "/photos/original_copy.jpg",
                "file_size": 500_000,
                "width": 1000,
                "height": 1000,
                "md5": "md5_original",
                "quality_status": "aligned",
                "ssim": 1.0,
                "ssim_color": 1.0,
                "pixel_ratio": 1.0,
                "is_low_confidence": 0,
            },
            # Smaller candidate keeper (q60, 400KB)
            {
                "id": 3,
                "file_path": "/photos/q60.jpg",
                "file_size": 400_000,
                "width": 1000,
                "height": 1000,
                "md5": "md5_q60",
                "quality_status": "aligned",
                "ssim": 0.985,
                "ssim_color": 0.980,
                "pixel_ratio": 1.0,
                "is_low_confidence": 0,
            },
        ]
        res = choose_keeper(members, mode="balanced")
        # Keeper is the smaller q60 candidate
        assert res["keeper_id"] == 3

        # Non-keeper original (id 1) is better than keeper (quality 1.0 vs 0.985) -> goes to review
        assert 1 in res["review_ids"]
        assert res["reasons"][1] == "better than recommended keeper"

        # Exact byte-identical copy of non-keeper original (id 2) is NOT stranded in review!
        # It is marked delete_safe
        assert 2 in res["delete_safe_ids"]
        assert "Byte-identical duplicate of original.jpg" in res["reasons"][2]


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


class TestConservativeUncertaintyPreservesByteIdentical:
    def test_conservative_mode_uncertain_preserves_byte_identical_as_delete_safe(self):
        """
        In conservative mode with reference_uncertain = 1, byte-identical copies
        of the keeper must still be delete_safe (they lose nothing whichever copy is the source).
        Only non-identical members go to review.
        """
        members = [
            {
                "id": 1,
                "file_path": "/photos/original.jpg",
                "file_size": 100_000,
                "width": 1000,
                "height": 1000,
                "md5": "md5_original",
                "quality_status": "reference",
                "ssim": 1.0,
                "ssim_color": 1.0,
                "pixel_ratio": 1.0,
                "is_low_confidence": 0,
            },
            {
                "id": 2,
                "file_path": "/photos/original_exact_copy.jpg",
                "file_size": 100_000,
                "width": 1000,
                "height": 1000,
                "md5": "md5_original",
                "quality_status": "aligned",
                "ssim": 1.0,
                "ssim_color": 1.0,
                "pixel_ratio": 1.0,
                "is_low_confidence": 0,
            },
            {
                "id": 3,
                "file_path": "/photos/q85_resave.jpg",
                "file_size": 90_000,
                "width": 1000,
                "height": 1000,
                "md5": "md5_q85",
                "quality_status": "aligned",
                "ssim": 0.985,
                "ssim_color": 0.98,
                "pixel_ratio": 1.0,
                "is_low_confidence": 0,
            },
        ]

        res = choose_keeper(members, mode="conservative", reference_uncertain=1)
        assert res["keeper_id"] == 1
        assert res["reference_uncertain"] == 1
        assert res["delete_safe_ids"] == [2]
        assert res["review_ids"] == [3]
        assert res["total_savings_bytes"] == 100_000
        assert res["reasons"][2] == "Byte-identical copy of recommended keeper"
        assert "uncertain" in res["reasons"][3].lower()


class TestByteIdenticalAnchorPrefersEarliestMtime:
    def test_anchor_prefers_earliest_mtime_over_path_and_renaming(self):
        """
        Anchor of a byte-identical group is chosen by:
        earliest mtime, then shortest path, then path, then id.
        Renaming files (or shorter path) does not change anchor when mtimes differ.
        """
        members = [
            {
                "id": 1,
                "file_path": "/photos/reference.jpg",
                "file_size": 200_000,
                "width": 1200,
                "height": 900,
                "quality_status": "reference",
                "ssim": 1.0,
                "ssim_color": 1.0,
                "pixel_ratio": 1.0,
                "is_low_confidence": 0,
            },
            # Non-keeper byte-identical duplicates with different mtimes:
            # id 2 has earlier mtime but very long path
            {
                "id": 2,
                "file_path": "/photos/deep/nested/subfolder/very_long_original_filename_2015.jpg",
                "file_size": 50_000,
                "width": 800,
                "height": 600,
                "md5": "md5_subgroup",
                "modified_time": 1400000000.0,  # Earlier mtime
                "quality_status": "aligned",
                "ssim": 0.95,
                "ssim_color": 0.95,
                "pixel_ratio": 0.8,
                "is_low_confidence": 0,
            },
            # id 3 has later mtime but very short path
            {
                "id": 3,
                "file_path": "/a.jpg",
                "file_size": 50_000,
                "width": 800,
                "height": 600,
                "md5": "md5_subgroup",
                "modified_time": 1700000000.0,  # Later mtime
                "quality_status": "aligned",
                "ssim": 0.95,
                "ssim_color": 0.95,
                "pixel_ratio": 0.8,
                "is_low_confidence": 0,
            },
        ]

        res = choose_keeper(members, mode="conservative")
        assert 3 in res["delete_safe_ids"] or 3 in res["review_ids"]
        assert "very_long_original_filename_2015.jpg" in res["reasons"][3]

