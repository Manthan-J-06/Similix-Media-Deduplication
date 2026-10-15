"""
tests/test_verify.py — Comprehensive tests for Stage 3 geometric keypoint matching
and RANSAC verification.
"""

from __future__ import annotations

from pathlib import Path
import pytest
from PIL import Image

from core.verify import (
    KeypointVerifier,
    MatchResult,
    load_proxy_grayscale,
    run_stage3_verification,
    verify_pair,
)
from db.connection import init_db


BENCH_DIR = Path("data/benchmark")


@pytest.fixture
def benchmark_available() -> bool:
    required = ["picsum_0008.jpg", "picsum_0009.jpg", "picsum_0001.jpg", "picsum_0005.jpg"]
    return all((BENCH_DIR / r).exists() for r in required)


class TestKeypointVerificationBenchmark:
    """
    Verify:
    (a) known-duplicate variants from the benchmark pass;
    (b) the known false-positive pairs (picsum_0008 vs 0009, picsum_0001 vs 0005) are rejected.
    """

    def test_known_duplicate_variants_pass(self, benchmark_available):
        if not benchmark_available:
            pytest.skip("Benchmark files not present in data/benchmark")

        orig_path = BENCH_DIR / "picsum_0008.jpg"
        variants = [
            "picsum_0008__exact.jpg",
            "picsum_0008__resize_128.jpg",
            "picsum_0008__resize_512.jpg",
            "picsum_0008__crop_05pct.jpg",
            "picsum_0008__rotate90.jpg",
            "picsum_0008__rotate180.jpg",
            "picsum_0008__rotate270.jpg",
            "picsum_0008__bright_085.jpg",
            "picsum_0008__contrast_114.jpg",
            "picsum_0008__fmt_png.png",
            "picsum_0008__fmt_webp.webp",
        ]

        cfg = {
            "stage3": {
                "algorithm": "sift",
                "min_inliers": 20,
                "min_inlier_ratio": 0.65,
                "transformation_model": "affine",
            }
        }
        verifier = KeypointVerifier(cfg)

        for v_name in variants:
            v_path = BENCH_DIR / v_name
            if not v_path.exists():
                continue
            res = verify_pair(orig_path, v_path, config=cfg, verifier=verifier)
            assert res.verified is True, f"Variant {v_name} failed: inliers={res.inlier_count}, ratio={res.inlier_ratio}"
            assert res.inlier_count >= 20
            assert res.inlier_ratio >= 0.65

    def test_known_false_positive_pairs_rejected(self, benchmark_available):
        if not benchmark_available:
            pytest.skip("Benchmark files not present in data/benchmark")

        cfg = {
            "stage3": {
                "algorithm": "sift",
                "min_inliers": 20,
                "min_inlier_ratio": 0.65,
                "transformation_model": "affine",
            }
        }
        verifier = KeypointVerifier(cfg)

        # 1. picsum_0008 vs picsum_0009
        res_8_9 = verify_pair(
            BENCH_DIR / "picsum_0008.jpg",
            BENCH_DIR / "picsum_0009.jpg",
            config=cfg,
            verifier=verifier,
        )
        assert res_8_9.verified is False
        assert res_8_9.inlier_ratio < 0.65

        # 2. picsum_0001 vs picsum_0005
        res_1_5 = verify_pair(
            BENCH_DIR / "picsum_0001.jpg",
            BENCH_DIR / "picsum_0005.jpg",
            config=cfg,
            verifier=verifier,
        )
        assert res_1_5.verified is False
        assert res_1_5.inlier_ratio < 0.65

    def test_synthetic_crops_rescaling_rotations(self, tmp_path):
        """Ensure verification tolerates synthetic crops, rescaling, and 90/180/270 rotations."""
        # Create a synthetic test pattern with high texture
        import numpy as np
        rng = np.random.RandomState(42)
        base_arr = (rng.rand(400, 400) * 255).astype(np.uint8)
        # Add geometric features (rectangles, circles)
        import cv2
        cv2.rectangle(base_arr, (50, 50), (150, 150), 0, -1)
        cv2.circle(base_arr, (300, 300), 50, 255, -1)
        cv2.line(base_arr, (50, 350), (350, 50), 128, 5)

        base_img = Image.fromarray(base_arr, mode="L")
        base_path = tmp_path / "base.jpg"
        base_img.save(base_path, quality=95)

        # 1. Rotations 90, 180, 270
        for angle in [90, 180, 270]:
            rot_img = base_img.rotate(angle, expand=True)
            rot_path = tmp_path / f"rot_{angle}.jpg"
            rot_img.save(rot_path, quality=95)
            res = verify_pair(base_path, rot_path)
            assert res.verified is True
            assert res.inlier_ratio >= 0.85

        # 2. Rescaling
        resize_img = base_img.resize((160, 160), Image.Resampling.LANCZOS)
        res_path = tmp_path / "resized.jpg"
        resize_img.save(res_path, quality=95)
        res_scale = verify_pair(base_path, res_path)
        assert res_scale.verified is True

        # 3. Crop (10% from each side)
        w, h = base_img.size
        crop_img = base_img.crop((int(w * 0.1), int(h * 0.1), int(w * 0.9), int(h * 0.9)))
        crop_path = tmp_path / "cropped.jpg"
        crop_img.save(crop_path, quality=95)
        res_crop = verify_pair(base_path, crop_path)
        assert res_crop.verified is True


class TestStage3ClusterPipelineSplitting:
    """
    Test run_stage3_verification database integration:
    - Failing members are split out of the cluster.
    - Surviving true duplicates remain in the cluster.
    - If a cluster drops below 2 members, it is disbanded.
    """

    def test_run_stage3_verification_splits_impostor(self, tmp_path, benchmark_available):
        if not benchmark_available:
            pytest.skip("Benchmark files not present in data/benchmark")

        db_path = tmp_path / "stage3_split.db"
        conn = init_db(db_path)

        # Create a cluster with:
        # Rep: picsum_0008.jpg
        # True duplicate 1: picsum_0008__exact.jpg
        # True duplicate 2: picsum_0008__rotate90.jpg
        # False positive: picsum_0009.jpg (different original!)
        p8 = str(BENCH_DIR / "picsum_0008.jpg")
        p8_exact = str(BENCH_DIR / "picsum_0008__exact.jpg")
        p8_rot90 = str(BENCH_DIR / "picsum_0008__rotate90.jpg")
        p9 = str(BENCH_DIR / "picsum_0009.jpg")

        c1 = conn.execute("INSERT INTO clusters (stage_detected, member_count) VALUES (2, 4)").lastrowid
        conn.execute("INSERT INTO images (id, file_path, cluster_id, is_keeper, status) VALUES (1, ?, ?, 1, 'clustered')", (p8, c1))
        conn.execute("UPDATE clusters SET representative_id = 1 WHERE id = ?", (c1,))

        conn.execute("INSERT INTO images (id, file_path, cluster_id, is_keeper, status) VALUES (2, ?, ?, 0, 'clustered')", (p8_exact, c1))
        conn.execute("INSERT INTO images (id, file_path, cluster_id, is_keeper, status) VALUES (3, ?, ?, 0, 'clustered')", (p8_rot90, c1))
        conn.execute("INSERT INTO images (id, file_path, cluster_id, is_keeper, status) VALUES (4, ?, ?, 0, 'clustered')", (p9, c1))
        conn.commit()

        cfg = {
            "stage3": {
                "algorithm": "sift",
                "min_inliers": 20,
                "min_inlier_ratio": 0.65,
                "transformation_model": "affine",
            }
        }

        res = run_stage3_verification(conn, cfg)

        assert res.pairs_evaluated == 3
        assert res.pairs_verified == 2
        assert res.members_split == 1
        assert res.clusters_disbanded == 0

        # Verify database state
        c_info = conn.execute("SELECT member_count FROM clusters WHERE id = ?", (c1,)).fetchone()
        assert c_info["member_count"] == 3

        # p9 must be split out
        p9_row = conn.execute("SELECT cluster_id, is_keeper FROM images WHERE id = 4").fetchone()
        assert p9_row["cluster_id"] is None
        assert p9_row["is_keeper"] == 0

        # p8, p8_exact, p8_rot90 remain in c1
        for img_id in (1, 2, 3):
            row = conn.execute("SELECT cluster_id FROM images WHERE id = ?", (img_id,)).fetchone()
            assert row["cluster_id"] == c1

    def test_run_stage3_disbands_cluster_when_all_members_fail(self, tmp_path, benchmark_available):
        if not benchmark_available:
            pytest.skip("Benchmark files not present in data/benchmark")

        db_path = tmp_path / "stage3_disband.db"
        conn = init_db(db_path)

        p1 = str(BENCH_DIR / "picsum_0001.jpg")
        p5 = str(BENCH_DIR / "picsum_0005.jpg")

        c1 = conn.execute("INSERT INTO clusters (stage_detected, member_count) VALUES (2, 2)").lastrowid
        conn.execute("INSERT INTO images (id, file_path, cluster_id, is_keeper, status) VALUES (1, ?, ?, 1, 'clustered')", (p1, c1))
        conn.execute("UPDATE clusters SET representative_id = 1 WHERE id = ?", (c1,))
        conn.execute("INSERT INTO images (id, file_path, cluster_id, is_keeper, status) VALUES (2, ?, ?, 0, 'clustered')", (p5, c1))
        conn.commit()

        cfg = {
            "stage3": {
                "algorithm": "sift",
                "min_inliers": 20,
                "min_inlier_ratio": 0.65,
                "transformation_model": "affine",
            }
        }

        res = run_stage3_verification(conn, cfg)

        assert res.pairs_evaluated == 1
        assert res.pairs_verified == 0
        assert res.members_split == 1
        assert res.clusters_disbanded == 1

        # Cluster must be deleted
        assert conn.execute("SELECT COUNT(*) FROM clusters").fetchone()[0] == 0
        # Both images unclustered
        assert conn.execute("SELECT COUNT(*) FROM images WHERE cluster_id IS NOT NULL").fetchone()[0] == 0
