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
    get_auto_delete_candidates,
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

    def test_failed_members_become_singletons_no_subclusters(self, tmp_path, benchmark_available):
        """
        A cluster containing scene A (rep) and unrelated scene B members must verify
        genuine scene A duplicates and split failing scene B members out as singletons,
        preventing non-deterministic sub-cluster cascades.
        """
        if not benchmark_available:
            pytest.skip("Benchmark files not present in data/benchmark")

        db_path = tmp_path / "stage3_reeval.db"
        conn = init_db(db_path)

        # Scene A: picsum_0008, picsum_0008__exact
        # Scene B: picsum_0009, picsum_0009__exact
        p8 = str(BENCH_DIR / "picsum_0008.jpg")
        p8_exact = str(BENCH_DIR / "picsum_0008__exact.jpg")
        p9 = str(BENCH_DIR / "picsum_0009.jpg")
        p9_exact = str(BENCH_DIR / "picsum_0009__exact.jpg")

        c1 = conn.execute("INSERT INTO clusters (stage_detected, member_count) VALUES (2, 4)").lastrowid
        conn.execute("INSERT INTO images (id, file_path, cluster_id, is_keeper, status) VALUES (1, ?, ?, 1, 'clustered')", (p8, c1))
        conn.execute("UPDATE clusters SET representative_id = 1 WHERE id = ?", (c1,))
        conn.execute("INSERT INTO images (id, file_path, cluster_id, is_keeper, status) VALUES (2, ?, ?, 0, 'clustered')", (p8_exact, c1))
        conn.execute("INSERT INTO images (id, file_path, cluster_id, is_keeper, status) VALUES (3, ?, ?, 0, 'clustered')", (p9, c1))
        conn.execute("INSERT INTO images (id, file_path, cluster_id, is_keeper, status) VALUES (4, ?, ?, 0, 'clustered')", (p9_exact, c1))
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

        # Exactly 1 surviving cluster (Scene A)
        all_clusters = conn.execute("SELECT id, representative_id, member_count FROM clusters ORDER BY id").fetchall()
        assert len(all_clusters) == 1, f"Expected 1 cluster, got {len(all_clusters)}"

        # Scene A members (ids 1, 2) remain in c1
        c1_members = {row["id"] for row in conn.execute("SELECT id FROM images WHERE cluster_id = ?", (c1,)).fetchall()}
        assert c1_members == {1, 2}

        # Failing Scene B members (ids 3, 4) become singletons (cluster_id IS NULL)
        p9_rows = conn.execute("SELECT id, cluster_id, is_keeper, status FROM images WHERE id IN (3, 4)").fetchall()
        for r in p9_rows:
            assert r["cluster_id"] is None
            assert r["is_keeper"] == 0
            assert r["status"] == "embedded"

    def test_inconclusive_verdict_low_keypoints_and_auto_delete_exclusion(self, tmp_path):
        """
        Pairs with fewer than N keypoints receive 'inconclusive' verdict, keep their cluster,
        receive is_low_confidence = 1, and are excluded from auto-delete recommendations.
        """
        import numpy as np
        # High texture base image
        rng = np.random.RandomState(42)
        tex_arr = (rng.rand(300, 300) * 255).astype(np.uint8)
        base_img = Image.fromarray(tex_arr, mode="L")
        base_path = tmp_path / "textured.jpg"
        base_img.save(base_path, quality=95)

        # Blank / smooth image (very low keypoints < 10)
        smooth_arr = np.full((300, 300), 128, dtype=np.uint8)
        smooth_img = Image.fromarray(smooth_arr, mode="L")
        smooth_path = tmp_path / "smooth.jpg"
        smooth_img.save(smooth_path, quality=95)

        cfg = {
            "stage3": {
                "algorithm": "sift",
                "min_keypoints": 50,
                "min_inliers": 20,
                "min_inlier_ratio": 0.65,
            }
        }
        verifier = KeypointVerifier(cfg)

        # 1. Direct match check
        res = verify_pair(base_path, smooth_path, config=cfg, verifier=verifier)
        assert res.status == "inconclusive"
        assert res.verified is True

        # 2. Database cluster run
        db_path = tmp_path / "stage3_inconclusive.db"
        conn = init_db(db_path)

        c1 = conn.execute("INSERT INTO clusters (stage_detected, member_count) VALUES (2, 2)").lastrowid
        conn.execute("INSERT INTO images (id, file_path, cluster_id, is_keeper, status) VALUES (1, ?, ?, 1, 'clustered')", (str(base_path), c1))
        conn.execute("UPDATE clusters SET representative_id = 1 WHERE id = ?", (c1,))
        conn.execute("INSERT INTO images (id, file_path, cluster_id, is_keeper, status) VALUES (2, ?, ?, 0, 'clustered')", (str(smooth_path), c1))
        conn.commit()

        s3_res = run_stage3_verification(conn, cfg)
        assert s3_res.pairs_inconclusive == 1

        # Inconclusive member must KEEP its existing cluster
        row_smooth = conn.execute("SELECT cluster_id, is_low_confidence, is_keeper FROM images WHERE id = 2").fetchone()
        assert row_smooth["cluster_id"] == c1
        assert row_smooth["is_low_confidence"] == 1

        # Must be EXCLUDED from auto-delete recommendations
        candidates = get_auto_delete_candidates(conn)
        candidate_ids = [c["id"] for c in candidates]
        assert 2 not in candidate_ids, "Low confidence image must be excluded from auto-delete candidates!"

    def test_keypoint_cache_persists_in_sqlite(self, tmp_path, benchmark_available):
        """Keypoints and descriptors are cached in SQLite so images are not re-detected."""
        if not benchmark_available:
            pytest.skip("Benchmark files not present in data/benchmark")

        db_path = tmp_path / "stage3_cache.db"
        conn = init_db(db_path)

        p8 = str(BENCH_DIR / "picsum_0008.jpg")
        p8_exact = str(BENCH_DIR / "picsum_0008__exact.jpg")

        c1 = conn.execute("INSERT INTO clusters (stage_detected, member_count) VALUES (2, 2)").lastrowid
        conn.execute("INSERT INTO images (id, file_path, cluster_id, is_keeper, status) VALUES (1, ?, ?, 1, 'clustered')", (p8, c1))
        conn.execute("UPDATE clusters SET representative_id = 1 WHERE id = ?", (c1,))
        conn.execute("INSERT INTO images (id, file_path, cluster_id, is_keeper, status) VALUES (2, ?, ?, 0, 'clustered')", (p8_exact, c1))
        conn.commit()

        cfg = {"stage3": {"algorithm": "sift", "min_inliers": 20, "min_inlier_ratio": 0.65}}
        run_stage3_verification(conn, cfg)

        # Check keypoint_cache table
        cache_count = conn.execute("SELECT COUNT(*) FROM keypoint_cache").fetchone()[0]
        assert cache_count == 2


class TestFusedDuplicateConfidenceIntegration:
    def test_update_cluster_duplicate_confidences(self, tmp_path: Path):
        """Verify update_cluster_duplicate_confidences sets 100 for MD5 and calibrated fused score for non-MD5."""
        from core.verify import update_cluster_duplicate_confidences
        from core.embeddings import serialize_embedding
        import numpy as np

        db_path = tmp_path / "test_conf.db"
        conn = init_db(db_path)

        # Create 1 cluster with representative (id 1) and 3 members:
        # id 2: exact MD5 match
        # id 3: high similarity non-MD5 (low phash, high cosine, high inliers)
        # id 4: marginal similarity non-MD5 (higher phash, lower cosine, few inliers)
        v_rep = np.ones(512, dtype=np.float32) / np.sqrt(512)
        v_high = v_rep.copy()  # cosine = 1.0
        v_low = np.zeros(512, dtype=np.float32)
        v_low[0] = 1.0  # low cosine ~ 0.044

        conn.execute("INSERT INTO clusters (id, representative_id, stage_detected, member_count) VALUES (1, NULL, 2, 4)")
        conn.execute(
            "INSERT INTO images (id, file_path, md5, phash, clip_vector, cluster_id, is_keeper) VALUES (1, 'orig.jpg', 'md5_1', '0000000000000000', ?, 1, 1)",
            (serialize_embedding(v_rep),)
        )
        conn.execute(
            "INSERT INTO images (id, file_path, md5, phash, clip_vector, cluster_id, is_keeper) VALUES (2, 'exact.jpg', 'md5_1', '0000000000000000', ?, 1, 0)",
            (serialize_embedding(v_rep),)
        )
        conn.execute(
            "INSERT INTO images (id, file_path, md5, phash, clip_vector, cluster_id, is_keeper) VALUES (3, 'high.jpg', 'md5_3', '0000000000000001', ?, 1, 0)",
            (serialize_embedding(v_high),)
        )
        conn.execute(
            "INSERT INTO images (id, file_path, md5, phash, clip_vector, cluster_id, is_keeper) VALUES (4, 'low.jpg', 'md5_4', 'ffffffffffffffff', ?, 1, 0)",
            (serialize_embedding(v_low),)
        )
        conn.execute("UPDATE clusters SET representative_id = 1 WHERE id = 1")

        # Cache SIFT transforms
        conn.execute(
            "INSERT INTO pair_transforms (source_id, target_id, transform_matrix, inlier_count, inlier_ratio) VALUES (1, 3, x'00', 200, 0.95)"
        )
        conn.execute(
            "INSERT INTO pair_transforms (source_id, target_id, transform_matrix, inlier_count, inlier_ratio) VALUES (1, 4, x'00', 5, 0.10)"
        )
        conn.commit()

        updated = update_cluster_duplicate_confidences(conn)
        assert updated == 3

        conf_rep = conn.execute("SELECT confidence FROM images WHERE id = 1").fetchone()[0]
        conf_exact = conn.execute("SELECT confidence FROM images WHERE id = 2").fetchone()[0]
        conf_high = conn.execute("SELECT confidence FROM images WHERE id = 3").fetchone()[0]
        conf_low = conn.execute("SELECT confidence FROM images WHERE id = 4").fetchone()[0]

        assert conf_rep == 1.0
        assert conf_exact == 1.0
        assert conf_high >= 0.99
        assert conf_low <= 0.15


class TestKeypointCacheCompressionAndPruning:
    def test_compressed_serialization_roundtrip_and_legacy_compat(self):
        import pickle
        import cv2
        import numpy as np
        from core.verify import (
            deserialize_descriptors,
            deserialize_keypoints,
            serialize_descriptors,
            serialize_keypoints,
        )

        # Integer SIFT float32 descriptors -> lossless compressed uint8 storage
        sift_des = np.random.randint(0, 256, size=(500, 128)).astype(np.float32)
        blob = serialize_descriptors(sift_des)
        assert len(blob) < len(pickle.dumps(sift_des))
        restored = deserialize_descriptors(blob)
        assert restored.dtype == np.float32
        assert np.array_equal(restored, sift_des)

        # Fractional float32 fallback -> lossless compressed float32
        frac_des = np.linspace(0.1, 1.5, 64, dtype=np.float32).reshape(4, 16)
        assert np.allclose(deserialize_descriptors(serialize_descriptors(frac_des)), frac_des)

        # ORB uint8 descriptors
        orb_des = np.random.randint(0, 256, size=(100, 32), dtype=np.uint8)
        restored_orb = deserialize_descriptors(serialize_descriptors(orb_des))
        assert restored_orb.dtype == np.uint8
        assert np.array_equal(restored_orb, orb_des)

        # Legacy pickle descriptors remain readable
        assert np.array_equal(deserialize_descriptors(pickle.dumps(sift_des)), sift_des)

        # Compressed keypoints round-trip + legacy uncompressed compatibility
        kps = [cv2.KeyPoint(x=float(i), y=float(i * 2), size=4.5, angle=90.0, response=0.5, octave=1, class_id=0) for i in range(50)]
        kp_blob = serialize_keypoints(kps)
        restored_kps = deserialize_keypoints(kp_blob)
        assert len(restored_kps) == 50
        assert restored_kps[10].pt == (10.0, 20.0)

        legacy_raw = np.array(
            [(kp.pt[0], kp.pt[1], kp.size, kp.angle, kp.response, kp.octave, kp.class_id) for kp in kps],
            dtype=np.float32,
        ).tobytes()
        assert len(deserialize_keypoints(legacy_raw)) == 50

    def test_prune_keypoint_cache_keeps_representatives_only(self, tmp_path: Path):
        from core.verify import prune_keypoint_cache

        db_path = tmp_path / "prune.db"
        conn = init_db(db_path)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS keypoint_cache (
                image_id INTEGER PRIMARY KEY,
                keypoints BLOB,
                descriptors BLOB,
                created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
            );
            """
        )
        conn.execute("INSERT INTO clusters (id, representative_id, stage_detected, member_count) VALUES (1, NULL, 2, 2)")
        conn.execute("INSERT INTO images (id, file_path, cluster_id, is_keeper) VALUES (1, 'rep.jpg', 1, 1)")
        conn.execute("INSERT INTO images (id, file_path, cluster_id, is_keeper) VALUES (2, 'dup.jpg', 1, 0)")
        conn.execute("INSERT INTO images (id, file_path, cluster_id, is_keeper) VALUES (3, 'uniq.jpg', NULL, 0)")
        conn.execute("UPDATE clusters SET representative_id = 1 WHERE id = 1")
        for img_id in (1, 2, 3):
            conn.execute("INSERT INTO keypoint_cache (image_id, keypoints, descriptors) VALUES (?, x'01', x'02')", (img_id,))
        conn.commit()

        deleted = prune_keypoint_cache(conn, keep_representatives=True, max_rep_rows=500, vacuum_threshold_mb=100.0)
        assert deleted == 2
        remaining = [r[0] for r in conn.execute("SELECT image_id FROM keypoint_cache").fetchall()]
        assert remaining == [1]
        conn.close()

