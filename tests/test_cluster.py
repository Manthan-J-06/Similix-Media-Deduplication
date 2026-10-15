"""
tests/test_cluster.py — Comprehensive tests for FAISS index, candidate retrieval,
and Stage 2 anti-chaining clustering.
"""

from __future__ import annotations

import sqlite3
import numpy as np
import pytest

from core.embeddings import serialize_embedding
from core.index import (
    FlatIPIndex,
    HNSWIndex,
    build_or_update_index,
    create_index,
    retrieve_candidate_pairs,
)
from core.cluster import run_stage2_clustering
from db.connection import init_db


def make_unit_vector(angle_rad: float, dim: int = 512) -> np.ndarray:
    """Create a 2D-rotated unit vector embedded in high-dimensional space."""
    v = np.zeros(dim, dtype=np.float32)
    v[0] = np.cos(angle_rad)
    v[1] = np.sin(angle_rad)
    return v


class TestVectorIndex:
    def test_flat_ip_index_search_and_persistence(self, tmp_path):
        dim = 128
        idx = FlatIPIndex(dim=dim)
        assert idx.ntotal == 0
        assert idx.get_indexed_ids() == set()

        # Add 3 unit vectors
        v1 = make_unit_vector(0.0, dim=dim)
        v2 = make_unit_vector(0.1, dim=dim)
        v3 = make_unit_vector(1.5, dim=dim)
        vecs = np.vstack([v1, v2, v3])
        ids = [10, 20, 30]

        idx.add(ids, vecs)
        assert idx.ntotal == 3
        assert idx.get_indexed_ids() == {10, 20, 30}

        # Search
        sims, result_ids = idx.search(v1.reshape(1, -1), k=2)
        assert result_ids[0, 0] == 10
        assert pytest.approx(sims[0, 0], abs=1e-5) == 1.0
        assert result_ids[0, 1] == 20
        assert pytest.approx(sims[0, 1], abs=1e-3) == float(np.cos(0.1))

        # Save and reload
        save_path = tmp_path / "test.faiss"
        idx.save(save_path)
        assert save_path.exists()

        loaded_idx = FlatIPIndex(dim=dim)
        loaded_idx.load(save_path)
        assert loaded_idx.ntotal == 3
        assert loaded_idx.get_indexed_ids() == {10, 20, 30}

        sims2, result_ids2 = loaded_idx.search(v1.reshape(1, -1), k=2)
        assert result_ids2[0, 0] == 10
        assert result_ids2[0, 1] == 20

    def test_hnsw_index_swappability(self):
        dim = 64
        idx = create_index(dim=dim, index_type="hnsw")
        assert isinstance(idx, HNSWIndex)
        v1 = make_unit_vector(0.0, dim=dim)
        v2 = make_unit_vector(0.05, dim=dim)
        idx.add([101, 102], np.vstack([v1, v2]))
        assert idx.ntotal == 2
        sims, result_ids = idx.search(v1.reshape(1, -1), k=2)
        assert result_ids[0, 0] == 101

    def test_incremental_indexing(self, tmp_path):
        db_path = tmp_path / "test.db"
        faiss_path = tmp_path / "test.faiss"
        conn = init_db(db_path)

        dim = 512
        cfg = {"stage2": {"faiss_index_path": str(faiss_path), "embedding_dim": dim}}

        # Insert 2 images
        v1 = make_unit_vector(0.0, dim)
        v2 = make_unit_vector(0.1, dim)
        conn.execute(
            """
            INSERT INTO images (file_path, clip_vector, status)
            VALUES ('img1.jpg', ?, 'embedded'), ('img2.jpg', ?, 'embedded')
            """,
            (serialize_embedding(v1), serialize_embedding(v2)),
        )
        conn.commit()

        # Run 1: index built from scratch
        index, added1 = build_or_update_index(conn, faiss_path, cfg)
        assert added1 == 2
        assert index.ntotal == 2

        # Run 2: no new images, nothing added
        index2, added2 = build_or_update_index(conn, faiss_path, cfg)
        assert added2 == 0
        assert index2.ntotal == 2

        # Add a 3rd image
        v3 = make_unit_vector(0.2, dim)
        conn.execute(
            "INSERT INTO images (file_path, clip_vector, status) VALUES ('img3.jpg', ?, 'embedded')",
            (serialize_embedding(v3),),
        )
        conn.commit()

        # Run 3: only the 3rd image added
        index3, added3 = build_or_update_index(conn, faiss_path, cfg)
        assert added3 == 1
        assert index3.ntotal == 3


class TestAntiChainingSynthetic:
    """
    Core requirement:
    A synthetic case where A~B and B~C but A!~C must not end up as one cluster.
    """

    def _setup_synthetic_db(
        self,
        tmp_path,
        sim_ab: float = 0.95,
        sim_bc: float = 0.95,
        sim_ac: float = 0.80,
        area_a: int = 1000 * 1000,
        area_b: int = 800 * 800,
        area_c: int = 600 * 600,
    ) -> tuple[sqlite3.Connection, dict]:
        db_path = tmp_path / "synthetic.db"
        faiss_path = tmp_path / "synthetic.faiss"
        conn = init_db(db_path)

        dim = 512
        # Construct 3 vectors in 2D plane:
        # A at angle 0.
        # B at angle theta1 where cos(theta1) = sim_ab.
        # C at angle theta1 + theta2 where cos(theta1 + theta2) = sim_ac,
        # and cos(theta2) = sim_bc.
        theta_ab = np.arccos(sim_ab)
        theta_bc = np.arccos(sim_bc)
        theta_ac = theta_ab + theta_bc  # cos(theta_ac) approx 0.80 when cos(theta) = 0.95

        vA = make_unit_vector(0.0, dim=dim)
        vB = make_unit_vector(theta_ab, dim=dim)
        vC = make_unit_vector(theta_ac, dim=dim)

        actual_sim_ab = float(np.dot(vA, vB))
        actual_sim_bc = float(np.dot(vB, vC))
        actual_sim_ac = float(np.dot(vA, vC))

        assert actual_sim_ab >= 0.92, f"Expected A~B >= 0.92, got {actual_sim_ab}"
        assert actual_sim_bc >= 0.92, f"Expected B~C >= 0.92, got {actual_sim_bc}"
        assert actual_sim_ac < 0.92, f"Expected A!~C < 0.92, got {actual_sim_ac}"

        # Insert A, B, C into DB
        conn.execute(
            """
            INSERT INTO images (file_path, width, height, clip_vector, status)
            VALUES ('img_A.jpg', ?, ?, ?, 'embedded')
            """,
            (int(np.sqrt(area_a)), int(np.sqrt(area_a)), serialize_embedding(vA)),
        )
        conn.execute(
            """
            INSERT INTO images (file_path, width, height, clip_vector, status)
            VALUES ('img_B.jpg', ?, ?, ?, 'embedded')
            """,
            (int(np.sqrt(area_b)), int(np.sqrt(area_b)), serialize_embedding(vB)),
        )
        conn.execute(
            """
            INSERT INTO images (file_path, width, height, clip_vector, status)
            VALUES ('img_C.jpg', ?, ?, ?, 'embedded')
            """,
            (int(np.sqrt(area_c)), int(np.sqrt(area_c)), serialize_embedding(vC)),
        )
        conn.commit()

        cfg = {
            "general": {"db_path": str(db_path)},
            "stage2": {
                "cosine_threshold": 0.92,
                "faiss_k": 10,
                "faiss_index_path": str(faiss_path),
                "embedding_dim": dim,
            },
        }
        return conn, cfg

    def test_anti_chaining_when_a_is_highest_resolution(self, tmp_path):
        """A is largest -> A is seed/rep. C is not similar to A. A and C must NOT be in the same cluster."""
        conn, cfg = self._setup_synthetic_db(
            tmp_path,
            area_a=1200 * 1200,
            area_b=800 * 800,
            area_c=600 * 600,
        )
        res = run_stage2_clustering(conn, cfg)

        # Check clusters in DB
        rows = conn.execute(
            "SELECT id, file_path, cluster_id, is_keeper, link_signal, confidence FROM images ORDER BY file_path"
        ).fetchall()
        row_map = {r["file_path"]: dict(r) for r in rows}

        c_A = row_map["img_A.jpg"]["cluster_id"]
        c_B = row_map["img_B.jpg"]["cluster_id"]
        c_C = row_map["img_C.jpg"]["cluster_id"]

        # A and B are clustered together
        assert c_A is not None
        assert c_A == c_B
        # A and C must NEVER be in the same cluster!
        assert c_A != c_C
        # C must be split out (not in cluster with A)
        assert c_C is None or c_C != c_A

    def test_anti_chaining_when_b_is_highest_resolution(self, tmp_path):
        """B is largest -> B is candidate rep with both A and C. A and C conflict. A and C must NOT be in the same cluster."""
        conn, cfg = self._setup_synthetic_db(
            tmp_path,
            area_a=800 * 800,
            area_b=1400 * 1400,  # B has highest resolution
            area_c=600 * 600,
        )
        res = run_stage2_clustering(conn, cfg)

        rows = conn.execute(
            "SELECT file_path, cluster_id FROM images ORDER BY file_path"
        ).fetchall()
        row_map = {r["file_path"]: r["cluster_id"] for r in rows}

        c_A = row_map["img_A.jpg"]
        c_B = row_map["img_B.jpg"]
        c_C = row_map["img_C.jpg"]

        # Crucial invariant: A and C must NOT be in the same cluster!
        assert c_A != c_C or c_A is None
        # B is clustered with one of them, but NOT both
        assert not (c_A == c_B and c_B == c_C and c_A is not None)

    def test_anti_chaining_when_c_is_highest_resolution(self, tmp_path):
        """C is largest -> C is seed/rep. A and C must NOT be in the same cluster."""
        conn, cfg = self._setup_synthetic_db(
            tmp_path,
            area_a=600 * 600,
            area_b=800 * 800,
            area_c=1200 * 1200,  # C has highest resolution
        )
        res = run_stage2_clustering(conn, cfg)

        rows = conn.execute(
            "SELECT file_path, cluster_id FROM images ORDER BY file_path"
        ).fetchall()
        row_map = {r["file_path"]: r["cluster_id"] for r in rows}

        c_A = row_map["img_A.jpg"]
        c_B = row_map["img_B.jpg"]
        c_C = row_map["img_C.jpg"]

        # Crucial invariant: A and C must NOT be in the same cluster!
        assert c_A != c_C or c_C is None


class TestUnifiedClustersAndLinkSignal:
    def test_combines_stage0_stage1_and_stage2(self, tmp_path):
        db_path = tmp_path / "unified.db"
        faiss_path = tmp_path / "unified.faiss"
        conn = init_db(db_path)

        dim = 512
        v_exact1 = make_unit_vector(0.0, dim)
        v_exact2 = make_unit_vector(0.0, dim)

        v_phash1 = make_unit_vector(0.5, dim)
        v_phash2 = make_unit_vector(0.51, dim)

        v_clip1 = make_unit_vector(1.0, dim)
        v_clip2 = make_unit_vector(1.02, dim)

        # Stage 0 exact pair
        conn.execute(
            """
            INSERT INTO images (file_path, file_size, md5, clip_vector, status)
            VALUES ('s0_1.jpg', 1024, 'md5_match', ?, 'hashed'), ('s0_2.jpg', 1024, 'md5_match', ?, 'hashed')
            """,
            (serialize_embedding(v_exact1), serialize_embedding(v_exact2)),
        )
        from core.stage0 import _build_exact_clusters
        _build_exact_clusters(conn)

        # Stage 1 phash pair
        conn.execute(
            """
            INSERT INTO images (file_path, phash, clip_vector, width, height, status)
            VALUES ('s1_1.jpg', '0000000000000000', ?, 800, 800, 'hashed'),
                   ('s1_2.jpg', '0000000000000001', ?, 600, 600, 'hashed')
            """,
            (serialize_embedding(v_phash1), serialize_embedding(v_phash2)),
        )
        from core.stage1 import cluster_stage1_candidates
        cluster_stage1_candidates(conn, {"stage1": {"hamming_threshold": 5}})

        # Stage 2 clip pair (unclustered so far)
        conn.execute(
            """
            INSERT INTO images (file_path, clip_vector, width, height, status)
            VALUES ('s2_1.jpg', ?, 1000, 1000, 'embedded'),
                   ('s2_2.jpg', ?, 800, 800, 'embedded')
            """,
            (serialize_embedding(v_clip1), serialize_embedding(v_clip2)),
        )
        conn.commit()

        cfg = {
            "general": {"db_path": str(db_path)},
            "stage2": {
                "cosine_threshold": 0.92,
                "faiss_k": 10,
                "faiss_index_path": str(faiss_path),
                "embedding_dim": dim,
            },
        }

        res = run_stage2_clustering(conn, cfg)

        # Verify unified clusters table
        clusters = conn.execute("SELECT id, representative_id, stage_detected, member_count FROM clusters ORDER BY id").fetchall()
        assert len(clusters) == 3
        stages = {c["stage_detected"] for c in clusters}
        assert stages == {0, 1, 2}

        # Check images table link_signal
        signals = conn.execute(
            "SELECT file_path, cluster_id, is_keeper, confidence, link_signal FROM images ORDER BY file_path"
        ).fetchall()
        sig_map = {r["file_path"]: dict(r) for r in signals}

        # Stage 0
        assert sig_map["s0_1.jpg"]["link_signal"] == "md5"
        assert sig_map["s0_2.jpg"]["link_signal"] == "md5"
        assert sig_map["s0_1.jpg"]["confidence"] == 100.0

        # Stage 1
        assert sig_map["s1_1.jpg"]["link_signal"] == "phash"
        assert sig_map["s1_2.jpg"]["link_signal"] == "phash"

        # Stage 2
        assert sig_map["s2_1.jpg"]["link_signal"] == "clip"
        assert sig_map["s2_2.jpg"]["link_signal"] == "clip"
        assert sig_map["s2_2.jpg"]["confidence"] >= 90.0

        # Resumability test: running Stage 2 again is idempotent
        res2 = run_stage2_clustering(conn, cfg)
        clusters_after = conn.execute("SELECT COUNT(*) FROM clusters").fetchone()[0]
        assert clusters_after == 3


class TestBestMatchingRepresentativeAssignment:
    """
    Ensure an unclustered image matches the cluster representative with the highest similarity,
    rather than being greedily captured by an earlier-visited cluster in arbitrary database order.
    """

    def test_unclustered_matches_highest_similarity_representative(self, tmp_path):
        db_path = tmp_path / "test_best_rep.db"
        conn = init_db(db_path)
        dim = 512

        def make_vec(angle: float) -> np.ndarray:
            v = np.zeros(dim, dtype=np.float32)
            v[0] = np.cos(angle)
            v[1] = np.sin(angle)
            return v

        # C1 representative at angle +0.40 (cos(0.40) ~= 0.921)
        # C2 representative at angle -0.10 (cos(-0.10) ~= 0.995)
        # C1 vs C2 distance is 0.50 rad (cos(0.50) ~= 0.877 < 0.90, so clusters do not merge)
        # Unclustered image U at angle 0.0 (sim with C1 is 0.921, with C2 is 0.995)
        v_u = make_vec(0.0)
        v_c1 = make_vec(0.40)
        v_c2 = make_vec(-0.10)

        # Cluster 1 (id=1, rep_id=1)
        c1 = conn.execute("INSERT INTO clusters (stage_detected, member_count) VALUES (1, 1)").lastrowid
        conn.execute(
            "INSERT INTO images (id, file_path, clip_vector, cluster_id, is_keeper, status) VALUES (1, 'c1_rep.jpg', ?, ?, 1, 'clustered')",
            (serialize_embedding(v_c1), c1),
        )
        conn.execute("UPDATE clusters SET representative_id = 1 WHERE id = ?", (c1,))

        # Cluster 2 (id=2, rep_id=2)
        c2 = conn.execute("INSERT INTO clusters (stage_detected, member_count) VALUES (1, 1)").lastrowid
        conn.execute(
            "INSERT INTO images (id, file_path, clip_vector, cluster_id, is_keeper, status) VALUES (2, 'c2_rep.jpg', ?, ?, 1, 'clustered')",
            (serialize_embedding(v_c2), c2),
        )
        conn.execute("UPDATE clusters SET representative_id = 2 WHERE id = ?", (c2,))

        # Unclustered image U (id=3)
        conn.execute(
            "INSERT INTO images (id, file_path, clip_vector, cluster_id, status) VALUES (3, 'u.jpg', ?, NULL, 'embedded')",
            (serialize_embedding(v_u),),
        )
        conn.commit()

        cfg = {
            "stage2": {
                "cosine_threshold": 0.90,
                "faiss_k": 10,
                "embedding_dim": dim,
            }
        }

        run_stage2_clustering(conn, cfg)

        u_row = conn.execute("SELECT cluster_id, confidence FROM images WHERE id = 3").fetchone()
        # Must be assigned to c2 (sim 0.995), NOT c1 (sim 0.913)
        assert u_row["cluster_id"] == c2
        assert u_row["confidence"] >= 99.0

