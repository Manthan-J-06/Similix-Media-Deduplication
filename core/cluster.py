"""
core/cluster.py — Stage 2: Anti-chaining duplicate clustering from CLIP embeddings.

Combines:
- Stage 0 exact-MD5 duplicate groups (link_signal = 'md5')
- Stage 1 pHash near-duplicate candidate groups (link_signal = 'phash')
- Stage 2 CLIP embedding candidate pairs (link_signal = 'clip')

Anti-chaining rule:
- A representative is chosen per candidate cluster (largest resolution, shortest path).
- A member is kept only if it is directly similar to the representative (cosine threshold
  and pHash distance where applicable) and mutually compatible with accepted members.
- Members that fail are split out and returned to the pool to be re-evaluated.
- One unified `clusters` table covers all duplicate groups across all stages.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import sqlite3
from typing import Any

import numpy as np

from core.embeddings import deserialize_embedding
from core.index import (
    VectorIndex,
    build_or_update_index,
    retrieve_candidate_pairs,
)


def _hamming_distance(h1: str | None, h2: str | None) -> int:
    """Compute Hamming distance between two hex hash strings."""
    if not h1 or not h2 or len(h1) != len(h2):
        return 999
    try:
        val1 = int(h1, 16)
        val2 = int(h2, 16)
        return bin(val1 ^ val2).count("1")
    except ValueError:
        return 999


def keeper_sort_key(img: dict[str, Any]) -> tuple:
    """
    Deterministic sort key for keeper/representative selection:
    1. Highest pixel resolution (width * height) descending
    2. Shortest absolute file path ascending
    3. Alphabetical file path ascending
    4. Image ID ascending
    """
    area = (img.get("width") or 0) * (img.get("height") or 0)
    file_path = img.get("file_path") or ""
    return (-area, len(file_path), file_path, img["id"])


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Cosine similarity between two unit-normalized vectors."""
    dot = float(np.dot(v1, v2))
    return max(-1.0, min(1.0, dot))


@dataclass
class Stage2Result:
    candidates_found: int = 0
    clusters_created: int = 0
    members_added_to_existing: int = 0
    clusters_merged: int = 0
    total_files_clustered: int = 0


def run_stage2_clustering(
    conn: sqlite3.Connection,
    config: dict | None = None,
    index: VectorIndex | None = None,
    index_path: Path | str | None = None,
) -> Stage2Result:
    """
    Execute Stage 2 duplicate clustering.

    Steps:
    1. Synchronize / load FAISS vector index.
    2. Backfill link_signal for Stage 0 ('md5') and Stage 1 ('phash') if missing.
    3. Retrieve candidate pairs above cosine_threshold.
    4. Match unclustered images against existing cluster representatives.
    5. Merge existing clusters where all members directly match the other's representative.
    6. Cluster remaining unclustered candidates with strict anti-chaining.
    7. Commit changes to SQLite.
    """
    config = config or {}
    stage2_cfg = config.get("stage2", {})
    cosine_threshold = float(stage2_cfg.get("cosine_threshold", 0.92))
    k = int(stage2_cfg.get("faiss_k", 20))
    hamming_threshold = int(config.get("stage1", {}).get("hamming_threshold", 10))

    if index_path is None:
        conn_db_path = None
        try:
            db_row = conn.execute("PRAGMA database_list").fetchone()
            if db_row:
                file_val = db_row["file"] if "file" in db_row.keys() else db_row[2]
                if file_val:
                    conn_db_path = file_val
        except Exception:
            pass
        if not conn_db_path:
            conn_db_path = config.get("general", {}).get("db_path", "argus.db")
        index_path = stage2_cfg.get("faiss_index_path") or str(Path(conn_db_path).with_suffix(".faiss"))

    # 1. Build or incrementally update the FAISS index
    if index is None:
        index, _ = build_or_update_index(conn, index_path, config)

    # 2. Backfill link_signal for existing clusters
    conn.execute(
        """
        UPDATE images
        SET link_signal = 'md5'
        WHERE cluster_id IN (SELECT id FROM clusters WHERE stage_detected = 0)
          AND link_signal IS NULL
        """
    )
    conn.execute(
        """
        UPDATE images
        SET link_signal = 'phash'
        WHERE cluster_id IN (SELECT id FROM clusters WHERE stage_detected = 1)
          AND link_signal IS NULL
        """
    )
    conn.commit()

    res = Stage2Result()

    # 3. Retrieve candidate pairs from FAISS
    candidate_pairs = retrieve_candidate_pairs(
        index, conn, config, k=k, cosine_threshold=cosine_threshold
    )

    # Load all embedded images metadata into memory for fast lookup
    rows = conn.execute(
        """
        SELECT id, file_path, width, height, phash, clip_vector,
               clip_vector_90, clip_vector_180, clip_vector_270,
               cluster_id, is_keeper, link_signal
        FROM images
        WHERE is_raw_pair = 0
        """
    ).fetchall()

    img_map: dict[int, dict[str, Any]] = {}
    vec_map: dict[int, list[np.ndarray]] = {}

    for r in rows:
        img_id = r["id"]
        img_dict = dict(r)
        img_map[img_id] = img_dict
        v_list = []
        for col in ("clip_vector", "clip_vector_90", "clip_vector_180", "clip_vector_270"):
            blob = r[col]
            if blob is not None:
                v_list.append(deserialize_embedding(blob).astype(np.float32))
        if v_list:
            vec_map[img_id] = v_list

    # Defensive check: ensure candidate pairs strictly reference existing images in img_map
    candidate_pairs = [
        (id1, id2, sim) for id1, id2, sim in candidate_pairs
        if id1 in img_map and id2 in img_map
    ]
    res.candidates_found = len(candidate_pairs)

    def get_sim(id1: int, id2: int) -> float:
        if id1 == id2:
            return 1.0
        v_list1 = vec_map.get(id1)
        v_list2 = vec_map.get(id2)
        if not v_list1 or not v_list2:
            return 0.0
        if len(v_list1) > 1 or len(v_list2) > 1:
            best_sim = -1.0
            for v1 in v_list1:
                for v2 in v_list2:
                    s = cosine_similarity(v1, v2)
                    if s > best_sim:
                        best_sim = s
            return best_sim
        else:
            return cosine_similarity(v_list1[0], v_list2[0])

    # 4. Match unclustered images against existing cluster representatives
    existing_clusters_cursor = conn.execute(
        """
        SELECT c.id, c.representative_id, c.member_count
        FROM clusters c
        WHERE c.representative_id IS NOT NULL
        """
    ).fetchall()

    rep_to_cluster = {
        c_row["representative_id"]: c_row["id"]
        for c_row in existing_clusters_cursor
        if c_row["representative_id"] in img_map
    }

    # Collect and rank candidate matches to existing representatives by direct similarity
    # descending to ensure each unclustered image joins its best-matching cluster.
    matches: list[tuple[float, int, int, int]] = []
    for id1, id2, _sim in candidate_pairs:
        rep_id = None
        unclustered_id = None
        if id1 in rep_to_cluster and img_map.get(id2, {}).get("cluster_id") is None:
            rep_id = id1
            unclustered_id = id2
        elif id2 in rep_to_cluster and img_map.get(id1, {}).get("cluster_id") is None:
            rep_id = id2
            unclustered_id = id1

        if rep_id is not None and unclustered_id is not None:
            direct_sim = get_sim(unclustered_id, rep_id)
            if direct_sim >= cosine_threshold:
                matches.append((direct_sim, unclustered_id, rep_id, rep_to_cluster[rep_id]))

    matches.sort(key=lambda x: -x[0])

    for direct_sim, unclustered_id, rep_id, c_id in matches:
        u_meta = img_map[unclustered_id]
        if u_meta.get("cluster_id") is not None:
            continue

        rep_meta = img_map[rep_id]
        rep_phash = rep_meta.get("phash")
        u_phash = u_meta.get("phash")
        if rep_phash and u_phash:
            pdist = _hamming_distance(rep_phash, u_phash)
            signal = "clip" if pdist > hamming_threshold else "phash"
        else:
            signal = "clip"

        conf = round(direct_sim * 100.0, 1)
        conn.execute(
            """
            UPDATE images
            SET cluster_id  = ?,
                is_keeper   = 0,
                confidence  = ?,
                link_signal = ?,
                status      = 'clustered',
                updated_at  = strftime('%Y-%m-%dT%H:%M:%SZ','now')
            WHERE id = ?
            """,
            (c_id, conf, signal, unclustered_id),
        )
        conn.execute(
            """
            UPDATE clusters
            SET member_count = member_count + 1
            WHERE id = ?
            """,
            (c_id,),
        )
        u_meta["cluster_id"] = c_id
        u_meta["is_keeper"] = 0
        u_meta["confidence"] = conf
        u_meta["link_signal"] = signal
        res.members_added_to_existing += 1
        res.total_files_clustered += 1

    conn.commit()

    # 5. Check if existing clusters can merge (anti-chaining: all members directly similar to rep)
    # Refresh cluster list
    active_clusters = conn.execute(
        """
        SELECT c.id, c.representative_id, c.member_count
        FROM clusters c
        WHERE c.representative_id IS NOT NULL
        ORDER BY c.id
        """
    ).fetchall()

    deleted_clusters: set[int] = set()

    for i in range(len(active_clusters)):
        c1 = active_clusters[i]
        c1_id = c1["id"]
        if c1_id in deleted_clusters:
            continue
        r1_id = c1["representative_id"]
        if r1_id not in img_map or img_map[r1_id].get("cluster_id") != c1_id:
            continue

        for j in range(i + 1, len(active_clusters)):
            if c1_id in deleted_clusters:
                break
            c2 = active_clusters[j]
            c2_id = c2["id"]
            if c2_id in deleted_clusters:
                continue
            r2_id = c2["representative_id"]
            if r2_id not in img_map or img_map[r2_id].get("cluster_id") != c2_id:
                continue

            # Check if representatives are candidate duplicates
            rep_sim = get_sim(r1_id, r2_id)
            if rep_sim < cosine_threshold:
                continue

            # Determine keeper between r1 and r2
            keeper_id = r1_id if keeper_sort_key(img_map[r1_id]) <= keeper_sort_key(img_map[r2_id]) else r2_id
            target_cid = c1_id if keeper_id == r1_id else c2_id
            donor_cid = c2_id if target_cid == c1_id else c1_id
            target_rep = keeper_id

            if target_cid in deleted_clusters or donor_cid in deleted_clusters:
                continue

            donor_members = [
                img_id for img_id, m in img_map.items() if m.get("cluster_id") == donor_cid
            ]

            # Anti-chaining: EVERY member in donor cluster must be directly similar to target representative
            all_pass = True
            member_sims: dict[int, float] = {}
            for dm in donor_members:
                sim_to_rep = get_sim(dm, target_rep)
                if sim_to_rep < cosine_threshold:
                    all_pass = False
                    break
                member_sims[dm] = sim_to_rep

            if all_pass and donor_members:
                # Merge donor cluster into target cluster
                for dm in donor_members:
                    sim_val = round(member_sims[dm] * 100.0, 1)
                    conn.execute(
                        """
                        UPDATE images
                        SET cluster_id  = ?,
                            is_keeper   = 0,
                            confidence  = ?,
                            link_signal = 'clip',
                            status      = 'clustered',
                            updated_at  = strftime('%Y-%m-%dT%H:%M:%SZ','now')
                        WHERE id = ?
                        """,
                        (target_cid, sim_val, dm),
                    )
                    img_map[dm]["cluster_id"] = target_cid
                    img_map[dm]["is_keeper"] = 0
                    img_map[dm]["confidence"] = sim_val
                    img_map[dm]["link_signal"] = "clip"

                conn.execute(
                    """
                    UPDATE clusters
                    SET member_count = member_count + ?
                    WHERE id = ?
                    """,
                    (len(donor_members), target_cid),
                )
                conn.execute("DELETE FROM clusters WHERE id = ?", (donor_cid,))
                deleted_clusters.add(donor_cid)
                res.clusters_merged += 1

                if donor_cid == c1_id:
                    break

    conn.commit()

    # 6. Group remaining unclustered images into new Stage 2 clusters with anti-chaining
    adj: dict[int, set[int]] = defaultdict(set)
    for id1, id2, sim in candidate_pairs:
        if id1 in img_map and id2 in img_map:
            if img_map[id1].get("cluster_id") is None and img_map[id2].get("cluster_id") is None:
                adj[id1].add(id2)
                adj[id2].add(id1)

    unclustered_pool = {img_id for img_id in adj if img_id in img_map and img_map[img_id].get("cluster_id") is None}

    while unclustered_pool:
        # Seed from the candidate with the highest keeper score
        seed_id = min(unclustered_pool, key=lambda x: keeper_sort_key(img_map[x]))
        candidate_neighbors = [nbr for nbr in adj[seed_id] if nbr in unclustered_pool]
        candidate_pool = [seed_id] + candidate_neighbors

        # Choose best representative among candidate pool
        representative_id = min(candidate_pool, key=lambda x: keeper_sort_key(img_map[x]))
        rep_meta = img_map[representative_id]
        rep_phash = rep_meta.get("phash")

        valid_members: list[tuple[int, float, str]] = []
        still_unclustered: list[int] = []

        for candidate_id in candidate_pool:
            if candidate_id == representative_id:
                continue

            # Anti-chaining check 1: direct similarity to representative
            direct_sim = get_sim(candidate_id, representative_id)
            if direct_sim < cosine_threshold:
                # Fails representative similarity: split out
                still_unclustered.append(candidate_id)
                continue

            # Anti-chaining check 2: mutual compatibility with already accepted members
            conflict = False
            for accepted_id, _, _ in valid_members:
                if get_sim(candidate_id, accepted_id) < cosine_threshold:
                    conflict = True
                    break

            if conflict:
                # Fails mutual compatibility: split out
                still_unclustered.append(candidate_id)
                continue

            # Determine link signal (phash if matching hamming distance, else clip)
            c_phash = img_map[candidate_id].get("phash")
            if rep_phash and c_phash:
                pdist = _hamming_distance(rep_phash, c_phash)
                signal = "clip" if pdist > hamming_threshold else "phash"
            else:
                signal = "clip"

            valid_members.append((candidate_id, direct_sim, signal))

        if valid_members:
            # We have a valid cluster!
            cur = conn.execute(
                """
                INSERT INTO clusters (representative_id, stage_detected, member_count)
                VALUES (?, 2, ?)
                """,
                (representative_id, len(valid_members) + 1),
            )
            cluster_id = cur.lastrowid

            # Update keeper
            conn.execute(
                """
                UPDATE images
                SET cluster_id  = ?,
                    is_keeper   = 1,
                    confidence  = 100.0,
                    link_signal = 'clip',
                    status      = 'clustered',
                    updated_at  = strftime('%Y-%m-%dT%H:%M:%SZ','now')
                WHERE id = ?
                """,
                (cluster_id, representative_id),
            )
            img_map[representative_id]["cluster_id"] = cluster_id
            img_map[representative_id]["is_keeper"] = 1
            img_map[representative_id]["confidence"] = 100.0
            img_map[representative_id]["link_signal"] = "clip"
            unclustered_pool.discard(representative_id)

            # Update duplicate members
            for m_id, sim_score, signal in valid_members:
                conf = round(sim_score * 100.0, 1)
                conn.execute(
                    """
                    UPDATE images
                    SET cluster_id  = ?,
                        is_keeper   = 0,
                        confidence  = ?,
                        link_signal = ?,
                        status      = 'clustered',
                        updated_at  = strftime('%Y-%m-%dT%H:%M:%SZ','now')
                    WHERE id = ?
                    """,
                    (cluster_id, conf, signal, m_id),
                )
                img_map[m_id]["cluster_id"] = cluster_id
                img_map[m_id]["is_keeper"] = 0
                img_map[m_id]["confidence"] = conf
                img_map[m_id]["link_signal"] = signal
                unclustered_pool.discard(m_id)

            res.clusters_created += 1
            res.total_files_clustered += len(valid_members) + 1
        else:
            # Seed could not form a cluster, remove it
            unclustered_pool.discard(seed_id)

    # Populate images.confidence using the fused logistic regression model
    from core.verify import update_cluster_duplicate_confidences
    update_cluster_duplicate_confidences(conn)

    conn.commit()
    return res
