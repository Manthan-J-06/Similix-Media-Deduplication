"""
core/index.py — Vector similarity index management using FAISS for Stage 2.

Features:
- Abstract VectorIndex interface to allow swapping between exact (FlatIP)
  and approximate (IVF/HNSW) indices when library size scales beyond ~200k images.
- FlatIPIndex: exact inner product (cosine similarity on L2-normalized vectors).
- HNSWIndex: approximate graph-based index for large-scale retrieval.
- Incremental indexing: persists to disk and only adds new images on subsequent runs.
- Top-K candidate pair retrieval above configurable cosine threshold.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import os
from pathlib import Path
import sqlite3
from typing import Sequence

# Avoid OpenMP multiple runtime conflict between PyTorch and FAISS on macOS
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import faiss
import numpy as np

from core.embeddings import deserialize_embedding


class VectorIndex(ABC):
    """Abstract interface for vector similarity indices in Argus."""

    @abstractmethod
    def add(self, ids: Sequence[int] | np.ndarray, vectors: np.ndarray) -> None:
        """Add vectors with corresponding integer image IDs."""
        ...

    @abstractmethod
    def search(
        self, queries: np.ndarray, k: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Search for top-K nearest neighbors.
        Returns (similarities, ids) of shape (N, actual_k).
        """
        ...

    @abstractmethod
    def save(self, path: Path | str) -> None:
        """Persist index to disk."""
        ...

    @abstractmethod
    def load(self, path: Path | str) -> None:
        """Load index from disk."""
        ...

    @abstractmethod
    def get_indexed_ids(self) -> set[int]:
        """Return the set of image IDs currently stored in the index."""
        ...

    @property
    @abstractmethod
    def ntotal(self) -> int:
        """Total number of vectors in the index."""
        ...


class FlatIPIndex(VectorIndex):
    """
    Exact inner-product index using FAISS IndexFlatIP with IndexIDMap2.
    Since embeddings are L2-normalized, inner-product equals cosine similarity.
    """

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim
        self._sub_index = faiss.IndexFlatIP(dim)
        self.index = faiss.IndexIDMap2(self._sub_index)

    def add(self, ids: Sequence[int] | np.ndarray, vectors: np.ndarray) -> None:
        if len(ids) == 0:
            return
        ids_arr = np.ascontiguousarray(ids, dtype=np.int64)
        vecs_arr = np.ascontiguousarray(vectors, dtype=np.float32)
        if vecs_arr.ndim == 1:
            vecs_arr = vecs_arr.reshape(1, -1)
        self.index.add_with_ids(vecs_arr, ids_arr)

    def search(
        self, queries: np.ndarray, k: int
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.ntotal == 0 or len(queries) == 0:
            n_queries = len(queries) if queries.ndim > 1 else (1 if len(queries) > 0 else 0)
            return (
                np.empty((n_queries, 0), dtype=np.float32),
                np.empty((n_queries, 0), dtype=np.int64),
            )
        q = np.ascontiguousarray(queries, dtype=np.float32)
        if q.ndim == 1:
            q = q.reshape(1, -1)
        actual_k = min(k, self.ntotal)
        sims, idxs = self.index.search(q, actual_k)
        return sims, idxs

    def save(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(p))

    def load(self, path: Path | str) -> None:
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"FAISS index file not found: {p}")
        self.index = faiss.read_index(str(p))
        self.dim = self.index.d

    def get_indexed_ids(self) -> set[int]:
        if self.ntotal == 0:
            return set()
        arr = faiss.vector_to_array(self.index.id_map)
        return set(arr.tolist())

    @property
    def ntotal(self) -> int:
        return self.index.ntotal


class HNSWIndex(VectorIndex):
    """
    Approximate graph-based index for large collections (>200k images).
    Uses FAISS IndexHNSWFlat with inner-product metric and IndexIDMap2.
    """

    def __init__(self, dim: int = 512, m: int = 32) -> None:
        self.dim = dim
        self.m = m
        self._sub_index = faiss.IndexHNSWFlat(dim, m, faiss.METRIC_INNER_PRODUCT)
        self.index = faiss.IndexIDMap2(self._sub_index)

    def add(self, ids: Sequence[int] | np.ndarray, vectors: np.ndarray) -> None:
        if len(ids) == 0:
            return
        ids_arr = np.ascontiguousarray(ids, dtype=np.int64)
        vecs_arr = np.ascontiguousarray(vectors, dtype=np.float32)
        if vecs_arr.ndim == 1:
            vecs_arr = vecs_arr.reshape(1, -1)
        self.index.add_with_ids(vecs_arr, ids_arr)

    def search(
        self, queries: np.ndarray, k: int
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.ntotal == 0 or len(queries) == 0:
            n_queries = len(queries) if queries.ndim > 1 else (1 if len(queries) > 0 else 0)
            return (
                np.empty((n_queries, 0), dtype=np.float32),
                np.empty((n_queries, 0), dtype=np.int64),
            )
        q = np.ascontiguousarray(queries, dtype=np.float32)
        if q.ndim == 1:
            q = q.reshape(1, -1)
        actual_k = min(k, self.ntotal)
        sims, idxs = self.index.search(q, actual_k)
        return sims, idxs

    def save(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(p))

    def load(self, path: Path | str) -> None:
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"FAISS index file not found: {p}")
        self.index = faiss.read_index(str(p))
        self.dim = self.index.d

    def get_indexed_ids(self) -> set[int]:
        if self.ntotal == 0:
            return set()
        arr = faiss.vector_to_array(self.index.id_map)
        return set(arr.tolist())

    @property
    def ntotal(self) -> int:
        return self.index.ntotal


def create_index(dim: int = 512, index_type: str = "exact") -> VectorIndex:
    """Factory to instantiate the appropriate VectorIndex subclass."""
    index_type = index_type.lower()
    if index_type in ("exact", "flat", "flat_ip"):
        return FlatIPIndex(dim=dim)
    elif index_type in ("hnsw", "approximate"):
        return HNSWIndex(dim=dim)
    else:
        raise ValueError(f"Unknown index_type: {index_type}. Must be 'exact' or 'hnsw'.")


def load_or_create_index(
    path: Path | str, dim: int = 512, index_type: str = "exact"
) -> VectorIndex:
    """Load an existing FAISS index from path, or initialize a new one if not present."""
    p = Path(path)
    idx = create_index(dim=dim, index_type=index_type)
    if p.is_file():
        idx.load(p)
    return idx


def build_or_update_index(
    conn: sqlite3.Connection,
    index_path: Path | str,
    config: dict | None = None,
    batch_size: int = 1000,
) -> tuple[VectorIndex, int]:
    """
    Build or incrementally update the FAISS vector index from stored embeddings in SQLite.

    - Loads index from disk if it exists.
    - Only selects and indexes images that have a clip_vector and are not yet in the index.
    - Saves updated index to disk.
    - Returns (index, newly_added_count).
    """
    config = config or {}
    stage2_cfg = config.get("stage2", {})
    index_type = stage2_cfg.get("index_type", "exact")
    dim = stage2_cfg.get("embedding_dim", 512)

    index = load_or_create_index(index_path, dim=dim, index_type=index_type)
    already_indexed = index.get_indexed_ids()

    cursor = conn.execute(
        """
        SELECT id, clip_vector
        FROM images
        WHERE clip_vector IS NOT NULL
          AND is_raw_pair = 0
        ORDER BY id
        """
    )

    ids_to_add: list[int] = []
    vecs_to_add: list[np.ndarray] = []
    total_added = 0

    while True:
        rows = cursor.fetchmany(batch_size)
        if not rows:
            break

        for row in rows:
            img_id = row["id"]
            if img_id in already_indexed:
                continue

            blob = row["clip_vector"]
            emb = deserialize_embedding(blob).astype(np.float32)
            ids_to_add.append(img_id)
            vecs_to_add.append(emb)

        if len(ids_to_add) >= batch_size:
            index.add(ids_to_add, np.vstack(vecs_to_add))
            total_added += len(ids_to_add)
            ids_to_add.clear()
            vecs_to_add.clear()

    if ids_to_add:
        index.add(ids_to_add, np.vstack(vecs_to_add))
        total_added += len(ids_to_add)

    if total_added > 0:
        index.save(index_path)

    return index, total_added


def retrieve_candidate_pairs(
    index: VectorIndex,
    conn: sqlite3.Connection,
    config: dict | None = None,
    k: int | None = None,
    cosine_threshold: float | None = None,
) -> list[tuple[int, int, float]]:
    """
    Retrieve candidate duplicate pairs from the index.

    For each image in the index:
    1. Queries top-K neighbors.
    2. Filters out self-matches and neighbors with similarity < cosine_threshold.
    3. If rotation embeddings exist and are enabled, also queries rotated vectors.
    4. Deduplicates pairs so that (A, B) and (B, A) are represented once as (min(A,B), max(A,B), sim).

    Returns:
        List of (image_id_1, image_id_2, cosine_similarity) sorted descending by similarity.
    """
    config = config or {}
    stage2_cfg = config.get("stage2", {})
    if k is None:
        k = stage2_cfg.get("faiss_k", 20)
    if cosine_threshold is None:
        cosine_threshold = stage2_cfg.get("cosine_threshold", 0.92)

    enable_rotations = stage2_cfg.get("enable_rotation_embeddings", False)

    # Query all vectors from DB for indexed images
    rows = conn.execute(
        """
        SELECT id, clip_vector, clip_vector_90, clip_vector_180, clip_vector_270
        FROM images
        WHERE clip_vector IS NOT NULL
          AND is_raw_pair = 0
        ORDER BY id
        """
    ).fetchall()

    if not rows or index.ntotal == 0:
        return []

    # Map of (min_id, max_id) -> max_similarity
    pair_dict: dict[tuple[int, int], float] = {}

    # Query in batches
    batch_size = 256
    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        q_ids = [r["id"] for r in batch]
        q_vecs = np.vstack([
            deserialize_embedding(r["clip_vector"]).astype(np.float32) for r in batch
        ])

        sims, idxs = index.search(q_vecs, k=k + 1)  # +1 to account for self

        for row_idx, q_id in enumerate(q_ids):
            for rank in range(sims.shape[1]):
                nbr_id = int(idxs[row_idx, rank])
                sim = float(sims[row_idx, rank])
                if nbr_id == q_id or nbr_id == -1:
                    continue
                if sim >= cosine_threshold:
                    key = (min(q_id, nbr_id), max(q_id, nbr_id))
                    if key not in pair_dict or sim > pair_dict[key]:
                        pair_dict[key] = sim

        # If rotation embeddings enabled, also query rotated variants
        if enable_rotations:
            for rot_col in ("clip_vector_90", "clip_vector_180", "clip_vector_270"):
                rot_items = [(r["id"], r[rot_col]) for r in batch if r[rot_col] is not None]
                if not rot_items:
                    continue
                rot_ids = [it[0] for it in rot_items]
                rot_vecs = np.vstack([
                    deserialize_embedding(it[1]).astype(np.float32) for it in rot_items
                ])
                rot_sims, rot_idxs = index.search(rot_vecs, k=k + 1)
                for row_idx, q_id in enumerate(rot_ids):
                    for rank in range(rot_sims.shape[1]):
                        nbr_id = int(rot_idxs[row_idx, rank])
                        sim = float(rot_sims[row_idx, rank])
                        if nbr_id == q_id or nbr_id == -1:
                            continue
                        if sim >= cosine_threshold:
                            key = (min(q_id, nbr_id), max(q_id, nbr_id))
                            if key not in pair_dict or sim > pair_dict[key]:
                                pair_dict[key] = sim

    candidates = [
        (pair[0], pair[1], round(score, 4))
        for pair, score in pair_dict.items()
    ]
    candidates.sort(key=lambda x: -x[2])
    return candidates
