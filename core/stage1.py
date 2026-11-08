"""
core/stage1.py — Stage 1: Perceptual hashing and near-duplicate clustering.

Pipeline position: Stage 1 (after Stage 0 exact match).
Input:  an ImageSource (or Path) + SQLite connection + config dict.
Output: Stage1Result + DB side-effects.

Principles applied:
- Memory discipline: relies on ImageSource.get_thumbnail (JPEG draft mode).
- Thresholds strictly from config.toml.
- Clustering rule: verify all cluster members against the cluster representative
  to avoid transitive chaining (A≈B and B≈C does NOT mean A≈C).
- RAW+JPEG pairs are excluded from deduplication.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import imagehash
from PIL import Image

from sources.base import ImageSource
from sources.local_folder import LocalFolderSource


@dataclass
class Stage1Result:
    hashes_computed: int = 0
    clusters_created: int = 0
    members_added_to_existing: int = 0
    files_clustered: int = 0
    files_skipped: int = 0
    skipped_reasons: list[tuple[str, str]] = field(default_factory=list)


def _compute_hash(
    thumb: Image.Image,
    algo: str = "phash",
    hash_size: int = 8,
) -> imagehash.ImageHash:
    """Compute perceptual hash using the configured algorithm."""
    algo = algo.lower()
    if algo == "dhash":
        return imagehash.dhash(thumb, hash_size=hash_size)
    elif algo == "ahash":
        return imagehash.average_hash(thumb, hash_size=hash_size)
    elif algo == "whash":
        return imagehash.whash(thumb, hash_size=hash_size)
    else:  # default "phash"
        return imagehash.phash(thumb, hash_size=hash_size)


def _hamming_distance(hash1_hex: str, hash2_hex: str) -> int:
    """Compute Hamming distance between two hex hash strings using fast integer XOR popcount."""
    try:
        return (int(hash1_hex, 16) ^ int(hash2_hex, 16)).bit_count()
    except Exception:
        h1 = imagehash.hex_to_hash(hash1_hex)
        h2 = imagehash.hex_to_hash(hash2_hex)
        return h1 - h2


def compute_stage1_hashes(
    source: ImageSource,
    conn: sqlite3.Connection,
    config: dict,
    result: Stage1Result | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> int:
    """
    Compute perceptual hashes for all images that don't have one yet.
    Processes in batches of 50, using an 8-worker thread pool for thumbnails
    and reporting "N of M" after every file.
    """
    thumb_size = config.get("general", {}).get("thumbnail_size", 256)
    batch_size = config.get("stage1", {}).get("batch_size", 50)
    hash_algo = config.get("stage1", {}).get("hash_algorithm", "phash")
    hash_size = config.get("stage1", {}).get("hash_size", 8)

    thumb_cache_dir_raw = config.get("sources", {}).get("thumb_cache_dir") or config.get("general", {}).get("thumb_cache_dir")
    if thumb_cache_dir_raw:
        thumb_cache_dir = Path(thumb_cache_dir_raw).expanduser().resolve()
    else:
        try:
            from core.config import resolve_data_dir
            thumb_cache_dir = resolve_data_dir() / "thumbnails"
        except Exception:
            thumb_cache_dir = None
    if thumb_cache_dir:
        try:
            thumb_cache_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

    total_to_hash = conn.execute(
        """
        SELECT COUNT(*)
        FROM images
        WHERE phash IS NULL
          AND is_raw_pair = 0
          AND (status IS NULL OR status != 'skipped')
        """
    ).fetchone()[0]

    if total_to_hash == 0:
        return 0

    hashes_computed = 0
    hashes_processed = 0

    while True:
        batch = conn.execute(
            """
            SELECT id, file_path
            FROM images
            WHERE phash IS NULL
              AND is_raw_pair = 0
              AND (status IS NULL OR status != 'skipped')
            ORDER BY id
            LIMIT ?
            """,
            (batch_size,),
        ).fetchall()

        if not batch:
            break

        file_ids = [row["file_path"] for row in batch]

        def _on_file_progress(done_in_batch: int, total_in_batch: int, fpath: str):
            cur_overall = hashes_processed + done_in_batch
            if progress_callback:
                progress_callback(cur_overall, total_to_hash, fpath)

        # Concurrently fetch thumbnails with 8 workers
        thumbs_map = source.fetch_thumbnails(
            file_ids,
            max_workers=8,
            progress_callback=_on_file_progress,
            size=(thumb_size, thumb_size),
        )

        updates: list[tuple[str, int]] = []
        failed_ids: list[int] = []

        for idx, row in enumerate(batch):
            fid = row["id"]
            fpath = row["file_path"]
            if progress_callback:
                try:
                    progress_callback(hashes_processed + idx, total_to_hash, fpath)
                except Exception:
                    pass
            thumb = thumbs_map.get(fpath)
            if thumb is None:
                # Concurrent fetch with retries already attempted this file; record skip without redundant blocking
                if result is not None:
                    result.files_skipped += 1
                    result.skipped_reasons.append((fpath, "Thumbnail download or decode failed"))
                failed_ids.append(fid)
                continue

            # Pre-cache thumbnail on ingestion to eliminate CPU starvation during UI browsing
            if thumb_cache_dir is not None and thumb is not None:
                try:
                    row_md5 = row["md5"] if "md5" in row.keys() and row["md5"] else None
                    cache_fname = f"{row_md5}_{thumb_size}.jpg" if row_md5 else f"{fid}_{thumb_size}.jpg"
                    pre_thumb_path = thumb_cache_dir / cache_fname
                    if not pre_thumb_path.exists():
                        thumb.save(str(pre_thumb_path), format="JPEG", quality=85)
                except Exception:
                    pass

            try:
                phash_val = str(_compute_hash(thumb, algo=hash_algo, hash_size=hash_size))
                updates.append((phash_val, fid))
            except Exception as exc:
                if result is not None:
                    result.files_skipped += 1
                    result.skipped_reasons.append((fpath, f"pHash computation failed: {exc}"))
                failed_ids.append(fid)
                continue

        if updates:
            conn.executemany(
                """
                UPDATE images
                SET phash = ?,
                    updated_at = strftime('%Y-%m-%dT%H:%M:%SZ','now')
                WHERE id = ?
                """,
                updates,
            )
            conn.commit()
            hashes_computed += len(updates)

        if failed_ids:
            conn.executemany(
                """
                UPDATE images
                SET status = 'skipped',
                    updated_at = strftime('%Y-%m-%dT%H:%M:%SZ','now')
                WHERE id = ?
                """,
                [(fid,) for fid in failed_ids],
            )
            conn.commit()

        hashes_processed += len(batch)

    return hashes_computed


def cluster_stage1_candidates(
    conn: sqlite3.Connection,
    config: dict,
) -> tuple[int, int, int]:
    """
    Group near-duplicate images based on perceptual hash Hamming distance.

    Enforces the representative verification rule:
    Each cluster member must be within hamming_threshold of the cluster
    representative (prevents chaining).

    Returns (new_clusters, members_added, total_clustered).
    """
    hamming_threshold: int = config.get("stage1", {}).get("hamming_threshold", 10)
    hash_size: int = config.get("stage1", {}).get("hash_size", 8)
    max_bits = hash_size * hash_size

    new_clusters = 0
    members_added = 0
    total_clustered = 0

    # 1. Check existing clusters and see if any unclustered images match their representative
    existing_clusters = conn.execute(
        """
        SELECT c.id, c.representative_id, i.phash
        FROM clusters c
        JOIN images i ON i.id = c.representative_id
        WHERE i.phash IS NOT NULL
        """
    ).fetchall()

    unclustered = conn.execute(
        """
        SELECT id, file_path, phash, width, height
        FROM images
        WHERE phash IS NOT NULL
          AND cluster_id IS NULL
          AND is_raw_pair = 0
        ORDER BY id
        """
    ).fetchall()

    def _to_phash_dict(r):
        d = dict(r)
        p = d.get("phash")
        try:
            d["phash_int"] = int(p, 16) if p else None
        except Exception:
            d["phash_int"] = None
        return d

    existing_clusters = [_to_phash_dict(c) for c in existing_clusters]
    unclustered_list = [_to_phash_dict(u) for u in unclustered]

    # Match unclustered against existing representatives
    for cluster in existing_clusters:
        cluster_id = cluster["id"]
        rep_phash = cluster["phash"]
        rep_int = cluster.get("phash_int")
        still_unclustered = []

        for img in unclustered_list:
            img_int = img.get("phash_int")
            if img_int is not None and rep_int is not None:
                dist = (img_int ^ rep_int).bit_count()
            else:
                dist = _hamming_distance(img["phash"], rep_phash)
            if dist <= hamming_threshold:
                # Add to existing cluster
                similarity = round((1.0 - (dist / max_bits)) * 100.0, 1)
                conn.execute(
                    """
                    UPDATE images
                    SET cluster_id  = ?,
                        confidence  = ?,
                        link_signal = 'phash',
                        status      = 'clustered',
                        updated_at  = strftime('%Y-%m-%dT%H:%M:%SZ','now')
                    WHERE id = ?
                    """,
                    (cluster_id, similarity, img["id"]),
                )
                conn.execute(
                    """
                    UPDATE clusters
                    SET member_count = member_count + 1
                    WHERE id = ?
                    """,
                    (cluster_id,),
                )
                members_added += 1
                total_clustered += 1
            else:
                still_unclustered.append(img)

        unclustered_list = still_unclustered

    # 2. Group remaining unclustered images into new clusters
    # We iterate and find near duplicates, verifying each new member against the representative
    i = 0
    while i < len(unclustered_list):
        candidate_rep = unclustered_list[i]
        rep_int = candidate_rep.get("phash_int")
        matched_members = []
        still_unclustered = []

        for j in range(i + 1, len(unclustered_list)):
            other = unclustered_list[j]
            other_int = other.get("phash_int")
            if rep_int is not None and other_int is not None:
                dist = (rep_int ^ other_int).bit_count()
            else:
                dist = _hamming_distance(candidate_rep["phash"], other["phash"])
            if dist <= hamming_threshold:
                matched_members.append((other, dist))
            else:
                still_unclustered.append(other)

        if matched_members:
            # We have a new near-duplicate cluster!
            # Representative candidate: choose between candidate_rep and matched_members
            # Deterministic keeper selection: largest resolution (w*h), then shortest path
            all_candidates = [candidate_rep] + [m[0] for m in matched_members]

            def keeper_sort_key(img_row):
                area = (img_row["width"] or 0) * (img_row["height"] or 0)
                # descending area, ascending path length
                return (-area, len(img_row["file_path"]), img_row["file_path"])

            sorted_candidates = sorted(all_candidates, key=keeper_sort_key)
            representative = sorted_candidates[0]
            rep_phash = representative["phash"]
            rep_phash_int = representative.get("phash_int")

            # Filter members to ensure strictly dist(member, representative) <= threshold
            valid_members = []
            for img in all_candidates:
                if img["id"] == representative["id"]:
                    continue
                img_int = img.get("phash_int")
                if img_int is not None and rep_phash_int is not None:
                    dist = (img_int ^ rep_phash_int).bit_count()
                else:
                    dist = _hamming_distance(img["phash"], rep_phash)
                if dist <= hamming_threshold:
                    valid_members.append((img, dist))
                else:
                    # Chaining prevented! Put back into unclustered
                    still_unclustered.append(img)

            if valid_members:
                cur = conn.execute(
                    """
                    INSERT INTO clusters (representative_id, stage_detected, member_count)
                    VALUES (?, 1, ?)
                    """,
                    (representative["id"], len(valid_members) + 1),
                )
                cluster_id = cur.lastrowid

                # Set keeper
                conn.execute(
                    """
                    UPDATE images
                    SET cluster_id  = ?,
                        is_keeper   = 1,
                        confidence  = 100.0,
                        link_signal = 'phash',
                        status      = 'clustered',
                        updated_at  = strftime('%Y-%m-%dT%H:%M:%SZ','now')
                    WHERE id = ?
                    """,
                    (cluster_id, representative["id"]),
                )

                # Set members
                for member, dist in valid_members:
                    similarity = round((1.0 - (dist / max_bits)) * 100.0, 1)
                    conn.execute(
                        """
                        UPDATE images
                        SET cluster_id  = ?,
                            is_keeper   = 0,
                            confidence  = ?,
                            link_signal = 'phash',
                            status      = 'clustered',
                            updated_at  = strftime('%Y-%m-%dT%H:%M:%SZ','now')
                        WHERE id = ?
                        """,
                        (cluster_id, similarity, member["id"]),
                    )

                new_clusters += 1
                total_clustered += len(valid_members) + 1

        unclustered_list = still_unclustered
        # Move forward since unclustered_list was filtered
        i = 0

    conn.commit()
    return new_clusters, members_added, total_clustered


def run_stage1(
    source_or_root: ImageSource | Path | str,
    conn: sqlite3.Connection,
    config: dict,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> Stage1Result:
    """Execute Stage 1 end-to-end."""
    result = Stage1Result()

    if isinstance(source_or_root, ImageSource):
        source = source_or_root
    else:
        source = LocalFolderSource(source_or_root)

    # Step 1: Compute pHash on thumbnails using draft mode and thread pool
    result.hashes_computed = compute_stage1_hashes(
        source, conn, config, result=result, progress_callback=progress_callback
    )

    # Step 2: Form near-duplicate clusters
    new_clusters, members_added, total_clustered = cluster_stage1_candidates(conn, config)
    result.clusters_created = new_clusters
    result.members_added_to_existing = members_added
    result.files_clustered = total_clustered

    return result
