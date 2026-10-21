"""
core/stage0.py — Stage 0: File scan, MD5 hashing, RAW-pair detection,
                           exact-duplicate clustering.

Pipeline position: Stage 0 (cheapest stage).
Input:  an ImageSource (or Path) + SQLite connection + config dict.
Output: Stage0Result + DB side-effects.

Status lifecycle:
    'discovered' -> 'hashed' -> 'clustered' (if duplicate) or remains 'hashed' (if unique)

Rescanning:
    Skips unchanged files when file_size and modified_time match the database record.
"""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import pillow_heif
pillow_heif.register_heif_opener()

from sources.base import ImageSource, ImageMetadata
from sources.local_folder import LocalFolderSource, RAW_EXTENSIONS

_MD5_CHUNK = 65_536  # 64 KB read chunks for MD5 hashing


@dataclass
class Stage0Result:
    files_discovered: int = 0        # new or updated files queued for hashing
    files_hashed: int = 0            # files whose MD5 was computed this run
    raw_pairs_marked: int = 0        # files flagged as RAW+JPEG siblings
    clusters_created: int = 0        # new exact-duplicate clusters
    files_clustered: int = 0         # total files assigned to clusters
    files_unchanged: int = 0         # files already in DB and unchanged
    files_skipped: int = 0           # files skipped / failed to decode
    skipped_reasons: list[tuple[str, str]] = field(default_factory=list) # (file_path, reason)


def _md5_file(path: Path) -> str:
    """Compute hex MD5 digest using chunks."""
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_MD5_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def _md5_bytes(data: bytes) -> str:
    """Compute hex MD5 digest from bytes."""
    return hashlib.md5(data).hexdigest()


_md5 = _md5_file


def _image_extensions(config: dict) -> frozenset[str]:
    from sources.local_folder import DEFAULT_EXTENSIONS
    raw_exts = {
        f".{e.lower().lstrip('.')}"
        for e in config.get("raw", {}).get("extensions", [])
    }
    return DEFAULT_EXTENSIONS | raw_exts


def _iter_image_files(root: Path, extensions: frozenset[str]) -> Iterator[Path]:
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in extensions:
            yield path


def _raw_extensions(config: dict) -> frozenset[str]:
    return frozenset(
        f".{e.lower().lstrip('.')}"
        for e in config.get("raw", {}).get("extensions", [])
    )


def _scan_and_sync_files(
    source: ImageSource,
    conn: sqlite3.Connection,
    raw_exts: frozenset[str] = RAW_EXTENSIONS,
) -> tuple[int, int, list[tuple[str, str]]]:
    """
    Sync source files with the database.
    - If file not in DB: inserts with status = 'discovered'
    - If file in DB and unchanged (size & mtime match): skips and counts as unchanged
    - If file in DB but modified: updates metadata and resets status = 'discovered'
    - If file fails to decode or has invalid header: records as skipped with reason

    Returns (discovered_count, unchanged_count, skipped_files).
    """
    existing_rows = conn.execute(
        "SELECT id, file_path, file_size, modified_time FROM images"
    ).fetchall()
    existing_map = {
        row["file_path"]: (row["id"], row["file_size"], row["modified_time"])
        for row in existing_rows
    }

    discovered_count = 0
    unchanged_count = 0
    skipped_files: list[tuple[str, str]] = []
    inserts: list[tuple[str, int, float, int | None, int | None, str | None]] = []
    updates: list[tuple[int, float, int | None, int | None, str | None, int]] = []

    for file_id in source.list_files():
        try:
            meta = source.get_metadata(file_id)
        except Exception as exc:
            skipped_files.append((file_id, str(exc)))
            continue

        if meta.file_size == 0:
            skipped_files.append((file_id, "Empty file (0 bytes)"))
            continue

        p_ext = Path(file_id).suffix.lower()
        is_remote = getattr(meta, "source", "local") != "local"
        if (meta.width is None or meta.height is None) and p_ext not in raw_exts and not is_remote:
            skipped_files.append((file_id, "Cannot identify image dimensions (corrupt or unreadable)"))
            continue

        if file_id in existing_map:
            img_id, cur_size, cur_mtime = existing_map[file_id]
            # Check if unchanged
            if cur_size == meta.file_size and cur_mtime is not None and abs(cur_mtime - meta.modified_time) < 1e-4:
                unchanged_count += 1
                continue

            # Modified file — reset for re-processing
            meta_md5 = getattr(meta, "md5", None)
            meta_source = getattr(meta, "source", "local")
            meta_drive_id = getattr(meta, "drive_id", None)
            status = "hashed" if meta_md5 else "discovered"
            updates.append((
                meta.file_size,
                meta.modified_time,
                meta.width,
                meta.height,
                meta.format,
                meta_md5,
                meta_source,
                meta_drive_id,
                status,
                img_id,
            ))
            discovered_count += 1
        else:
            meta_md5 = getattr(meta, "md5", None)
            meta_source = getattr(meta, "source", "local")
            meta_drive_id = getattr(meta, "drive_id", None)
            status = "hashed" if meta_md5 else "discovered"
            inserts.append((
                meta.file_id,
                meta.file_size,
                meta.modified_time,
                meta.width,
                meta.height,
                meta.format,
                meta_md5,
                meta_source,
                meta_drive_id,
                status,
            ))
            discovered_count += 1

    if inserts:
        conn.executemany(
            """
            INSERT OR IGNORE INTO images (
                file_path, file_size, modified_time, width, height, format,
                md5, source, drive_id, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            inserts,
        )

    if updates:
        conn.executemany(
            """
            UPDATE images
            SET file_size = ?,
                modified_time = ?,
                width = ?,
                height = ?,
                format = ?,
                md5 = ?,
                source = ?,
                drive_id = ?,
                phash = NULL,
                clip_vector = NULL,
                cluster_id = NULL,
                is_keeper = 0,
                confidence = NULL,
                status = ?,
                updated_at = strftime('%Y-%m-%dT%H:%M:%SZ','now')
            WHERE id = ?
            """,
            updates,
        )

    conn.commit()
    return discovered_count, unchanged_count, skipped_files


def _detect_raw_pairs(conn: sqlite3.Connection, raw_exts: frozenset[str]) -> int:
    """
    Mark files as RAW+JPEG siblings when the same directory contains both a
    RAW file and a non-RAW file with an identical stem (case-insensitive).
    """
    rows = conn.execute(
        "SELECT id, file_path FROM images"
    ).fetchall()

    from collections import defaultdict
    stem_map: dict[tuple[str, str], list[tuple[int, str]]] = defaultdict(list)
    for row in rows:
        p = Path(row["file_path"])
        key = (str(p.parent), p.stem.lower())
        stem_map[key].append((row["id"], p.suffix.lower()))

    ids_to_flag: set[int] = set()
    for (_dir, _stem), members in stem_map.items():
        exts = {ext for (_id, ext) in members}
        has_raw = bool(exts & raw_exts)
        has_non_raw = bool(exts - raw_exts)
        if has_raw and has_non_raw:
            for (img_id, _ext) in members:
                ids_to_flag.add(img_id)

    if not ids_to_flag:
        return 0

    conn.executemany(
        """
        UPDATE images
        SET is_raw_pair = 1,
            updated_at  = strftime('%Y-%m-%dT%H:%M:%SZ','now')
        WHERE id = ?
        """,
        [(img_id,) for img_id in ids_to_flag],
    )
    conn.commit()
    return len(ids_to_flag)


def _build_exact_clusters(conn: sqlite3.Connection) -> tuple[int, int]:
    """
    Find all (file_size, md5) groups with >1 non-RAW-pair member and create a
    cluster row for each. Assigns cluster_id + marks keeper and updates status to 'clustered'.

    Keeper rule: shortest absolute file_path (deterministic tiebreak).
    """
    groups = conn.execute(
        """
        SELECT file_size, md5, COUNT(*) AS cnt
        FROM images
        WHERE status IN ('hashed', 'stage0_done')
          AND is_raw_pair = 0
          AND md5 IS NOT NULL
          AND cluster_id IS NULL
        GROUP BY file_size, md5
        HAVING cnt > 1
        """
    ).fetchall()

    clusters_created = 0
    files_clustered = 0

    for group in groups:
        members = conn.execute(
            """
            SELECT id, file_path
            FROM images
            WHERE file_size = ? AND md5 = ? AND is_raw_pair = 0 AND cluster_id IS NULL
            ORDER BY LENGTH(file_path), file_path
            """,
            (group["file_size"], group["md5"]),
        ).fetchall()

        if len(members) < 2:
            continue

        keeper_id = members[0]["id"]

        cur = conn.execute(
            """
            INSERT INTO clusters (representative_id, stage_detected, member_count)
            VALUES (?, 0, ?)
            """,
            (keeper_id, len(members)),
        )
        cluster_id = cur.lastrowid

        conn.executemany(
            """
            UPDATE images
            SET cluster_id  = ?,
                is_keeper   = CASE WHEN id = ? THEN 1 ELSE 0 END,
                confidence  = 100.0,
                link_signal = 'md5',
                status      = 'clustered',
                updated_at  = strftime('%Y-%m-%dT%H:%M:%SZ','now')
            WHERE id = ?
            """,
            [(cluster_id, keeper_id, row["id"]) for row in members],
        )

        clusters_created += 1
        files_clustered += len(members)

    conn.commit()
    return clusters_created, files_clustered


def run_stage0(
    source_or_root: ImageSource | Path | str,
    conn: sqlite3.Connection,
    config: dict,
) -> Stage0Result:
    """
    Execute Stage 0 end-to-end.
    """
    result = Stage0Result()
    batch_size: int = config.get("general", {}).get("batch_size", 500)
    raw_exts = _raw_extensions(config)

    if isinstance(source_or_root, ImageSource):
        source = source_or_root
    else:
        source = LocalFolderSource(source_or_root)

    # Step 1: Scan and sync files (skipping unchanged files)
    discovered, unchanged, skipped = _scan_and_sync_files(source, conn, raw_exts)
    result.files_discovered = discovered
    result.files_unchanged = unchanged
    result.files_skipped = len(skipped)
    result.skipped_reasons.extend(skipped)

    # Step 2: Compute MD5 for discovered files in batches
    while True:
        batch = conn.execute(
            "SELECT id, file_path FROM images WHERE status = 'discovered' LIMIT ?",
            (batch_size,),
        ).fetchall()
        if not batch:
            break

        updates: list[tuple[str, int]] = []
        failed_ids: list[int] = []
        for row in batch:
            path_str = row["file_path"]
            try:
                p = Path(path_str)
                if p.is_file():
                    digest = _md5_file(p)
                else:
                    digest = _md5_bytes(source.get_bytes(path_str))
                updates.append((digest, row["id"]))
            except Exception as exc:
                result.files_skipped += 1
                result.skipped_reasons.append((path_str, f"MD5 read failed: {exc}"))
                failed_ids.append(row["id"])
                continue

        if updates:
            conn.executemany(
                """
                UPDATE images
                SET md5 = ?,
                    status = 'hashed',
                    updated_at = strftime('%Y-%m-%dT%H:%M:%SZ','now')
                WHERE id = ?
                """,
                updates,
            )
            conn.commit()
            result.files_hashed += len(updates)

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

    # Step 3: Flag RAW+JPEG sibling pairs
    result.raw_pairs_marked = _detect_raw_pairs(conn, raw_exts)

    # Step 4: Build exact-duplicate clusters (MD5 + file_size match)
    clusters, clustered = _build_exact_clusters(conn)
    result.clusters_created = clusters
    result.files_clustered = clustered

    return result
