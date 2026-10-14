"""
core/stage0.py — Stage 0: File scan, MD5 hashing, RAW-pair detection,
                           exact-duplicate clustering.

Pipeline position: first and cheapest stage.
Input:  a root directory path + an open SQLite connection + config dict.
Output: Stage0Result (stats) + side-effects in the DB.

Steps
-----
1. Walk the directory tree; insert every image file as 'pending' (idempotent).
2. For each 'pending' row (in batches): record file_size + MD5, mark 'stage0_done'.
3. Detect RAW+JPEG sibling pairs (same stem, same directory); mark is_raw_pair=1.
4. Group non-RAW-pair files by (file_size, md5); any group with >1 member is an
   exact-duplicate cluster. Create a cluster row and assign all members to it.
5. Within each cluster pick the keeper: shortest absolute path (deterministic
   tiebreak; quality-aware selection happens at Stage 4).

Resumability
------------
Step 1 uses INSERT OR IGNORE, so re-running after a crash just skips already-
inserted files. Step 2 only queries rows WHERE status = 'pending', so completed
rows are never re-hashed.
"""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# All extensions this stage will ingest (lowercased, with dot).
# RAW extensions are also listed here so they appear in the DB;
# they are excluded from deduplication via the is_raw_pair flag.
_IMAGE_EXTS: frozenset[str] = frozenset({
    ".jpg", ".jpeg", ".png", ".webp", ".gif",
    ".bmp", ".tiff", ".tif", ".heic", ".heif",
    # RAW formats (kept in sync with config.toml [raw] extensions)
    ".arw", ".cr2", ".cr3", ".nef", ".orf",
    ".rw2", ".dng", ".raf", ".pef",
})

_MD5_CHUNK = 65_536  # 64 KB read chunks for MD5 hashing


# ---------------------------------------------------------------------------
# Public result type
# ---------------------------------------------------------------------------

@dataclass
class Stage0Result:
    files_discovered: int = 0        # total files inserted (new this run)
    files_hashed: int = 0            # files whose MD5 was computed this run
    raw_pairs_marked: int = 0        # files flagged as RAW+JPEG siblings
    clusters_created: int = 0        # new exact-duplicate clusters
    files_clustered: int = 0         # total files assigned to clusters


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _image_extensions(config: dict) -> frozenset[str]:
    """
    Merge the hardcoded extension list with any extras from config.
    Ensures all extensions are lowercase and start with a dot.
    """
    raw_exts = {
        f".{e.lower().lstrip('.')}"
        for e in config.get("raw", {}).get("extensions", [])
    }
    return _IMAGE_EXTS | raw_exts


def _iter_image_files(root: Path, extensions: frozenset[str]) -> Iterator[Path]:
    """Recursively yield image files under *root* matching *extensions*."""
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in extensions:
            yield path


def _md5(path: Path) -> str:
    """Return the hex MD5 digest of a file's contents."""
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_MD5_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# DB operations
# ---------------------------------------------------------------------------

def _insert_files_pending(
    conn: sqlite3.Connection,
    paths: list[Path],
) -> int:
    """
    Bulk-insert image paths as 'pending'. Existing paths are silently ignored
    (INSERT OR IGNORE), making this step safe to re-run after a crash.

    Returns the number of newly inserted rows.
    """
    rows = [(str(p),) for p in paths]
    cur = conn.executemany(
        "INSERT OR IGNORE INTO images (file_path) VALUES (?)", rows
    )
    conn.commit()
    return cur.rowcount  # -1 for executemany on some drivers; see note below


def _count_pending(conn: sqlite3.Connection) -> int:
    row = conn.execute(
        "SELECT COUNT(*) FROM images WHERE status = 'pending'"
    ).fetchone()
    return row[0]


def _fetch_pending_batch(
    conn: sqlite3.Connection, batch_size: int
) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT id, file_path FROM images WHERE status = 'pending' LIMIT ?",
        (batch_size,),
    ).fetchall()


def _apply_md5_batch(
    conn: sqlite3.Connection,
    updates: list[tuple[int, int, str]],  # (id, file_size, md5)
) -> None:
    """Write file_size + md5 and advance status to 'stage0_done'."""
    conn.executemany(
        """
        UPDATE images
        SET file_size = ?,
            md5       = ?,
            status    = 'stage0_done',
            updated_at = strftime('%Y-%m-%dT%H:%M:%SZ','now')
        WHERE id = ?
        """,
        [(size, md5, img_id) for (img_id, size, md5) in updates],
    )
    conn.commit()


# ---------------------------------------------------------------------------
# RAW-pair detection
# ---------------------------------------------------------------------------

def _raw_extensions(config: dict) -> frozenset[str]:
    return frozenset(
        f".{e.lower().lstrip('.')}"
        for e in config.get("raw", {}).get("extensions", [])
    )


def _detect_raw_pairs(conn: sqlite3.Connection, raw_exts: frozenset[str]) -> int:
    """
    Mark files as RAW+JPEG siblings when the same directory contains both a
    RAW file and a non-RAW file with an identical stem (case-insensitive).

    Example: /photos/IMG_001.ARW  +  /photos/IMG_001.JPG → both flagged.

    Returns the total number of files newly flagged.
    """
    # Load all stage0_done paths into memory (stems only; avoids N+1 queries).
    rows = conn.execute(
        "SELECT id, file_path FROM images WHERE status = 'stage0_done'"
    ).fetchall()

    # Build a map: (parent_dir, lowercase_stem) → list of (id, ext)
    from collections import defaultdict
    stem_map: dict[tuple[str, str], list[tuple[int, str]]] = defaultdict(list)
    for row in rows:
        p = Path(row["file_path"])
        key = (str(p.parent), p.stem.lower())
        stem_map[key].append((row["id"], p.suffix.lower()))

    # Find groups that contain at least one RAW and at least one non-RAW member.
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


# ---------------------------------------------------------------------------
# Exact-duplicate clustering
# ---------------------------------------------------------------------------

def _build_exact_clusters(conn: sqlite3.Connection) -> tuple[int, int]:
    """
    Find all (file_size, md5) groups with >1 non-RAW-pair member and create a
    cluster row for each. Assigns cluster_id + marks the keeper.

    Keeper rule (Stage 0): shortest absolute file_path (deterministic tiebreak).
    Final quality-aware keeper selection happens at Stage 4.

    Returns (clusters_created, files_clustered).
    """
    # Find duplicate groups — only among files not yet assigned to a cluster.
    groups = conn.execute(
        """
        SELECT file_size, md5, COUNT(*) AS cnt
        FROM images
        WHERE status     = 'stage0_done'
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
            WHERE file_size = ? AND md5 = ? AND is_raw_pair = 0
            ORDER BY LENGTH(file_path), file_path   -- deterministic keeper
            """,
            (group["file_size"], group["md5"]),
        ).fetchall()

        if len(members) < 2:
            continue

        keeper_id = members[0]["id"]

        # Create the cluster; representative = the keeper.
        cur = conn.execute(
            """
            INSERT INTO clusters (representative_id, stage_detected, member_count)
            VALUES (?, 0, ?)
            """,
            (keeper_id, len(members)),
        )
        cluster_id = cur.lastrowid

        # Assign all members to the cluster.
        conn.executemany(
            """
            UPDATE images
            SET cluster_id = ?,
                is_keeper  = CASE WHEN id = ? THEN 1 ELSE 0 END,
                updated_at = strftime('%Y-%m-%dT%H:%M:%SZ','now')
            WHERE id = ?
            """,
            [(cluster_id, keeper_id, row["id"]) for row in members],
        )

        clusters_created += 1
        files_clustered += len(members)

    conn.commit()
    return clusters_created, files_clustered


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_stage0(
    root: Path,
    conn: sqlite3.Connection,
    config: dict,
) -> Stage0Result:
    """
    Execute Stage 0 end-to-end.

    Parameters
    ----------
    root:   Directory to scan for images.
    conn:   Initialised SQLite connection (from db.connection.init_db).
    config: Parsed config dict (from core.config.load_config).

    Returns a Stage0Result with counts for logging/display.
    """
    result = Stage0Result()
    batch_size: int = config.get("general", {}).get("batch_size", 500)
    extensions = _image_extensions(config)
    raw_exts = _raw_extensions(config)

    # ── Step 1: discover all image files and insert as 'pending' ─────────────
    paths = list(_iter_image_files(root, extensions))
    if paths:
        _insert_files_pending(conn, paths)
        # Count how many are actually new (pending) after the insert.
        result.files_discovered = _count_pending(conn)

    # ── Step 2: hash every pending file in batches ────────────────────────────
    while True:
        batch = _fetch_pending_batch(conn, batch_size)
        if not batch:
            break

        updates: list[tuple[int, int, str]] = []
        for row in batch:
            p = Path(row["file_path"])
            try:
                size = p.stat().st_size
                digest = _md5(p)
                updates.append((row["id"], size, digest))
            except OSError:
                # File disappeared or unreadable; leave as 'pending' to retry.
                continue

        if updates:
            _apply_md5_batch(conn, updates)
            result.files_hashed += len(updates)

    # ── Step 3: mark RAW+JPEG sibling pairs ───────────────────────────────────
    result.raw_pairs_marked = _detect_raw_pairs(conn, raw_exts)

    # ── Step 4: build exact-duplicate clusters ────────────────────────────────
    clusters, clustered = _build_exact_clusters(conn)
    result.clusters_created = clusters
    result.files_clustered = clustered

    return result
