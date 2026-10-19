"""
db/connection.py — SQLite connection factory for Argus.

Usage:
    from db.connection import init_db
    conn = init_db("argus.db")
"""

import sqlite3
from pathlib import Path

from db.schema import ALL_DDL


def get_connection(db_path: str | Path) -> sqlite3.Connection:
    """
    Open a SQLite connection with sensible defaults:
    - WAL journal mode  → concurrent reads during writes
    - Foreign key enforcement
    - Row factory       → rows accessible as dicts / by column name
    """
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.execute("PRAGMA synchronous=NORMAL;")  # safe with WAL, faster than FULL
    return conn


def init_db(db_path: str | Path) -> sqlite3.Connection:
    """
    Open a connection and ensure the schema exists.
    Safe to call multiple times (all DDL uses IF NOT EXISTS).
    Applies non-destructive migrations for any new columns.
    """
    conn = get_connection(db_path)
    conn.executescript(ALL_DDL)

    # Safe column migrations for existing databases
    cursor = conn.execute("PRAGMA table_info(images);")
    existing_cols = {row["name"] for row in cursor.fetchall()}
    for col in ("clip_vector_90", "clip_vector_180", "clip_vector_270"):
        if col not in existing_cols:
            conn.execute(f"ALTER TABLE images ADD COLUMN {col} BLOB;")
    if "link_signal" not in existing_cols:
        conn.execute("ALTER TABLE images ADD COLUMN link_signal TEXT;")
    if "is_low_confidence" not in existing_cols:
        conn.execute("ALTER TABLE images ADD COLUMN is_low_confidence INTEGER NOT NULL DEFAULT 0;")
    for col, col_type in (
        ("ssim", "REAL"),
        ("ssim_color", "REAL"),
        ("sharpness_ratio", "REAL"),
        ("pixel_ratio", "REAL"),
        ("bytes_ratio", "REAL"),
        ("jpeg_quality_est", "INTEGER"),
        ("source_likelihood", "REAL"),
        ("quality_status", "TEXT"),
        ("category_top1", "TEXT"),
        ("category_top1_prob", "REAL"),
        ("category_top3", "TEXT"),
        ("match_pct", "REAL"),
        ("resolution_pct", "REAL"),
        ("is_cluster_anchor", "INTEGER NOT NULL DEFAULT 0"),
        ("is_recommended_keeper", "INTEGER NOT NULL DEFAULT 0"),
        ("delete_safe", "INTEGER NOT NULL DEFAULT 0"),
        ("review_flag", "INTEGER NOT NULL DEFAULT 0"),
        ("quality_retained_pct", "REAL"),
        ("bytes_saved_if_deleted", "INTEGER DEFAULT 0"),
        ("reason", "TEXT"),
    ):
        if col not in existing_cols:
            conn.execute(f"ALTER TABLE images ADD COLUMN {col} {col_type};")

    existing_cluster_cols = {
        col[1] for col in conn.execute("PRAGMA table_info(clusters)").fetchall()
    }
    if "reference_uncertain" not in existing_cluster_cols:
        conn.execute("ALTER TABLE clusters ADD COLUMN reference_uncertain INTEGER NOT NULL DEFAULT 0;")
    if "category" not in existing_cluster_cols:
        conn.execute("ALTER TABLE clusters ADD COLUMN category TEXT;")

    # Ensure category indexes exist
    conn.execute("CREATE INDEX IF NOT EXISTS idx_images_category ON images(category_top1);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_clusters_category ON clusters(category);")

    # Backfill link_signal for previously clustered rows if unset
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
    return conn

