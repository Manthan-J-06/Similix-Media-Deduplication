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
    conn = sqlite3.connect(str(db_path))
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

