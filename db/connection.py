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
    """
    conn = get_connection(db_path)
    conn.executescript(ALL_DDL)
    conn.commit()
    return conn
