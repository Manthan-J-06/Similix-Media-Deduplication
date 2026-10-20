"""
tests/test_db.py — Tests for database initialization, schema migrations, and backward compatibility.
"""

from __future__ import annotations

from pathlib import Path
import shutil
import sqlite3
import pytest
from fastapi.testclient import TestClient

from api.server import create_app
from db.connection import get_connection, init_db


class TestDatabaseMigrations:
    def test_init_db_fresh_database(self, tmp_path: Path):
        """init_db on a fresh path creates all tables and indexes."""
        db_path = tmp_path / "fresh.db"
        conn = init_db(db_path)
        tables = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "images" in tables
        assert "clusters" in tables
        assert "drive_bandwidth" in tables
        assert "deletion_journal" in tables

        indexes = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            ).fetchall()
        }
        assert "idx_images_drive_id" in indexes
        assert "idx_images_source" in indexes
        assert "idx_images_category" in indexes
        assert "idx_clusters_category" in indexes
        conn.close()

    def test_old_schema_migration_and_viewer_api_list_clusters(self, tmp_path: Path):
        """
        Builds a DB using the oldest schema (copying real ~/dedupe_test/demo_backup2/demo.db)
        and opens it with init_db without error, and the viewer API can list clusters from it.
        """
        real_backup = Path.home() / "dedupe_test/demo_backup2/demo.db"
        test_db = tmp_path / "old_demo.db"

        if real_backup.exists():
            shutil.copyfile(real_backup, test_db)
            # Verify it has the older schema before init_db
            raw_conn = sqlite3.connect(test_db)
            cols_before = {r[1] for r in raw_conn.execute("PRAGMA table_info(images)").fetchall()}
            raw_conn.close()
            assert "drive_id" not in cols_before
            assert "source" not in cols_before
            assert "quality_basis" not in cols_before
        else:
            # Build oldest schema from scratch with clusters and images
            raw_conn = sqlite3.connect(test_db)
            raw_conn.executescript(
                """
                CREATE TABLE images (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_path     TEXT NOT NULL UNIQUE,
                    file_size     INTEGER,
                    modified_time REAL,
                    width         INTEGER,
                    height        INTEGER,
                    format        TEXT,
                    md5           TEXT,
                    phash         TEXT,
                    clip_vector   BLOB,
                    status        TEXT NOT NULL DEFAULT 'discovered',
                    cluster_id    INTEGER,
                    is_keeper     INTEGER NOT NULL DEFAULT 0,
                    created_at    TEXT NOT NULL,
                    updated_at    TEXT NOT NULL
                );
                CREATE TABLE clusters (
                    id                INTEGER PRIMARY KEY AUTOINCREMENT,
                    representative_id INTEGER,
                    stage_detected    INTEGER NOT NULL,
                    member_count      INTEGER NOT NULL DEFAULT 0,
                    created_at        TEXT NOT NULL
                );
                INSERT INTO clusters (id, representative_id, stage_detected, member_count, created_at)
                VALUES (1, NULL, 0, 2, '2026-01-01T00:00:00Z');
                INSERT INTO images (id, file_path, file_size, modified_time, width, height, format, md5, status, cluster_id, is_keeper, created_at, updated_at)
                VALUES (1, '/path/img1.jpg', 1000, 1.0, 100, 100, 'JPEG', 'md5_1', 'clustered', 1, 1, '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z');
                INSERT INTO images (id, file_path, file_size, modified_time, width, height, format, md5, status, cluster_id, is_keeper, created_at, updated_at)
                VALUES (2, '/path/img2.jpg', 1000, 1.0, 100, 100, 'JPEG', 'md5_1', 'clustered', 1, 0, '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z');
                """
            )
            raw_conn.commit()
            raw_conn.close()

        # Opening with init_db must succeed without error
        conn = init_db(test_db)

        # Verify columns were migrated
        cols_after = {r[1] for r in conn.execute("PRAGMA table_info(images)").fetchall()}
        assert "drive_id" in cols_after
        assert "source" in cols_after
        assert "quality_basis" in cols_after
        assert "match_pct" in cols_after
        assert "resolution_pct" in cols_after
        assert "category_top1" in cols_after
        conn.close()

        # Verify viewer API can list clusters from the migrated DB
        app = create_app(db_path=test_db)
        client = TestClient(app)
        resp = client.get("/clusters")
        assert resp.status_code == 200
        clusters = resp.json()
        assert isinstance(clusters, list)
        assert len(clusters) > 0
        assert "id" in clusters[0]
        assert "keeper" in clusters[0]
