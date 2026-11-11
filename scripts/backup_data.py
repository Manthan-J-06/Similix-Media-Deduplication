"""
scripts/backup_data.py — Safe online backup for SIMILIX /data directory.

Snapshots every SQLite database using sqlite3's online backup API (safe under
active WAL writes) alongside user libraries, thumbnails, and job states into a
timestamped .tar.gz archive, pruning old backups by age.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import sqlite3
import tarfile
import tempfile
import time


def _backup_sqlite_db(src_db: Path, dst_db: Path) -> None:
    """Create a transactionally consistent copy of a live SQLite database."""
    dst_db.parent.mkdir(parents=True, exist_ok=True)
    src_conn = sqlite3.connect(str(src_db), timeout=60.0)
    try:
        dst_conn = sqlite3.connect(str(dst_db))
        try:
            src_conn.backup(dst_conn)
        finally:
            dst_conn.close()
    finally:
        src_conn.close()


def create_backup(
    data_dir: Path,
    backup_dir: Path,
    retain_days: int = 30,
) -> Path:
    """Create a safe online backup archive of data_dir and return archive path."""
    data_dir = Path(data_dir).resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    backup_dir = Path(backup_dir).resolve()
    backup_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    archive_path = backup_dir / f"similix_data_{timestamp}.tar.gz"
    tmp_archive = backup_dir / f".similix_data_{timestamp}.tar.gz.tmp"

    with tempfile.TemporaryDirectory(prefix="similix_bkp_") as stage_tmp:
        stage_dir = Path(stage_tmp)

        for root, dirnames, filenames in os.walk(data_dir):
            rel_root = Path(root).relative_to(data_dir)
            # Skip ephemeral temporary upload staging folders
            dirnames[:] = [d for d in dirnames if d not in ("tmp", "purge")]

            for fname in filenames:
                src_file = Path(root) / fname
                if src_file.is_symlink():
                    continue
                if fname.endswith(("-wal", "-shm", "-journal")):
                    continue

                rel_file = rel_root / fname
                dst_file = stage_dir / rel_file
                dst_file.parent.mkdir(parents=True, exist_ok=True)

                if fname.endswith(".db") and src_file.stat().st_size > 0:
                    _backup_sqlite_db(src_file, dst_file)
                else:
                    shutil.copy2(src_file, dst_file)

        with tarfile.open(tmp_archive, "w:gz") as tar:
            tar.add(stage_dir, arcname=".")

    tmp_archive.replace(archive_path)

    if retain_days > 0:
        cutoff = time.time() - (retain_days * 86400)
        for old_bkp in backup_dir.glob("similix_data_*.tar.gz"):
            try:
                if old_bkp != archive_path and old_bkp.stat().st_mtime < cutoff:
                    old_bkp.unlink()
            except OSError:
                pass

    return archive_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Safe online backup for SIMILIX /data directory")
    parser.add_argument("--data-dir", default=os.environ.get("SIMILIX_DATA_DIR", "/data"))
    parser.add_argument("--backup-dir", default="/opt/backups/similix")
    parser.add_argument("--retain-days", type=int, default=30)
    args = parser.parse_args()

    out = create_backup(Path(args.data_dir), Path(args.backup_dir), retain_days=args.retain_days)
    print(f"Backup created successfully: {out}")


if __name__ == "__main__":
    main()
