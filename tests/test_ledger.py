"""
tests/test_ledger.py — Unit tests for core/ledger.py safe deletion & trash system.

Tests:
1. plan does not touch files on disk
2. execute moves files to trash with relative path preserved
3. undo restores files byte-identically
4. a modified file is refused at execute time
5. the keeper is never moved
6. a whole cluster can never be moved (at least one copy survives)
7. review_flag / low_confidence / unalignable items are refused without manual choice
8. undo conflict handled (target already exists, refuses to overwrite)
9. purge refuses recent batch (< 30 days) and succeeds on older batch
10. undo_file restores single file safely
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import sqlite3
import time

import pytest
from PIL import Image

from core.ledger import (
    compute_file_md5,
    dry_run,
    ensure_ledger_tables,
    execute,
    get_history,
    plan,
    purge,
    undo,
    undo_file,
)
from db.connection import init_db


def _create_test_image(path: Path, color: tuple[int, int, int] = (100, 100, 100), size: tuple[int, int] = (200, 200)) -> None:
    im = Image.new("RGB", size, color=color)
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path, format="JPEG", quality=90)


@pytest.fixture
def test_setup(tmp_path: Path):
    """Setup a test folder, db, and trash directory in tmp_path."""
    db_file = tmp_path / "test_argus.db"
    conn = init_db(db_file)
    ensure_ledger_tables(conn)
    trash_dir = tmp_path / "trash"
    trash_dir.mkdir(parents=True, exist_ok=True)

    img_dir = tmp_path / "photos"
    img_dir.mkdir(parents=True, exist_ok=True)

    # Cluster 1: 1 keeper original, 1 delete_safe exact copy, 1 lower-res copy
    p_orig = img_dir / "orig.jpg"
    p_copy = img_dir / "sub" / "orig_copy.jpg"
    p_low = img_dir / "orig_low.jpg"

    _create_test_image(p_orig, color=(200, 50, 50), size=(800, 600))
    # Write identical bytes for p_copy
    p_copy.parent.mkdir(parents=True, exist_ok=True)
    p_copy.write_bytes(p_orig.read_bytes())
    _create_test_image(p_low, color=(200, 50, 50), size=(400, 300))

    md5_orig = compute_file_md5(p_orig)
    md5_low = compute_file_md5(p_low)

    now = time.time()
    conn.execute("INSERT INTO clusters (id, representative_id, stage_detected, member_count) VALUES (1, NULL, 0, 3)")

    conn.execute(
        """
        INSERT INTO images (
            id, cluster_id, file_path, file_size, width, height, format, md5,
            is_keeper, is_recommended_keeper, delete_safe, review_flag,
            is_low_confidence, quality_status, ssim, ssim_color, sharpness_ratio,
            pixel_ratio, bytes_ratio, modified_time
        ) VALUES
          (1, 1, ?, ?, 800, 600, 'JPEG', ?, 1, 1, 0, 0, 0, 'reference', 1.0, 1.0, 1.0, 1.0, 1.0, ?),
          (2, 1, ?, ?, 800, 600, 'JPEG', ?, 0, 0, 1, 0, 0, 'aligned', 1.0, 1.0, 1.0, 1.0, 1.0, ?),
          (3, 1, ?, ?, 400, 300, 'JPEG', ?, 0, 0, 1, 0, 0, 'aligned', 0.96, 0.96, 1.0, 0.25, 0.25, ?)
        """,
        (
            str(p_orig), p_orig.stat().st_size, md5_orig, now - 100,
            str(p_copy), p_copy.stat().st_size, md5_orig, now - 50,
            str(p_low), p_low.stat().st_size, md5_low, now - 10,
        ),
    )
    conn.execute("UPDATE clusters SET representative_id = 1 WHERE id = 1")
    conn.commit()

    return {
        "conn": conn,
        "tmp_path": tmp_path,
        "trash_dir": trash_dir,
        "p_orig": p_orig,
        "p_copy": p_copy,
        "p_low": p_low,
        "md5_orig": md5_orig,
    }


class TestLedgerCore:

    def test_plan_does_not_touch_files(self, test_setup):
        """1. plan() takes member IDs or clusters and returns summary; does NOT touch files on disk."""
        conn = test_setup["conn"]
        p_orig = test_setup["p_orig"]
        p_copy = test_setup["p_copy"]
        p_low = test_setup["p_low"]
        trash_dir = test_setup["trash_dir"]

        mtime_orig_before = p_orig.stat().st_mtime
        mtime_copy_before = p_copy.stat().st_mtime

        summary = plan(conn, mode="conservative", trash_dir=trash_dir)

        assert summary.batch_id.startswith("batch_")
        assert summary.file_count >= 1
        assert summary.bytes_freed > 0
        assert 1 in summary.per_cluster_keeper
        assert summary.per_cluster_keeper[1]["keeper_id"] == 1

        # Check files on disk are completely untouched
        assert p_orig.exists()
        assert p_copy.exists()
        assert p_low.exists()
        assert p_orig.stat().st_mtime == mtime_orig_before
        assert p_copy.stat().st_mtime == mtime_copy_before

        # Nothing in trash yet
        batch_trash = Path(summary.trash_dir) / summary.batch_id
        assert not batch_trash.exists()

    def test_execute_moves_files(self, test_setup):
        """2. execute(confirm=True) moves delete_safe files to trash folder preserving relative path."""
        conn = test_setup["conn"]
        p_orig = test_setup["p_orig"]
        p_copy = test_setup["p_copy"]
        trash_dir = test_setup["trash_dir"]

        summary = plan(conn, member_ids=[2], mode="conservative", trash_dir=trash_dir)
        batch_id = summary.batch_id

        # Require confirm=True
        with pytest.raises(ValueError, match="confirm=True"):
            execute(conn, batch_id, confirm=False)

        res = execute(conn, batch_id, confirm=True)
        assert res.moved_count == 1
        assert res.refused_count == 0
        assert res.failed_count == 0

        # Original copy moved, keeper left intact
        assert not p_copy.exists()
        assert p_orig.exists()

        # Check trash path exists and preserves relative path hierarchy
        entry = conn.execute("SELECT * FROM deletion_journal WHERE batch_id = ? AND image_id = 2", (batch_id,)).fetchone()
        assert entry["status"] == "moved"
        trash_path = Path(entry["trash_path"])
        assert trash_path.exists()
        assert "sub" in trash_path.parts  # preserved relative folder structure

        # Image status in DB is trashed
        img_st = conn.execute("SELECT status FROM images WHERE id = 2").fetchone()[0]
        assert img_st == "trashed"

    def test_undo_restores_them_byte_identically(self, test_setup):
        """3. undo(batch_id) restores every file byte-identically to its original path."""
        conn = test_setup["conn"]
        p_copy = test_setup["p_copy"]
        md5_orig = test_setup["md5_orig"]
        trash_dir = test_setup["trash_dir"]

        summary = plan(conn, member_ids=[2], mode="conservative", trash_dir=trash_dir)
        batch_id = summary.batch_id
        execute(conn, batch_id, confirm=True)
        assert not p_copy.exists()

        undo_res = undo(conn, batch_id)
        assert undo_res.restored_count == 1
        assert undo_res.conflict_count == 0
        assert str(p_copy) in undo_res.restored_files

        # File restored and matches original MD5 byte-identically
        assert p_copy.exists()
        assert compute_file_md5(p_copy) == md5_orig

        # DB status updated to restored / clustered
        entry = conn.execute("SELECT status FROM deletion_journal WHERE batch_id = ? AND image_id = 2", (batch_id,)).fetchone()
        assert entry["status"] == "restored"
        batch_row = conn.execute("SELECT status FROM ledger_batches WHERE batch_id = ?", (batch_id,)).fetchone()
        assert batch_row["status"] == "undone"

    def test_modified_file_is_refused(self, test_setup):
        """4. A file modified after plan time (size or MD5 mismatch) is strictly refused at execute time."""
        conn = test_setup["conn"]
        p_copy = test_setup["p_copy"]
        trash_dir = test_setup["trash_dir"]

        summary = plan(conn, member_ids=[2], mode="conservative", trash_dir=trash_dir)
        batch_id = summary.batch_id

        # Tamper with file on disk before execute
        p_copy.write_bytes(p_copy.read_bytes() + b"\x00\x00_TAMPERED")

        res = execute(conn, batch_id, confirm=True)
        assert res.moved_count == 0
        assert res.refused_count == 1
        assert "modified" in res.refusals[0]["reason"].lower()

        # File was left on disk untouched
        assert p_copy.exists()
        entry = conn.execute("SELECT status, reason FROM deletion_journal WHERE batch_id = ? AND image_id = 2", (batch_id,)).fetchone()
        assert entry["status"] == "refused"

    def test_keeper_is_never_moved(self, test_setup):
        """5. Cluster's keeper is never planned, and cannot be moved even if forcefully queued."""
        conn = test_setup["conn"]
        p_orig = test_setup["p_orig"]
        trash_dir = test_setup["trash_dir"]

        # Attempt to plan keeper (id 1)
        summary = plan(conn, member_ids=[1], mode="conservative", trash_dir=trash_dir)
        # Keeper must be excluded from planned entries
        assert summary.file_count == 0
        assert len(summary.entries) == 0

        # If an entry somehow has keeper_id == image_id in planned journal:
        batch_id = "batch_manual_hack"
        conn.execute(
            """
            INSERT INTO ledger_batches (batch_id, created_at, mode, status, file_count, bytes_freed, trash_dir)
            VALUES (?, strftime('%Y-%m-%dT%H:%M:%SZ','now'), 'conservative', 'planned', 1, 100, ?)
            """,
            (batch_id, str(trash_dir)),
        )
        conn.execute(
            """
            INSERT INTO deletion_journal (
                batch_id, original_path, trash_path, md5, size, cluster_id, keeper_id, mode, status, is_manual, image_id
            ) VALUES (?, ?, ?, ?, ?, 1, 1, 'conservative', 'planned', 0, 1)
            """,
            (batch_id, str(p_orig), str(trash_dir / "hack" / "orig.jpg"), test_setup["md5_orig"], p_orig.stat().st_size),
        )
        conn.commit()

        res = execute(conn, batch_id, confirm=True)
        assert res.moved_count == 0
        assert res.refused_count == 1
        assert "keeper" in res.refusals[0]["reason"].lower()
        assert p_orig.exists()

    def test_whole_cluster_can_never_be_moved(self, test_setup):
        """6. Refuse to delete every member of any cluster (at least one copy must survive)."""
        conn = test_setup["conn"]
        trash_dir = test_setup["trash_dir"]
        tmp_path = test_setup["tmp_path"]

        # Create isolated 2-member cluster 99
        p1 = tmp_path / "c99_a.jpg"
        p2 = tmp_path / "c99_b.jpg"
        _create_test_image(p1, color=(1, 2, 3))
        _create_test_image(p2, color=(1, 2, 3))

        conn.execute("INSERT INTO clusters (id, representative_id, stage_detected, member_count) VALUES (99, NULL, 0, 2)")
        conn.execute(
            """
            INSERT INTO images (id, cluster_id, file_path, file_size, width, height, format, md5, is_keeper, delete_safe)
            VALUES (101, 99, ?, ?, 200, 200, 'JPEG', 'md5_c99', 1, 0),
                   (102, 99, ?, ?, 200, 200, 'JPEG', 'md5_c99', 0, 1)
            """,
            (str(p1), p1.stat().st_size, str(p2), p2.stat().st_size),
        )
        conn.execute("UPDATE clusters SET representative_id = 101 WHERE id = 99")
        conn.commit()

        # If a planned batch attempts to delete both members (101 and 102)
        batch_id = "batch_kill_cluster"
        conn.execute(
            "INSERT INTO ledger_batches (batch_id, status, trash_dir) VALUES (?, 'planned', ?)",
            (batch_id, str(trash_dir)),
        )
        conn.execute(
            """
            INSERT INTO deletion_journal (batch_id, original_path, trash_path, md5, size, cluster_id, keeper_id, status, is_manual, image_id)
            VALUES (?, ?, ?, 'md5_c99', ?, 99, 101, 'planned', 1, 101),
                   (?, ?, ?, 'md5_c99', ?, 99, 101, 'planned', 1, 102)
            """,
            (
                batch_id, str(p1), str(trash_dir / "t1.jpg"), p1.stat().st_size,
                batch_id, str(p2), str(trash_dir / "t2.jpg"), p2.stat().st_size,
            ),
        )
        conn.commit()

        res = execute(conn, batch_id, confirm=True)
        assert res.moved_count == 0
        assert res.refused_count == 2
        assert "whole cluster deletion prohibited" in res.refusals[0]["reason"].lower()
        assert p1.exists()
        assert p2.exists()

    def test_review_flag_items_refused_without_manual(self, test_setup):
        """7. Refuse items with review_flag, is_low_confidence, or unalignable unless is_manual=True."""
        conn = test_setup["conn"]
        trash_dir = test_setup["trash_dir"]
        tmp_path = test_setup["tmp_path"]

        p_rev = tmp_path / "review_item.jpg"
        _create_test_image(p_rev, color=(50, 50, 50))
        md5_rev = compute_file_md5(p_rev)

        conn.execute("INSERT INTO clusters (id, representative_id, stage_detected, member_count) VALUES (50, NULL, 1, 2)")
        conn.execute(
            """
            INSERT INTO images (id, cluster_id, file_path, file_size, width, height, format, md5, is_keeper, delete_safe, review_flag, is_low_confidence, quality_status)
            VALUES (501, 50, '/k.jpg', 100, 200, 200, 'JPEG', 'k_md5', 1, 0, 0, 0, 'reference'),
                   (502, 50, ?, ?, 200, 200, 'JPEG', ?, 0, 0, 1, 0, 'aligned')
            """,
            (str(p_rev), p_rev.stat().st_size, md5_rev),
        )
        conn.execute("UPDATE clusters SET representative_id = 501 WHERE id = 50")
        conn.commit()

        # Batch 1: Automatic plan -> refused
        b_auto = "batch_auto_review"
        conn.execute("INSERT INTO ledger_batches (batch_id, status, trash_dir) VALUES (?, 'planned', ?)", (b_auto, str(trash_dir)))
        conn.execute(
            """
            INSERT INTO deletion_journal (batch_id, original_path, trash_path, md5, size, cluster_id, keeper_id, status, is_manual, image_id)
            VALUES (?, ?, ?, ?, ?, 50, 501, 'planned', 0, 502)
            """,
            (b_auto, str(p_rev), str(trash_dir / "r.jpg"), md5_rev, p_rev.stat().st_size),
        )
        conn.commit()

        res_auto = execute(conn, b_auto, confirm=True)
        assert res_auto.moved_count == 0
        assert res_auto.refused_count == 1
        assert "review" in res_auto.refusals[0]["reason"].lower()
        assert p_rev.exists()

        # Batch 2: Manual user choice (is_manual = 1) -> accepted
        b_man = "batch_manual_review"
        conn.execute("INSERT INTO ledger_batches (batch_id, status, trash_dir) VALUES (?, 'planned', ?)", (b_man, str(trash_dir)))
        conn.execute(
            """
            INSERT INTO deletion_journal (batch_id, original_path, trash_path, md5, size, cluster_id, keeper_id, status, is_manual, image_id)
            VALUES (?, ?, ?, ?, ?, 50, 501, 'planned', 1, 502)
            """,
            (b_man, str(p_rev), str(trash_dir / "r_man.jpg"), md5_rev, p_rev.stat().st_size),
        )
        conn.commit()

        res_man = execute(conn, b_man, confirm=True)
        assert res_man.moved_count == 1
        assert not p_rev.exists()

    def test_undo_conflict_handled(self, test_setup):
        """8. undo() refuses to overwrite an existing file and reports conflict."""
        conn = test_setup["conn"]
        p_copy = test_setup["p_copy"]
        trash_dir = test_setup["trash_dir"]

        summary = plan(conn, member_ids=[2], mode="conservative", trash_dir=trash_dir)
        batch_id = summary.batch_id
        execute(conn, batch_id, confirm=True)
        assert not p_copy.exists()

        # Create a new file at original_path before undo
        p_copy.write_bytes(b"NEW_CONFLICTING_FILE")

        undo_res = undo(conn, batch_id)
        assert undo_res.restored_count == 0
        assert undo_res.conflict_count == 1
        assert "refusing to overwrite" in undo_res.conflicts[0][1].lower()

        # Conflicting file remains untouched
        assert p_copy.read_bytes() == b"NEW_CONFLICTING_FILE"

    def test_purge_refuses_recent_batch_and_purges_old(self, test_setup):
        """9. purge() permanently removes trashed files only with confirm=True and only for batches older than 30 days."""
        conn = test_setup["conn"]
        trash_dir = test_setup["trash_dir"]

        summary = plan(conn, member_ids=[2], mode="conservative", trash_dir=trash_dir)
        batch_id = summary.batch_id
        execute(conn, batch_id, confirm=True)

        entry = conn.execute("SELECT trash_path FROM deletion_journal WHERE batch_id = ?", (batch_id,)).fetchone()
        trash_file = Path(entry["trash_path"])
        assert trash_file.exists()

        # 1. Require confirm=True
        with pytest.raises(ValueError, match="confirm=True"):
            purge(conn, batch_id, confirm=False)

        # 2. Refuse recent batch (< 30 days)
        with pytest.raises(ValueError, match="purge requires batches older than 30 days"):
            purge(conn, batch_id, confirm=True, max_age_days=30)
        assert trash_file.exists()

        # 3. Purge succeeds when age meets threshold (e.g. max_age_days=0)
        p_res = purge(conn, batch_id, confirm=True, max_age_days=0)
        assert p_res.purged_count == 1
        assert not trash_file.exists()
        batch_status = conn.execute("SELECT status FROM ledger_batches WHERE batch_id = ?", (batch_id,)).fetchone()[0]
        assert batch_status == "purged"

    def test_undo_file(self, test_setup):
        """10. undo_file() safely restores an individual file."""
        conn = test_setup["conn"]
        p_copy = test_setup["p_copy"]
        trash_dir = test_setup["trash_dir"]

        summary = plan(conn, member_ids=[2], mode="conservative", trash_dir=trash_dir)
        batch_id = summary.batch_id
        execute(conn, batch_id, confirm=True)
        assert not p_copy.exists()

        ok, msg = undo_file(conn, original_path=p_copy)
        assert ok is True
        assert p_copy.exists()
        assert "Successfully restored" in msg

    def test_dry_run_checks_everything(self, test_setup):
        """dry_run simulates execution and reports ready vs refused counts."""
        conn = test_setup["conn"]
        trash_dir = test_setup["trash_dir"]

        summary = plan(conn, member_ids=[2], mode="conservative", trash_dir=trash_dir)
        d_res = dry_run(conn, summary.batch_id)
        assert d_res.total_planned == 1
        assert d_res.ready_count == 1
        assert d_res.refused_count == 0

    def test_history(self, test_setup):
        """get_history returns list of recorded batches."""
        conn = test_setup["conn"]
        trash_dir = test_setup["trash_dir"]

        summary = plan(conn, member_ids=[2], mode="conservative", trash_dir=trash_dir)
        hist = get_history(conn)
        assert len(hist) >= 1
        assert hist[0]["batch_id"] == summary.batch_id

    def test_drive_files_excluded_from_plan_before_confirm(self, test_setup):
        """Google Drive files are read-only and must be excluded in plan() under skipped_entries."""
        conn = test_setup["conn"]
        trash_dir = test_setup["trash_dir"]

        conn.execute("INSERT INTO clusters (id, representative_id, stage_detected, member_count) VALUES (99, NULL, 0, 2)")
        now = time.time()
        conn.execute(
            """
            INSERT INTO images (
                id, cluster_id, file_path, file_size, width, height, format, md5,
                is_keeper, is_recommended_keeper, delete_safe, review_flag,
                is_low_confidence, quality_status, ssim, ssim_color, sharpness_ratio,
                pixel_ratio, bytes_ratio, modified_time
            ) VALUES
              (901, 99, 'drive://root/oxford_keeper.jpg', 50000, 800, 600, 'JPEG', 'md5_1', 1, 1, 0, 0, 0, 'reference', 1.0, 1.0, 1.0, 1.0, 1.0, ?),
              (902, 99, 'drive://root/oxford_copy.jpg', 48000, 800, 600, 'JPEG', 'md5_1', 0, 0, 1, 0, 0, 'aligned', 1.0, 1.0, 1.0, 1.0, 1.0, ?)
            """,
            (now, now),
        )
        conn.execute("UPDATE clusters SET representative_id = 901 WHERE id = 99")
        conn.commit()

        summary = plan(conn, member_ids=[902], mode="conservative", trash_dir=trash_dir)
        # Should NOT plan any files into deletion journal
        assert summary.file_count == 0
        assert summary.bytes_freed == 0
        # Should be recorded in skipped_entries with read-only explanation
        drive_skips = [s for s in summary.skipped_entries if s["image_id"] == 902]
        assert len(drive_skips) == 1
        assert "read-only cloud source" in drive_skips[0]["reason"]

    def test_drive_files_refused_with_drive_reason_in_execute(self, test_setup):
        """If a drive:// entry exists in a deletion batch, execute() and dry_run() refuse with Drive reason."""
        conn = test_setup["conn"]
        trash_dir = test_setup["trash_dir"]
        batch_id = "batch_test_drive_refusal"
        conn.execute(
            """
            INSERT INTO ledger_batches (batch_id, file_count, bytes_freed, status, mode, trash_dir)
            VALUES (?, 1, 1000, 'planned', 'conservative', ?)
            """,
            (batch_id, str(trash_dir)),
        )
        conn.execute(
            """
            INSERT INTO deletion_journal (batch_id, original_path, trash_path, md5, size, cluster_id, keeper_id, status, is_manual, image_id)
            VALUES (?, 'drive://root/sample.jpg', 'trash/sample.jpg', 'dummy_md5', 1000, 1, 1, 'planned', 0, 2)
            """,
            (batch_id,),
        )
        conn.commit()

        d_res = dry_run(conn, batch_id)
        assert d_res.refused_count == 1
        assert "read-only cloud source" in d_res.refusals[0]["reason"]

        exec_res = execute(conn, batch_id, confirm=True)
        assert exec_res.refused_count == 1
        assert exec_res.moved_count == 0
        assert "read-only cloud source" in exec_res.refusals[0]["reason"]


class TestLedgerCLI:
    def test_cli_lifecycle_plan_apply_undo_history(self, test_setup, monkeypatch, capsys):
        """Test full CLI lifecycle: plan -> apply without confirm (fails) -> apply with confirm -> history -> undo."""
        from core.cli import main
        tmp_path = test_setup["tmp_path"]
        db_file = tmp_path / "test_argus.db"
        trash_dir = test_setup["trash_dir"]
        p_copy = test_setup["p_copy"]

        # 1. argus plan
        main(["plan", "--db", str(db_file), "--mode", "conservative", "--trash-dir", str(trash_dir)])
        captured = capsys.readouterr().out
        assert "Batch ID" in captured
        assert "Files planned for trash" in captured

        # Extract batch ID from output
        batch_id = None
        for line in captured.splitlines():
            if "Batch ID" in line:
                batch_id = line.split(":")[-1].strip()
        assert batch_id is not None
        assert p_copy.exists()  # Plan touches nothing

        # 2. argus apply without --confirm must exit with code 1
        with pytest.raises(SystemExit) as exc_info:
            main(["apply", batch_id, "--db", str(db_file)])
        assert exc_info.value.code == 1
        err_out = capsys.readouterr().err
        assert "--confirm" in err_out
        assert p_copy.exists()

        # 3. argus apply with --confirm
        main(["apply", batch_id, "--confirm", "--db", str(db_file)])
        apply_out = capsys.readouterr().out
        assert "Successfully moved: 2" in apply_out
        assert not p_copy.exists()

        # 4. argus history
        main(["history", "--db", str(db_file)])
        hist_out = capsys.readouterr().out
        assert batch_id in hist_out
        assert "executed" in hist_out

        # 5. argus undo
        main(["undo", batch_id, "--db", str(db_file)])
        undo_out = capsys.readouterr().out
        assert "Files restored   : 2" in undo_out
        assert p_copy.exists()

    def test_cli_unknown_batch_id_error_handling(self, test_setup, capsys):
        """execute, undo, and purge print one-line error and exit non-zero for unknown batch id."""
        from core.cli import main
        tmp_path = test_setup["tmp_path"]
        db_file = tmp_path / "test_argus.db"
        fake_id = "batch_nonexistent_12345"

        # 1. apply
        with pytest.raises(SystemExit) as exc:
            main(["apply", fake_id, "--confirm", "--db", str(db_file)])
        assert exc.value.code != 0
        err = capsys.readouterr().err
        assert f"Batch '{fake_id}' not found; run `similix history`" in err

        # 2. apply --dry-run
        with pytest.raises(SystemExit) as exc:
            main(["apply", fake_id, "--dry-run", "--db", str(db_file)])
        assert exc.value.code != 0
        err = capsys.readouterr().err
        assert f"Batch '{fake_id}' not found; run `similix history`" in err

        # 3. undo
        with pytest.raises(SystemExit) as exc:
            main(["undo", fake_id, "--db", str(db_file)])
        assert exc.value.code != 0
        err = capsys.readouterr().err
        assert f"Batch '{fake_id}' not found; run `similix history`" in err

        # 4. purge
        with pytest.raises(SystemExit) as exc:
            main(["purge", fake_id, "--confirm", "--db", str(db_file)])
        assert exc.value.code != 0
        err = capsys.readouterr().err
        assert f"Batch '{fake_id}' not found; run `similix history`" in err
