"""
core/ledger.py — Safe deletion ledger and trash management.

This is the ONLY module in Argus permitted to remove or move files.

Core Principles:
1. Never permanently delete by default: files are moved to a batch trash folder
   (default: ~/.argus/trash/<batch_id>/), preserving full relative directory hierarchy.
   If a move fails, the file is left in place and failure recorded.
2. Batch lifecycle: plan -> dry_run -> confirm -> execute.
   plan() computes candidate deletions and returns summary.
   Nothing is touched on disk until execute(batch_id, confirm=True).
3. Defense in depth at execute time:
   - Re-verifies file exists on disk.
   - Re-verifies size and MD5 hash match recorded values.
   - Re-verifies file is not the cluster's keeper.
   - Refuses to delete all members of any cluster (at least one copy must survive).
   - Refuses review_flag, is_low_confidence, or quality_status 'unalignable'
     unless the plan explicitly flagged it as a manual user selection.
4. SQLite Journaling:
   - ledger_batches: batch metadata (batch_id, created_at, executed_at, mode, status, file_count, bytes_freed, trash_dir).
   - deletion_journal: per-file journal (batch_id, timestamp, original_path, trash_path, md5, size,
     cluster_id, keeper_id, mode, reason, status, is_manual, image_id).
5. Undo & Rollback:
   - undo(batch_id): Restores all moved files from trash, refusing to overwrite existing files.
   - undo_file(): Restores an individual file entry safely.
6. Purge:
   - purge(batch_id): Permanently removes trashed files only with explicit confirm=True
     and only if the batch is older than configurable days (default 30).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import shutil
import sqlite3
import time
from typing import Any
import uuid

from core.config import load_config
from core.keeper import choose_keeper


def compute_file_md5(p: str | Path) -> str:
    """Compute MD5 hash in 64KB chunks."""
    h = hashlib.md5()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def _parse_timestamp(ts: str) -> datetime:
    """Parse various SQLite / ISO timestamp representations into UTC datetime."""
    ts = ts.strip().replace("Z", "+00:00")
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(ts.split("+")[0], fmt)
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    try:
        return datetime.fromisoformat(ts).replace(tzinfo=timezone.utc)
    except Exception:
        return datetime.now(timezone.utc)


def ensure_ledger_tables(conn: sqlite3.Connection) -> None:
    """Ensure ledger_batches and deletion_journal tables and indexes exist."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS ledger_batches (
            batch_id     TEXT PRIMARY KEY,
            created_at   TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
            executed_at  TEXT,
            mode         TEXT NOT NULL DEFAULT 'balanced',
            status       TEXT NOT NULL DEFAULT 'planned',
            file_count   INTEGER NOT NULL DEFAULT 0,
            bytes_freed  INTEGER NOT NULL DEFAULT 0,
            trash_dir    TEXT NOT NULL
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS deletion_journal (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            batch_id      TEXT NOT NULL REFERENCES ledger_batches(batch_id) ON DELETE CASCADE,
            timestamp     TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
            original_path TEXT NOT NULL,
            trash_path    TEXT,
            md5           TEXT,
            size          INTEGER,
            cluster_id    INTEGER REFERENCES clusters(id) ON DELETE SET NULL,
            keeper_id     INTEGER REFERENCES images(id) ON DELETE SET NULL,
            mode          TEXT,
            reason        TEXT,
            status        TEXT NOT NULL DEFAULT 'planned',
            is_manual     INTEGER NOT NULL DEFAULT 0,
            image_id      INTEGER REFERENCES images(id) ON DELETE SET NULL
        );
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_journal_batch ON deletion_journal(batch_id);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_journal_status ON deletion_journal(status);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_journal_path ON deletion_journal(original_path);")
    conn.commit()


@dataclass
class PlanEntry:
    image_id: int
    cluster_id: int
    keeper_id: int
    original_path: str
    trash_path: str
    file_size: int
    md5: str
    reason: str
    is_manual: bool = False


@dataclass
class PlanSummary:
    batch_id: str
    mode: str
    file_count: int
    bytes_freed: int
    per_cluster_keeper: dict[int, dict[str, Any]]
    entries: list[PlanEntry] = field(default_factory=list)
    trash_dir: str = ""


@dataclass
class DryRunResult:
    batch_id: str
    total_planned: int
    ready_count: int
    refused_count: int
    refusals: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ExecuteResult:
    batch_id: str
    total_planned: int
    moved_count: int
    refused_count: int
    failed_count: int
    bytes_moved: int
    refusals: list[dict[str, Any]] = field(default_factory=list)
    failures: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class UndoResult:
    batch_id: str
    restored_count: int
    conflict_count: int
    conflicts: list[tuple[str, str]] = field(default_factory=list)
    restored_files: list[str] = field(default_factory=list)


@dataclass
class PurgeResult:
    batch_id: str
    purged_count: int
    bytes_purged: int


def _resolve_trash_dir(config: dict | None = None, trash_dir: str | Path | None = None) -> Path:
    """Resolve base trash directory from argument, config, or ~/.argus/trash."""
    if trash_dir is not None:
        return Path(trash_dir).expanduser().resolve()
    if config:
        cfg_trash = config.get("ledger", {}).get("trash_dir")
        if cfg_trash:
            return Path(cfg_trash).expanduser().resolve()
    return (Path.home() / ".argus" / "trash").resolve()


def _get_relative_trash_path(batch_trash_dir: Path, original_path: str | Path) -> Path:
    """Preserve full relative directory hierarchy inside batch trash folder."""
    orig_p = Path(original_path).resolve()
    # Strip filesystem root/drive to build unique relative path
    rel_p = Path(str(orig_p).lstrip(os.sep))
    return batch_trash_dir / rel_p


def plan(
    conn: sqlite3.Connection,
    member_ids: list[int] | None = None,
    cluster_ids: list[int] | None = None,
    mode: str = "balanced",
    manual_member_ids: list[int] | None = None,
    config: dict | None = None,
    trash_dir: str | Path | None = None,
) -> PlanSummary:
    """
    Plan a deletion batch without modifying any files on disk.
    
    Evaluates clusters and determines keeper and delete_safe members according to policy mode.
    Stores planned batch and journal rows in SQLite.
    Returns PlanSummary containing file count, bytes freed, and per-cluster keeper.
    """
    ensure_ledger_tables(conn)
    if config is None:
        try:
            config = load_config()
        except Exception:
            config = {}

    base_trash = _resolve_trash_dir(config, trash_dir)
    batch_id = f"batch_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    batch_trash = base_trash / batch_id

    manual_set = set(manual_member_ids or [])
    requested_member_set = set(member_ids) if member_ids is not None else None

    # Determine clusters to inspect
    if cluster_ids is not None:
        target_cluster_ids = list(cluster_ids)
    elif requested_member_set is not None:
        if not requested_member_set:
            target_cluster_ids = []
        else:
            q_marks = ",".join("?" for _ in requested_member_set)
            rows = conn.execute(
                f"SELECT DISTINCT cluster_id FROM images WHERE id IN ({q_marks}) AND cluster_id IS NOT NULL",
                list(requested_member_set),
            ).fetchall()
            target_cluster_ids = [r[0] for r in rows]
    else:
        rows = conn.execute("SELECT id FROM clusters ORDER BY id").fetchall()
        target_cluster_ids = [r[0] for r in rows]

    per_cluster_keeper: dict[int, dict[str, Any]] = {}
    planned_entries: list[PlanEntry] = []

    for cid in target_cluster_ids:
        c_row = conn.execute(
            "SELECT id, representative_id, reference_uncertain FROM clusters WHERE id = ?",
            (cid,),
        ).fetchone()
        if not c_row:
            continue

        ref_unc = c_row["reference_uncertain"] if "reference_uncertain" in c_row.keys() else 0
        members = conn.execute(
            """
            SELECT id, cluster_id, file_path, width, height, file_size, format, md5,
                   is_keeper, is_low_confidence, ssim, ssim_color, sharpness_ratio,
                   pixel_ratio, bytes_ratio, quality_status, review_flag, delete_safe,
                   reason, source_likelihood, jpeg_quality_est
            FROM images
            WHERE cluster_id = ?
            ORDER BY id
            """,
            (cid,),
        ).fetchall()

        if len(members) < 2:
            continue

        member_dicts = [dict(m) for m in members]
        id_to_member = {m["id"]: m for m in member_dicts}

        keeper_res = choose_keeper(
            member_dicts,
            mode=mode,
            config=config,
            reference_uncertain=ref_unc,
        )

        keeper_id = keeper_res["keeper_id"]
        k_m = id_to_member.get(keeper_id, {})
        per_cluster_keeper[cid] = {
            "keeper_id": keeper_id,
            "keeper_path": k_m.get("file_path", ""),
            "keeper_size": k_m.get("file_size", 0),
        }

        del_safe_ids = set(keeper_res["delete_safe_ids"])
        reasons = keeper_res["reasons"]

        # Determine which members to plan for deletion in this cluster
        for m in member_dicts:
            mid = m["id"]
            if mid == keeper_id:
                # Keeper is never planned for deletion
                continue

            if requested_member_set is not None and mid not in requested_member_set:
                continue

            is_manual = mid in manual_set

            if is_manual:
                entry_reason = f"Manual user selection ({reasons.get(mid, 'Manual choice')})"
            else:
                if mid not in del_safe_ids:
                    # In review or unalignable and not explicitly manual
                    continue
                entry_reason = reasons.get(mid, f"Delete-safe in {mode} mode")

            p = m["file_path"]
            trash_dest = str(_get_relative_trash_path(batch_trash, p))
            planned_entries.append(
                PlanEntry(
                    image_id=mid,
                    cluster_id=cid,
                    keeper_id=keeper_id,
                    original_path=p,
                    trash_path=trash_dest,
                    file_size=m.get("file_size") or 0,
                    md5=m.get("md5") or "",
                    reason=entry_reason,
                    is_manual=is_manual,
                )
            )

    # Invariant: Never plan to delete every member of any cluster
    cluster_member_counts: dict[int, int] = {}
    for entry in planned_entries:
        cluster_member_counts[entry.cluster_id] = cluster_member_counts.get(entry.cluster_id, 0) + 1

    valid_planned_entries: list[PlanEntry] = []
    for entry in planned_entries:
        cid = entry.cluster_id
        total_cluster_members = conn.execute(
            "SELECT COUNT(*) FROM images WHERE cluster_id = ?",
            (cid,),
        ).fetchone()[0]

        if cluster_member_counts[cid] >= total_cluster_members:
            # Whole cluster deletion forbidden
            continue
        valid_planned_entries.append(entry)

    total_bytes_freed = sum(e.file_size for e in valid_planned_entries)
    file_count = len(valid_planned_entries)

    # Persist planned batch and journal records in SQLite
    conn.execute(
        """
        INSERT INTO ledger_batches (batch_id, created_at, mode, status, file_count, bytes_freed, trash_dir)
        VALUES (?, strftime('%Y-%m-%dT%H:%M:%SZ','now'), ?, 'planned', ?, ?, ?)
        """,
        (batch_id, mode, file_count, total_bytes_freed, str(base_trash)),
    )

    for e in valid_planned_entries:
        conn.execute(
            """
            INSERT INTO deletion_journal (
                batch_id, timestamp, original_path, trash_path, md5, size,
                cluster_id, keeper_id, mode, reason, status, is_manual, image_id
            )
            VALUES (?, strftime('%Y-%m-%dT%H:%M:%SZ','now'), ?, ?, ?, ?, ?, ?, ?, ?, 'planned', ?, ?)
            """,
            (
                batch_id,
                e.original_path,
                e.trash_path,
                e.md5,
                e.file_size,
                e.cluster_id,
                e.keeper_id,
                mode,
                e.reason,
                1 if e.is_manual else 0,
                e.image_id,
            ),
        )
    conn.commit()

    return PlanSummary(
        batch_id=batch_id,
        mode=mode,
        file_count=file_count,
        bytes_freed=total_bytes_freed,
        per_cluster_keeper=per_cluster_keeper,
        entries=valid_planned_entries,
        trash_dir=str(base_trash),
    )


def dry_run(conn: sqlite3.Connection, batch_id: str) -> DryRunResult:
    """
    Simulate execution of a planned batch without touching any files on disk.
    Evaluates all execute-time safety checks and returns potential refusals.
    """
    ensure_ledger_tables(conn)
    batch = conn.execute("SELECT * FROM ledger_batches WHERE batch_id = ?", (batch_id,)).fetchone()
    if not batch:
        raise ValueError(f"Batch '{batch_id}' not found in ledger")

    entries = conn.execute(
        "SELECT * FROM deletion_journal WHERE batch_id = ? AND status = 'planned' ORDER BY id",
        (batch_id,),
    ).fetchall()

    refusals: list[dict[str, Any]] = []
    ready_count = 0

    # Group entries by cluster to check whole-cluster rule
    by_cluster: dict[int, list[sqlite3.Row]] = {}
    for e in entries:
        cid = e["cluster_id"]
        by_cluster.setdefault(cid, []).append(e)

    for cid, c_entries in by_cluster.items():
        surviving = conn.execute(
            "SELECT COUNT(*) FROM images WHERE cluster_id = ? AND status != 'trashed'",
            (cid,),
        ).fetchone()[0]

        if len(c_entries) >= surviving:
            for e in c_entries:
                refusals.append({
                    "entry_id": e["id"],
                    "path": e["original_path"],
                    "reason": "Refused: whole cluster deletion prohibited; at least one copy must survive",
                })
            continue

        for e in c_entries:
            orig_p = Path(e["original_path"])
            # 1. Existence check
            if not orig_p.exists():
                refusals.append({
                    "entry_id": e["id"],
                    "path": e["original_path"],
                    "reason": "File does not exist at original path",
                })
                continue

            # 2. Size check
            actual_sz = orig_p.stat().st_size
            if actual_sz != e["size"]:
                refusals.append({
                    "entry_id": e["id"],
                    "path": e["original_path"],
                    "reason": f"File size modified (recorded {e['size']}, actual {actual_sz})",
                })
                continue

            # 3. MD5 check
            actual_md5 = compute_file_md5(orig_p)
            if actual_md5 != e["md5"]:
                refusals.append({
                    "entry_id": e["id"],
                    "path": e["original_path"],
                    "reason": f"File content modified (MD5 mismatch: recorded {e['md5']}, actual {actual_md5})",
                })
                continue

            # 4. Keeper check
            c_row = conn.execute("SELECT representative_id FROM clusters WHERE id = ?", (cid,)).fetchone()
            cur_keeper_id = c_row["representative_id"] if c_row else None
            if e["image_id"] == cur_keeper_id or e["image_id"] == e["keeper_id"]:
                refusals.append({
                    "entry_id": e["id"],
                    "path": e["original_path"],
                    "reason": "File is recommended cluster keeper",
                })
                continue

            # 5. Review / low-confidence / unalignable check
            if not e["is_manual"]:
                img = conn.execute(
                    "SELECT review_flag, is_low_confidence, quality_status FROM images WHERE id = ?",
                    (e["image_id"],),
                ).fetchone()
                if img:
                    if img["review_flag"] == 1:
                        refusals.append({
                            "entry_id": e["id"],
                            "path": e["original_path"],
                            "reason": "Member flagged for review; manual override required",
                        })
                        continue
                    if img["is_low_confidence"] == 1:
                        refusals.append({
                            "entry_id": e["id"],
                            "path": e["original_path"],
                            "reason": "Member is low confidence; manual override required",
                        })
                        continue
                    if img["quality_status"] == "unalignable":
                        refusals.append({
                            "entry_id": e["id"],
                            "path": e["original_path"],
                            "reason": "Member is unalignable; manual override required",
                        })
                        continue

            ready_count += 1

    return DryRunResult(
        batch_id=batch_id,
        total_planned=len(entries),
        ready_count=ready_count,
        refused_count=len(refusals),
        refusals=refusals,
    )


def execute(
    conn: sqlite3.Connection,
    batch_id: str,
    confirm: bool = False,
) -> ExecuteResult:
    """
    Execute a planned deletion batch by moving safe files to trash.
    
    Requires explicit confirm=True flag.
    Performs comprehensive safety checks at execution time:
    - Verifies file existence, size, and MD5 match recorded values.
    - Ensures file is not cluster keeper.
    - Prevents deleting every member of any cluster.
    - Excludes review_flag, is_low_confidence, or unalignable members unless marked manual.
    - If move fails, leaves file in place and records failure.
    """
    if not confirm:
        raise ValueError("Execution requires explicit confirmation (confirm=True)")

    ensure_ledger_tables(conn)
    batch = conn.execute("SELECT * FROM ledger_batches WHERE batch_id = ?", (batch_id,)).fetchone()
    if not batch:
        raise ValueError(f"Batch '{batch_id}' not found in ledger")
    if batch["status"] != "planned":
        raise ValueError(f"Batch '{batch_id}' has status '{batch['status']}'; only 'planned' batches can be executed")

    entries = conn.execute(
        "SELECT * FROM deletion_journal WHERE batch_id = ? AND status = 'planned' ORDER BY id",
        (batch_id,),
    ).fetchall()

    if not entries:
        return ExecuteResult(
            batch_id=batch_id,
            total_planned=0,
            moved_count=0,
            refused_count=0,
            failed_count=0,
            bytes_moved=0,
        )

    refusals: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    moved_count = 0
    bytes_moved = 0

    # Group by cluster for whole-cluster rule
    by_cluster: dict[int, list[sqlite3.Row]] = {}
    for e in entries:
        cid = e["cluster_id"]
        by_cluster.setdefault(cid, []).append(e)

    for cid, c_entries in by_cluster.items():
        surviving = conn.execute(
            "SELECT COUNT(*) FROM images WHERE cluster_id = ? AND status != 'trashed'",
            (cid,),
        ).fetchone()[0]

        if len(c_entries) >= surviving:
            for e in c_entries:
                msg = "Refused: whole cluster deletion prohibited; at least one copy must survive"
                conn.execute(
                    "UPDATE deletion_journal SET status = 'refused', reason = ? WHERE id = ?",
                    (msg, e["id"]),
                )
                refusals.append({"entry_id": e["id"], "path": e["original_path"], "reason": msg})
            continue

        for e in c_entries:
            orig_p = Path(e["original_path"])
            # 1. Existence check
            if not orig_p.exists():
                msg = "File does not exist at original path"
                conn.execute(
                    "UPDATE deletion_journal SET status = 'refused', reason = ? WHERE id = ?",
                    (msg, e["id"]),
                )
                refusals.append({"entry_id": e["id"], "path": e["original_path"], "reason": msg})
                continue

            # 2. Size check
            actual_sz = orig_p.stat().st_size
            if actual_sz != e["size"]:
                msg = f"File size modified (recorded {e['size']}, actual {actual_sz})"
                conn.execute(
                    "UPDATE deletion_journal SET status = 'refused', reason = ? WHERE id = ?",
                    (msg, e["id"]),
                )
                refusals.append({"entry_id": e["id"], "path": e["original_path"], "reason": msg})
                continue

            # 3. MD5 check
            actual_md5 = compute_file_md5(orig_p)
            if actual_md5 != e["md5"]:
                msg = f"File content modified (MD5 mismatch: recorded {e['md5']}, actual {actual_md5})"
                conn.execute(
                    "UPDATE deletion_journal SET status = 'refused', reason = ? WHERE id = ?",
                    (msg, e["id"]),
                )
                refusals.append({"entry_id": e["id"], "path": e["original_path"], "reason": msg})
                continue

            # 4. Keeper check
            c_row = conn.execute("SELECT representative_id FROM clusters WHERE id = ?", (cid,)).fetchone()
            cur_keeper_id = c_row["representative_id"] if c_row else None
            if e["image_id"] == cur_keeper_id or e["image_id"] == e["keeper_id"]:
                msg = "File is recommended cluster keeper"
                conn.execute(
                    "UPDATE deletion_journal SET status = 'refused', reason = ? WHERE id = ?",
                    (msg, e["id"]),
                )
                refusals.append({"entry_id": e["id"], "path": e["original_path"], "reason": msg})
                continue

            # 5. Review / low-confidence / unalignable check
            if not e["is_manual"]:
                img = conn.execute(
                    "SELECT review_flag, is_low_confidence, quality_status FROM images WHERE id = ?",
                    (e["image_id"],),
                ).fetchone()
                if img:
                    if img["review_flag"] == 1:
                        msg = "Member flagged for review; manual override required"
                        conn.execute(
                            "UPDATE deletion_journal SET status = 'refused', reason = ? WHERE id = ?",
                            (msg, e["id"]),
                        )
                        refusals.append({"entry_id": e["id"], "path": e["original_path"], "reason": msg})
                        continue
                    if img["is_low_confidence"] == 1:
                        msg = "Member is low confidence; manual override required"
                        conn.execute(
                            "UPDATE deletion_journal SET status = 'refused', reason = ? WHERE id = ?",
                            (msg, e["id"]),
                        )
                        refusals.append({"entry_id": e["id"], "path": e["original_path"], "reason": msg})
                        continue
                    if img["quality_status"] == "unalignable":
                        msg = "Member is unalignable; manual override required"
                        conn.execute(
                            "UPDATE deletion_journal SET status = 'refused', reason = ? WHERE id = ?",
                            (msg, e["id"]),
                        )
                        refusals.append({"entry_id": e["id"], "path": e["original_path"], "reason": msg})
                        continue

            # 6. Perform the move to trash
            trash_p = Path(e["trash_path"])
            trash_p.parent.mkdir(parents=True, exist_ok=True)
            try:
                try:
                    os.replace(str(orig_p), str(trash_p))
                except OSError:
                    shutil.move(str(orig_p), str(trash_p))

                conn.execute("UPDATE deletion_journal SET status = 'moved' WHERE id = ?", (e["id"],))
                if e["image_id"]:
                    conn.execute("UPDATE images SET status = 'trashed' WHERE id = ?", (e["image_id"],))
                moved_count += 1
                bytes_moved += (e["size"] or 0)
            except Exception as exc:
                fail_msg = f"Move failed: {exc}"
                conn.execute(
                    "UPDATE deletion_journal SET status = 'failed', reason = ? WHERE id = ?",
                    (fail_msg, e["id"]),
                )
                failures.append({"entry_id": e["id"], "path": e["original_path"], "error": fail_msg})

    final_status = "executed" if moved_count > 0 else ("refused" if len(refusals) > 0 else "failed")
    conn.execute(
        """
        UPDATE ledger_batches
        SET status = ?, executed_at = strftime('%Y-%m-%dT%H:%M:%SZ','now'), bytes_freed = ?
        WHERE batch_id = ?
        """,
        (final_status, bytes_moved, batch_id),
    )
    conn.commit()

    return ExecuteResult(
        batch_id=batch_id,
        total_planned=len(entries),
        moved_count=moved_count,
        refused_count=len(refusals),
        failed_count=len(failures),
        bytes_moved=bytes_moved,
        refusals=refusals,
        failures=failures,
    )


def undo(conn: sqlite3.Connection, batch_id: str) -> UndoResult:
    """
    Restore all moved files from trash back to their original locations.
    
    Refuses to overwrite any existing file and reports conflicts.
    Updates journal entry status to 'restored' upon successful restore.
    """
    ensure_ledger_tables(conn)
    batch = conn.execute("SELECT * FROM ledger_batches WHERE batch_id = ?", (batch_id,)).fetchone()
    if not batch:
        raise ValueError(f"Batch '{batch_id}' not found in ledger")

    entries = conn.execute(
        "SELECT * FROM deletion_journal WHERE batch_id = ? AND status = 'moved' ORDER BY id",
        (batch_id,),
    ).fetchall()

    conflicts: list[tuple[str, str]] = []
    restored_files: list[str] = []
    restored_count = 0

    for e in entries:
        orig_p = Path(e["original_path"])
        trash_p = Path(e["trash_path"])

        if not trash_p.exists():
            conflicts.append((str(orig_p), f"Trash file missing at {trash_p}"))
            continue

        if orig_p.exists():
            conflicts.append((str(orig_p), "Target original path already exists; refusing to overwrite"))
            continue

        orig_p.parent.mkdir(parents=True, exist_ok=True)
        try:
            try:
                os.replace(str(trash_p), str(orig_p))
            except OSError:
                shutil.move(str(trash_p), str(orig_p))

            conn.execute("UPDATE deletion_journal SET status = 'restored' WHERE id = ?", (e["id"],))
            if e["image_id"]:
                conn.execute("UPDATE images SET status = 'clustered' WHERE id = ?", (e["image_id"],))
            restored_count += 1
            restored_files.append(str(orig_p))
        except Exception as exc:
            conflicts.append((str(orig_p), f"Restore move failed: {exc}"))

    # If all moved files restored without conflict, update batch status
    if restored_count == len(entries) and not conflicts:
        conn.execute("UPDATE ledger_batches SET status = 'undone' WHERE batch_id = ?", (batch_id,))

    conn.commit()
    return UndoResult(
        batch_id=batch_id,
        restored_count=restored_count,
        conflict_count=len(conflicts),
        conflicts=conflicts,
        restored_files=restored_files,
    )


def undo_file(
    conn: sqlite3.Connection,
    entry_id: int | None = None,
    original_path: str | Path | None = None,
) -> tuple[bool, str]:
    """
    Restore an individual trashed file back to its original location.
    
    Refuses to overwrite if target file already exists.
    Returns (success: bool, message: str).
    """
    ensure_ledger_tables(conn)
    if entry_id is not None:
        row = conn.execute(
            "SELECT * FROM deletion_journal WHERE id = ? AND status = 'moved'",
            (entry_id,),
        ).fetchone()
    elif original_path is not None:
        row = conn.execute(
            """
            SELECT * FROM deletion_journal
            WHERE original_path = ? AND status = 'moved'
            ORDER BY id DESC LIMIT 1
            """,
            (str(original_path),),
        ).fetchone()
    else:
        return False, "Must provide entry_id or original_path"

    if not row:
        return False, "No active moved entry found for given specifier"

    orig_p = Path(row["original_path"])
    trash_p = Path(row["trash_path"])

    if not trash_p.exists():
        return False, f"Trash file does not exist at {trash_p}"
    if orig_p.exists():
        return False, f"Original path already exists at {orig_p}; refusing to overwrite"

    orig_p.parent.mkdir(parents=True, exist_ok=True)
    try:
        try:
            os.replace(str(trash_p), str(orig_p))
        except OSError:
            shutil.move(str(trash_p), str(orig_p))

        conn.execute("UPDATE deletion_journal SET status = 'restored' WHERE id = ?", (row["id"],))
        if row["image_id"]:
            conn.execute("UPDATE images SET status = 'clustered' WHERE id = ?", (row["image_id"],))
        conn.commit()
        return True, f"Successfully restored {orig_p}"
    except Exception as exc:
        return False, f"Restore move failed: {exc}"


def purge(
    conn: sqlite3.Connection,
    batch_id: str,
    confirm: bool = False,
    max_age_days: int = 30,
) -> PurgeResult:
    """
    Permanently purge trashed files for a batch.
    
    Requires explicit confirm=True and enforces that batch age is older than max_age_days.
    """
    if not confirm:
        raise ValueError("Purge requires explicit confirmation (confirm=True)")

    ensure_ledger_tables(conn)
    batch = conn.execute("SELECT * FROM ledger_batches WHERE batch_id = ?", (batch_id,)).fetchone()
    if not batch:
        raise ValueError(f"Batch '{batch_id}' not found in ledger")

    ts_str = batch["executed_at"] or batch["created_at"]
    batch_time = _parse_timestamp(ts_str)
    now = datetime.now(timezone.utc)
    age_days = (now - batch_time).total_seconds() / 86400.0

    if age_days < max_age_days:
        raise ValueError(
            f"Batch '{batch_id}' is only {age_days:.1f} days old; "
            f"purge requires batches older than {max_age_days} days"
        )

    entries = conn.execute(
        "SELECT * FROM deletion_journal WHERE batch_id = ? AND status = 'moved' ORDER BY id",
        (batch_id,),
    ).fetchall()

    purged_count = 0
    bytes_purged = 0

    for e in entries:
        trash_p = Path(e["trash_path"])
        if trash_p.exists():
            trash_p.unlink()
        conn.execute("UPDATE deletion_journal SET status = 'purged' WHERE id = ?", (e["id"],))
        purged_count += 1
        bytes_purged += (e["size"] or 0)

    # Clean up empty batch directory in trash
    batch_trash_dir = Path(batch["trash_dir"]) / batch_id
    if batch_trash_dir.exists():
        shutil.rmtree(batch_trash_dir, ignore_errors=True)

    conn.execute("UPDATE ledger_batches SET status = 'purged' WHERE batch_id = ?", (batch_id,))
    conn.commit()

    return PurgeResult(
        batch_id=batch_id,
        purged_count=purged_count,
        bytes_purged=bytes_purged,
    )


def get_history(conn: sqlite3.Connection, limit: int = 50) -> list[dict[str, Any]]:
    """Retrieve history of deletion batches from ledger."""
    ensure_ledger_tables(conn)
    rows = conn.execute(
        """
        SELECT batch_id, created_at, executed_at, mode, status, file_count, bytes_freed, trash_dir
        FROM ledger_batches
        ORDER BY created_at DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    return [dict(r) for r in rows]
