"""
core/worker.py — Standalone background pipeline job worker.

Runs in a separate OS process so PyTorch, SIFT, and OpenCV computation
never starves the FastAPI request-handling event loop or holds its GIL.
Limits thread pools (OpenMP, MKL, OpenCV, PyTorch) to prevent CPU starvation.
"""

from __future__ import annotations

import os
import sys

# Limit thread pools BEFORE importing libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

try:
    import cv2
    cv2.setNumThreads(2)
except Exception:
    pass



import argparse
import json
import logging
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any
import uuid

from core.config import load_config, resolve_data_dir
from db.connection import get_connection, init_db

logger = logging.getLogger("similix.worker")


class WorkerState:
    """Live state tracked by worker and monitored by watchdog thread."""

    def __init__(self):
        self.stage: str = "scan"
        self.stage_display: str = "Scanning files"
        self.current_file_path: str | None = None
        self.last_progress_time: float = time.time()
        self.is_stalled: bool = False


class JobHeartbeatThread(threading.Thread):
    """Background thread that writes a heartbeat timestamp to the job file and monitors for stalled stages (>60s)."""

    def __init__(self, job_file: Path, interval: float = 5.0, worker_state: WorkerState | None = None, stall_timeout: float = 900.0):
        super().__init__(daemon=True, name="JobHeartbeat")
        self.job_file = job_file
        self.interval = interval
        self.worker_state = worker_state
        self.stall_timeout = stall_timeout
        self._stop_event = threading.Event()

    def run(self) -> None:
        while not self._stop_event.wait(self.interval):
            try:
                now = time.time()
                if self.worker_state is not None:
                    # Check if stage has stalled
                    elapsed_since_prog = now - self.worker_state.last_progress_time
                    if self.stall_timeout > 0 and elapsed_since_prog > self.stall_timeout and not self.worker_state.is_stalled:
                        self.worker_state.is_stalled = True
                        stalled_stage = self.worker_state.stage
                        stalled_file = self.worker_state.current_file_path or "unknown"
                        stall_err = f"Stage '{stalled_stage}' stalled on file '{stalled_file}' (no progress for {int(self.stall_timeout)}s)"
                        logger.error(stall_err)
                        print(f"PIPELINE STALL: {stall_err}", file=sys.stderr)
                        update_job_file(
                            self.job_file,
                            status="failed",
                            stage=stalled_stage,
                            stage_display=f"Stalled: {stalled_stage}",
                            error=stall_err,
                            message=f"Pipeline failed: {stall_err}",
                            failed_stage=stalled_stage,
                            failed_file=stalled_file,
                            actionable_advice="Processing stalled. Check network connectivity, source availability, or file integrity.",
                        )
                    elif self.worker_state.is_stalled and elapsed_since_prog <= self.stall_timeout:
                        # Resumed progress: auto-recover
                        self.worker_state.is_stalled = False
                        update_job_file(
                            self.job_file,
                            status="running",
                            error=None,
                            failed_stage=None,
                            failed_file=None,
                        )

                if self.job_file.is_file():
                    update_job_file(self.job_file, heartbeat_at=now)
            except Exception:
                pass

    def stop(self) -> None:
        self._stop_event.set()


def update_job_file(
    job_file: Path,
    status: str | None = None,
    stage: str | None = None,
    stage_display: str | None = None,
    stage_progress: float | None = None,
    current_file: int | None = None,
    total_files: int | None = None,
    current_file_path: str | None = None,
    progress: float | None = None,
    message: str | None = None,
    warnings: list[str] | None = None,
    error: str | None = None,
    pid: int | None = None,
    heartbeat_at: float | None = None,
    start_time: float | None = None,
    stage_timings: dict[str, float] | None = None,
    slowest_clusters: list[dict[str, Any]] | None = None,
    images_added: int | None = None,
    clusters_created: int | None = None,
    skipped_files: list[str] | None = None,
    skipped_count: int | None = None,
    total_images: int | None = None,
    actionable_advice: str | None = None,
    failed_stage: str | None = None,
    failed_file: str | None = None,
    **extra: Any,
) -> None:
    """Atomically update a job's JSON state file."""
    data: dict[str, Any] = {}
    if job_file.is_file():
        try:
            data = json.loads(job_file.read_text())
        except Exception:
            data = {}

    if not data.get("id"):
        data["id"] = job_file.stem
    if pid is not None:
        data["pid"] = pid

    now = time.time()
    if not data.get("start_time"):
        data["start_time"] = start_time or now
    elif start_time is not None:
        data["start_time"] = start_time

    data["elapsed_seconds"] = round(now - data["start_time"], 1)

    if status is not None:
        data["status"] = status
        if status == "running":
            if error is None:
                data["error"] = None
            data.pop("failed_stage", None)
            data.pop("failed_file", None)
    elif data.get("status") in ("failed", "stalled", None) and stage is not None and stage not in ("complete", "failed"):
        data["status"] = "running"
        data["error"] = None
        data.pop("failed_stage", None)
        data.pop("failed_file", None)
    if stage is not None:
        data["stage"] = stage
    if stage_display is not None:
        data["stage_display"] = stage_display
    if stage_progress is not None:
        data["stage_progress"] = stage_progress
    if current_file is not None:
        data["current_file"] = current_file
    if total_files is not None:
        data["total_files"] = total_files
    if current_file_path is not None:
        data["current_file_path"] = current_file_path
    if progress is not None:
        data["progress"] = progress
    if message is not None:
        data["message"] = message
    if warnings is not None:
        data["warnings"] = warnings
    if error is not None:
        data["error"] = error
    if stage_timings is not None:
        data["stage_timings"] = stage_timings
    if slowest_clusters is not None:
        data["slowest_clusters"] = slowest_clusters
    if images_added is not None:
        data["images_added"] = images_added
    if clusters_created is not None:
        data["clusters_created"] = clusters_created
    if skipped_files is not None:
        data["skipped_files"] = skipped_files
    if skipped_count is not None:
        data["skipped_count"] = skipped_count
    if total_images is not None:
        data["total_images"] = total_images
    if actionable_advice is not None:
        data["actionable_advice"] = actionable_advice
    if failed_stage is not None:
        data["failed_stage"] = failed_stage
    if failed_file is not None:
        data["failed_file"] = failed_file

    if current_file is not None or progress is not None or stage_progress is not None:
        data["last_progress_at"] = now

    for k, v in extra.items():
        if v is not None:
            data[k] = v

    data["updated_at"] = now
    data["heartbeat_at"] = heartbeat_at if heartbeat_at is not None else now
    job_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = job_file.with_suffix(f".tmp_{uuid.uuid4().hex[:6]}")
    tmp_file.write_text(json.dumps(data, indent=2))
    os.replace(tmp_file, job_file)


def _stage_needs_scan(conn: "sqlite3.Connection") -> bool:
    """Return True if Stage 0/1 (scan+hash) still has work to do."""
    # If there are images that haven't been hashed yet, or none at all, we need to scan.
    undiscovered = conn.execute(
        "SELECT COUNT(*) FROM images WHERE md5 IS NULL AND (status IS NULL OR status != 'skipped')"
    ).fetchone()[0]
    total = conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]
    return undiscovered > 0 or total == 0


def _stage_needs_embed(conn: "sqlite3.Connection") -> bool:
    return conn.execute(
        """SELECT COUNT(*) FROM images
           WHERE clip_vector IS NULL AND is_raw_pair = 0
             AND (status IS NULL OR status != 'skipped')"""
    ).fetchone()[0] > 0


def _stage_needs_cluster(conn: "sqlite3.Connection") -> bool:
    # Re-cluster whenever there are embedded images not yet in a cluster
    # (or previously clustered but now needing re-evaluation after new imports)
    return conn.execute(
        """SELECT COUNT(*) FROM images
           WHERE clip_vector IS NOT NULL AND cluster_id IS NULL
             AND is_raw_pair = 0 AND (status IS NULL OR status != 'skipped')"""
    ).fetchone()[0] > 0


def _stage_needs_verify(conn: "sqlite3.Connection") -> bool:
    return conn.execute(
        """SELECT COUNT(*) FROM images
           WHERE cluster_id IS NOT NULL
             AND quality_status IS NULL
             AND (status IS NULL OR status != 'skipped')"""
    ).fetchone()[0] > 0


def _stage_needs_quality(conn: "sqlite3.Connection") -> bool:
    return conn.execute(
        """SELECT COUNT(*) FROM images
           WHERE cluster_id IS NOT NULL
             AND quality_status IS NULL
             AND (status IS NULL OR status != 'skipped')"""
    ).fetchone()[0] > 0


def _stage_needs_categorize(conn: "sqlite3.Connection") -> bool:
    return conn.execute(
        """SELECT COUNT(*) FROM images
           WHERE category_top1 IS NULL
             AND (status IS NULL OR status != 'skipped')"""
    ).fetchone()[0] > 0


def run_pipeline(
    job_id: str,
    db_path: str | Path,
    config_path: str | Path | None = None,
    source_type: str = "local",
    folder: str | Path | None = None,
    drive_folder_id: str | None = None,
    job_file: Path | None = None,
) -> None:
    """Execute the full deduplication pipeline in small batches with progress reporting.

    Resumable: each stage is skipped if already complete, so a failed run can be
    continued by re-invoking run_pipeline with the same arguments.
    """
    if job_file is None:
        job_file = resolve_data_dir() / "jobs" / f"{job_id}.json"

    resolved_db = Path(db_path).resolve()
    logger.info("Worker targeting per-user database: %s", resolved_db)
    print(f"Worker targeting per-user database: {resolved_db}")
    cfg = load_config(config_path) if config_path else load_config()
    conn = init_db(db_path)
    images_before = conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]
    clusters_before = conn.execute("SELECT COUNT(*) FROM clusters").fetchone()[0]

    all_warnings: list[str] = []
    current_stage = "scan"
    current_file_path: str | None = None
    slowest_5_records: list[dict[str, Any]] = []
    stage_timings: dict[str, float] = {}
    job_start_time = time.time()

    worker_state = WorkerState()
    stall_timeout = float(cfg.get("general", {}).get("stall_timeout_seconds", 900.0))
    heartbeat = JobHeartbeatThread(job_file, interval=5.0, worker_state=worker_state, stall_timeout=stall_timeout)
    heartbeat.start()

    try:
        from core.categories import run_categorization
        from core.cluster import run_stage2_clustering
        from core.embeddings import run_embeddings
        from core.quality import run_quality_pipeline
        from core.stage0 import run_stage0
        from core.stage1 import run_stage1
        from core.verify import prune_keypoint_cache, run_stage3_verification

        update_job_file(
            job_file,
            pid=os.getpid(),
            status="running",
            stage="scan",
            stage_display="Scanning files",
            stage_progress=0.0,
            current_file=0,
            total_files=0,
            progress=5.0,
            start_time=job_start_time,
            message="Scanning files...",
        )

        # ── Stage 0: Discovery & MD5 ──
        t_stage = time.time()
        current_stage = "scan"
        worker_state.stage = "scan"
        worker_state.stage_display = "Scanning files"
        worker_state.last_progress_time = time.time()

        if source_type == "drive":
            from sources.google_drive import GoogleDriveSource
            if not drive_folder_id:
                fid_file = resolved_db.parent / "drive_folder_id.txt"
                if fid_file.exists():
                    try:
                        drive_folder_id = fid_file.read_text(encoding="utf-8").strip() or None
                    except Exception:
                        pass
                if not drive_folder_id:
                    try:
                        row_fid = conn.execute("SELECT drive_id FROM images WHERE source = 'gdrive' AND drive_id IS NOT NULL LIMIT 1").fetchone()
                        if row_fid and row_fid[0]:
                            drive_folder_id = row_fid[0].split(":", 1)[0]
                    except Exception:
                        pass
            if not drive_folder_id or drive_folder_id == "root":
                raise ValueError("Security violation: A specific Google Drive folder ID is required. Scanning root or entire drive is prohibited.")
            user_tok = resolved_db.parent / "drive_token.json"
            tok_path = user_tok if user_tok.exists() else None
            cache_p = resolved_db.parent / "drive_cache"
            source0 = GoogleDriveSource(
                folder_id=drive_folder_id,
                conn=conn,
                stage="stage0",
                token_path=tok_path,
                cache_dir=cache_p,
                is_public=bool(not tok_path),
            )

            def on_drive_scan(cur: int, tot: int, fpath: str):
                worker_state.current_file_path = fpath
                worker_state.last_progress_time = time.time()
                stg_pct = (cur / max(1, tot)) * 100 if tot > 0 else 50.0
                update_job_file(
                    job_file,
                    stage="scan",
                    stage_display="Scanning files",
                    stage_progress=round(stg_pct, 1),
                    current_file=cur,
                    total_files=tot,
                    current_file_path=fpath,
                    progress=5.0,
                    message=f"Scanning Drive folder ({cur} files discovered)",
                )

            s0_res = run_stage0(source0, conn, cfg, progress_callback=on_drive_scan)
            for p, r in s0_res.skipped_reasons:
                all_warnings.append(f"{Path(p).name}: {r}")
            if s0_res.files_discovered == 0:
                all_warnings.append(f"No supported image files found in Google Drive folder '{drive_folder_id or 'root'}'.")
            source_active = source0
            source_active.stage = "stage1"
        else:
            from sources.local_folder import LocalFolderSource
            scan_dir = Path(folder or cfg.get("sources", {}).get("library_dir", resolve_data_dir() / "library")).expanduser().resolve()
            source_active = LocalFolderSource(scan_dir)
            files = list(source_active.list_files())
            total_files = len(files)

            update_job_file(
                job_file,
                stage="scan",
                stage_display="Scanning files",
                stage_progress=0.0,
                current_file=0,
                total_files=total_files,
                progress=5.0,
                message=f"Scanning (0 of {total_files} files)",
            )

            def on_local_scan(cur: int, tot: int, fpath: str):
                worker_state.current_file_path = fpath
                worker_state.last_progress_time = time.time()
                stg_pct = (cur / max(1, tot)) * 100 if tot > 0 else 50.0
                pct = 5.0 + 10.0 * (cur / max(1, tot)) if tot > 0 else 5.0
                update_job_file(
                    job_file,
                    stage="scan",
                    stage_display="Scanning files",
                    stage_progress=round(stg_pct, 1),
                    current_file=cur,
                    total_files=tot,
                    current_file_path=fpath,
                    progress=round(pct, 1),
                    message=f"Scanning ({cur} of {tot} files)",
                )

            if _stage_needs_scan(conn) or total_files > conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]:
                s0_res = run_stage0(source_active, conn, cfg, progress_callback=on_local_scan)
                for p, r in s0_res.skipped_reasons:
                    all_warnings.append(f"{Path(p).name}: {r}")
            else:
                logger.info("Stage 0 (scan) already complete — skipping.")
                class _DummyResult:
                    files_discovered = 0
                    files_unchanged = total_files
                    skipped_reasons: list[tuple[str, str]] = []
                s0_res = _DummyResult()

        stage_timings["scan"] = round(time.time() - t_stage, 3)

        # ── Stage 1: Perceptual Hashing ──
        t_stage = time.time()
        current_stage = "hash"
        worker_state.stage = "hash"
        worker_state.stage_display = "Perceptual hashing"
        worker_state.last_progress_time = time.time()

        total_to_hash = conn.execute(
            """
            SELECT COUNT(*) FROM images
            WHERE phash IS NULL
              AND is_raw_pair = 0
              AND (status IS NULL OR status != 'skipped')
            """
        ).fetchone()[0]

        update_job_file(
            job_file,
            stage="hash",
            stage_display="Perceptual hashing",
            stage_progress=0.0,
            current_file=0,
            total_files=total_to_hash,
            progress=15.0,
            message=f"Perceptual hashing (0 of {total_to_hash} files)" if total_to_hash > 0 else "Perceptual hashing up-to-date",
        )

        def on_hash_progress(cur: int, tot: int, fpath: str):
            worker_state.current_file_path = fpath
            worker_state.last_progress_time = time.time()
            stg_pct = (cur / max(1, tot)) * 100 if tot > 0 else 100.0
            pct = 15.0 + 10.0 * (cur / max(1, tot)) if tot > 0 else 25.0
            update_job_file(
                job_file,
                stage="hash",
                stage_display="Perceptual hashing",
                stage_progress=round(stg_pct, 1),
                current_file=cur,
                total_files=tot,
                current_file_path=fpath,
                progress=round(pct, 1),
                message=f"Perceptual hashing ({cur} of {tot} files)",
            )

        s1_res = run_stage1(source_active, conn, cfg, progress_callback=on_hash_progress)
        for p, r in s1_res.skipped_reasons:
            all_warnings.append(f"{Path(p).name}: {r}")

        stage_timings["hash"] = round(time.time() - t_stage, 3)

        # ── Stage 2: Embeddings ──
        t_stage = time.time()
        current_stage = "embed"
        worker_state.stage = "embed"
        worker_state.stage_display = "Embedding"
        worker_state.last_progress_time = time.time()

        pending_embed = conn.execute(
            """
            SELECT COUNT(*) FROM images
            WHERE clip_vector IS NULL
              AND is_raw_pair = 0
              AND (status IS NULL OR status != 'skipped')
            """
        ).fetchone()[0]

        update_job_file(
            job_file,
            stage="embed",
            stage_display="Embedding",
            stage_progress=0.0,
            current_file=0,
            total_files=pending_embed,
            progress=25.0,
            message=f"Embedding (0 of {pending_embed} files)" if pending_embed > 0 else "Embeddings up-to-date",
        )

        def on_embed_progress(cur: int, tot: int, fpath: str | None = None):
            if fpath:
                worker_state.current_file_path = fpath
            worker_state.last_progress_time = time.time()
            stg_pct = (cur / max(1, tot)) * 100 if tot > 0 else 100.0
            pct = 25.0 + 33.0 * (cur / max(1, tot)) if tot > 0 else 58.0
            update_job_file(
                job_file,
                stage="embed",
                stage_display="Embedding",
                stage_progress=round(stg_pct, 1),
                current_file=cur,
                total_files=tot,
                current_file_path=fpath,
                progress=round(pct, 1),
                message=f"Embedding ({cur} of {tot} files)",
            )

        if hasattr(source_active, "stage"):
            source_active.stage = "stage2"
        if pending_embed > 0:
            run_embeddings(source_active, conn, cfg, progress_callback=on_embed_progress)
        else:
            logger.info("Stage 2 (embed) already complete — skipping.")

        stage_timings["embed"] = round(time.time() - t_stage, 3)

        # ── Stage 2: Clustering ──
        t_stage = time.time()
        current_stage = "cluster"
        worker_state.stage = "cluster"
        worker_state.stage_display = "Clustering"
        worker_state.last_progress_time = time.time()

        update_job_file(
            job_file,
            stage="cluster",
            stage_display="Clustering",
            stage_progress=50.0,
            progress=58.0,
            message="Clustering candidate duplicates...",
        )
        if _stage_needs_cluster(conn):
            user_index_path = Path(db_path).with_suffix(".faiss")
            run_stage2_clustering(conn, cfg, index_path=user_index_path)
        else:
            logger.info("Stage 2 (cluster) already complete — skipping.")

        stage_timings["cluster"] = round(time.time() - t_stage, 3)

        # ── Stage 3: Verification ──
        t_stage = time.time()
        current_stage = "verify"
        worker_state.stage = "verify"
        worker_state.stage_display = "Geometric verification"
        worker_state.last_progress_time = time.time()

        cand_count = conn.execute(
            "SELECT COUNT(*) FROM images WHERE cluster_id IS NOT NULL AND is_keeper = 0"
        ).fetchone()[0]

        update_job_file(
            job_file,
            stage="verify",
            stage_display="Geometric verification",
            stage_progress=0.0,
            current_file=0,
            total_files=cand_count,
            progress=65.0,
            message=f"Geometric verification (0 of {cand_count} pairs)" if cand_count > 0 else "Geometric verification up-to-date",
        )

        def on_verify_progress(cur: int, tot: int, fpath: str):
            worker_state.current_file_path = fpath
            worker_state.last_progress_time = time.time()
            stg_pct = (cur / max(1, tot)) * 100 if tot > 0 else 100.0
            pct = 65.0 + 10.0 * (cur / max(1, tot)) if tot > 0 else 75.0
            update_job_file(
                job_file,
                stage="verify",
                stage_display="Geometric verification",
                stage_progress=round(stg_pct, 1),
                current_file=cur,
                total_files=tot,
                current_file_path=fpath,
                progress=round(pct, 1),
                message=f"Geometric verification ({cur} of {tot} pairs)",
            )

        if hasattr(source_active, "stage"):
            source_active.stage = "stage3"
        if _stage_needs_verify(conn):
            run_stage3_verification(conn, cfg, source=source_active, progress_callback=on_verify_progress)
        else:
            logger.info("Stage 3 (verify) already complete — skipping.")

        stage_timings["verify"] = round(time.time() - t_stage, 3)

        # ── Stage 4: Quality & Keeper Selection ──
        t_stage = time.time()
        current_stage = "quality"
        worker_state.stage = "quality"
        worker_state.stage_display = "Evaluating quality"
        worker_state.last_progress_time = time.time()

        cluster_count = conn.execute("SELECT COUNT(*) FROM clusters").fetchone()[0]
        update_job_file(
            job_file,
            stage="quality",
            stage_display=f"Evaluating quality (0 of {cluster_count} clusters)" if cluster_count > 0 else "Quality up-to-date",
            stage_progress=0.0,
            current_file=0,
            total_files=cluster_count,
            progress=75.0,
            message=f"Evaluating quality (0 of {cluster_count} clusters)" if cluster_count > 0 else "Quality up-to-date",
        )

        def on_quality_progress(cur: int, tot: int, cid: int):
            worker_state.last_progress_time = time.time()
            stg_pct = (cur / max(1, tot)) * 100 if tot > 0 else 100.0
            pct = 75.0 + 13.0 * (cur / max(1, tot)) if tot > 0 else 88.0
            update_job_file(
                job_file,
                stage="quality",
                stage_display=f"Evaluating quality (cluster {cur} of {tot}, id #{cid})",
                stage_progress=round(stg_pct, 1),
                current_file=cur,
                total_files=tot,
                progress=round(pct, 1),
                message=f"Evaluating quality (cluster {cur} of {tot}, id #{cid})",
            )

        if hasattr(source_active, "stage"):
            source_active.stage = "stage4"
        if _stage_needs_quality(conn):
            q_res = run_quality_pipeline(conn, cfg, source=source_active, progress_callback=on_quality_progress)
            if hasattr(q_res, "slowest_clusters") and q_res.slowest_clusters:
                slowest_5 = q_res.slowest_clusters[:5]
                slowest_5_records = [{"cluster_id": cid, "seconds": round(sec, 2)} for cid, sec in slowest_5]
                slowest_str = ", ".join(f"#{cid} ({sec:.2f}s)" for cid, sec in slowest_5)
                print(f"Top 5 slowest clusters in quality evaluation: {slowest_str}")
                logger.info("Top 5 slowest clusters in quality evaluation: %s", slowest_str)
        else:
            logger.info("Stage 4 (quality) already complete — skipping.")

        stage_timings["quality"] = round(time.time() - t_stage, 3)

        # ── Stage 4.5: Keeper Selection ──
        t_stage = time.time()
        current_stage = "keeper"
        worker_state.stage = "keeper"
        worker_state.stage_display = "Selecting keepers"
        worker_state.last_progress_time = time.time()
        try:
            from core.keeper import run_keeper_pipeline
            run_keeper_pipeline(conn, mode="balanced", config=cfg)
        except Exception as exc:
            logger.warning("Stage 4.5 keeper pipeline failed: %s", exc)
        stage_timings["keeper"] = round(time.time() - t_stage, 3)

        # ── Stage 5: Semantic Categorization ──
        t_stage = time.time()
        current_stage = "categorize"
        worker_state.stage = "categorize"
        worker_state.stage_display = "Categorizing"
        worker_state.last_progress_time = time.time()

        pending_cat = conn.execute(
            """
            SELECT COUNT(*) FROM images
            WHERE category_top1 IS NULL
              AND (status IS NULL OR status != 'skipped')
            """
        ).fetchone()[0]

        update_job_file(
            job_file,
            stage="categorize",
            stage_display="Categorizing",
            stage_progress=0.0,
            current_file=0,
            total_files=pending_cat,
            progress=88.0,
            message=f"Categorizing (0 of {pending_cat} files)" if pending_cat > 0 else "Categories up-to-date",
        )

        def on_cat_progress(cur: int, tot: int):
            worker_state.last_progress_time = time.time()
            stg_pct = (cur / max(1, tot)) * 100 if tot > 0 else 100.0
            pct = 88.0 + 10.0 * (cur / max(1, tot)) if tot > 0 else 98.0
            update_job_file(
                job_file,
                stage="categorize",
                stage_display="Categorizing",
                stage_progress=round(stg_pct, 1),
                current_file=cur,
                total_files=tot,
                progress=round(pct, 1),
                message=f"Categorizing ({cur} of {tot} files)",
            )

        if pending_cat > 0:
            run_categorization(conn, cfg, force=False, progress_callback=on_cat_progress)
        else:
            logger.info("Stage 5 (categorize) already complete — skipping.")

        stage_timings["categorize"] = round(time.time() - t_stage, 3)

        # ── Finalize ──
        conn.commit()
        try:
            pruned_kp = prune_keypoint_cache(conn, keep_representatives=True, max_rep_rows=500, vacuum_threshold_mb=10.0)
            if pruned_kp > 0:
                logger.info("Pruned %d stale keypoint_cache rows after pipeline completion.", pruned_kp)
        except Exception as exc:
            logger.warning("Non-fatal error pruning keypoint_cache: %s", exc)

        # Timing Summary Report
        total_pipeline_time = round(time.time() - job_start_time, 3)
        print("\n" + "=" * 54)
        print("  SIMILIX Deduplication Pipeline Stage Timings")
        print("=" * 54)
        for stg, dur in stage_timings.items():
            print(f"  - {stg.capitalize():<24} : {dur:>8.3f}s")
        print(f"  Total Pipeline Time      : {total_pipeline_time:>8.3f}s")
        print("=" * 54 + "\n")

        # Verification step (Requirement 4): Open a fresh connection, verify all writes are committed
        # and readable from an independent connection before reporting 'completed'.
        verify_conn = get_connection(db_path)
        try:
            verify_conn.execute("PRAGMA wal_checkpoint(PASSIVE);")
            v_images = verify_conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]
            v_clusters = verify_conn.execute("SELECT COUNT(*) FROM clusters").fetchone()[0]
            v_categories = verify_conn.execute(
                "SELECT COUNT(*) FROM images WHERE category_top1 IS NOT NULL AND (status IS NULL OR status != 'skipped')"
            ).fetchone()[0]
        finally:
            verify_conn.close()

        images_after = v_images
        clusters_after = v_clusters
        images_added = max(0, images_after - images_before)
        clusters_created = max(0, clusters_after - clusters_before)
        skipped_count = len(all_warnings)

        summary_msg = f"{images_added} images added, {clusters_created} duplicate clusters found"
        if skipped_count > 0:
            summary_msg += f" ({skipped_count} skipped)"
        if slowest_5_records:
            slowest_str = ", ".join(f"#{r['cluster_id']} ({r['seconds']}s)" for r in slowest_5_records)
            summary_msg += f". Slowest clusters: {slowest_str}"

        final_status = "completed_with_warnings" if all_warnings else "completed"
        update_job_file(
            job_file,
            status=final_status,
            stage="done",
            stage_display="Completed with warnings" if all_warnings else "Completed",
            progress=100.0,
            stage_progress=100.0,
            message=f"Pipeline complete: {summary_msg}",
            warnings=all_warnings,
            slowest_clusters=slowest_5_records if slowest_5_records else None,
            images_added=images_added,
            clusters_created=clusters_created,
            skipped_files=all_warnings,
            skipped_count=skipped_count,
            total_images=images_after,
            verified_images=v_images,
            verified_clusters=v_clusters,
            verified_categories=v_categories,
            stage_timings=stage_timings,
            start_time=job_start_time,
        )
    except Exception as exc:
        if worker_state.is_stalled:
            return

        import traceback
        print(f"PIPELINE EXCEPTION at stage '{current_stage}': {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        logger.exception("Pipeline failed at stage '%s': %s", current_stage, exc)
        err_detail = str(exc)
        exc_type = type(exc).__name__
        if not err_detail:
            err_msg = exc_type
        elif exc_type in ("KeyError", "IndexError", "ValueError", "TypeError") and not err_detail.startswith(exc_type):
            err_msg = f"{exc_type}: {err_detail}"
        else:
            err_msg = err_detail

        # Actionable advice construction
        advice_parts = []
        if current_stage in ("scan", "hash"):
            advice_parts.append("Check file permissions, verify that paths are valid images, and ensure files are not corrupted or empty.")
        elif current_stage == "embed":
            advice_parts.append("Check system memory and verify CLIP model cache is accessible.")
        elif current_stage in ("cluster", "verify"):
            advice_parts.append("Check feature matching parameters and database integrity.")
        elif current_stage == "quality":
            advice_parts.append("Verify original image files are accessible on disk for quality analysis.")
        elif current_stage == "categorize":
            advice_parts.append("Check category configuration in config.toml.")

        if source_type == "drive":
            advice_parts.append("Verify Google Drive folder permissions, credentials, and network connectivity.")

        actionable_advice = " ".join(advice_parts) if advice_parts else "Review error logs and retry the operation."

        active_file = worker_state.current_file_path or current_file_path
        file_info = f" on file '{active_file}'" if active_file else ""
        formatted_message = f"Pipeline failed at stage '{current_stage}'{file_info}: {err_msg}. Action: {actionable_advice}"

        update_job_file(
            job_file,
            status="failed",
            stage=current_stage,
            error=err_msg,
            message=formatted_message,
            failed_stage=current_stage,
            failed_file=active_file,
            actionable_advice=actionable_advice,
            slowest_clusters=slowest_5_records if slowest_5_records else None,
            stage_timings=stage_timings if stage_timings else None,
            start_time=job_start_time,
        )
    finally:
        heartbeat.stop()
        try:
            conn.close()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description="SIMILIX pipeline background worker")
    parser.add_argument("--job-id", required=True, help="Unique job ID")
    parser.add_argument("--db", required=True, help="Database path")
    parser.add_argument("--config", default=None, help="Config path")
    parser.add_argument("--source-type", default="local", help="Source type: local | drive")
    parser.add_argument("--folder", default=None, help="Target folder for local scanning")
    parser.add_argument("--drive-folder-id", default=None, help="Google Drive folder ID")
    parser.add_argument("--job-file", default=None, help="Path to job JSON state file")

    args = parser.parse_args()
    job_file = Path(args.job_file) if args.job_file else None

    run_pipeline(
        job_id=args.job_id,
        db_path=args.db,
        config_path=args.config,
        source_type=args.source_type,
        folder=args.folder,
        drive_folder_id=args.drive_folder_id,
        job_file=job_file,
    )


if __name__ == "__main__":
    main()
