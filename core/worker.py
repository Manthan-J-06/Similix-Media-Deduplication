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

from core.config import load_config
from db.connection import get_connection, init_db

logger = logging.getLogger("argus.worker")


class JobHeartbeatThread(threading.Thread):
    """Background thread that writes a heartbeat timestamp to the job file every interval seconds."""

    def __init__(self, job_file: Path, interval: float = 5.0):
        super().__init__(daemon=True, name="JobHeartbeat")
        self.job_file = job_file
        self.interval = interval
        self._stop_event = threading.Event()

    def run(self) -> None:
        while not self._stop_event.wait(self.interval):
            try:
                if self.job_file.is_file():
                    update_job_file(self.job_file, heartbeat_at=time.time())
            except Exception:
                pass

    def stop(self) -> None:
        self._stop_event.set()


def update_job_file(
    job_file: Path,
    status: str | None = None,
    stage: str | None = None,
    stage_display: str | None = None,
    current_file: int | None = None,
    total_files: int | None = None,
    progress: float | None = None,
    message: str | None = None,
    warnings: list[str] | None = None,
    error: str | None = None,
    pid: int | None = None,
    heartbeat_at: float | None = None,
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

    if status is not None:
        data["status"] = status
    if stage is not None:
        data["stage"] = stage
    if stage_display is not None:
        data["stage_display"] = stage_display
    if current_file is not None:
        data["current_file"] = current_file
    if total_files is not None:
        data["total_files"] = total_files
    if progress is not None:
        data["progress"] = progress
    if message is not None:
        data["message"] = message
    if warnings is not None:
        data["warnings"] = warnings
    if error is not None:
        data["error"] = error
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

    for k, v in extra.items():
        if v is not None:
            data[k] = v

    now = time.time()
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
        job_file = Path.home() / ".argus" / "jobs" / f"{job_id}.json"

    resolved_db = Path(db_path).resolve()
    logger.info("Worker targeting per-user database: %s", resolved_db)
    print(f"Worker targeting per-user database: {resolved_db}")
    cfg = load_config(config_path) if config_path else load_config()
    init_db(db_path)
    conn = get_connection(db_path)
    images_before = conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]
    clusters_before = conn.execute("SELECT COUNT(*) FROM clusters").fetchone()[0]

    all_warnings: list[str] = []
    current_stage = "scan"
    current_file_path: str | None = None
    slowest_5_records: list[dict[str, Any]] = []

    heartbeat = JobHeartbeatThread(job_file, interval=5.0)
    heartbeat.start()

    try:
        from core.categories import run_categorization
        from core.cluster import run_stage2_clustering
        from core.embeddings import run_embeddings
        from core.quality import run_quality_pipeline
        from core.stage0 import run_stage0
        from core.stage1 import run_stage1
        from core.verify import run_stage3_verification

        update_job_file(
            job_file,
            pid=os.getpid(),
            status="running",
            stage="scan",
            stage_display="Scanning files",
            current_file=0,
            total_files=0,
            progress=5.0,
            message="Scanning files...",
        )

        # ── Stage 0 & 1: Discovery, MD5, and pHash ──
        if source_type == "drive":
            from sources.google_drive import GoogleDriveSource
            current_stage = "scan"
            source0 = GoogleDriveSource(folder_id=drive_folder_id, conn=conn, stage="stage0")
            s0_res = run_stage0(source0, conn, cfg)
            for p, r in s0_res.skipped_reasons:
                all_warnings.append(f"{Path(p).name}: {r}")

            update_job_file(
                job_file,
                progress=15.0,
                message="Perceptual hashing (Drive thumbnails)...",
            )
            source_active = GoogleDriveSource(folder_id=drive_folder_id, conn=conn, stage="stage1")
            s1_res = run_stage1(source_active, conn, cfg)
            for p, r in s1_res.skipped_reasons:
                all_warnings.append(f"{Path(p).name}: {r}")
        else:
            from sources.local_folder import LocalFolderSource
            scan_dir = Path(folder or cfg.get("sources", {}).get("library_dir", "~/.argus/library")).expanduser().resolve()
            source_active = LocalFolderSource(scan_dir)
            files = list(source_active.list_files())
            total_files = len(files)

            update_job_file(
                job_file,
                stage="scan",
                stage_display="Scanning files",
                current_file=0,
                total_files=total_files,
                progress=5.0,
                message=f"Scanning (0 of {total_files} files)",
            )

            current_stage = "scan"
            if _stage_needs_scan(conn) or total_files > conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]:
                s0_res = run_stage0(source_active, conn, cfg)
                for p, r in s0_res.skipped_reasons:
                    all_warnings.append(f"{Path(p).name}: {r}")
            else:
                logger.info("Stage 0 (scan) already complete — skipping.")
                # Reconstruct a dummy result for progress display
                class _DummyResult:
                    files_discovered = 0
                    files_unchanged = total_files
                s0_res = _DummyResult()

            update_job_file(
                job_file,
                progress=15.0,
                current_file=s0_res.files_discovered + s0_res.files_unchanged,
                total_files=total_files,
                message=f"Perceptual hashing (0 of {total_files} files)",
            )

            current_stage = "hash"
            s1_res = run_stage1(source_active, conn, cfg)
            for p, r in s1_res.skipped_reasons:
                all_warnings.append(f"{Path(p).name}: {r}")

        # ── Stage 2: Embeddings ──
        current_stage = "embed"
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
            current_file=0,
            total_files=pending_embed,
            progress=25.0,
            message=f"Embedding (0 of {pending_embed} files)" if pending_embed > 0 else "Embeddings up-to-date",
        )

        def on_embed_progress(cur: int, tot: int):
            pct = 25.0 + 30.0 * (cur / max(1, tot))
            update_job_file(
                job_file,
                stage="embed",
                stage_display="Embedding",
                current_file=cur,
                total_files=tot,
                progress=round(pct, 1),
                message=f"Embedding ({cur} of {tot} files)",
            )

        if pending_embed > 0:
            run_embeddings(source_active, conn, cfg, progress_callback=on_embed_progress)
        else:
            logger.info("Stage 2 (embed) already complete — skipping.")

        # ── Stage 2: Clustering ──
        current_stage = "cluster"
        update_job_file(
            job_file,
            stage="cluster",
            stage_display="Clustering",
            progress=58.0,
            message="Clustering candidate duplicates...",
        )
        if _stage_needs_cluster(conn):
            user_index_path = Path(db_path).with_suffix(".faiss")
            run_stage2_clustering(conn, cfg, index_path=user_index_path)
        else:
            logger.info("Stage 2 (cluster) already complete — skipping.")

        # ── Stage 3: Verification ──
        current_stage = "verify"
        cand_count = conn.execute(
            "SELECT COUNT(*) FROM images WHERE cluster_id IS NOT NULL"
        ).fetchone()[0]

        update_job_file(
            job_file,
            stage="verify",
            stage_display="Geometric verification",
            current_file=0,
            total_files=cand_count,
            progress=70.0,
            message=f"Geometric verification ({cand_count} candidates)",
        )
        if _stage_needs_verify(conn):
            run_stage3_verification(conn, cfg, source=source_active)
        else:
            logger.info("Stage 3 (verify) already complete — skipping.")

        # ── Stage 4: Quality & Keeper Selection ──
        current_stage = "quality"
        cluster_count = conn.execute("SELECT COUNT(*) FROM clusters").fetchone()[0]
        update_job_file(
            job_file,
            stage="quality",
            stage_display=f"Evaluating quality (0 of {cluster_count} clusters)" if cluster_count > 0 else "Quality up-to-date",
            current_file=0,
            total_files=cluster_count,
            progress=85.0,
            message=f"Evaluating quality (0 of {cluster_count} clusters)" if cluster_count > 0 else "Quality up-to-date",
        )

        def on_quality_progress(cur: int, tot: int, cid: int):
            pct = 85.0 + 7.0 * (cur / max(1, tot))
            update_job_file(
                job_file,
                stage="quality",
                stage_display=f"Evaluating quality (cluster {cur} of {tot}, id #{cid})",
                current_file=cur,
                total_files=tot,
                progress=round(pct, 1),
                message=f"Evaluating quality (cluster {cur} of {tot}, id #{cid})",
            )

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

        # ── Stage 5: Semantic Categorization ──
        current_stage = "categorize"
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
            current_file=0,
            total_files=pending_cat,
            progress=92.0,
            message=f"Categorizing (0 of {pending_cat} files)" if pending_cat > 0 else "Categories up-to-date",
        )

        def on_cat_progress(cur: int, tot: int):
            pct = 92.0 + 7.0 * (cur / max(1, tot))
            update_job_file(
                job_file,
                stage="categorize",
                stage_display="Categorizing",
                current_file=cur,
                total_files=tot,
                progress=round(pct, 1),
                message=f"Categorizing ({cur} of {tot} files)",
            )

        if pending_cat > 0:
            run_categorization(conn, cfg, force=False, progress_callback=on_cat_progress)
        else:
            logger.info("Stage 5 (categorize) already complete — skipping.")

        # ── Finalize ──
        images_after = conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]
        clusters_after = conn.execute("SELECT COUNT(*) FROM clusters").fetchone()[0]
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
            message=f"Pipeline complete: {summary_msg}",
            warnings=all_warnings,
            slowest_clusters=slowest_5_records if slowest_5_records else None,
            images_added=images_added,
            clusters_created=clusters_created,
            skipped_files=all_warnings,
            skipped_count=skipped_count,
            total_images=images_after,
        )
    except Exception as exc:
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

        file_info = f" on file '{current_file_path}'" if current_file_path else ""
        formatted_message = f"Pipeline failed at stage '{current_stage}'{file_info}: {err_msg}. Action: {actionable_advice}"

        update_job_file(
            job_file,
            status="failed",
            stage=current_stage,
            error=err_msg,
            message=formatted_message,
            failed_stage=current_stage,
            failed_file=current_file_path,
            actionable_advice=actionable_advice,
            slowest_clusters=slowest_5_records if slowest_5_records else None,
        )
    finally:
        heartbeat.stop()
        conn.close()


def main():
    parser = argparse.ArgumentParser(description="Argus pipeline background worker")
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
