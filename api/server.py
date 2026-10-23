"""
api/server.py — FastAPI REST API for Argus.

Binds to 127.0.0.1 only. No CORS for external origins.
All file moves/deletions are strictly routed through core/ledger.
"""

from __future__ import annotations

import io
import json
import logging
import os
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
from typing import Any, Optional
import urllib.parse
import urllib.request
import uuid
import zipfile

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger("argus.api")

# In-process synchronization locks for concurrent thumbnail generation
_thumb_locks: dict[tuple[int, int], threading.Lock] = {}
_thumb_global_lock = threading.Lock()


def _get_thumb_lock(image_id: int, size: int) -> threading.Lock:
    with _thumb_global_lock:
        key = (image_id, size)
        if key not in _thumb_locks:
            _thumb_locks[key] = threading.Lock()
        return _thumb_locks[key]


def _create_placeholder_jpeg() -> bytes:
    """Generate a small neutral gray JPEG placeholder for failed thumbnails."""
    buf = io.BytesIO()
    im = Image.new("RGB", (64, 64), color=(180, 180, 180))
    im.save(buf, format="JPEG", quality=75)
    return buf.getvalue()


PLACEHOLDER_JPEG_BYTES = _create_placeholder_jpeg()

from core.categories import search_images_by_text
from core.config import load_config
from core.keeper import choose_keeper
from core.ledger import (
    ensure_ledger_tables,
    execute,
    get_history,
    plan,
    undo,
)
from db.connection import get_connection, init_db


class PlanRequest(BaseModel):
    mode: str = "conservative"
    approved_swap_cluster_ids: list[int] = Field(default_factory=list)
    excluded_ids: list[int] = Field(default_factory=list)
    model_config = ConfigDict(extra="ignore")


class ApplyRequest(BaseModel):
    batch_id: str
    confirm: bool = False
    model_config = ConfigDict(extra="ignore")


class UndoRequest(BaseModel):
    batch_id: str
    model_config = ConfigDict(extra="ignore")


class DriveSourceRequest(BaseModel):
    link: str
    project: str = "default"
    model_config = ConfigDict(extra="ignore")


class DatasetSourceRequest(BaseModel):
    url: str
    project: str = "default"
    model_config = ConfigDict(extra="ignore")


def extract_drive_folder_id(link: str) -> str:
    """Extract and validate Google Drive folder ID from link or raw ID string."""
    link = link.strip()
    if link.startswith("http://") or link.startswith("https://"):
        m = re.search(r"/folders/([a-zA-Z0-9_-]+)", link)
        if m:
            return m.group(1)
        m = re.search(r"[?&]id=([a-zA-Z0-9_-]+)", link)
        if m:
            return m.group(1)
        raise HTTPException(status_code=400, detail=f"Invalid Drive folder link: {link}")

    # Raw ID: Google Drive folder IDs are typically 28-45 alphanumeric/underscore/hyphen characters
    if re.fullmatch(r"[a-zA-Z0-9_-]{20,60}", link):
        return link
    raise HTTPException(status_code=400, detail=f"Invalid Drive folder link or ID: {link}")


def write_job_state(job_file: Path, data: dict) -> None:
    """Write job state JSON atomically."""
    job_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = job_file.with_suffix(f".tmp_{uuid.uuid4().hex[:6]}")
    data["updated_at"] = time.time()
    tmp_file.write_text(json.dumps(data, indent=2))
    os.replace(tmp_file, job_file)


def read_job_state(job_file: Path) -> dict | None:
    """Read job state JSON safely."""
    if not job_file.is_file():
        return None
    try:
        return json.loads(job_file.read_text())
    except Exception:
        return None


def launch_pipeline_job(
    app: FastAPI,
    source_type: str = "local",
    folder: Path | None = None,
    drive_folder_id: str | None = None,
) -> str:
    """
    Launch pipeline execution in a separate OS process so heavy PyTorch/SIFT
    computation never starves FastAPI's event loop or request handling.
    """
    job_id = uuid.uuid4().hex[:12]
    jobs_dir = app.state.jobs_dir
    jobs_dir.mkdir(parents=True, exist_ok=True)
    job_file = jobs_dir / f"{job_id}.json"

    initial_job_data = {
        "id": job_id,
        "status": "running",
        "stage": "queued",
        "stage_display": "Queued",
        "current_file": 0,
        "total_files": 0,
        "progress": 0.0,
        "message": "Job queued",
        "source_type": source_type,
        "target_folder": str(folder) if folder else None,
        "warnings": [],
        "error": None,
        "created_at": time.time(),
        "updated_at": time.time(),
    }
    write_job_state(job_file, initial_job_data)
    app.state.jobs[job_id] = initial_job_data

    # Allow custom in-process worker override for testing
    custom_worker = getattr(app.state, "pipeline_worker", None)
    if custom_worker is not None:
        t = threading.Thread(
            target=custom_worker,
            args=(job_id, app.state, source_type, folder, drive_folder_id),
            daemon=True,
        )
        t.start()
        return job_id

    cmd = [
        sys.executable,
        "-m",
        "core.worker",
        "--job-id", job_id,
        "--db", str(app.state.db_path),
        "--source-type", source_type,
        "--job-file", str(job_file),
    ]
    if folder:
        cmd.extend(["--folder", str(folder)])
    if drive_folder_id:
        cmd.extend(["--drive-folder-id", drive_folder_id])
    if getattr(app.state, "config_path", None):
        cmd.extend(["--config", str(app.state.config_path)])

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "2"
    env["VECLIB_MAXIMUM_THREADS"] = "2"
    env["NUMEXPR_NUM_THREADS"] = "2"
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    initial_job_data["pid"] = proc.pid
    write_job_state(job_file, initial_job_data)
    app.state.jobs[job_id] = initial_job_data
    return job_id


def get_db(request: Request):
    """Provide a per-request SQLite connection with row_factory enabled."""
    db_path = request.app.state.db_path
    conn = get_connection(db_path)
    try:
        yield conn
    finally:
        conn.close()


def create_app(
    db_path: str | Path | None = None,
    trash_dir: str | Path | None = None,
    thumb_cache_dir: str | Path | None = None,
    library_dir: str | Path | None = None,
    jobs_dir: str | Path | None = None,
    config: dict | None = None,
    config_path: str | Path | None = None,
) -> FastAPI:
    """Create and configure the Argus FastAPI application."""
    if config is None:
        try:
            config = load_config(config_path) if config_path else load_config()
        except Exception:
            config = {}

    if db_path is None:
        db_path = os.environ.get("ARGUS_DB_PATH") or config.get("database", {}).get("path") or "argus.db"
    resolved_db_path = Path(db_path).expanduser().resolve()

    # Ensure schema and ledger tables are initialized
    init_conn = init_db(resolved_db_path)
    ensure_ledger_tables(init_conn)
    init_conn.close()

    if trash_dir is None:
        cfg_trash = config.get("ledger", {}).get("trash_dir")
        trash_dir = os.environ.get("ARGUS_TRASH_DIR") or cfg_trash or (Path.home() / ".argus" / "trash")
    resolved_trash_dir = Path(trash_dir).expanduser().resolve()

    if thumb_cache_dir is None:
        thumb_cache_dir = os.environ.get("ARGUS_THUMB_CACHE_DIR") or (Path.home() / ".argus" / "thumbnails")
    resolved_thumb_cache_dir = Path(thumb_cache_dir).expanduser().resolve()
    try:
        resolved_thumb_cache_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass

    if library_dir is None:
        cfg_library = config.get("sources", {}).get("library_dir")
        library_dir = os.environ.get("ARGUS_LIBRARY_DIR") or cfg_library or (Path.home() / ".argus" / "library")
    resolved_library_dir = Path(library_dir).expanduser().resolve()
    try:
        resolved_library_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass

    app = FastAPI(title="Argus API", version="0.1.0")

    # Strict local-only CORS: never allow external origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://127.0.0.1",
            "http://127.0.0.1:8000",
            "http://localhost",
            "http://localhost:8000",
        ],
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    if jobs_dir is None:
        cfg_jobs = config.get("general", {}).get("jobs_dir")
        jobs_dir = os.environ.get("ARGUS_JOBS_DIR") or cfg_jobs or (Path.home() / ".argus" / "jobs")
    resolved_jobs_dir = Path(jobs_dir).expanduser().resolve()
    try:
        resolved_jobs_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass

    app.state.db_path = resolved_db_path
    app.state.trash_dir = resolved_trash_dir
    app.state.thumb_cache_dir = resolved_thumb_cache_dir
    app.state.library_dir = resolved_library_dir
    app.state.jobs_dir = resolved_jobs_dir
    app.state.config = config
    app.state.config_path = config_path
    app.state.jobs = {}

    ui_file = Path(__file__).resolve().parent.parent / "ui" / "index.html"

    @app.get("/")
    def serve_ui():
        """Serve the single-file Argus web UI."""
        if not ui_file.is_file():
            raise HTTPException(status_code=404, detail="UI index.html not found")
        return FileResponse(str(ui_file), media_type="text/html")

    @app.get("/clusters")
    def list_clusters(
        mode: str = Query("conservative"),
        category: Optional[str] = Query(None),
        min_savings: Optional[float] = Query(None),
        sort: str = Query("id"),
        request: Request = None,
        conn: sqlite3.Connection = Depends(get_db),
    ) -> list[dict[str, Any]]:
        if mode not in ("conservative", "balanced", "aggressive"):
            raise HTTPException(
                status_code=400,
                detail="Invalid mode. Must be one of: conservative, balanced, aggressive",
            )

        c_rows = conn.execute(
            "SELECT id, representative_id, reference_uncertain, category FROM clusters ORDER BY id"
        ).fetchall()

        results: list[dict[str, Any]] = []
        for c_row in c_rows:
            cid = c_row["id"]
            ref_unc = c_row["reference_uncertain"] if "reference_uncertain" in c_row.keys() else 0
            members = conn.execute(
                """
                SELECT id, cluster_id, file_path, width, height, file_size, format, md5,
                       is_keeper, is_low_confidence, ssim, ssim_color, sharpness_ratio,
                       pixel_ratio, bytes_ratio, quality_status, review_flag, delete_safe,
                       reason, source_likelihood, jpeg_quality_est, category_top1,
                       category_top1_prob, match_pct, resolution_pct, confidence, status
                FROM images
                WHERE cluster_id = ? AND (status IS NULL OR status != 'trashed')
                ORDER BY id
                """,
                (cid,),
            ).fetchall()

            live_members = [
                m for m in members
                if str(m["file_path"]).startswith("drive://") or os.path.exists(m["file_path"])
            ]
            if len(live_members) < 2:
                continue

            member_dicts = [dict(m) for m in live_members]
            keeper_res = choose_keeper(
                member_dicts,
                mode=mode,
                config=request.app.state.config,
                reference_uncertain=ref_unc,
            )

            keeper_id = keeper_res["keeper_id"]
            del_safe_ids = set(keeper_res["delete_safe_ids"])
            review_ids = set(keeper_res["review_ids"])
            swap_prop = keeper_res.get("swap_proposal")

            del_safe_members = [m for m in member_dicts if m["id"] in del_safe_ids]
            savings_bytes = sum(m["file_size"] or 0 for m in del_safe_members)
            total_bytes = sum(m["file_size"] or 0 for m in member_dicts)
            savings_ratio = round(savings_bytes / total_bytes, 4) if total_bytes > 0 else 0.0

            keeper_m = next((m for m in member_dicts if m["id"] == keeper_id), None)
            cluster_cat = c_row["category"] or (keeper_m.get("category_top1") if keeper_m else None)

            # Category filter
            if category is not None and (cluster_cat or "").lower() != category.lower():
                continue

            # Minimum savings filter
            if min_savings is not None:
                if 0 < min_savings <= 1.0:
                    if savings_ratio < min_savings:
                        continue
                else:
                    if savings_bytes < min_savings:
                        continue

            results.append({
                "id": cid,
                "keeper": {
                    "id": keeper_id,
                    "file_path": keeper_m["file_path"] if keeper_m else "",
                    "file_size": keeper_m["file_size"] if keeper_m else 0,
                    "format": keeper_m["format"] if keeper_m else "",
                },
                "savings_bytes": savings_bytes,
                "reclaimable_bytes": savings_bytes,
                "savings_ratio": savings_ratio,
                "member_count": len(member_dicts),
                "delete_safe_count": len(del_safe_ids),
                "review_count": len(review_ids),
                "category": cluster_cat,
                "has_swap_proposal": swap_prop is not None,
                "swap_proposal": swap_prop,
            })

        # Sorting
        if sort in ("savings", "-savings"):
            results.sort(key=lambda x: x["savings_bytes"], reverse=True)
        elif sort == "+savings":
            results.sort(key=lambda x: x["savings_bytes"], reverse=False)
        elif sort in ("count", "counts", "-count", "-counts"):
            results.sort(key=lambda x: x["member_count"], reverse=True)
        elif sort in ("+count", "+counts"):
            results.sort(key=lambda x: x["member_count"], reverse=False)
        elif sort in ("category", "+category"):
            results.sort(key=lambda x: (x["category"] or "").lower())
        elif sort == "-category":
            results.sort(key=lambda x: (x["category"] or "").lower(), reverse=True)
        elif sort == "-id":
            results.sort(key=lambda x: x["id"], reverse=True)
        else:
            results.sort(key=lambda x: x["id"])

        return results

    @app.get("/clusters/{cluster_id}")
    def get_cluster(
        cluster_id: int,
        mode: str = Query("conservative"),
        request: Request = None,
        conn: sqlite3.Connection = Depends(get_db),
    ) -> dict[str, Any]:
        if mode not in ("conservative", "balanced", "aggressive"):
            raise HTTPException(
                status_code=400,
                detail="Invalid mode. Must be one of: conservative, balanced, aggressive",
            )

        c_row = conn.execute(
            "SELECT id, representative_id, reference_uncertain, category FROM clusters WHERE id = ?",
            (cluster_id,),
        ).fetchone()
        if not c_row:
            raise HTTPException(status_code=404, detail=f"Cluster {cluster_id} not found")

        members = conn.execute(
            """
            SELECT id, cluster_id, file_path, width, height, file_size, format, md5,
                   is_keeper, is_low_confidence, ssim, ssim_color, sharpness_ratio,
                   pixel_ratio, bytes_ratio, quality_status, review_flag, delete_safe,
                   reason, source_likelihood, jpeg_quality_est, category_top1,
                   category_top1_prob, match_pct, resolution_pct, confidence, status
            FROM images
            WHERE cluster_id = ? AND (status IS NULL OR status != 'trashed')
            ORDER BY id
            """,
            (cluster_id,),
        ).fetchall()

        live_members = [
            m for m in members
            if str(m["file_path"]).startswith("drive://") or os.path.exists(m["file_path"])
        ]

        if len(live_members) < 2:
            raise HTTPException(status_code=404, detail=f"Cluster {cluster_id} has fewer than 2 live members on disk")

        member_dicts = [dict(m) for m in live_members]
        ref_unc = c_row["reference_uncertain"] if "reference_uncertain" in c_row.keys() else 0
        keeper_res = choose_keeper(
            member_dicts,
            mode=mode,
            config=request.app.state.config,
            reference_uncertain=ref_unc,
        )

        keeper_id = keeper_res["keeper_id"]
        del_safe_ids = set(keeper_res["delete_safe_ids"])
        review_ids = set(keeper_res["review_ids"])
        reasons = keeper_res["reasons"]

        formatted_members = []
        for m in member_dicts:
            mid = m["id"]
            formatted_members.append({
                "id": mid,
                "file_path": m["file_path"],
                "size": m["file_size"],
                "width": m["width"],
                "height": m["height"],
                "format": m["format"],
                "match_pct": m["match_pct"],
                "resolution_pct": m["resolution_pct"],
                "confidence": m["confidence"],
                "category": m["category_top1"],
                "category_prob": m["category_top1_prob"],
                "is_recommended_keeper": mid == keeper_id,
                "delete_safe": mid in del_safe_ids,
                "review": mid in review_ids,
                "reason": reasons.get(mid, ""),
            })

        keeper_m = next((m for m in formatted_members if m["is_recommended_keeper"]), None)
        cluster_cat = c_row["category"] or (keeper_m.get("category") if keeper_m else None)

        return {
            "id": c_row["id"],
            "mode": mode,
            "category": cluster_cat,
            "reference_uncertain": bool(ref_unc),
            "swap_proposal": keeper_res.get("swap_proposal"),
            "members": formatted_members,
        }

    @app.get("/thumb/{image_id}")
    def get_thumbnail(
        image_id: int,
        size: int = Query(256, ge=32, le=2048),
        request: Request = None,
        conn: sqlite3.Connection = Depends(get_db),
    ):
        """Serve a cached JPEG thumbnail for the given image ID. Never accepts file paths."""
        try:
            row = conn.execute(
                "SELECT id, file_path, source, drive_id, modified_time FROM images WHERE id = ?",
                (image_id,),
            ).fetchone()
            if not row:
                raise HTTPException(status_code=404, detail=f"Image {image_id} not found")

            file_path_str = str(row["file_path"])
            orig_path: Path | None = None

            # Resolve Google Drive virtual path from drive_cache if available
            is_drive = file_path_str.startswith("drive://") or row["source"] == "gdrive" or bool(row["drive_id"])
            if is_drive:
                drive_id = row["drive_id"]
                if not drive_id and file_path_str.startswith("drive://"):
                    try:
                        drive_id = file_path_str[8:].split("/")[0]
                    except Exception:
                        drive_id = None

                cfg = getattr(request.app.state, "config", {}) or {}
                drive_cache_dir = Path(
                    cfg.get("sources", {})
                    .get("google_drive", {})
                    .get("cache_dir", "~/.argus/drive_cache")
                ).expanduser()

                if drive_id and drive_cache_dir.is_dir():
                    mtime = row["modified_time"]
                    if mtime is not None:
                        try:
                            candidate = drive_cache_dir / f"{drive_id}_{int(mtime)}.jpg"
                            if candidate.is_file():
                                orig_path = candidate
                        except Exception:
                            pass
                    if orig_path is None:
                        matches = sorted(drive_cache_dir.glob(f"{drive_id}_*.jpg"))
                        if not matches:
                            matches = sorted(drive_cache_dir.glob(f"{drive_id}.jpg"))
                        if matches:
                            orig_path = matches[0]

            if orig_path is None:
                candidate_path = Path(file_path_str)
                if candidate_path.exists():
                    orig_path = candidate_path

            if orig_path is None:
                logger.warning(f"Image {image_id} source file not found on disk: {file_path_str}")
                return Response(content=PLACEHOLDER_JPEG_BYTES, media_type="image/jpeg", status_code=200)

            cache_dir = request.app.state.thumb_cache_dir
            cache_path = cache_dir / f"{image_id}_{size}.jpg"

            # Check existing valid cache
            if cache_path.is_file():
                try:
                    st = cache_path.stat()
                    if st.st_size > 0 and st.st_mtime >= orig_path.stat().st_mtime:
                        return FileResponse(str(cache_path), media_type="image/jpeg")
                except OSError:
                    pass

            # Concurrency synchronization per (image_id, size)
            lock = _get_thumb_lock(image_id, size)
            with lock:
                # Re-check inside lock in case another thread just completed generation
                if cache_path.is_file():
                    try:
                        st = cache_path.stat()
                        if st.st_size > 0 and st.st_mtime >= orig_path.stat().st_mtime:
                            return FileResponse(str(cache_path), media_type="image/jpeg")
                    except OSError:
                        pass

                cache_dir.mkdir(parents=True, exist_ok=True)
                tmp_path = cache_dir / f".tmp_{image_id}_{size}_{uuid.uuid4().hex}.jpg"
                try:
                    with Image.open(orig_path) as img:
                        if img.mode != "RGB":
                            img = img.convert("RGB")
                        img.thumbnail((size, size), Image.Resampling.LANCZOS)
                        img.save(str(tmp_path), format="JPEG", quality=85)
                    os.replace(str(tmp_path), str(cache_path))
                except Exception as exc:
                    if tmp_path.exists():
                        try:
                            tmp_path.unlink()
                        except OSError:
                            pass
                    logger.exception(f"Thumbnail generation failed for image {image_id} ({orig_path}): {exc}")
                    return Response(content=PLACEHOLDER_JPEG_BYTES, media_type="image/jpeg", status_code=200)

                return FileResponse(str(cache_path), media_type="image/jpeg")
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception(f"Unhandled error in get_thumbnail for image {image_id}: {exc}")
            return Response(content=PLACEHOLDER_JPEG_BYTES, media_type="image/jpeg", status_code=200)

    @app.get("/search")
    def search_images(
        q: str = Query(..., description="Natural language search query"),
        limit: int = Query(10, ge=1, le=100),
        request: Request = None,
        conn: sqlite3.Connection = Depends(get_db),
    ) -> list[dict[str, Any]]:
        if not q.strip():
            return []
        try:
            return search_images_by_text(
                query=q,
                conn=conn,
                config=request.app.state.config,
                limit=limit,
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Search failed: {exc}")

    @app.get("/categories")
    def get_categories(
        conn: sqlite3.Connection = Depends(get_db),
    ) -> dict[str, int]:
        rows = conn.execute(
            """
            SELECT category_top1, COUNT(*) AS count
            FROM images
            WHERE category_top1 IS NOT NULL AND (status IS NULL OR status != 'trashed')
            GROUP BY category_top1
            ORDER BY count DESC
            """
        ).fetchall()
        return {r["category_top1"]: r["count"] for r in rows}

    @app.post("/plan")
    def create_plan(
        req: PlanRequest,
        request: Request = None,
        conn: sqlite3.Connection = Depends(get_db),
    ) -> dict[str, Any]:
        if req.mode not in ("conservative", "balanced", "aggressive"):
            raise HTTPException(
                status_code=400,
                detail="Invalid mode. Must be one of: conservative, balanced, aggressive",
            )

        # Enforce that all excluded_ids are valid members of a cluster
        if req.excluded_ids:
            placeholders = ",".join("?" for _ in req.excluded_ids)
            rows = conn.execute(
                f"SELECT id FROM images WHERE id IN ({placeholders}) AND cluster_id IS NOT NULL",
                req.excluded_ids,
            ).fetchall()
            found_ids = {r[0] for r in rows}
            invalid_ids = [mid for mid in req.excluded_ids if mid not in found_ids]
            if invalid_ids:
                raise HTTPException(
                    status_code=400,
                    detail=f"Image ID(s) not currently a member of a cluster: {invalid_ids}",
                )

        if req.approved_swap_cluster_ids:
            c_placeholders = ",".join("?" for _ in req.approved_swap_cluster_ids)
            c_rows = conn.execute(
                f"SELECT id FROM clusters WHERE id IN ({c_placeholders})",
                req.approved_swap_cluster_ids,
            ).fetchall()
            found_c_ids = {r[0] for r in c_rows}
            invalid_c = [cid for cid in req.approved_swap_cluster_ids if cid not in found_c_ids]
            if invalid_c:
                raise HTTPException(
                    status_code=404,
                    detail=f"Cluster ID(s) not found: {invalid_c}",
                )

        try:
            summary = plan(
                conn=conn,
                mode=req.mode,
                approved_swap_cluster_ids=req.approved_swap_cluster_ids,
                excluded_ids=req.excluded_ids,
                config=request.app.state.config,
                trash_dir=request.app.state.trash_dir,
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        return {
            "batch_id": summary.batch_id,
            "mode": summary.mode,
            "file_count": summary.file_count,
            "bytes_freed": summary.bytes_freed,
            "trash_dir": summary.trash_dir,
            "per_cluster_keeper": summary.per_cluster_keeper,
            "entries": [
                {
                    "image_id": e.image_id,
                    "cluster_id": e.cluster_id,
                    "keeper_id": e.keeper_id,
                    "original_path": e.original_path,
                    "file_size": e.file_size,
                    "reason": e.reason,
                    "is_manual": e.is_manual,
                }
                for e in summary.entries
            ],
        }

    @app.post("/apply")
    def apply_batch(
        req: ApplyRequest,
        conn: sqlite3.Connection = Depends(get_db),
    ) -> dict[str, Any]:
        if not req.confirm:
            raise HTTPException(
                status_code=400,
                detail="Execution requires explicit confirmation (confirm: true)",
            )

        batch = conn.execute(
            "SELECT status FROM ledger_batches WHERE batch_id = ?",
            (req.batch_id,),
        ).fetchone()
        if not batch:
            raise HTTPException(status_code=404, detail=f"Batch '{req.batch_id}' not found in ledger")
        if batch["status"] != "planned":
            raise HTTPException(
                status_code=400,
                detail=f"Batch '{req.batch_id}' has status '{batch['status']}'; only 'planned' batches can be executed",
            )

        try:
            res = execute(conn, batch_id=req.batch_id, confirm=True)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        return {
            "batch_id": res.batch_id,
            "total_planned": res.total_planned,
            "moved_count": res.moved_count,
            "refused_count": res.refused_count,
            "failed_count": res.failed_count,
            "bytes_moved": res.bytes_moved,
            "refusals": res.refusals,
            "failures": res.failures,
        }

    @app.post("/undo")
    def undo_batch(
        req: UndoRequest,
        conn: sqlite3.Connection = Depends(get_db),
    ) -> dict[str, Any]:
        batch = conn.execute(
            "SELECT status FROM ledger_batches WHERE batch_id = ?",
            (req.batch_id,),
        ).fetchone()
        if not batch:
            raise HTTPException(status_code=404, detail=f"Batch '{req.batch_id}' not found in ledger")

        try:
            res = undo(conn, batch_id=req.batch_id)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        return {
            "batch_id": res.batch_id,
            "restored_count": res.restored_count,
            "conflict_count": res.conflict_count,
            "conflicts": res.conflicts,
            "restored_files": res.restored_files,
        }

    @app.get("/history")
    def batch_history(
        limit: int = Query(50, ge=1, le=500),
        conn: sqlite3.Connection = Depends(get_db),
    ) -> list[dict[str, Any]]:
        return get_history(conn, limit=limit)

    @app.post("/sources/upload")
    async def upload_sources(
        files: list[UploadFile] = File(...),
        project: str = Query("default"),
        request: Request = None,
    ) -> dict[str, Any]:
        """
        Upload images into ~/.argus/library/<project>/ with path sanitization
        and size caps. Never modifies or removes user original files.
        """
        clean_project = re.sub(r"[^a-zA-Z0-9_\-]", "_", project.strip()) or "default"
        sources_cfg = request.app.state.config.get("sources", {})
        upload_cfg = sources_cfg.get("upload", {})
        library_base = request.app.state.library_dir
        dest_dir = library_base / clean_project
        dest_dir.mkdir(parents=True, exist_ok=True)

        max_batch_files = upload_cfg.get("max_batch_files", 100)
        if len(files) > max_batch_files:
            raise HTTPException(
                status_code=400,
                detail=f"Upload batch exceeds {max_batch_files} files cap (received {len(files)})",
            )

        allowed_exts = {e.lower().lstrip(".") for e in upload_cfg.get("allowed_extensions", ["jpg", "jpeg", "png", "webp", "heic"])}
        per_file_cap = upload_cfg.get("per_file_cap_mb", 50) * 1024 * 1024
        total_cap = upload_cfg.get("total_cap_mb", 1000) * 1024 * 1024

        saved_files = []
        for upload in files:
            orig_name = upload.filename or ""
            # Path traversal rejection
            if not orig_name or "/" in orig_name or "\\" in orig_name or ".." in orig_name or orig_name != Path(orig_name).name:
                raise HTTPException(status_code=400, detail=f"Path traversal rejected: {orig_name}")

            # Non-image extension rejection
            ext = Path(orig_name).suffix.lower().lstrip(".")
            if ext not in allowed_exts:
                raise HTTPException(status_code=400, detail=f"Non-image rejected: {orig_name}")

            dest_path = dest_dir / orig_name
            total_bytes = 0
            with open(dest_path, "wb") as f_out:
                while True:
                    chunk = await upload.read(65536)
                    if not chunk:
                        break
                    total_bytes += len(chunk)
                    if total_bytes > per_file_cap:
                        f_out.close()
                        if dest_path.exists():
                            dest_path.unlink()
                        raise HTTPException(status_code=400, detail=f"File exceeds {per_file_cap // (1024*1024)} MB cap: {orig_name}")
                    f_out.write(chunk)
            saved_files.append(str(dest_path))

        # Check total directory size against total_cap
        dir_size = sum(f.stat().st_size for f in dest_dir.glob("*") if f.is_file())
        if dir_size > total_cap:
            for sf in saved_files:
                p = Path(sf)
                if p.exists():
                    p.unlink()
            raise HTTPException(status_code=400, detail=f"Upload exceeds total library cap ({total_cap // (1024*1024)} MB)")

        job_id = launch_pipeline_job(request.app, source_type="local", folder=dest_dir)
        return {
            "job_id": job_id,
            "uploaded_count": len(saved_files),
            "project": clean_project,
            "files": [Path(p).name for p in saved_files],
        }

    @app.post("/sources/drive")
    def add_drive_source(
        req: DriveSourceRequest,
        request: Request = None,
    ) -> dict[str, Any]:
        """
        Validate Google Drive link, ensure read-only auth is active,
        and kick off read-only scan.
        """
        folder_id = extract_drive_folder_id(req.link)

        # Verify Google Drive authentication status
        from sources.google_drive import DEFAULT_TOKEN_PATH, get_drive_credentials
        token_path = Path(request.app.state.config.get("sources", {}).get("google_drive", {}).get("token_path", DEFAULT_TOKEN_PATH)).expanduser()
        if not token_path.exists():
            raise HTTPException(
                status_code=400,
                detail="Google Drive authentication required. Please run 'argus drive-auth' first.",
            )

        try:
            creds = get_drive_credentials(token_path=token_path, prompt_if_missing=False)
            if not creds or not creds.valid:
                raise HTTPException(
                    status_code=400,
                    detail="Google Drive authentication expired or invalid. Please re-run 'argus drive-auth'.",
                )
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Google Drive authentication error: {exc}. Please run 'argus drive-auth' first.",
            )

        job_id = launch_pipeline_job(request.app, source_type="drive", drive_folder_id=folder_id)
        return {
            "job_id": job_id,
            "folder_id": folder_id,
            "status": "queued",
        }

    @app.post("/sources/dataset")
    def add_dataset_source(
        req: DatasetSourceRequest,
        request: Request = None,
    ) -> dict[str, Any]:
        """
        Download dataset from allow-listed HTTPS host, streaming with size and
        file-count caps. Rejects non-images and archives with traversal or non-images.
        """
        url = req.url.strip()
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme.lower() != "https":
            raise HTTPException(status_code=400, detail="HTTPS only: HTTP and other schemes are rejected")

        dataset_cfg = request.app.state.config.get("sources", {}).get("dataset", {})
        allowed_hosts = [h.lower().strip() for h in dataset_cfg.get("allowed_hosts", ["huggingface.co"])]
        host = (parsed.netloc.split(":")[0]).lower()
        if not any(host == ah or host.endswith("." + ah) for ah in allowed_hosts):
            raise HTTPException(status_code=400, detail=f"Host '{host}' outside allow-list ({allowed_hosts}) rejected")

        size_cap_bytes = dataset_cfg.get("size_cap_mb", 500) * 1024 * 1024
        file_count_cap = dataset_cfg.get("file_count_cap", 1000)

        class SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, req_obj, fp, code, msg, headers, newurl):
                p = urllib.parse.urlparse(newurl)
                if p.scheme.lower() != "https":
                    raise HTTPException(status_code=400, detail="Redirect to non-HTTPS URL rejected")
                h = (p.netloc.split(":")[0]).lower()
                if not any(h == ah or h.endswith("." + ah) for ah in allowed_hosts):
                    raise HTTPException(status_code=400, detail=f"Redirect off allow-list to '{h}' rejected")
                return super().redirect_request(req_obj, fp, code, msg, headers, newurl)

        opener = urllib.request.build_opener(SafeRedirectHandler())
        req_head = urllib.request.Request(url, headers={"User-Agent": "Argus/0.1.0"})
        try:
            with opener.open(req_head, timeout=30) as resp:
                content_len = resp.headers.get("Content-Length")
                if content_len and int(content_len) > size_cap_bytes:
                    raise HTTPException(status_code=400, detail=f"Dataset exceeds {size_cap_bytes // (1024*1024)} MB size cap")

                content_type = (resp.headers.get("Content-Type") or "").lower()

                # Stream with size cap
                data = bytearray()
                while True:
                    chunk = resp.read(65536)
                    if not chunk:
                        break
                    data.extend(chunk)
                    if len(data) > size_cap_bytes:
                        raise HTTPException(status_code=400, detail=f"Dataset exceeds {size_cap_bytes // (1024*1024)} MB size cap")
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Failed to download dataset: {exc}")

        clean_project = re.sub(r"[^a-zA-Z0-9_\-]", "_", req.project.strip()) or "default"
        dest_dir = request.app.state.library_dir / clean_project
        dest_dir.mkdir(parents=True, exist_ok=True)

        is_zip = data.startswith(b"PK\x03\x04") or "zip" in content_type or url.lower().endswith(".zip")
        extracted_files = []

        if is_zip:
            try:
                zf = zipfile.ZipFile(io.BytesIO(data))
            except Exception as exc:
                raise HTTPException(status_code=400, detail=f"Invalid zip archive: {exc}")

            entries = zf.infolist()
            image_entries = [e for e in entries if not e.is_dir()]
            if len(image_entries) > file_count_cap:
                raise HTTPException(status_code=400, detail=f"Archive exceeds {file_count_cap} file cap (found {len(image_entries)})")

            total_uncompressed = sum(e.file_size for e in image_entries)
            if total_uncompressed > size_cap_bytes:
                raise HTTPException(status_code=400, detail=f"Archive uncompressed size exceeds {size_cap_bytes // (1024*1024)} MB cap")

            allowed_img_exts = {"jpg", "jpeg", "png", "webp", "heic"}
            for e in image_entries:
                ename = e.filename
                if ".." in ename or ename.startswith("/") or ename.startswith("\\"):
                    raise HTTPException(status_code=400, detail=f"Path traversal in archive rejected: {ename}")
                ext = Path(ename).suffix.lower().lstrip(".")
                if ext not in allowed_img_exts:
                    raise HTTPException(status_code=400, detail=f"Non-image file in archive rejected: {ename}")

            for e in image_entries:
                safe_name = Path(e.filename).name
                out_path = dest_dir / safe_name
                out_path.write_bytes(zf.read(e))
                extracted_files.append(str(out_path))
        else:
            # Direct image file
            try:
                im = Image.open(io.BytesIO(data))
                im.verify()
                fmt = (im.format or "JPEG").lower()
            except Exception:
                raise HTTPException(status_code=400, detail="Non-image content type rejected")

            base_name = Path(urllib.parse.urlparse(url).path).name or f"dataset_{uuid.uuid4().hex[:8]}.{fmt}"
            if "." not in base_name:
                base_name = f"{base_name}.{fmt}"
            out_path = dest_dir / base_name
            out_path.write_bytes(data)
            extracted_files.append(str(out_path))

        job_id = launch_pipeline_job(request.app, source_type="local", folder=dest_dir)
        return {
            "job_id": job_id,
            "project": clean_project,
            "extracted_count": len(extracted_files),
        }

    @app.post("/sources/rescan")
    def rescan_sources(
        request: Request = None,
        conn: sqlite3.Connection = Depends(get_db),
    ) -> dict[str, Any]:
        """Re-run the deduplication pipeline on existing sources."""
        has_gdrive = conn.execute("SELECT 1 FROM images WHERE source = 'gdrive' LIMIT 1").fetchone() is not None
        source_type = "drive" if has_gdrive else "local"
        job_id = launch_pipeline_job(request.app, source_type=source_type, folder=request.app.state.library_dir)
        return {
            "job_id": job_id,
            "status": "queued",
            "source_type": source_type,
        }

    @app.get("/jobs/active")
    def get_active_job(
        request: Request = None,
    ) -> dict[str, Any]:
        """Return the currently running or queued pipeline job if one is active."""
        now = time.time()
        jobs_dir = request.app.state.jobs_dir
        if jobs_dir.is_dir():
            try:
                files = sorted(jobs_dir.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True)
                for jf in files:
                    state = read_job_state(jf)
                    if state and state.get("status") in ("running", "queued"):
                        pid = state.get("pid")
                        is_alive = True
                        if pid is not None:
                            try:
                                os.kill(pid, 0)
                            except OSError:
                                is_alive = False
                        else:
                            # Stale fallback: if no PID and no update in > 60s, job is dead
                            if (now - state.get("updated_at", 0)) > 60:
                                is_alive = False

                        if not is_alive:
                            state["status"] = "failed"
                            state["error"] = "Worker process is no longer running"
                            write_job_state(jf, state)
                            continue

                        job_id = state.get("id") or state.get("job_id") or jf.stem
                        stage = state.get("stage_display") or state.get("stage") or "running"
                        done = int(state.get("current_file", 0))
                        total = int(state.get("total_files", 0))
                        return {
                            "active": True,
                            "job_id": str(job_id),
                            "stage": str(stage),
                            "done": done,
                            "total": total,
                        }
            except Exception:
                pass

        for j_id, j_data in sorted(
            request.app.state.jobs.items(),
            key=lambda item: item[1].get("created_at", 0),
            reverse=True,
        ):
            if j_data.get("status") in ("running", "queued"):
                job_id = j_data.get("id") or j_data.get("job_id") or j_id
                stage = j_data.get("stage_display") or j_data.get("stage") or "running"
                done = int(j_data.get("current_file", 0))
                total = int(j_data.get("total_files", 0))
                return {
                    "active": True,
                    "job_id": str(job_id),
                    "stage": str(stage),
                    "done": done,
                    "total": total,
                }

        return {"active": False}

    @app.get("/jobs/{job_id}")
    def get_job_status(
        job_id: str,
        request: Request = None,
    ) -> dict[str, Any]:
        """Poll the status and progress of a background pipeline job."""
        jobs_dir = request.app.state.jobs_dir
        job_file = jobs_dir / f"{job_id}.json"
        disk_job = read_job_state(job_file)
        if disk_job:
            return disk_job

        job = request.app.state.jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
        return job

    return app


# Module-level app instance with protected initialization
try:
    app = create_app()
except Exception:
    app = FastAPI(title="Argus API", version="0.1.0")

