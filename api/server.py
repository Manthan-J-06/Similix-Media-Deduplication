"""
api/server.py — FastAPI REST API for Argus.

Binds to 127.0.0.1 only. No CORS for external origins.
All file moves/deletions are strictly routed through core/ledger.
Authentication, session management, and per-user isolation enforced.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
import base64
import hashlib
import io
import json
import logging
import math
import os
from pathlib import Path
import re
import secrets
import shutil
import signal
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Optional
import urllib.parse
import urllib.request
import uuid
import zipfile

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse, StreamingResponse
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from core.auth import (
    authenticate_user,
    check_login_rate_limit,
    create_session,
    delete_user_account,
    get_session_user,
    get_user_dirs,
    hash_password,
    init_auth_db,
    invalidate_session,
    register_user,
    validate_confined_user_path,
)
from core.categories import search_images_by_text
from core.config import load_config
from core.keeper import choose_keeper
from core.ledger import (
    ensure_ledger_tables,
    execute,
    get_history,
    normalize_refusal_reason,
    plan,
    undo,
)
from db.connection import get_connection, init_db

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


def _thumb_cache_key(image_id: int, md5: str | None, mtime: float | None, size: int) -> str:
    """Content-derived cache filename: {image_id}_{md5[:12]}_{mtime_int}_{size}.jpg
    Falls back to zeros when md5/mtime are absent (e.g., Drive images without metadata).
    """
    md5_part = (md5 or "")[:12] or "000000000000"
    mtime_int = int(mtime or 0)
    return f"{image_id}_{md5_part}_{mtime_int}_{size}.jpg"


def _create_placeholder_jpeg() -> bytes:
    """Generate a small neutral gray JPEG placeholder for failed thumbnails."""
    buf = io.BytesIO()
    im = Image.new("RGB", (64, 64), color=(180, 180, 180))
    im.save(buf, format="JPEG", quality=75)
    return buf.getvalue()


PLACEHOLDER_JPEG_BYTES = _create_placeholder_jpeg()


class RegisterRequest(BaseModel):
    email: str
    password: str
    model_config = ConfigDict(extra="ignore")


class LoginRequest(BaseModel):
    email: str
    password: str
    model_config = ConfigDict(extra="ignore")


class DeleteAccountRequest(BaseModel):
    password: str
    model_config = ConfigDict(extra="ignore")


class PlanRequest(BaseModel):
    mode: str = "conservative"
    approved_swap_cluster_ids: list[int] = Field(default_factory=list)
    excluded_ids: list[int] = Field(default_factory=list)
    manual_member_ids: list[int] = Field(default_factory=list)
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


class ClusterCategoryRequest(BaseModel):
    category: Optional[str] = None
    model_config = ConfigDict(extra="ignore")


class DownloadSelectedRequest(BaseModel):
    image_ids: list[int] = Field(..., min_length=1)
    model_config = ConfigDict(extra="ignore")


@dataclass
class UserContext:
    id: int
    email: str
    csrf_token: str
    root: Path
    db_path: Path
    library_dir: Path
    trash_dir: Path
    thumb_cache_dir: Path
    jobs_dir: Path
    drive_cache_dir: Path

    @property
    def user_id(self) -> int:
        return self.id


def get_user_storage_usage(user_root: Path) -> int:
    """Calculate total byte size of non-symlink files in user directory tree."""
    total = 0
    if not user_root.is_dir():
        return 0
    for dirpath, _, filenames in os.walk(user_root):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):
                try:
                    total += os.path.getsize(fp)
                except OSError:
                    pass
    return total


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

    if re.fullmatch(r"[a-zA-Z0-9_-]{20,60}", link):
        return link
    raise HTTPException(status_code=400, detail=f"Invalid Drive folder link or ID: {link}")


def find_drive_client_secret(config: dict | None = None, data_dir: Path | None = None) -> tuple[Path | None, dict | None, list[Path]]:
    """Find and read OAuth client_secret.json across configuration and data directories.

    Returns a 3-tuple: (path, info_dict, searched_paths_list).
    ``path`` and ``info_dict`` are ``None`` when no valid file is found.
    ``searched_paths_list`` always contains every candidate that was checked.
    """
    cfg = config or {}
    candidates: list[Path] = []
    cfg_path = cfg.get("sources", {}).get("google_drive", {}).get("client_secret_path")
    if cfg_path:
        candidates.append(Path(cfg_path).expanduser())
    if data_dir:
        candidates.append(data_dir / "client_secret.json")
    env_data_dir = os.environ.get("ARGUS_DATA_DIR")
    if env_data_dir:
        candidates.append(Path(env_data_dir).expanduser() / "client_secret.json")
    candidates.append(Path.home() / ".argus" / "client_secret.json")
    candidates.append(Path("client_secret.json").resolve())

    for cand in candidates:
        if cand.is_file():
            try:
                with open(cand, "r", encoding="utf-8") as f:
                    data = json.load(f)
                info = data.get("web") or data.get("installed") or {}
                if "client_id" in info:
                    return cand, info, candidates
            except Exception:
                continue
    return None, None, candidates


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


def count_active_jobs_global(app: FastAPI) -> tuple[int, int]:
    """Count running and queued jobs across all users."""
    running = 0
    queued = 0
    now = time.time()

    users_dir = getattr(app.state, "users_dir", None)
    job_dirs: list[Path] = []
    if users_dir and users_dir.is_dir():
        for u in users_dir.iterdir():
            if u.is_dir():
                jd = u / "jobs"
                if jd.is_dir():
                    job_dirs.append(jd)

    custom_jd = getattr(app.state, "jobs_dir", None)
    if custom_jd and custom_jd.is_dir() and custom_jd not in job_dirs:
        job_dirs.append(custom_jd)

    for jd in job_dirs:
        for jf in jd.glob("*.json"):
            st = read_job_state(jf)
            if not st:
                continue
            status = st.get("status")
            if status == "running":
                pid = st.get("pid")
                alive = True
                if pid:
                    try:
                        os.kill(pid, 0)
                    except OSError:
                        alive = False
                last_hb = st.get("heartbeat_at") or st.get("updated_at", 0)
                if alive and (now - last_hb) <= 60:
                    running += 1
            elif status == "queued":
                queued += 1

    return running, queued


def _start_job_process(
    app: FastAPI,
    job_id: str,
    job_file: Path,
    db_path: Path,
    source_type: str,
    folder: Path | None,
    drive_folder_id: str | None,
) -> None:
    custom_worker = getattr(app.state, "pipeline_worker", None)
    if custom_worker is not None:
        t = threading.Thread(
            target=custom_worker,
            args=(job_id, app.state, source_type, folder, drive_folder_id),
            daemon=True,
        )
        t.start()
        return

    cmd = [
        sys.executable,
        "-m",
        "core.worker",
        "--job-id", job_id,
        "--db", str(db_path),
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

    log_file_path = job_file.with_suffix(".log")
    log_f = open(log_file_path, "a", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    log_f.close()
    st = read_job_state(job_file) or {}
    st["pid"] = proc.pid
    write_job_state(job_file, st)
    app.state.jobs[job_id] = st


def maybe_start_next_queued_job(app: FastAPI) -> None:
    """Check if any global worker slot is free and promote the oldest queued job."""
    with app.state.job_queue_lock:
        max_concurrent = app.state.config.get("general", {}).get("max_concurrent_jobs", 1)
        running_count, _ = count_active_jobs_global(app)
        if running_count >= max_concurrent:
            return

        users_dir = getattr(app.state, "users_dir", None)
        job_dirs: list[Path] = []
        if users_dir and users_dir.is_dir():
            for u in users_dir.iterdir():
                if u.is_dir():
                    jd = u / "jobs"
                    if jd.is_dir():
                        job_dirs.append(jd)
        custom_jd = getattr(app.state, "jobs_dir", None)
        if custom_jd and custom_jd.is_dir() and custom_jd not in job_dirs:
            job_dirs.append(custom_jd)

        queued_items = []
        for jd in job_dirs:
            for jf in jd.glob("*.json"):
                st = read_job_state(jf)
                if st and st.get("status") == "queued":
                    queued_items.append((st.get("created_at", 0), jf, st))

        queued_items.sort(key=lambda x: x[0])
        if not queued_items:
            return

        _, next_jf, next_st = queued_items[0]
        next_job_id = next_st["id"]
        next_st["status"] = "running"
        next_st["stage"] = "queued"
        next_st["stage_display"] = "Queued"
        next_st["message"] = "Job starting"
        write_job_state(next_jf, next_st)
        app.state.jobs[next_job_id] = next_st

        target_folder = Path(next_st["target_folder"]) if next_st.get("target_folder") else None
        db_path = Path(next_st.get("db_path", app.state.db_path))
        _start_job_process(
            app=app,
            job_id=next_job_id,
            job_file=next_jf,
            db_path=db_path,
            source_type=next_st.get("source_type", "local"),
            folder=target_folder,
            drive_folder_id=next_st.get("drive_folder_id"),
        )



def stop_user_jobs(app: FastAPI, user_id: int, jobs_dir: Path) -> None:
    """Terminate running worker processes and cancel queued jobs for the user before account deletion."""
    with app.state.job_queue_lock:
        if jobs_dir.is_dir():
            for jf in jobs_dir.glob("*.json"):
                st = read_job_state(jf)
                if not st:
                    continue
                pid = st.get("pid")
                if pid:
                    try:
                        os.kill(pid, signal.SIGTERM)
                    except OSError:
                        pass
                st["status"] = "canceled"
                st["stage"] = "canceled"
                st["message"] = "Canceled due to account deletion"
                write_job_state(jf, st)

        for jid, st in list(app.state.jobs.items()):
            if st.get("user_id") == user_id:
                pid = st.get("pid")
                if pid:
                    try:
                        os.kill(pid, signal.SIGTERM)
                    except OSError:
                        pass
                del app.state.jobs[jid]

    maybe_start_next_queued_job(app)


def launch_pipeline_job(
    app: FastAPI,
    user: UserContext | None = None,
    source_type: str = "local",
    folder: Path | None = None,
    drive_folder_id: str | None = None,
) -> str:
    """
    Launch pipeline execution in a separate OS process so heavy PyTorch/SIFT
    computation never starves FastAPI's event loop or request handling.
    Enforces 1 running job per user and global max concurrent jobs with a queue.
    """
    if user is None:
        overrides = getattr(app.state, "custom_user_overrides", {}).get(1)
        user_dirs = get_user_dirs(app.state.users_dir, 1, custom_overrides=overrides)
        jobs_dir = user_dirs["jobs_dir"]
        db_path = user_dirs["db_path"]
        user_id = 1
    else:
        jobs_dir = user.jobs_dir
        db_path = user.db_path
        user_id = user.id

    now = time.time()
    # Check if user already has an active (running/queued) job
    if jobs_dir.is_dir():
        for jf in jobs_dir.glob("*.json"):
            st = read_job_state(jf)
            if st and st.get("status") in ("running", "queued"):
                pid = st.get("pid")
                alive = True
                if pid:
                    try:
                        os.kill(pid, 0)
                    except OSError:
                        alive = False
                last_hb = st.get("heartbeat_at") or st.get("updated_at", 0)
                if alive and (now - last_hb) <= 60:
                    raise HTTPException(
                        status_code=409,
                        detail="A pipeline job is already active for this user",
                    )

    job_id = uuid.uuid4().hex[:12]
    jobs_dir.mkdir(parents=True, exist_ok=True)
    job_file = jobs_dir / f"{job_id}.json"

    max_concurrent = app.state.config.get("general", {}).get("max_concurrent_jobs", 1)
    with app.state.job_queue_lock:
        running_count, queued_count = count_active_jobs_global(app)
        if running_count < max_concurrent:
            status = "running"
            stage = "queued"
            stage_display = "Queued"
            msg = "Job queued"
        else:
            status = "queued"
            stage = "queued"
            queue_pos = queued_count + 1
            stage_display = f"Queued (position {queue_pos})"
            msg = f"Queued (position {queue_pos})"

        initial_job_data = {
            "id": job_id,
            "user_id": user_id,
            "status": status,
            "stage": stage,
            "stage_display": stage_display,
            "current_file": 0,
            "total_files": 0,
            "progress": 0.0,
            "message": msg,
            "source_type": source_type,
            "target_folder": str(folder) if folder else None,
            "drive_folder_id": drive_folder_id,
            "db_path": str(db_path),
            "warnings": [],
            "error": None,
            "created_at": time.time(),
            "updated_at": time.time(),
        }
        write_job_state(job_file, initial_job_data)
        app.state.jobs[job_id] = initial_job_data

        if status == "running":
            _start_job_process(
                app=app,
                job_id=job_id,
                job_file=job_file,
                db_path=db_path,
                source_type=source_type,
                folder=folder,
                drive_folder_id=drive_folder_id,
            )

    return job_id


def get_current_user(request: Request) -> UserContext:
    """Validate session cookie or Bearer token, enforce CSRF on POST, and resolve user dirs."""
    token = request.cookies.get("argus_session")
    if not token:
        auth_h = request.headers.get("Authorization", "")
        if auth_h.startswith("Bearer "):
            token = auth_h[7:].strip()

    if not token:
        logger.warning("Auth failure (401): No session cookie or Bearer token on %s %s", request.method, request.url.path)
        raise HTTPException(status_code=401, detail="Authentication required")

    auth_conn = getattr(request.app.state, "auth_conn", None)
    if not auth_conn:
        raise HTTPException(status_code=500, detail="Auth database connection not available")

    session_data = get_session_user(auth_conn, token)
    if not session_data:
        logger.warning("Auth failure (401): Invalid or expired session token on %s %s", request.method, request.url.path)
        raise HTTPException(status_code=401, detail="Invalid or expired session")

    # Enforce CSRF token on every POST request except public login/register
    if request.method == "POST":
        path = request.url.path
        if path not in ("/auth/login", "/auth/register", "/auth/csrf"):
            csrf_header = request.headers.get("X-CSRF-Token") or request.headers.get("x-csrf-token")
            if not csrf_header or not secrets.compare_digest(csrf_header, session_data["csrf_token"]):
                logger.warning("Auth failure (403): Missing or invalid CSRF token on %s %s (has_csrf: %s)", request.method, request.url.path, bool(csrf_header))
                raise HTTPException(status_code=403, detail="Invalid or missing CSRF token")

    user_id = session_data["user_id"]
    overrides = getattr(request.app.state, "custom_user_overrides", {}).get(user_id)
    user_dirs = get_user_dirs(request.app.state.users_dir, user_id, custom_overrides=overrides)

    db_p = user_dirs["db_path"]
    if not hasattr(request.app.state, "_initialized_user_dbs"):
        request.app.state._initialized_user_dbs = set()
    if db_p not in request.app.state._initialized_user_dbs:
        uconn = init_db(db_p)
        ensure_ledger_tables(uconn)
        uconn.close()
        request.app.state._initialized_user_dbs.add(db_p)

    return UserContext(
        id=user_id,
        email=session_data["email"],
        csrf_token=session_data["csrf_token"],
        root=user_dirs["root"],
        db_path=user_dirs["db_path"],
        library_dir=user_dirs["library_dir"],
        trash_dir=user_dirs["trash_dir"],
        thumb_cache_dir=user_dirs["thumb_cache_dir"],
        jobs_dir=user_dirs["jobs_dir"],
        drive_cache_dir=user_dirs["drive_cache_dir"],
    )


def get_optional_user(request: Request) -> UserContext | None:
    """Return UserContext if session is valid, otherwise None without raising HTTPException."""
    try:
        return get_current_user(request)
    except HTTPException:
        return None


def get_db(request: Request, user: UserContext = Depends(get_current_user)):
    """Provide a per-request SQLite connection for the authenticated user."""
    conn = get_connection(user.db_path)
    try:
        yield conn
    finally:
        conn.close()


def _get_user_gdrive_source(request: Request, user: UserContext, conn: sqlite3.Connection):
    """Instantiate GoogleDriveSource for the authenticated user if credentials exist."""
    try:
        from sources.google_drive import GoogleDriveSource, DEFAULT_TOKEN_PATH
        cfg = getattr(request.app.state, "config", {}) or {}
        configured_drive_cache = cfg.get("sources", {}).get("google_drive", {}).get("cache_dir")
        drive_cache_dir = Path(configured_drive_cache).expanduser() if configured_drive_cache else user.drive_cache_dir
        user_tok_path = user.root / "drive_token.json"
        global_tok_path = Path(cfg.get("sources", {}).get("google_drive", {}).get("token_path", DEFAULT_TOKEN_PATH)).expanduser()
        tok_file = user_tok_path if user_tok_path.exists() else (global_tok_path if global_tok_path.exists() else None)
        if tok_file and tok_file.exists():
            return GoogleDriveSource(
                conn=conn,
                cache_dir=drive_cache_dir,
                token_path=tok_file,
                config=cfg,
                stage="cluster_verification",
            )
    except Exception as exc:
        logger.warning("Could not instantiate GoogleDriveSource: %s", exc)
    return None


def _collect_download_file(
    file_path_str: str,
    arcname_default: str,
    user: UserContext,
    request: Request,
    gdrive_source: Any = None,
) -> tuple[Path | bytes, str] | None:
    """Resolve a local path or drive:// URI to (Path | bytes, arcname)."""
    if file_path_str.startswith("drive://"):
        from sources.google_drive import parse_drive_uri
        drive_id, filename = parse_drive_uri(file_path_str)
        arcname = filename or arcname_default or f"{drive_id}.jpg"
        if gdrive_source is not None:
            try:
                b = gdrive_source.get_bytes(file_path_str)
                if b:
                    return (b, arcname)
            except Exception as exc:
                logger.warning("Could not fetch full Google Drive file %s: %s", file_path_str, exc)
        cfg = getattr(request.app.state, "config", {}) or {}
        configured_drive_cache = cfg.get("sources", {}).get("google_drive", {}).get("cache_dir")
        drive_cache_dir = Path(configured_drive_cache).expanduser() if configured_drive_cache else user.drive_cache_dir
        if drive_cache_dir.is_dir():
            matches = sorted(drive_cache_dir.glob(f"{drive_id}_*.jpg"))
            if not matches:
                matches = sorted(drive_cache_dir.glob(f"{drive_id}.jpg"))
            if not matches:
                matches = sorted(drive_cache_dir.glob(f"{drive_id}*"))
            if matches and matches[0].is_file():
                return (matches[0], arcname)
        return None

    fp = Path(file_path_str)
    if fp.is_file() and not fp.is_symlink():
        try:
            validate_confined_user_path(fp, user.library_dir)
            return (fp, arcname_default or fp.name)
        except Exception:
            try:
                validate_confined_user_path(fp, user.root)
                return (fp, arcname_default or fp.name)
            except Exception:
                pass
    return None


def stream_zip_archive(files_to_zip: list[tuple[Path | bytes, str]], zip_filename: str):
    """Create and stream a zip archive from a list of (Path | bytes, arcname) tuples."""
    if not files_to_zip:
        raise HTTPException(status_code=404, detail="No downloadable files found")

    max_download_size = 2 * 1024 * 1024 * 1024  # 2 GB cap
    total_size = 0
    for item, _ in files_to_zip:
        if isinstance(item, (bytes, bytearray)):
            total_size += len(item)
        elif isinstance(item, Path) and item.is_file() and not item.is_symlink():
            try:
                total_size += item.stat().st_size
            except OSError:
                pass

    if total_size > max_download_size:
        raise HTTPException(
            status_code=400,
            detail=f"Total download size ({total_size // (1024*1024)} MB) exceeds 2GB cap",
        )

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
    temp_path = Path(temp_file.name)
    temp_file.close()

    try:
        with zipfile.ZipFile(temp_path, "w", zipfile.ZIP_DEFLATED) as zf:
            seen_arcnames: set[str] = set()
            for item, arcname in files_to_zip:
                final_arcname = arcname
                count = 1
                while final_arcname in seen_arcnames:
                    p = Path(arcname)
                    final_arcname = f"{p.stem}_{count}{p.suffix}"
                    count += 1
                seen_arcnames.add(final_arcname)
                if isinstance(item, (bytes, bytearray)):
                    zf.writestr(final_arcname, item)
                elif isinstance(item, Path) and item.is_file() and not item.is_symlink():
                    zf.write(item, arcname=final_arcname)
    except Exception as exc:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass
        raise HTTPException(status_code=500, detail=f"Failed to create zip archive: {exc}")

    def iter_file():
        try:
            with open(temp_path, "rb") as f:
                while chunk := f.read(65536):
                    yield chunk
        finally:
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception:
                    pass

    return StreamingResponse(
        iter_file(),
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="{zip_filename}"',
            "Cache-Control": "no-store",
        },
    )


def create_app(
    db_path: str | Path | None = None,
    trash_dir: str | Path | None = None,
    thumb_cache_dir: str | Path | None = None,
    library_dir: str | Path | None = None,
    jobs_dir: str | Path | None = None,
    config: dict | None = None,
    config_path: str | Path | None = None,
    auth_db_path: str | Path | None = None,
    users_dir: str | Path | None = None,
) -> FastAPI:
    """Create and configure the Argus FastAPI application with per-user isolation."""
    if config is None:
        try:
            config = load_config(config_path) if config_path else load_config()
        except Exception:
            config = {}

    raw_data_dir = os.environ.get("ARGUS_DATA_DIR") or config.get("general", {}).get("data_dir")
    if raw_data_dir:
        data_root = Path(raw_data_dir).expanduser().resolve()
        data_root.mkdir(parents=True, exist_ok=True)
    else:
        cand = Path.home() / ".argus"
        try:
            cand.mkdir(parents=True, exist_ok=True)
            test_f = cand / ".perm_check"
            test_f.touch()
            test_f.unlink()
            data_root = cand.resolve()
        except (OSError, PermissionError):
            data_root = Path.cwd() / ".argus"
            data_root.mkdir(parents=True, exist_ok=True)

    explicit_db_path = db_path is not None
    if not explicit_db_path:
        db_path = os.environ.get("ARGUS_DB_PATH") or (data_root / "argus.db" if raw_data_dir else (config.get("database", {}).get("path") or (data_root / "argus.db")))
    resolved_db_path = Path(db_path).expanduser().resolve()

    if trash_dir is None:
        trash_dir = os.environ.get("ARGUS_TRASH_DIR") or (data_root / "trash" if raw_data_dir else (config.get("ledger", {}).get("trash_dir") or (data_root / "trash")))
    resolved_trash_dir = Path(trash_dir).expanduser().resolve()
    resolved_trash_dir.mkdir(parents=True, exist_ok=True)

    if thumb_cache_dir is None:
        thumb_cache_dir = os.environ.get("ARGUS_THUMB_CACHE_DIR") or (data_root / "thumbnails")
    resolved_thumb_cache_dir = Path(thumb_cache_dir).expanduser().resolve()
    resolved_thumb_cache_dir.mkdir(parents=True, exist_ok=True)

    if library_dir is None:
        library_dir = os.environ.get("ARGUS_LIBRARY_DIR") or (data_root / "library" if raw_data_dir else (config.get("sources", {}).get("library_dir") or (data_root / "library")))
    resolved_library_dir = Path(library_dir).expanduser().resolve()
    resolved_library_dir.mkdir(parents=True, exist_ok=True)

    if jobs_dir is None:
        jobs_dir = os.environ.get("ARGUS_JOBS_DIR") or (data_root / "jobs" if raw_data_dir else (config.get("general", {}).get("jobs_dir") or (data_root / "jobs")))
    resolved_jobs_dir = Path(jobs_dir).expanduser().resolve()
    resolved_jobs_dir.mkdir(parents=True, exist_ok=True)

    # Per-user storage directories root
    if users_dir is None:
        users_dir = os.environ.get("ARGUS_USERS_DIR") or (data_root / "users" if raw_data_dir else (config.get("general", {}).get("users_dir") or (data_root / "users")))
    resolved_users_dir = Path(users_dir).expanduser().resolve()
    resolved_users_dir.mkdir(parents=True, exist_ok=True)

    # Auth DB initialization
    if auth_db_path is None:
        auth_db_path = os.environ.get("ARGUS_ACCOUNTS_DB") or os.environ.get("ARGUS_AUTH_DB") or (data_root / "auth.db" if raw_data_dir else (config.get("auth", {}).get("db_path") or (data_root / "auth.db")))
    resolved_auth_db_path = Path(auth_db_path).expanduser().resolve()
    resolved_auth_db_path.parent.mkdir(parents=True, exist_ok=True)
    auth_conn = init_auth_db(resolved_auth_db_path)

    logger.info("Argus fixed data root: %s", data_root)
    logger.info("Auth DB: %s, Users dir: %s", resolved_auth_db_path, resolved_users_dir)

    # Initialize default single-user / test DB
    init_conn = init_db(resolved_db_path)
    ensure_ledger_tables(init_conn)
    init_conn.close()

    # Hosted mode validation: must refuse to start without SECRET_KEY (minimum 32 chars, no default anywhere)
    is_hosted = bool(
        config.get("hosted_mode", False)
        or config.get("general", {}).get("hosted_mode", False)
        or os.environ.get("HOSTED_MODE", "").lower() in ("true", "1", "yes")
    )
    secret_key = os.environ.get("SECRET_KEY") or config.get("auth", {}).get("secret_key")
    if is_hosted:
        if not secret_key or not str(secret_key).strip():
            sys.stderr.write("ERROR: Hosted mode requires SECRET_KEY environment variable (minimum 32 characters). Refusing to start.\n")
            raise RuntimeError("Hosted mode requires SECRET_KEY to be set in environment or config (minimum 32 characters). Refusing to start.")
        if len(str(secret_key).strip()) < 32:
            sys.stderr.write(f"ERROR: SECRET_KEY must be at least 32 characters in hosted mode (got {len(str(secret_key).strip())}). Refusing to start.\n")
            raise RuntimeError(f"SECRET_KEY must be at least 32 characters in hosted mode (got {len(str(secret_key).strip())}). Refusing to start.")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        secret_is_set = "yes" if bool(os.environ.get("SECRET_KEY") or config.get("auth", {}).get("secret_key")) else "no"
        sys.stdout.write(
            f"\n==================================================\n"
            f"  ARGUS Startup Self-Check\n"
            f"==================================================\n"
            f"  Data Directory : {data_root}\n"
            f"  DB Path        : {resolved_db_path}\n"
            f"  Secret Key Set : {secret_is_set}\n"
            f"==================================================\n\n"
        )
        sys.stdout.flush()
        yield

    app = FastAPI(
        title="Argus API",
        version="0.1.0",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def add_security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; "
            "connect-src 'self'; "
            "frame-ancestors 'none'; "
            "base-uri 'self'; "
            "form-action 'self';"
        )
        # Requirement 3: All JSON API responses (/clusters, /clusters/{id}, /categories, /jobs/*,
        # /history, /auth/me) must send Cache-Control: no-store. Only /thumb may be cached.
        path = request.url.path
        if not path.startswith("/thumb/"):
            content_type = response.headers.get("content-type", "")
            if (
                path.startswith(("/clusters", "/categories", "/jobs", "/history", "/auth", "/plan", "/apply", "/undo", "/sources", "/library", "/download"))
                or "application/json" in content_type
            ):
                response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
                response.headers["Pragma"] = "no-cache"
        return response

    allowed_origins = [
        "http://127.0.0.1",
        "http://127.0.0.1:8000",
        "http://localhost",
        "http://localhost:8000",
    ]
    base_url = os.environ.get("BASE_URL")
    if base_url:
        clean_base = base_url.strip().rstrip("/")
        if clean_base and clean_base not in allowed_origins:
            allowed_origins.append(clean_base)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    app.state.data_root = data_root
    app.state.auth_conn = auth_conn
    app.state.auth_db_path = resolved_auth_db_path
    app.state.users_dir = resolved_users_dir
    app.state.db_path = resolved_db_path
    app.state.trash_dir = resolved_trash_dir
    app.state.thumb_cache_dir = resolved_thumb_cache_dir
    app.state.library_dir = resolved_library_dir
    app.state.jobs_dir = resolved_jobs_dir
    app.state.config = config
    app.state.config_path = config_path
    app.state.jobs = {}
    app.state.job_queue_lock = threading.Lock()
    app.state._initialized_user_dbs = {resolved_db_path}

    if explicit_db_path:
        app.state.custom_user_overrides = {
            1: {
                "root": (resolved_db_path.parent if resolved_db_path else resolved_library_dir),
                "db_path": resolved_db_path,
                "trash_dir": resolved_trash_dir,
                "thumb_cache_dir": resolved_thumb_cache_dir,
                "library_dir": resolved_library_dir,
                "jobs_dir": resolved_jobs_dir,
                "drive_cache_dir": resolved_thumb_cache_dir.parent / "drive_cache",
            }
        }
    else:
        app.state.custom_user_overrides = {}

    ui_file = Path(__file__).resolve().parent.parent / "ui" / "index.html"
    login_file = Path(__file__).resolve().parent.parent / "ui" / "login.html"

    # ==========================
    # Health & Authentication Endpoints
    # ==========================

    @app.get("/healthz")
    def health_check():
        """Public health check endpoint for reverse proxies and orchestrators."""
        return {"status": "ok", "service": "argus"}

    @app.get("/login")
    def serve_login():
        """Serve the standalone login/registration web page (public)."""
        if not login_file.is_file():
            raise HTTPException(status_code=404, detail="Login page not found")
        return FileResponse(str(login_file), media_type="text/html")

    @app.get("/auth/csrf")
    def get_csrf(request: Request):
        """Public endpoint returning CSRF token for the active session (if any)."""
        token = request.cookies.get("argus_session")
        csrf_tok = ""
        if token and hasattr(request.app.state, "auth_conn"):
            s = get_session_user(request.app.state.auth_conn, token)
            if s:
                csrf_tok = s["csrf_token"]
        return {"csrf_token": csrf_tok}

    @app.post("/auth/register")
    def register(req: RegisterRequest, request: Request):
        """Register a new user in auth.db (public, unless invite-only configured)."""
        cfg = getattr(request.app.state, "config", {}) or {}
        allow_reg = cfg.get("auth", {}).get("allow_registration", True)
        if cfg.get("invite_only", False) or cfg.get("auth", {}).get("invite_only", False):
            allow_reg = False
        if "ALLOW_REGISTRATION" in os.environ:
            allow_reg = os.environ.get("ALLOW_REGISTRATION", "").lower() in ("true", "1", "yes")

        res = register_user(
            request.app.state.auth_conn,
            email=req.email,
            password=req.password,
            allow_registration=allow_reg,
        )
        return {"status": "registered", "user": res}

    @app.post("/auth/login")
    def login(req: LoginRequest, request: Request, response: Response):
        """Authenticate user, set HttpOnly session cookie and return CSRF token (public)."""
        client_ip = request.client.host if request.client else "127.0.0.1"
        user_data = authenticate_user(
            request.app.state.auth_conn,
            email=req.email,
            password=req.password,
            ip=client_ip,
        )
        raw_token, csrf_tok = create_session(request.app.state.auth_conn, user_data["id"])

        proto = request.headers.get("x-forwarded-proto", "").lower()
        is_secure = (proto == "https") or (request.url.scheme == "https")
        response.set_cookie(
            key="argus_session",
            value=raw_token,
            max_age=7 * 24 * 3600,
            httponly=True,
            samesite="lax",
            secure=is_secure,
        )
        response.set_cookie(
            key="argus_csrf",
            value=csrf_tok,
            max_age=7 * 24 * 3600,
            httponly=False,
            samesite="lax",
            secure=is_secure,
        )
        return {
            "status": "authenticated",
            "user": user_data,
            "token": raw_token,
            "csrf_token": csrf_tok,
        }

    @app.post("/auth/logout")
    def logout(
        request: Request,
        response: Response,
        user: UserContext = Depends(get_current_user),
    ):
        """Invalidate session in auth.db and clear auth cookies."""
        token = request.cookies.get("argus_session")
        if not token:
            auth_h = request.headers.get("Authorization", "")
            if auth_h.startswith("Bearer "):
                token = auth_h[7:].strip()
        if token:
            invalidate_session(request.app.state.auth_conn, token)
        response.delete_cookie("argus_session")
        response.delete_cookie("argus_csrf")
        return {"status": "logged_out"}

    @app.get("/auth/me")
    def get_me(user: UserContext = Depends(get_current_user)):
        """Return information about the currently authenticated user."""
        return {
            "id": user.id,
            "email": user.email,
            "csrf_token": user.csrf_token,
        }

    @app.post("/auth/delete-account")
    def delete_account(
        req: DeleteAccountRequest,
        request: Request = None,
        response: Response = None,
        user: UserContext = Depends(get_current_user),
    ) -> dict[str, Any]:
        """Delete user account, isolated directory on disk, and DB rows after password confirmation."""
        def _pre_delete():
            stop_user_jobs(request.app, user.id, user.jobs_dir)
            if hasattr(request.app.state, "_initialized_user_dbs"):
                request.app.state._initialized_user_dbs.discard(user.db_path)
            # Clear per-user thumbnail cache
            try:
                shutil.rmtree(user.thumb_cache_dir, ignore_errors=True)
            except Exception:
                pass

        res = delete_user_account(
            conn=request.app.state.auth_conn,
            user_id=user.user_id,
            password=req.password,
            users_dir=request.app.state.users_dir,
            on_before_delete=_pre_delete,
        )

        if response is not None:
            response.delete_cookie("argus_session")
            response.delete_cookie("argus_csrf")
        return res

    @app.delete("/auth/me")
    def delete_account_me(
        req: DeleteAccountRequest,
        request: Request = None,
        response: Response = None,
        user: UserContext = Depends(get_current_user),
    ) -> dict[str, Any]:
        """Alias for /auth/delete-account using HTTP DELETE."""
        return delete_account(req, request=request, response=response, user=user)

    @app.get("/auth/export-data")
    def export_user_data(user: UserContext = Depends(get_current_user)):
        """Generate and stream a zip archive of the authenticated user's library directory."""
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            if user.library_dir.is_dir():
                for root, _, files in os.walk(user.library_dir):
                    for f in files:
                        fp = Path(root) / f
                        if not fp.is_symlink():
                            try:
                                arcname = fp.relative_to(user.library_dir)
                                zf.write(fp, arcname=str(arcname))
                            except Exception:
                                pass
        buf.seek(0)
        return Response(
            content=buf.getvalue(),
            media_type="application/zip",
            headers={"Content-Disposition": 'attachment; filename="argus_library.zip"'},
        )

    # ==========================
    # Downloads & Library
    # ==========================

    @app.get("/download/cluster/{cluster_id}")
    def download_cluster(
        cluster_id: int,
        request: Request = None,
        user: UserContext = Depends(get_current_user),
        conn: sqlite3.Connection = Depends(get_db),
    ):
        """Stream a zip archive of all live files in a cluster (local or Google Drive)."""
        c_row = conn.execute("SELECT id FROM clusters WHERE id = ?", (cluster_id,)).fetchone()
        if not c_row:
            raise HTTPException(status_code=404, detail=f"Cluster {cluster_id} not found")

        rows = conn.execute(
            "SELECT id, file_path, file_size FROM images WHERE cluster_id = ? AND (status IS NULL OR status != 'deleted')",
            (cluster_id,),
        ).fetchall()

        gdrive_source = _get_user_gdrive_source(request, user, conn)
        files_to_zip = []
        for r in rows:
            fp_str = str(r["file_path"])
            item = _collect_download_file(fp_str, Path(fp_str).name, user, request, gdrive_source)
            if item:
                files_to_zip.append(item)

        return stream_zip_archive(files_to_zip, f"cluster_{cluster_id}.zip")

    @app.post("/download/selected")
    def download_selected(
        req: DownloadSelectedRequest,
        request: Request = None,
        user: UserContext = Depends(get_current_user),
        conn: sqlite3.Connection = Depends(get_db),
    ):
        """Stream a zip archive of selected image IDs (local or Google Drive)."""
        if not req.image_ids:
            raise HTTPException(status_code=400, detail="No image IDs provided")

        placeholders = ",".join(["?"] * len(req.image_ids))
        rows = conn.execute(
            f"SELECT id, file_path, file_size FROM images WHERE id IN ({placeholders}) AND (status IS NULL OR status != 'deleted')",
            req.image_ids,
        ).fetchall()

        gdrive_source = _get_user_gdrive_source(request, user, conn)
        files_to_zip = []
        for r in rows:
            fp_str = str(r["file_path"])
            item = _collect_download_file(fp_str, Path(fp_str).name, user, request, gdrive_source)
            if item:
                files_to_zip.append(item)

        return stream_zip_archive(files_to_zip, "selected_images.zip")

    @app.get("/download/keepers")
    def download_keepers(
        mode: str = Query("balanced"),
        request: Request = None,
        user: UserContext = Depends(get_current_user),
        conn: sqlite3.Connection = Depends(get_db),
    ):
        """Stream a zip archive of the complete deduplicated dataset: all unique images plus the recommended keeper from each cluster (local or Google Drive)."""
        clusters = conn.execute("SELECT id, reference_uncertain FROM clusters").fetchall()
        gdrive_source = _get_user_gdrive_source(request, user, conn)

        keeper_files = []
        cluster_keeper_ids: set[int] = set()

        for c in clusters:
            cid = c["id"]
            ref_unc = c["reference_uncertain"] if "reference_uncertain" in c.keys() else 0
            members = conn.execute(
                "SELECT * FROM images WHERE cluster_id = ? AND (status IS NULL OR status NOT IN ('trashed', 'deleted'))",
                (cid,),
            ).fetchall()
            live_members = [
                m for m in members
                if str(m["file_path"]).startswith("drive://") or (os.path.exists(m["file_path"]) and not os.path.islink(m["file_path"]))
            ]
            if len(live_members) == 1:
                m = live_members[0]
                cluster_keeper_ids.add(m["id"])
                fp_str = str(m["file_path"])
                item = _collect_download_file(fp_str, Path(fp_str).name, user, request, gdrive_source)
                if item:
                    keeper_files.append(item)
                continue
            if len(live_members) < 2:
                continue

            member_dicts = [dict(m) for m in live_members]
            res = choose_keeper(
                member_dicts,
                mode=mode,
                config=request.app.state.config,
                reference_uncertain=ref_unc,
            )
            k_id = res.get("keeper_id")
            if k_id:
                cluster_keeper_ids.add(k_id)
                k_row = next((m for m in live_members if m["id"] == k_id), None)
                if k_row:
                    fp_str = str(k_row["file_path"])
                    item = _collect_download_file(fp_str, Path(fp_str).name, user, request, gdrive_source)
                    if item:
                        keeper_files.append(item)

        # Collect all unique images (not belonging to any duplicate cluster)
        unique_members = conn.execute(
            """
            SELECT id, file_path FROM images
            WHERE (cluster_id IS NULL OR cluster_id = 0)
              AND (status IS NULL OR status NOT IN ('trashed', 'deleted'))
            ORDER BY id
            """
        ).fetchall()
        for u in unique_members:
            if u["id"] in cluster_keeper_ids:
                continue
            fp_str = str(u["file_path"])
            item = _collect_download_file(fp_str, Path(fp_str).name, user, request, gdrive_source)
            if item:
                keeper_files.append(item)

        return stream_zip_archive(keeper_files, "recommended_keepers.zip")

    @app.get("/library")
    def list_library_images(
        page: int = Query(1, ge=1),
        limit: int = Query(60, ge=1, le=200),
        search: str | None = Query(None),
        category: str | None = Query(None),
        request: Request = None,
        user: UserContext = Depends(get_current_user),
        conn: sqlite3.Connection = Depends(get_db),
    ) -> dict[str, Any]:
        """
        List all images for the authenticated user (including unique, unclustered images),
        with pagination (default 60/page), search by filename, and category filtering.
        """
        conditions = ["(status IS NULL OR status NOT IN ('deleted', 'trashed'))"]
        params: list[Any] = []

        if search and search.strip():
            s = f"%{search.strip()}%"
            conditions.append("(file_path LIKE ?)")
            params.append(s)

        if category and category.strip():
            cat = category.strip()
            if cat in ("Needs label", "uncertain"):
                conditions.append("(category_top1 IS NULL OR category_top1 = 'Needs label' OR category_top1 = 'uncertain')")
            else:
                conditions.append("(category_top1 = ? OR category_top2 = ?)")
                params.extend([cat, cat])

        where_clause = " WHERE " + " AND ".join(conditions)

        total_row = conn.execute(f"SELECT COUNT(*) as cnt FROM images{where_clause}", params).fetchone()
        total = total_row["cnt"] if total_row else 0

        offset = (page - 1) * limit
        query_params = list(params) + [limit, offset]
        rows = conn.execute(
            f"""
            SELECT id, file_path, file_size, width, height, format, md5,
                   cluster_id, is_keeper, is_cluster_anchor,
                   category_top1, category_top1_prob, category_top2, category_top2_prob,
                   created_at
            FROM images
            {where_clause}
            ORDER BY id DESC
            LIMIT ? OFFSET ?
            """,
            query_params,
        ).fetchall()

        items = []
        for r in rows:
            top1 = r["category_top1"]
            top1_prob = r["category_top1_prob"]
            top2 = r["category_top2"]
            top2_prob = r["category_top2_prob"]
            cat_label = "Needs label" if top1 in ("uncertain", "Needs label") or not top1 else top1
            items.append({
                "id": r["id"],
                "filename": Path(r["file_path"]).name,
                "file_path": r["file_path"],
                "file_size": r["file_size"],
                "width": r["width"],
                "height": r["height"],
                "format": r["format"],
                "md5": r["md5"],
                "cluster_id": r["cluster_id"],
                "is_keeper": bool(r["is_keeper"]),
                "category": cat_label,
                "category_prob": top1_prob,
                "category_top1": top1,
                "category_top1_prob": top1_prob,
                "category_top2": top2,
                "category_top2_prob": top2_prob,
                "created_at": r["created_at"],
            })

        return {
            "total": total,
            "page": page,
            "limit": limit,
            "total_pages": math.ceil(total / limit) if limit > 0 else 1,
            "images": items,
        }

    # ==========================
    # Main Application UI
    # ==========================

    @app.get("/")
    def serve_ui(request: Request):
        """Serve the single-file Argus web UI if authenticated, else redirect to /login (302)."""
        user = get_optional_user(request)
        if not user:
            return RedirectResponse(url="/login", status_code=302)
        if not ui_file.is_file():
            raise HTTPException(status_code=404, detail="UI index.html not found")
        return FileResponse(str(ui_file), media_type="text/html")

    # ==========================
    # Clusters & Images
    # ==========================

    @app.get("/clusters")
    def list_clusters(
        mode: str = Query("conservative"),
        category: Optional[str] = Query(None),
        min_savings: Optional[float] = Query(None),
        sort: str = Query("id"),
        request: Request = None,
        response: Response = None,
        user: UserContext = Depends(get_current_user),
        conn: sqlite3.Connection = Depends(get_db),
    ) -> list[dict[str, Any]]:
        if mode not in ("conservative", "balanced", "aggressive"):
            raise HTTPException(
                status_code=400,
                detail="Invalid mode. Must be one of: conservative, balanced, aggressive",
            )

        total_images = conn.execute(
            "SELECT COUNT(*) FROM images WHERE status IS NULL OR status != 'trashed'"
        ).fetchone()[0]
        if response is not None:
            response.headers["X-Total-Images"] = str(total_images)

        c_rows = conn.execute(
            "SELECT id, representative_id, reference_uncertain, category, manual_category FROM clusters ORDER BY id"
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
                       category_top1_prob, category_top2, category_top2_prob, match_pct,
                       resolution_pct, confidence, status
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

            manual_cat = c_row["manual_category"] if "manual_category" in c_row.keys() else None
            keeper_m = next((m for m in member_dicts if m["id"] == keeper_id), None)
            if manual_cat:
                display_category = manual_cat
            else:
                cluster_cat = c_row["category"] or (keeper_m.get("category_top1") if keeper_m else None)

                # Majority-vote display category
                member_cats = [m["category_top1"] for m in member_dicts if m.get("category_top1")]
                if member_cats:
                    from collections import Counter
                    most_common, occ = Counter(member_cats).most_common(1)[0]
                    display_category = most_common if occ > len(member_cats) / 2 else "mixed"
                else:
                    display_category = cluster_cat

            if display_category in ("uncertain", "Needs label"):
                display_category = "Needs label"

            if category is not None:
                cat_norm = category.strip().lower()
                disp_norm = (display_category or "").strip().lower()
                if cat_norm in ("needs label", "uncertain", "needs_label"):
                    if disp_norm not in ("needs label", "uncertain"):
                        continue
                elif disp_norm != cat_norm:
                    continue

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
                    "md5": keeper_m.get("md5", "") if keeper_m else "",
                },
                "savings_bytes": savings_bytes,
                "reclaimable_bytes": savings_bytes,
                "savings_ratio": savings_ratio,
                "member_count": len(member_dicts),
                "delete_safe_count": len(del_safe_ids),
                "review_count": len(review_ids),
                "category": display_category,
                "category_top1": keeper_m.get("category_top1") if keeper_m else None,
                "category_top1_prob": keeper_m.get("category_top1_prob") if keeper_m else None,
                "category_top2": keeper_m.get("category_top2") if keeper_m else None,
                "category_top2_prob": keeper_m.get("category_top2_prob") if keeper_m else None,
                "manual_category": manual_cat,
                "has_swap_proposal": swap_prop is not None,
                "swap_proposal": swap_prop,
            })

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
        user: UserContext = Depends(get_current_user),
        conn: sqlite3.Connection = Depends(get_db),
    ) -> dict[str, Any]:
        if mode not in ("conservative", "balanced", "aggressive"):
            raise HTTPException(
                status_code=400,
                detail="Invalid mode. Must be one of: conservative, balanced, aggressive",
            )

        c_row = conn.execute(
            "SELECT id, representative_id, reference_uncertain, category, manual_category FROM clusters WHERE id = ?",
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
                   category_top1_prob, category_top2, category_top2_prob, match_pct,
                   resolution_pct, confidence, status
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
            top1 = m.get("category_top1")
            top1_prob = m.get("category_top1_prob")
            top2 = m.get("category_top2")
            top2_prob = m.get("category_top2_prob")
            cat_label = "Needs label" if top1 in ("uncertain", "Needs label") else top1
            formatted_members.append({
                "id": mid,
                "file_path": m["file_path"],
                "size": m["file_size"],
                "width": m["width"],
                "height": m["height"],
                "format": m["format"],
                "md5": m.get("md5", ""),
                "match_pct": m["match_pct"],
                "resolution_pct": m["resolution_pct"],
                "confidence": m["confidence"],
                "category": cat_label,
                "category_prob": top1_prob,
                "category_top1": top1,
                "category_top1_prob": top1_prob,
                "category_top2": top2,
                "category_top2_prob": top2_prob,
                "is_recommended_keeper": mid == keeper_id,
                "delete_safe": mid in del_safe_ids,
                "review": mid in review_ids,
                "reason": reasons.get(mid, ""),
            })

        keeper_m = next((m for m in formatted_members if m["is_recommended_keeper"]), None)
        manual_cat = c_row["manual_category"] if "manual_category" in c_row.keys() else None
        if manual_cat:
            display_category = manual_cat
        else:
            cluster_cat = c_row["category"] or (keeper_m.get("category") if keeper_m else None)

            member_cats = [m.get("category") for m in formatted_members if m.get("category")]
            if member_cats:
                from collections import Counter
                most_common, occ = Counter(member_cats).most_common(1)[0]
                display_category = most_common if occ > len(member_cats) / 2 else "mixed"
            else:
                display_category = cluster_cat

        if display_category in ("uncertain", "Needs label"):
            display_category = "Needs label"

        return {
            "id": c_row["id"],
            "mode": mode,
            "category": display_category,
            "category_top1": keeper_m.get("category_top1") if keeper_m else None,
            "category_top1_prob": keeper_m.get("category_top1_prob") if keeper_m else None,
            "category_top2": keeper_m.get("category_top2") if keeper_m else None,
            "category_top2_prob": keeper_m.get("category_top2_prob") if keeper_m else None,
            "manual_category": manual_cat,
            "reference_uncertain": bool(keeper_res.get("reference_uncertain", ref_unc)),
            "swap_proposal": keeper_res.get("swap_proposal"),
            "no_swap_reason": keeper_res.get("no_swap_reason"),
            "members": formatted_members,
        }

    @app.post("/clusters/{cluster_id}/category")
    def set_cluster_category(
        cluster_id: int,
        req: ClusterCategoryRequest,
        user: UserContext = Depends(get_current_user),
        conn: sqlite3.Connection = Depends(get_db),
    ) -> dict[str, Any]:
        c_row = conn.execute("SELECT id FROM clusters WHERE id = ?", (cluster_id,)).fetchone()
        if not c_row:
            raise HTTPException(status_code=404, detail=f"Cluster {cluster_id} not found")

        clean_cat = req.category.strip() if req.category else ""
        if not clean_cat:
            conn.execute(
                "UPDATE clusters SET manual_category = NULL WHERE id = ?",
                (cluster_id,),
            )
            conn.commit()
            return {
                "cluster_id": cluster_id,
                "category": None,
                "manual_category": None,
            }

        conn.execute(
            "UPDATE clusters SET manual_category = ?, category = ? WHERE id = ?",
            (clean_cat, clean_cat, cluster_id),
        )
        conn.commit()
        return {
            "cluster_id": cluster_id,
            "category": clean_cat,
            "manual_category": clean_cat,
        }

    @app.delete("/clusters/{cluster_id}/category")
    def delete_cluster_category(
        cluster_id: int,
        user: UserContext = Depends(get_current_user),
        conn: sqlite3.Connection = Depends(get_db),
    ) -> dict[str, Any]:
        c_row = conn.execute("SELECT id FROM clusters WHERE id = ?", (cluster_id,)).fetchone()
        if not c_row:
            raise HTTPException(status_code=404, detail=f"Cluster {cluster_id} not found")

        conn.execute(
            "UPDATE clusters SET manual_category = NULL WHERE id = ?",
            (cluster_id,),
        )
        conn.commit()
        return {
            "cluster_id": cluster_id,
            "category": None,
            "manual_category": None,
        }

    @app.get("/thumb/{image_id}")
    def get_thumbnail(
        image_id: int,
        size: int = Query(256, ge=32, le=2048),
        v: str | None = None,
        request: Request = None,
        user: UserContext = Depends(get_current_user),
        conn: sqlite3.Connection = Depends(get_db),
    ):
        """Serve a cached JPEG thumbnail for the given image ID. Strictly confined to user data."""
        try:
            row = conn.execute(
                "SELECT id, file_path, source, drive_id, modified_time, md5 FROM images WHERE id = ?",
                (image_id,),
            ).fetchone()
            if not row:
                raise HTTPException(status_code=404, detail=f"Image {image_id} not found")

            file_path_str = str(row["file_path"])
            orig_path: Path | None = None

            is_drive = file_path_str.startswith("drive://") or row["source"] == "gdrive" or bool(row["drive_id"])
            if is_drive:
                drive_id = row["drive_id"]
                if not drive_id and file_path_str.startswith("drive://"):
                    try:
                        drive_id = file_path_str[8:].split("/")[0]
                    except Exception:
                        drive_id = None

                cfg = getattr(request.app.state, "config", {}) or {}
                configured_drive_cache = cfg.get("sources", {}).get("google_drive", {}).get("cache_dir")
                drive_cache_dir = Path(configured_drive_cache).expanduser() if configured_drive_cache else user.drive_cache_dir

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
                        try:
                            from sources.google_drive import GoogleDriveSource
                            gdrive_source = GoogleDriveSource(conn=conn, cache_dir=drive_cache_dir, config=cfg, stage="thumbnail")
                            thumb_im = gdrive_source.get_thumbnail(file_path_str, size=(size, size))
                            matches = sorted(drive_cache_dir.glob(f"{drive_id}_*.jpg"))
                            if not matches:
                                matches = sorted(drive_cache_dir.glob(f"{drive_id}.jpg"))
                            if matches:
                                orig_path = matches[0]
                            else:
                                cache_dir = user.thumb_cache_dir
                                cache_path = cache_dir / f"{image_id}_{size}.jpg"
                                cache_dir.mkdir(parents=True, exist_ok=True)
                                thumb_im.save(str(cache_path), format="JPEG", quality=85)
                                _thumb_headers = {
                                    "Cache-Control": "private, max-age=3600",
                                    "ETag": f'"{row["md5"] or image_id}"',
                                }
                                return FileResponse(str(cache_path), media_type="image/jpeg", headers=_thumb_headers)
                        except Exception as exc:
                            logger.warning(f"Could not refetch Drive thumbnail for image {image_id} ({drive_id}): {exc}")

            if orig_path is None:
                candidate_path = Path(file_path_str)
                # Verify that local file is confined strictly to the user boundary
                validate_confined_user_path(candidate_path, user.root)
                if candidate_path.exists():
                    orig_path = candidate_path

            if orig_path is None:
                logger.warning(f"Image {image_id} source file not found on disk: {file_path_str}")
                return Response(content=PLACEHOLDER_JPEG_BYTES, media_type="image/jpeg", status_code=200)

            cache_dir = user.thumb_cache_dir
            validate_confined_user_path(cache_dir, user.root)
            cache_key = _thumb_cache_key(image_id, row["md5"], row["modified_time"], size)
            cache_path = cache_dir / cache_key

            # Evict stale thumbnails for the same image_id/size but different content key
            try:
                for stale in cache_dir.glob(f"{image_id}_*_{size}.jpg"):
                    if stale != cache_path:
                        try:
                            stale.unlink()
                        except OSError:
                            pass
            except Exception:
                pass

            etag = f'"{row["md5"] or image_id}"'
            _thumb_headers = {
                "Cache-Control": "private, max-age=3600",
                "ETag": etag,
            }
            if request and request.headers.get("if-none-match") == etag and cache_path.is_file():
                return Response(status_code=304, headers=_thumb_headers)

            if cache_path.is_file():
                try:
                    st = cache_path.stat()
                    if st.st_size > 0 and st.st_mtime >= orig_path.stat().st_mtime:
                        return FileResponse(str(cache_path), media_type="image/jpeg", headers=_thumb_headers)
                except OSError:
                    pass

            lock = _get_thumb_lock(image_id, size)
            with lock:
                if cache_path.is_file():
                    try:
                        st = cache_path.stat()
                        if st.st_size > 0 and st.st_mtime >= orig_path.stat().st_mtime:
                            _thumb_headers = {
                                "Cache-Control": "private, max-age=3600",
                                "ETag": f'"{row["md5"] or image_id}"',
                            }
                            return FileResponse(str(cache_path), media_type="image/jpeg", headers=_thumb_headers)
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

                _thumb_headers = {
                    "Cache-Control": "private, max-age=3600",
                    "ETag": f'"{row["md5"] or image_id}"',
                }
                return FileResponse(str(cache_path), media_type="image/jpeg", headers=_thumb_headers)
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
        user: UserContext = Depends(get_current_user),
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
        mode: str = Query("conservative"),
        user: UserContext = Depends(get_current_user),
        conn: sqlite3.Connection = Depends(get_db),
    ) -> dict[str, int]:
        """Return cluster counts per category (same visibility rules as /clusters).
        Only clusters with >=2 live members are counted. Zero-count categories are omitted.
        """
        from collections import Counter
        c_rows = conn.execute(
            "SELECT id, representative_id, reference_uncertain, category, manual_category FROM clusters ORDER BY id"
        ).fetchall()

        cat_counts: Counter = Counter()
        for c_row in c_rows:
            cid = c_row["id"]
            members = conn.execute(
                """
                SELECT id, file_path, category_top1
                FROM images
                WHERE cluster_id = ? AND (status IS NULL OR status != 'trashed')
                """,
                (cid,),
            ).fetchall()
            live_members = [
                m for m in members
                if str(m["file_path"]).startswith("drive://") or os.path.exists(m["file_path"])
            ]
            if len(live_members) < 2:
                continue

            manual_cat = c_row["manual_category"] if "manual_category" in c_row.keys() else None
            if manual_cat:
                display_cat = manual_cat
            else:
                # Majority category among live members
                member_cats = [m["category_top1"] for m in live_members if m["category_top1"]]
                if member_cats:
                    from collections import Counter as _Counter
                    most_common, occ = _Counter(member_cats).most_common(1)[0]
                    display_cat = most_common if occ > len(member_cats) / 2 else "mixed"
                else:
                    display_cat = c_row["category"] or "uncategorized"

            if display_cat in ("uncertain", "Needs label"):
                display_cat = "Needs label"

            if display_cat:
                cat_counts[display_cat] += 1

        return {cat: cnt for cat, cnt in sorted(cat_counts.items(), key=lambda x: -x[1]) if cnt > 0}

    # ==========================
    # Ledger Planning & Execution
    # ==========================

    @app.post("/plan")
    def create_plan(
        req: PlanRequest,
        request: Request = None,
        user: UserContext = Depends(get_current_user),
        conn: sqlite3.Connection = Depends(get_db),
    ) -> dict[str, Any]:
        if req.mode not in ("conservative", "balanced", "aggressive"):
            raise HTTPException(
                status_code=400,
                detail="Invalid mode. Must be one of: conservative, balanced, aggressive",
            )

        validate_confined_user_path(user.trash_dir, user.root)

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

        if req.manual_member_ids:
            placeholders = ",".join("?" for _ in req.manual_member_ids)
            rows = conn.execute(
                f"SELECT id FROM images WHERE id IN ({placeholders}) AND cluster_id IS NOT NULL",
                req.manual_member_ids,
            ).fetchall()
            found_ids = {r[0] for r in rows}
            invalid_ids = [mid for mid in req.manual_member_ids if mid not in found_ids]
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
                manual_member_ids=req.manual_member_ids,
                approved_swap_cluster_ids=req.approved_swap_cluster_ids,
                excluded_ids=req.excluded_ids,
                config=request.app.state.config,
                trash_dir=user.trash_dir,
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        for entry in summary.entries:
            if not str(entry.original_path).startswith("drive://"):
                validate_confined_user_path(entry.original_path, user.root)

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
            "skipped_entries": summary.skipped_entries,
        }

    @app.post("/apply")
    def apply_batch(
        req: ApplyRequest,
        user: UserContext = Depends(get_current_user),
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

        entries = conn.execute(
            "SELECT original_path, trash_path FROM deletion_journal WHERE batch_id = ?",
            (req.batch_id,),
        ).fetchall()
        for e in entries:
            if not str(e["original_path"]).startswith("drive://"):
                validate_confined_user_path(e["original_path"], user.root)
            if e["trash_path"]:
                validate_confined_user_path(e["trash_path"], user.root)

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
            "refusals": [
                {
                    "entry_id": r.get("entry_id"),
                    "path": r.get("path"),
                    "reason": normalize_refusal_reason(r.get("reason")),
                    "raw_reason": r.get("reason"),
                }
                for r in res.refusals
            ],
            "failures": res.failures,
        }

    @app.post("/undo")
    def undo_batch(
        req: UndoRequest,
        user: UserContext = Depends(get_current_user),
        conn: sqlite3.Connection = Depends(get_db),
    ) -> dict[str, Any]:
        batch = conn.execute(
            "SELECT status FROM ledger_batches WHERE batch_id = ?",
            (req.batch_id,),
        ).fetchone()
        if not batch:
            raise HTTPException(status_code=404, detail=f"Batch '{req.batch_id}' not found in ledger")

        entries = conn.execute(
            "SELECT original_path, trash_path FROM deletion_journal WHERE batch_id = ?",
            (req.batch_id,),
        ).fetchall()
        for e in entries:
            if not str(e["original_path"]).startswith("drive://"):
                validate_confined_user_path(e["original_path"], user.root)
            if e["trash_path"]:
                validate_confined_user_path(e["trash_path"], user.root)

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

    # ==========================
    # Sources Ingestion
    # ==========================

    @app.post("/sources/upload")
    async def upload_sources(
        files: list[UploadFile] = File(...),
        project: str = Query("default"),
        request: Request = None,
        user: UserContext = Depends(get_current_user),
    ) -> dict[str, Any]:
        """
        Upload images into user's library with path sanitization, 50MB per-file cap,
        and 2GB per-user quota enforcement. Confined to ~/.argus/users/<id>/library/<project>/.
        """
        clean_project = re.sub(r"[^a-zA-Z0-9_\-]", "_", project.strip()) or "default"
        sources_cfg = request.app.state.config.get("sources", {})
        upload_cfg = sources_cfg.get("upload", {})
        quotas_cfg = request.app.state.config.get("quotas", {})

        dest_dir = user.library_dir / clean_project
        validate_confined_user_path(dest_dir, user.root)
        dest_dir.mkdir(parents=True, exist_ok=True)

        max_batch_files = upload_cfg.get("max_batch_files", 100)
        if len(files) > max_batch_files:
            raise HTTPException(
                status_code=400,
                detail=f"Upload batch exceeds {max_batch_files} files cap (received {len(files)})",
            )

        allowed_exts = {e.lower().lstrip(".") for e in upload_cfg.get("allowed_extensions", ["jpg", "jpeg", "png", "webp", "heic"])}
        per_file_cap = upload_cfg.get("per_file_cap_mb", 50) * 1024 * 1024
        user_storage_cap = quotas_cfg.get("user_storage_cap_mb", 2048) * 1024 * 1024

        current_user_storage = get_user_storage_usage(user.root)
        if current_user_storage >= user_storage_cap:
            raise HTTPException(
                status_code=400,
                detail=f"Upload exceeds user storage quota ({user_storage_cap // (1024*1024)} MB)",
            )

        saved_files = []
        for upload in files:
            orig_name = upload.filename or ""
            if not orig_name or "/" in orig_name or "\\" in orig_name or ".." in orig_name or orig_name != Path(orig_name).name:
                raise HTTPException(status_code=400, detail=f"Path traversal rejected: {orig_name}")

            ext = Path(orig_name).suffix.lower().lstrip(".")
            if ext not in allowed_exts:
                raise HTTPException(status_code=400, detail=f"Non-image rejected: {orig_name}")

            dest_path = dest_dir / orig_name
            validate_confined_user_path(dest_path, user.root)

            file_bytes = 0
            with open(dest_path, "wb") as f_out:
                while True:
                    chunk = await upload.read(65536)
                    if not chunk:
                        break
                    file_bytes += len(chunk)
                    current_user_storage += len(chunk)
                    if file_bytes > per_file_cap:
                        f_out.close()
                        if dest_path.exists():
                            dest_path.unlink()
                        raise HTTPException(
                            status_code=400,
                            detail=f"File exceeds {per_file_cap // (1024*1024)} MB cap: {orig_name}",
                        )
                    if current_user_storage > user_storage_cap:
                        f_out.close()
                        if dest_path.exists():
                            dest_path.unlink()
                        raise HTTPException(
                            status_code=400,
                            detail=f"Upload exceeds user storage quota ({user_storage_cap // (1024*1024)} MB)",
                        )
                    f_out.write(chunk)
            saved_files.append(str(dest_path))

        # Immediately record uploaded files into the user's DB so they appear in /library without delay
        try:
            with sqlite3.connect(user.db_path) as conn:
                for fp in saved_files:
                    p = Path(fp)
                    sz = p.stat().st_size if p.exists() else 0
                    mtime = p.stat().st_mtime if p.exists() else time.time()
                    w, h, fmt = None, None, None
                    try:
                        with Image.open(p) as img:
                            w, h = img.size
                            fmt = img.format
                    except Exception:
                        pass
                    conn.execute(
                        """
                        INSERT OR IGNORE INTO images (
                            file_path, file_size, modified_time, width, height, format, status, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, 'discovered', strftime('%Y-%m-%dT%H:%M:%SZ','now'), strftime('%Y-%m-%dT%H:%M:%SZ','now'))
                        """,
                        (str(p.resolve()), sz, mtime, w, h, fmt),
                    )
                conn.commit()
        except Exception as exc:
            logging.getLogger("argus.api").warning("Could not pre-insert uploaded files into DB: %s", exc)

        job_id = launch_pipeline_job(request.app, user=user, source_type="local", folder=dest_dir)
        return {
            "job_id": job_id,
            "uploaded_count": len(saved_files),
            "project": clean_project,
            "files": [Path(p).name for p in saved_files],
        }

    # ==========================
    # Google Drive Web OAuth
    # ==========================

    @app.get("/drive/status")
    def get_drive_status(
        request: Request,
        user: UserContext = Depends(get_current_user),
    ) -> dict[str, Any]:
        cfg = getattr(request.app.state, "config", {}) or {}
        hosted_mode = bool(cfg.get("hosted_mode", False) or cfg.get("general", {}).get("hosted_mode", False))
        if "HOSTED_MODE" in os.environ:
            hosted_mode = os.environ.get("HOSTED_MODE", "").lower() in ("true", "1", "yes")

        if hosted_mode and os.environ.get("ARGUS_ENABLE_DRIVE") != "1":
            return {
                "enabled": False,
                "configured": False,
                "connected": False,
                "email": None,
                "account_email": None,
                "message": "Google Drive integration is disabled in hosted mode.",
            }

        data_root = getattr(request.app.state, "data_root", None)
        sec_path, _, searched_paths = find_drive_client_secret(cfg, data_root)
        if not sec_path:
            return {
                "enabled": True,
                "configured": False,
                "connected": False,
                "email": None,
                "account_email": None,
                "message": (
                    "Google Drive integration is not configured. "
                    "An administrator must provide client_secret.json in the data directory."
                ),
            }

        user_tok_path = user.root / "drive_token.json"
        from sources.google_drive import DEFAULT_TOKEN_PATH
        global_tok_path = Path(cfg.get("sources", {}).get("google_drive", {}).get("token_path", DEFAULT_TOKEN_PATH)).expanduser()
        tok_file = user_tok_path if user_tok_path.exists() else (global_tok_path if global_tok_path.exists() else None)

        connected = False
        account_email = None
        if tok_file and tok_file.exists():
            try:
                with open(tok_file, "r", encoding="utf-8") as f:
                    tok_data = json.load(f)
                account_email = tok_data.get("account_email")
                connected = bool(tok_data.get("token") or tok_data.get("refresh_token"))
            except Exception:
                connected = False

        return {
            "enabled": True,
            "configured": True,
            "connected": connected,
            "email": account_email,
            "account_email": account_email,
            "message": None,
        }

    @app.get("/drive/auth-url")
    def get_drive_auth_url(
        request: Request,
        user: UserContext = Depends(get_current_user),
    ) -> dict[str, Any]:
        cfg = getattr(request.app.state, "config", {}) or {}
        hosted_mode = bool(cfg.get("hosted_mode", False) or cfg.get("general", {}).get("hosted_mode", False))
        if "HOSTED_MODE" in os.environ:
            hosted_mode = os.environ.get("HOSTED_MODE", "").lower() in ("true", "1", "yes")

        if hosted_mode and os.environ.get("ARGUS_ENABLE_DRIVE") != "1":
            raise HTTPException(status_code=403, detail="Google Drive integration is disabled in hosted mode.")

        data_root = getattr(request.app.state, "data_root", None)
        sec_path, sec_info, searched_paths = find_drive_client_secret(cfg, data_root)
        if not sec_path or not sec_info:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Google Drive integration is not configured. "
                    "An administrator must provide client_secret.json in the data directory."
                ),
            )

        client_id = sec_info.get("client_id")
        if not client_id:
            raise HTTPException(
                status_code=400,
                detail="Invalid client_secret.json: missing client_id.",
            )

        # Generate PKCE verifier + S256 challenge
        code_verifier = secrets.token_urlsafe(64)
        digest = hashlib.sha256(code_verifier.encode("ascii")).digest()
        code_challenge = base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")
        state = secrets.token_urlsafe(32)

        base = os.environ.get("BASE_URL", "").rstrip("/")
        req_base = str(request.base_url).rstrip("/")
        if not base or ("localhost" in base and "127.0.0.1" in req_base) or ("127.0.0.1" in base and "localhost" in req_base):
            base = req_base
        redirect_uri = f"{base}/drive/callback"

        if not hasattr(request.app.state, "drive_oauth_states"):
            request.app.state.drive_oauth_states = {}

        now = time.time()
        request.app.state.drive_oauth_states = {
            k: v for k, v in request.app.state.drive_oauth_states.items()
            if now - v.get("created_at", 0) < 900
        }

        request.app.state.drive_oauth_states[state] = {
            "user_id": user.id,
            "code_verifier": code_verifier,
            "redirect_uri": redirect_uri,
            "created_at": now,
        }

        params = {
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": "https://www.googleapis.com/auth/drive.readonly https://www.googleapis.com/auth/userinfo.email",
            "access_type": "offline",
            "include_granted_scopes": "true",
            "prompt": "consent",
            "state": state,
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
        }
        auth_url = f"https://accounts.google.com/o/oauth2/v2/auth?{urllib.parse.urlencode(params)}"
        return {"auth_url": auth_url, "state": state}

    @app.get("/drive/callback")
    def drive_callback(
        code: str = Query(None),
        state: str = Query(None),
        error: str = Query(None),
        request: Request = None,
    ) -> Response:
        if error:
            logger.warning("Google Drive OAuth error returned: %s", error)
            return RedirectResponse(
                url=f"/#drive=error&detail={urllib.parse.quote(error)}",
                status_code=303,
            )

        if not code or not state:
            raise HTTPException(status_code=400, detail="Missing code or state parameter")

        states = getattr(request.app.state, "drive_oauth_states", {})
        state_info = states.pop(state, None)
        if not state_info or (time.time() - state_info.get("created_at", 0) > 900):
            raise HTTPException(status_code=400, detail="Invalid or expired OAuth state")

        user_id = state_info["user_id"]

        # Validate caller session if cookie is present; prevent user hijacking
        current_user = None
        try:
            current_user = get_current_user(request)
            if current_user and current_user.id != user_id:
                raise HTTPException(status_code=403, detail="OAuth user mismatch")
        except HTTPException as exc:
            if exc.status_code == 403:
                raise
            # Session cookie may have been omitted by browser across cross-origin/redirect navigation
            current_user = None

        code_verifier = state_info["code_verifier"]
        redirect_uri = state_info["redirect_uri"]

        cfg = getattr(request.app.state, "config", {}) or {}
        data_root = getattr(request.app.state, "data_root", None)
        _, sec_info, _ = find_drive_client_secret(cfg, data_root)
        if not sec_info:
            raise HTTPException(status_code=500, detail="OAuth credentials configuration missing")

        client_id = sec_info.get("client_id")
        client_secret = sec_info.get("client_secret")

        token_payload = urllib.parse.urlencode({
            "code": code,
            "client_id": client_id,
            "client_secret": client_secret,
            "redirect_uri": redirect_uri,
            "grant_type": "authorization_code",
            "code_verifier": code_verifier,
        }).encode("utf-8")

        token_req = urllib.request.Request(
            "https://oauth2.googleapis.com/token",
            data=token_payload,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        try:
            with urllib.request.urlopen(token_req, timeout=20) as token_resp:
                token_data = json.loads(token_resp.read().decode("utf-8"))
        except Exception as exc:
            logger.error("Failed to exchange OAuth code: %s", type(exc).__name__)
            raise HTTPException(status_code=400, detail="OAuth token exchange failed")

        access_token = token_data.get("access_token")
        refresh_token = token_data.get("refresh_token")

        account_email = None
        if access_token:
            try:
                userinfo_req = urllib.request.Request(
                    "https://www.googleapis.com/oauth2/v2/userinfo",
                    headers={"Authorization": f"Bearer {access_token}"},
                )
                with urllib.request.urlopen(userinfo_req, timeout=10) as u_resp:
                    u_info = json.loads(u_resp.read().decode("utf-8"))
                    account_email = u_info.get("email")
            except Exception:
                pass

        overrides = getattr(request.app.state, "custom_user_overrides", {}).get(user_id)
        user_dirs = get_user_dirs(request.app.state.users_dir, user_id, custom_overrides=overrides)
        user_tok_path = user_dirs["root"] / "drive_token.json"

        stored_tok = {
            "token": access_token,
            "refresh_token": refresh_token,
            "token_uri": "https://oauth2.googleapis.com/token",
            "client_id": client_id,
            "client_secret": client_secret,
            "scopes": ["https://www.googleapis.com/auth/drive.readonly"],
            "account_email": account_email,
        }

        user_tok_path.parent.mkdir(parents=True, exist_ok=True)
        user_tok_path.write_text(json.dumps(stored_tok, indent=2))
        try:
            user_tok_path.chmod(0o600)
        except OSError:
            pass

        resp = RedirectResponse(url="/#drive=connected", status_code=303)
        if not current_user and hasattr(request.app.state, "auth_conn") and request.app.state.auth_conn:
            try:
                raw_token, _ = create_session(request.app.state.auth_conn, user_id)
                proto = request.headers.get("x-forwarded-proto", "").lower()
                is_secure = (proto == "https") or (request.url.scheme == "https")
                resp.set_cookie(
                    key="argus_session",
                    value=raw_token,
                    max_age=7 * 24 * 3600,
                    httponly=True,
                    samesite="lax",
                    secure=is_secure,
                )
            except Exception:
                pass
        return resp

    @app.post("/drive/disconnect")
    def disconnect_drive(
        user: UserContext = Depends(get_current_user),
    ) -> dict[str, Any]:
        user_tok_path = user.root / "drive_token.json"
        if user_tok_path.exists():
            user_tok_path.unlink(missing_ok=True)
        return {"connected": False}

    @app.post("/drive/configure")
    async def configure_drive(
        file: UploadFile = File(...),
        request: Request = None,
        user: UserContext = Depends(get_current_user),
    ) -> dict[str, Any]:
        """Upload and configure Google OAuth client_secret.json directly from the UI."""
        cfg = getattr(request.app.state, "config", {}) or {}
        hosted_mode = bool(cfg.get("hosted_mode", False) or cfg.get("general", {}).get("hosted_mode", False))
        if "HOSTED_MODE" in os.environ:
            hosted_mode = os.environ.get("HOSTED_MODE", "").lower() in ("true", "1", "yes")
        if hosted_mode and os.environ.get("ARGUS_ENABLE_DRIVE") != "1":
            raise HTTPException(status_code=403, detail="Google Drive integration is disabled in hosted mode.")

        try:
            content = await file.read(1024 * 1024)
            data = json.loads(content.decode("utf-8"))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid JSON file: {exc}")

        info = data.get("web") or data.get("installed") or {}
        if not info.get("client_id") or not info.get("client_secret"):
            raise HTTPException(
                status_code=400,
                detail="Invalid Google OAuth secret file: missing client_id or client_secret.",
            )

        data_root = getattr(request.app.state, "data_root", None)
        target_dir = data_root if data_root else Path(os.environ.get("ARGUS_DATA_DIR", "~/.argus")).expanduser()
        target_dir.mkdir(parents=True, exist_ok=True)
        dest_file = target_dir / "client_secret.json"

        dest_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
        try:
            dest_file.chmod(0o600)
        except OSError:
            pass

        return {
            "configured": True,
            "message": "Google Drive integration configured successfully.",
        }

    @app.post("/sources/drive")
    def add_drive_source(
        req: DriveSourceRequest,
        request: Request = None,
        user: UserContext = Depends(get_current_user),
    ) -> dict[str, Any]:
        """Validate Drive link and start read-only scan under user context."""
        cfg = getattr(request.app.state, "config", {}) or {}
        hosted_mode = bool(cfg.get("hosted_mode", False) or cfg.get("general", {}).get("hosted_mode", False))
        if "HOSTED_MODE" in os.environ:
            hosted_mode = os.environ.get("HOSTED_MODE", "").lower() in ("true", "1", "yes")
        if hosted_mode and os.environ.get("ARGUS_ENABLE_DRIVE") != "1":
            raise HTTPException(status_code=403, detail="Google Drive integration is disabled in hosted mode.")

        folder_id = extract_drive_folder_id(req.link)

        from sources.google_drive import DEFAULT_TOKEN_PATH, get_drive_credentials
        user_tok = user.root / "drive_token.json"
        global_tok = Path(request.app.state.config.get("sources", {}).get("google_drive", {}).get("token_path", DEFAULT_TOKEN_PATH)).expanduser()
        token_path = user_tok if user_tok.exists() else (global_tok if global_tok.exists() else None)
        if not token_path or not token_path.exists():
            raise HTTPException(
                status_code=400,
                detail="Google Drive authentication required. Please connect your Google Drive account first.",
            )

        try:
            creds = get_drive_credentials(token_path=token_path, prompt_if_missing=False)
            if not creds or not creds.valid:
                raise HTTPException(
                    status_code=400,
                    detail="Google Drive authentication expired or invalid. Please reconnect your account.",
                )
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Google Drive authentication error: {exc}. Please reconnect your account.",
            )

        job_id = launch_pipeline_job(request.app, user=user, source_type="drive", drive_folder_id=folder_id)
        return {
            "job_id": job_id,
            "folder_id": folder_id,
            "status": "queued",
        }

    @app.post("/sources/dataset")
    def add_dataset_source(
        req: DatasetSourceRequest,
        request: Request = None,
        user: UserContext = Depends(get_current_user),
    ) -> dict[str, Any]:
        """Download dataset from allow-listed HTTPS host. Disabled in hosted mode."""
        cfg = getattr(request.app.state, "config", {}) or {}
        hosted_mode = bool(cfg.get("hosted_mode", False) or cfg.get("general", {}).get("hosted_mode", False))
        if "HOSTED_MODE" in os.environ:
            hosted_mode = os.environ.get("HOSTED_MODE", "").lower() in ("true", "1", "yes")
        if hosted_mode:
            raise HTTPException(status_code=403, detail="Dataset download is disabled in hosted mode")

        url = req.url.strip()
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme.lower() != "https":
            raise HTTPException(status_code=400, detail="HTTPS only: HTTP and other schemes are rejected")

        dataset_cfg = cfg.get("sources", {}).get("dataset", {})
        allowed_hosts = [h.lower().strip() for h in dataset_cfg.get("allowed_hosts", ["huggingface.co", "hf.co", "github.com", "raw.githubusercontent.com"])]
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
        temp_dir = user.root / "tmp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        tmp_download = tempfile.NamedTemporaryFile(prefix="dataset_stream_", dir=temp_dir, delete=False)
        tmp_download_path = Path(tmp_download.name)
        total_downloaded = 0
        content_type = ""

        try:
            with opener.open(req_head, timeout=30) as resp:
                content_len = resp.headers.get("Content-Length")
                if content_len and int(content_len) > size_cap_bytes:
                    raise HTTPException(status_code=400, detail=f"Dataset exceeds {size_cap_bytes // (1024*1024)} MB size cap")

                content_type = (resp.headers.get("Content-Type") or "").lower()
                while True:
                    chunk = resp.read(65536)
                    if not chunk:
                        break
                    total_downloaded += len(chunk)
                    if total_downloaded > size_cap_bytes:
                        raise HTTPException(status_code=400, detail=f"Dataset exceeds {size_cap_bytes // (1024*1024)} MB size cap")
                    tmp_download.write(chunk)
            tmp_download.flush()
            tmp_download.close()

            quotas_cfg = cfg.get("quotas", {})
            user_storage_cap = quotas_cfg.get("user_storage_cap_mb", 2048) * 1024 * 1024
            if get_user_storage_usage(user.root) + total_downloaded > user_storage_cap:
                raise HTTPException(status_code=400, detail="Dataset download exceeds user storage quota (2 GB)")

            clean_project = re.sub(r"[^a-zA-Z0-9_\-]", "_", req.project.strip()) or "default"
            dest_dir = user.library_dir / clean_project
            validate_confined_user_path(dest_dir, user.root)
            dest_dir.mkdir(parents=True, exist_ok=True)

            is_zip = False
            with open(tmp_download_path, "rb") as fh:
                magic = fh.read(4)
                if magic.startswith(b"PK\x03\x04") or "zip" in content_type or url.lower().endswith(".zip"):
                    is_zip = True

            extracted_files: list[str] = []

            if is_zip:
                try:
                    zf = zipfile.ZipFile(tmp_download_path)
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

                # Stream and extract in batches of 50
                conn = get_connection(user.db_path)
                try:
                    BATCH_SIZE = 50
                    for b_idx in range(0, len(image_entries), BATCH_SIZE):
                        batch = image_entries[b_idx : b_idx + BATCH_SIZE]
                        db_rows = []
                        for e in batch:
                            safe_name = Path(e.filename).name
                            out_path = dest_dir / safe_name
                            validate_confined_user_path(out_path, user.root)
                            out_path.write_bytes(zf.read(e))
                            extracted_files.append(str(out_path))
                            stat = out_path.stat()
                            db_rows.append((str(out_path), stat.st_size, stat.st_mtime, "local"))

                        conn.executemany(
                            """
                            INSERT OR IGNORE INTO images (file_path, file_size, modified_time, source, status, created_at)
                            VALUES (?, ?, ?, ?, 'discovered', strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
                            """,
                            db_rows,
                        )
                        conn.commit()
                finally:
                    conn.close()
            else:
                data = tmp_download_path.read_bytes()
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
                validate_confined_user_path(out_path, user.root)
                out_path.write_bytes(data)
                extracted_files.append(str(out_path))

                stat = out_path.stat()
                conn = get_connection(user.db_path)
                try:
                    conn.execute(
                        """
                        INSERT OR IGNORE INTO images (file_path, file_size, modified_time, source, status, created_at)
                        VALUES (?, ?, ?, ?, 'discovered', strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
                        """,
                        (str(out_path), stat.st_size, stat.st_mtime, "local"),
                    )
                    conn.commit()
                finally:
                    conn.close()

            job_id = launch_pipeline_job(request.app, user=user, source_type="local", folder=dest_dir)
            return {
                "job_id": job_id,
                "project": clean_project,
                "extracted_count": len(extracted_files),
            }
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Failed to download dataset: {exc}")
        finally:
            if tmp_download_path.exists():
                tmp_download_path.unlink(missing_ok=True)

    @app.post("/sources/rescan")
    def rescan_sources(
        request: Request = None,
        user: UserContext = Depends(get_current_user),
        conn: sqlite3.Connection = Depends(get_db),
    ) -> dict[str, Any]:
        """Re-run the deduplication pipeline on existing user sources."""
        has_gdrive = conn.execute("SELECT 1 FROM images WHERE source = 'gdrive' LIMIT 1").fetchone() is not None
        source_type = "drive" if has_gdrive else "local"
        job_id = launch_pipeline_job(request.app, user=user, source_type=source_type, folder=user.library_dir)
        return {
            "job_id": job_id,
            "status": "queued",
            "source_type": source_type,
        }

    @app.post("/categories/run")
    def run_categorization_job(
        request: Request = None,
        user: UserContext = Depends(get_current_user),
        conn: sqlite3.Connection = Depends(get_db),
    ) -> dict[str, Any]:
        """Trigger semantic categorization stage for the user's images."""
        has_gdrive = conn.execute("SELECT 1 FROM images WHERE source = 'gdrive' LIMIT 1").fetchone() is not None
        source_type = "drive" if has_gdrive else "local"
        job_id = launch_pipeline_job(request.app, user=user, source_type=source_type, folder=user.library_dir)
        return {
            "job_id": job_id,
            "status": "queued",
            "source_type": source_type,
        }

    # ==========================
    # Background Pipeline Jobs
    # ==========================

    @app.get("/jobs/active")
    def get_active_job(
        request: Request = None,
        user: UserContext = Depends(get_current_user),
    ) -> dict[str, Any]:
        """Return the currently running or queued pipeline job for the authenticated user."""
        maybe_start_next_queued_job(request.app)
        now = time.time()
        jobs_dir = user.jobs_dir
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
                        last_heartbeat = state.get("heartbeat_at") or state.get("updated_at", 0)
                        last_progress = state.get("last_progress_at") or state.get("updated_at", 0)
                        is_stalled = ((now - last_heartbeat) > 60) or ((now - last_progress) > 60)

                        if not is_alive and state.get("status") == "running":
                            state["status"] = "failed"
                            state["error"] = "Worker process is no longer running"
                            write_job_state(jf, state)
                            job_id = state.get("id") or state.get("job_id") or jf.stem
                            return {
                                "active": False,
                                "status": "failed",
                                "job_id": str(job_id),
                                "error": "Worker process is no longer running",
                                "message": "Worker process is no longer running",
                            }

                        if is_stalled and state.get("status") == "running":
                            curr_file = state.get("current_file_path") or state.get("failed_file")
                            stg = state.get("stage") or "pipeline"
                            if curr_file:
                                err_msg = f"Stage '{stg}' stalled on file '{curr_file}' (no progress for 60s)"
                            else:
                                err_msg = "Worker heartbeat timed out (>60s)"
                            state["status"] = "stalled"
                            state["error"] = err_msg
                            state["message"] = f"Pipeline stalled: {err_msg}"
                            if curr_file:
                                state["failed_stage"] = stg
                                state["failed_file"] = curr_file
                            write_job_state(jf, state)
                            job_id = state.get("id") or state.get("job_id") or jf.stem
                            return {
                                "active": False,
                                "status": "stalled",
                                "job_id": str(job_id),
                                "error": err_msg,
                                "message": f"Pipeline stalled: {err_msg}",
                            }

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

        return {"active": False}

    @app.get("/jobs/{job_id}")
    def get_job_status(
        job_id: str,
        request: Request = None,
        user: UserContext = Depends(get_current_user),
    ) -> dict[str, Any]:
        """Poll the status and progress of a background pipeline job for the current user."""
        jobs_dir = user.jobs_dir
        job_file = jobs_dir / f"{job_id}.json"
        disk_job = read_job_state(job_file)
        if disk_job:
            return disk_job

        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    return app


# Module-level app instance
app = create_app()
