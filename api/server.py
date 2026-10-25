"""
api/server.py — FastAPI REST API for Argus.

Binds to 127.0.0.1 only. No CORS for external origins.
All file moves/deletions are strictly routed through core/ledger.
Authentication, session management, and per-user isolation enforced.
"""

from __future__ import annotations

from dataclasses import dataclass
import io
import json
import logging
import os
from pathlib import Path
import re
import secrets
import shutil
import signal
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
from fastapi.responses import FileResponse, RedirectResponse
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

    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
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
        raise HTTPException(status_code=401, detail="Authentication required")

    auth_conn = getattr(request.app.state, "auth_conn", None)
    if not auth_conn:
        raise HTTPException(status_code=500, detail="Auth database connection not available")

    session_data = get_session_user(auth_conn, token)
    if not session_data:
        raise HTTPException(status_code=401, detail="Invalid or expired session")

    # Enforce CSRF token on every POST request except public login/register
    if request.method == "POST":
        path = request.url.path
        if path not in ("/auth/login", "/auth/register", "/auth/csrf"):
            csrf_header = request.headers.get("X-CSRF-Token") or request.headers.get("x-csrf-token")
            if not csrf_header or not secrets.compare_digest(csrf_header, session_data["csrf_token"]):
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

    explicit_db_path = db_path is not None
    if not explicit_db_path:
        db_path = os.environ.get("ARGUS_DB_PATH") or config.get("database", {}).get("path")
        if not db_path:
            cand = Path.home() / ".argus" / "argus.db"
            try:
                cand.parent.mkdir(parents=True, exist_ok=True)
                test_f = cand.parent / ".perm_check"
                test_f.touch()
                test_f.unlink()
                db_path = cand
            except (OSError, PermissionError):
                db_path = Path("/tmp/argus") / "argus.db"
    resolved_db_path = Path(db_path).expanduser().resolve()

    if trash_dir is None:
        cfg_trash = config.get("ledger", {}).get("trash_dir")
        trash_dir = os.environ.get("ARGUS_TRASH_DIR") or cfg_trash
        if not trash_dir:
            cand = Path.home() / ".argus" / "trash"
            try:
                cand.mkdir(parents=True, exist_ok=True)
                trash_dir = cand
            except (OSError, PermissionError):
                trash_dir = Path("/tmp/argus") / "trash"
    resolved_trash_dir = Path(trash_dir).expanduser().resolve()

    if thumb_cache_dir is None:
        thumb_cache_dir = os.environ.get("ARGUS_THUMB_CACHE_DIR")
        if not thumb_cache_dir:
            cand = Path.home() / ".argus" / "thumbnails"
            try:
                cand.mkdir(parents=True, exist_ok=True)
                thumb_cache_dir = cand
            except (OSError, PermissionError):
                thumb_cache_dir = Path("/tmp/argus") / "thumbnails"
    resolved_thumb_cache_dir = Path(thumb_cache_dir).expanduser().resolve()
    try:
        resolved_thumb_cache_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass

    if library_dir is None:
        cfg_library = config.get("sources", {}).get("library_dir")
        library_dir = os.environ.get("ARGUS_LIBRARY_DIR") or cfg_library
        if not library_dir:
            cand = Path.home() / ".argus" / "library"
            try:
                cand.mkdir(parents=True, exist_ok=True)
                library_dir = cand
            except (OSError, PermissionError):
                library_dir = Path("/tmp/argus") / "library"
    resolved_library_dir = Path(library_dir).expanduser().resolve()
    try:
        resolved_library_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass

    if jobs_dir is None:
        cfg_jobs = config.get("general", {}).get("jobs_dir")
        jobs_dir = os.environ.get("ARGUS_JOBS_DIR") or cfg_jobs
        if not jobs_dir:
            cand = Path.home() / ".argus" / "jobs"
            try:
                cand.mkdir(parents=True, exist_ok=True)
                jobs_dir = cand
            except (OSError, PermissionError):
                jobs_dir = Path("/tmp/argus") / "jobs"
    resolved_jobs_dir = Path(jobs_dir).expanduser().resolve()
    try:
        resolved_jobs_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass

    # Per-user storage directories root
    if users_dir is None:
        cfg_users = config.get("general", {}).get("users_dir")
        users_dir = os.environ.get("ARGUS_USERS_DIR") or cfg_users
        if not users_dir:
            cand = Path.home() / ".argus" / "users"
            try:
                cand.mkdir(parents=True, exist_ok=True)
                users_dir = cand
            except (OSError, PermissionError):
                users_dir = Path("/tmp/argus") / "users"
    resolved_users_dir = Path(users_dir).expanduser().resolve()
    try:
        resolved_users_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass

    # Auth DB initialization
    if auth_db_path is None:
        cfg_auth = config.get("auth", {}).get("db_path")
        auth_db_path = os.environ.get("ARGUS_ACCOUNTS_DB") or os.environ.get("ARGUS_AUTH_DB") or cfg_auth
        if not auth_db_path:
            cand = Path.home() / ".argus" / "auth.db"
            try:
                cand.parent.mkdir(parents=True, exist_ok=True)
                test_f = cand.parent / ".perm_check"
                test_f.touch()
                test_f.unlink()
                auth_db_path = cand
            except (OSError, PermissionError):
                auth_db_path = Path("/tmp/argus") / "auth.db"
    resolved_auth_db_path = Path(auth_db_path).expanduser().resolve()
    auth_conn = init_auth_db(resolved_auth_db_path)

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

    app = FastAPI(
        title="Argus API",
        version="0.1.0",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
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
        is_secure = (proto == "https") or (request.url.scheme == "https") or (request.url.hostname not in ("127.0.0.1", "localhost", "testserver"))
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
        user: UserContext = Depends(get_current_user),
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

            # Majority-vote display category
            member_cats = [m["category_top1"] for m in member_dicts if m.get("category_top1")]
            if member_cats:
                from collections import Counter
                most_common, occ = Counter(member_cats).most_common(1)[0]
                display_category = most_common if occ > len(member_cats) / 2 else "mixed"
            else:
                display_category = cluster_cat

            if category is not None and (display_category or "").lower() != category.lower():
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
                },
                "savings_bytes": savings_bytes,
                "reclaimable_bytes": savings_bytes,
                "savings_ratio": savings_ratio,
                "member_count": len(member_dicts),
                "delete_safe_count": len(del_safe_ids),
                "review_count": len(review_ids),
                "category": display_category,
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

        member_cats = [m.get("category") for m in formatted_members if m.get("category")]
        if member_cats:
            from collections import Counter
            most_common, occ = Counter(member_cats).most_common(1)[0]
            display_category = most_common if occ > len(member_cats) / 2 else "mixed"
        else:
            display_category = cluster_cat

        return {
            "id": c_row["id"],
            "mode": mode,
            "category": display_category,
            "reference_uncertain": bool(ref_unc),
            "swap_proposal": keeper_res.get("swap_proposal"),
            "members": formatted_members,
        }

    @app.get("/thumb/{image_id}")
    def get_thumbnail(
        image_id: int,
        size: int = Query(256, ge=32, le=2048),
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
            "SELECT id, representative_id, reference_uncertain, category FROM clusters ORDER BY id"
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
            # Majority category among live members
            member_cats = [m["category_top1"] for m in live_members if m["category_top1"]]
            if member_cats:
                from collections import Counter as _Counter
                most_common, occ = _Counter(member_cats).most_common(1)[0]
                display_cat = most_common if occ > len(member_cats) / 2 else "mixed"
            else:
                display_cat = c_row["category"] or "uncategorized"
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
            "refusals": res.refusals,
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

        job_id = launch_pipeline_job(request.app, user=user, source_type="local", folder=dest_dir)
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
        user: UserContext = Depends(get_current_user),
    ) -> dict[str, Any]:
        """Validate Drive link and start read-only scan under user context."""
        cfg = getattr(request.app.state, "config", {}) or {}
        hosted_mode = bool(cfg.get("hosted_mode", False) or cfg.get("general", {}).get("hosted_mode", False))
        if "HOSTED_MODE" in os.environ:
            hosted_mode = os.environ.get("HOSTED_MODE", "").lower() in ("true", "1", "yes")
        if hosted_mode:
            raise HTTPException(status_code=403, detail="Google Drive sources are disabled in hosted mode")

        folder_id = extract_drive_folder_id(req.link)

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

        quotas_cfg = cfg.get("quotas", {})
        user_storage_cap = quotas_cfg.get("user_storage_cap_mb", 2048) * 1024 * 1024
        if get_user_storage_usage(user.root) + len(data) > user_storage_cap:
            raise HTTPException(status_code=400, detail="Dataset download exceeds user storage quota (2 GB)")

        clean_project = re.sub(r"[^a-zA-Z0-9_\-]", "_", req.project.strip()) or "default"
        dest_dir = user.library_dir / clean_project
        validate_confined_user_path(dest_dir, user.root)
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
                validate_confined_user_path(out_path, user.root)
                out_path.write_bytes(zf.read(e))
                extracted_files.append(str(out_path))
        else:
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

        job_id = launch_pipeline_job(request.app, user=user, source_type="local", folder=dest_dir)
        return {
            "job_id": job_id,
            "project": clean_project,
            "extracted_count": len(extracted_files),
        }

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
                        is_stalled = (now - last_heartbeat) > 60

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
                            state["status"] = "stalled"
                            state["error"] = "Worker heartbeat timed out (>60s)"
                            write_job_state(jf, state)
                            job_id = state.get("id") or state.get("job_id") or jf.stem
                            return {
                                "active": False,
                                "status": "stalled",
                                "job_id": str(job_id),
                                "error": "Worker heartbeat timed out (>60s)",
                                "message": "Worker heartbeat timed out (>60s)",
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
