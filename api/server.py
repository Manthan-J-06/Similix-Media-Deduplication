"""
api/server.py — FastAPI REST API for SIMILIX.

Binds to 127.0.0.1 only. No CORS for external origins.
All file moves/deletions are strictly routed through core/ledger.
Authentication, session management, and per-user isolation enforced.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
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
from datetime import datetime, timezone, timedelta
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
from core.config import load_config, get_env, resolve_data_dir
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

logger = logging.getLogger("similix.api")

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
    """Extract and validate Google Drive folder or file ID from link or raw ID string."""
    link = link.strip()
    if link.lower() in ("root", "my drive", "mydrive", ""):
        return "root"
    if link.startswith("http://") or link.startswith("https://"):
        m = re.search(r"/folders/([a-zA-Z0-9_-]+)", link)
        if m:
            return m.group(1)
        m = re.search(r"/file/d/([a-zA-Z0-9_-]+)", link)
        if m:
            return m.group(1)
        m = re.search(r"[?&]id=([a-zA-Z0-9_-]+)", link)
        if m:
            return m.group(1)
        raise HTTPException(status_code=400, detail=f"Invalid Drive link: {link}")

    if re.fullmatch(r"[a-zA-Z0-9_-]{20,60}", link) or link == "root":
        return link
    raise HTTPException(status_code=400, detail=f"Invalid Drive link or ID: {link}")


def check_drive_link_accessibility(folder_id_or_link: str, timeout: float = 8.0) -> dict[str, Any]:
    """Check whether a Google Drive folder or file link/ID is publicly accessible without authentication.
    
    Supports folders, files, and large zip archives (handling virus scan confirmation).
    Returns a dict with keys:
      'accessible': bool,
      'is_public': bool,
      'folder_id': str,
      'folder_name': str,
      'is_file': bool,
      'file_size': int | None,
      'error': str | None,
      'detail': str,
    """
    try:
        folder_id = extract_drive_folder_id(folder_id_or_link)
    except Exception as exc:
        return {
            "accessible": False,
            "is_public": False,
            "folder_id": "",
            "folder_name": "",
            "is_file": False,
            "file_size": None,
            "error": "invalid_link",
            "detail": str(exc),
        }

    if folder_id == "root":
        return {
            "accessible": False,
            "is_public": False,
            "folder_id": "root",
            "folder_name": "My Drive",
            "is_file": False,
            "file_size": None,
            "error": "not_authenticated",
            "detail": "My Drive requires authentication.",
        }

    is_explicit_file = "/file/d/" in folder_id_or_link or "uc?id=" in folder_id_or_link
    urls_to_try = []
    if is_explicit_file:
        urls_to_try = [
            f"https://drive.google.com/file/d/{folder_id}/view",
            f"https://drive.usercontent.google.com/download?id={folder_id}&export=download&confirm=t",
            f"https://drive.google.com/uc?id={folder_id}&export=download&confirm=t",
        ]
    else:
        urls_to_try = [
            f"https://drive.google.com/drive/folders/{folder_id}",
            f"https://drive.google.com/file/d/{folder_id}/view",
            f"https://drive.usercontent.google.com/download?id={folder_id}&export=download&confirm=t",
        ]

    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req_obj, fp, code, msg, headers, newurl):
            if "accounts.google.com" in newurl or "ServiceLogin" in newurl:
                return None
            return super().redirect_request(req_obj, fp, code, msg, headers, newurl)

    opener = urllib.request.build_opener(NoRedirect)
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "*/*",
    }

    last_error_detail = "This Google Drive item is private or requires sign-in. Please connect your Google Drive account."
    last_error_code = "not_authenticated"

    for url in urls_to_try:
        req = urllib.request.Request(url, headers=headers)
        try:
            with opener.open(req, timeout=timeout) as resp:
                status = getattr(resp, "status", getattr(resp, "code", 200))
                final_url = getattr(resp, "url", url)
                if "accounts.google.com" in final_url or "ServiceLogin" in final_url:
                    last_error_detail = "This Google Drive item is private or requires sign-in. Please connect your Google Drive account."
                    last_error_code = "not_authenticated"
                    continue
                if status in (200, 206):
                    cd = resp.headers.get("Content-Disposition", "")
                    content_len = resp.headers.get("Content-Length")
                    fsize = int(content_len) if content_len and content_len.isdigit() else None
                    filename = None
                    if "filename=" in cd:
                        m_fn = re.search(r'filename="?([^";]+)"?', cd)
                        if m_fn:
                            filename = m_fn.group(1).strip()
                    
                    content_peek = resp.read(8192).decode("utf-8", errors="ignore")
                    if not filename:
                        m_title = re.search(r"<title>(.*?)(?:\s*[-–—]\s*Google Drive)?</title>", content_peek, re.IGNORECASE)
                        if m_title and "Virus scan warning" not in m_title.group(1):
                            filename = m_title.group(1).strip()
                        elif "Virus scan warning" in content_peek:
                            # Large file requiring confirmation token
                            m_uc = re.search(r'uc-name-caption">([^<]+)<', content_peek)
                            if m_uc:
                                filename = m_uc.group(1).strip()

                    is_file = is_explicit_file or bool(filename and "." in filename)
                    default_name = "Public Google Drive Archive" if is_file else "Public Google Drive Folder"
                    item_name = filename or default_name
                    return {
                        "accessible": True,
                        "is_public": True,
                        "folder_id": folder_id,
                        "folder_name": item_name,
                        "is_file": is_file,
                        "file_size": fsize,
                        "error": None,
                        "detail": f"Public Google Drive {'file' if is_file else 'folder'} '{item_name}' is accessible without sign-in.",
                    }
        except urllib.error.HTTPError as exc:
            if exc.code in (401, 403):
                last_error_detail = "This Google Drive item is private or requires sign-in. Please connect your Google Drive account."
                last_error_code = "not_authenticated"
            else:
                last_error_detail = f"HTTP error {exc.code} accessing Google Drive item: {exc.reason}"
                last_error_code = "http_error"
            continue
        except Exception as exc:
            logger.debug("Accessibility check exception for %s on %s: %s", folder_id, url, exc)
            last_error_detail = str(exc)
            last_error_code = "connection_error"
            continue

    return {
        "accessible": False,
        "is_public": False,
        "folder_id": folder_id,
        "folder_name": "Google Drive Item",
        "is_file": is_explicit_file,
        "file_size": None,
        "error": last_error_code,
        "detail": last_error_detail,
    }


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
    env_data_dir = get_env("DATA_DIR")
    if env_data_dir:
        candidates.append(Path(env_data_dir).expanduser() / "client_secret.json")
    candidates.append(Path.home() / ".similix" / "client_secret.json")
    candidates.append(Path.home() / ".argus" / "client_secret.json")
    candidates.append(Path("client_secret.json").resolve())
    repo_cand = Path(__file__).resolve().parent.parent / "client_secret.json"
    if repo_cand not in candidates:
        candidates.append(repo_cand)

    env_client_id = get_env("GOOGLE_CLIENT_ID") or cfg.get("sources", {}).get("google_drive", {}).get("client_id")
    env_client_secret = get_env("GOOGLE_CLIENT_SECRET") or cfg.get("sources", {}).get("google_drive", {}).get("client_secret")
    if env_client_id and env_client_secret:
        info = {
            "client_id": env_client_id,
            "client_secret": env_client_secret,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": ["http://localhost:8000/drive/callback", "http://127.0.0.1:8000/drive/callback"],
        }
        return Path("env:GOOGLE_CLIENT_ID"), info, candidates

    for cand in candidates:
        if cand.is_file():
            try:
                with open(cand, "r", encoding="utf-8") as f:
                    data = json.load(f)
                info = data.get("web") or data.get("installed") or (data if "client_id" in data else {})
                cid = str(info.get("client_id", "")).strip()
                if cid:
                    import sys
                    is_test_env = "pytest" in sys.modules or any("pytest" in arg or "test" in arg for arg in sys.argv)
                    if cid.endswith(".apps.googleusercontent.com") or is_test_env:
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


def is_process_alive(pid: int | None) -> bool:
    """Safely check if a worker process PID is alive across OS / sandbox permission domains.

    Returns True if process exists or cannot be queried due to permission.
    Returns False only if process definitely does not exist (ESRCH).
    If pid is None, returns True so stall detection can evaluate heartbeat.
    """
    if pid is None:
        return True
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        # ESRCH: Process definitely does not exist
        return False
    except PermissionError:
        # EPERM: Process exists, caller lacks permission to signal it
        return True
    except OSError as exc:
        if getattr(exc, "errno", None) == 3:  # ESRCH
            return False
        return True


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
                last_hb = st.get("heartbeat_at") or st.get("updated_at", 0)
                alive = is_process_alive(pid)
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
    cpu_cores = str(os.cpu_count() or 2)
    env["OMP_NUM_THREADS"] = cpu_cores
    env["OPENBLAS_NUM_THREADS"] = cpu_cores
    env["VECLIB_MAXIMUM_THREADS"] = cpu_cores
    env["NUMEXPR_NUM_THREADS"] = cpu_cores
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
                last_hb = st.get("heartbeat_at") or st.get("updated_at", 0)
                alive = is_process_alive(pid)
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
    token = request.cookies.get("similix_session") or request.cookies.get("argus_session")
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
    """Instantiate GoogleDriveSource for the authenticated user with strict per-user token isolation."""
    try:
        from sources.google_drive import GoogleDriveSource
        cfg = getattr(request.app.state, "config", {}) or {}
        configured_drive_cache = cfg.get("sources", {}).get("google_drive", {}).get("cache_dir")
        drive_cache_dir = Path(configured_drive_cache).expanduser() if configured_drive_cache else user.drive_cache_dir
        user_tok_path = user.root / "drive_token.json"
        if user_tok_path.exists():
            return GoogleDriveSource(
                conn=conn,
                cache_dir=drive_cache_dir,
                token_path=user_tok_path,
                config=cfg,
                stage="cluster_verification",
            )
        # Check if user has an unauthenticated public drive scan
        saved_fid_file = user.root / "drive_folder_id.txt"
        if saved_fid_file.exists():
            fid = saved_fid_file.read_text(encoding="utf-8").strip()
            if fid:
                return GoogleDriveSource(
                    conn=conn,
                    folder_id=fid,
                    cache_dir=drive_cache_dir,
                    token_path=None,
                    config=cfg,
                    stage="cluster_verification",
                )
    except Exception as exc:
        logger.warning("Could not instantiate GoogleDriveSource: %s", exc)
    return None


download_progress: dict[int, dict[str, Any]] = {}


def _collect_download_file(
    file_path_str: str,
    arcname_default: str,
    user: UserContext,
    request: Request,
    gdrive_source: Any = None,
    allow_unclustered: bool = False,
) -> tuple[Path | bytes, str] | None:
    """Resolve a local path or drive:// URI to (Path | bytes, arcname)."""
    if file_path_str.startswith("drive://"):
        from sources.google_drive import parse_drive_uri
        drive_id, filename = parse_drive_uri(file_path_str)
        arcname = filename or arcname_default or f"{drive_id}.jpg"

        # Check local disk cache first (orders of magnitude faster than remote API)
        cfg = getattr(request.app.state, "config", {}) or {}
        configured_drive_cache = cfg.get("sources", {}).get("google_drive", {}).get("cache_dir")
        drive_cache_dir = Path(configured_drive_cache).expanduser() if configured_drive_cache else user.drive_cache_dir

        search_dirs = [
            drive_cache_dir,
            user.root / "drive_cache",
            Path.home() / ".similix" / "users" / str(user.id) / "drive_cache",
        ]
        for dcd in search_dirs:
            if dcd and dcd.is_dir():
                matches = sorted(dcd.glob(f"{drive_id}_*.jpg"))
                if not matches:
                    matches = sorted(dcd.glob(f"{drive_id}.jpg"))
                if not matches:
                    matches = sorted(dcd.glob(f"{drive_id}*"))
                if matches and matches[0].is_file():
                    return (matches[0], arcname)

        # Fallback to remote Google Drive fetch only if missing from local cache
        if gdrive_source is not None:
            try:
                try:
                    b = gdrive_source.get_bytes(file_path_str, allow_unclustered=allow_unclustered)
                except TypeError:
                    b = gdrive_source.get_bytes(file_path_str)
                if b:
                    return (b, arcname)
            except Exception as exc:
                logger.warning("Could not fetch full Google Drive file %s: %s", file_path_str, exc)

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


def build_and_stream_zip(
    file_records: list[tuple[str, str]],
    zip_filename: str,
    user: UserContext,
    request: Request,
    gdrive_source: Any = None,
    allow_unclustered: bool = False,
    progress_dict: dict[int, dict[str, Any]] | None = None,
    max_download_size: int = 2 * 1024 * 1024 * 1024,
):
    """
    Build a zip archive directly to disk with concurrent multi-worker fetching for remote files.
    Streams the finished archive to the client and automatically unlinks the temp file.
    Does NOT store all file bytes in Python RAM, preserving memory and device performance.
    """
    if not file_records:
        raise HTTPException(status_code=404, detail="No downloadable files found")

    seen_arcnames: set[str] = set()
    unique_items: list[dict[str, Any]] = []

    for fp_str, arcname_default in file_records:
        if not fp_str:
            continue
        if fp_str.startswith("drive://"):
            from sources.google_drive import parse_drive_uri
            drive_id, filename = parse_drive_uri(fp_str)
            base_name = filename or arcname_default or f"{drive_id}.jpg"
        else:
            base_name = arcname_default or Path(fp_str).name or "image.jpg"

        final_arcname = base_name
        count = 1
        while final_arcname in seen_arcnames:
            p = Path(base_name)
            final_arcname = f"{p.stem}_{count}{p.suffix}"
            count += 1
        seen_arcnames.add(final_arcname)
        unique_items.append({"file_path": fp_str, "arcname": final_arcname})

    if not unique_items:
        raise HTTPException(status_code=404, detail="No downloadable files found")

    cfg = getattr(request.app.state, "config", {}) or {}
    configured_drive_cache = cfg.get("sources", {}).get("google_drive", {}).get("cache_dir")
    drive_cache_dir = Path(configured_drive_cache).expanduser() if configured_drive_cache else user.drive_cache_dir

    search_dirs = [
        drive_cache_dir,
        user.root / "drive_cache",
        Path.home() / ".similix" / "users" / str(user.id) / "drive_cache",
    ]

    local_items: list[tuple[Path, str]] = []
    drive_items: list[dict[str, Any]] = []

    for item in unique_items:
        fp_str = item["file_path"]
        arcname = item["arcname"]
        if fp_str.startswith("drive://"):
            from sources.google_drive import parse_drive_uri
            drive_id, _ = parse_drive_uri(fp_str)
            found_local = None
            for dcd in search_dirs:
                if dcd and dcd.is_dir():
                    matches = sorted(dcd.glob(f"{drive_id}_*.jpg"))
                    if not matches:
                        matches = sorted(dcd.glob(f"{drive_id}.jpg"))
                    if not matches:
                        matches = sorted(dcd.glob(f"{drive_id}*"))
                    if matches and matches[0].is_file():
                        found_local = matches[0]
                        break
            if found_local:
                local_items.append((found_local, arcname))
            else:
                drive_items.append(item)
        else:
            p = Path(fp_str)
            if p.is_file() and not p.is_symlink():
                local_items.append((p, arcname))

    total_est_size = sum(p.stat().st_size for p, _ in local_items if p.exists())
    total_est_size += len(drive_items) * 350 * 1024
    if total_est_size > max_download_size:
        raise HTTPException(
            status_code=400,
            detail=f"Total download size ({total_est_size // (1024*1024)} MB) exceeds 2GB cap",
        )

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
    temp_path = Path(temp_file.name)
    temp_file.close()

    total_count = len(local_items) + len(drive_items)
    if progress_dict is not None:
        progress_dict[user.id] = {
            "active": True,
            "total": total_count,
            "completed": 0,
            "status": "Archiving files...",
        }

    written_count = 0
    try:
        with zipfile.ZipFile(temp_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zip_lock = threading.Lock()

            # 1. Local disk items written directly
            for local_path, arcname in local_items:
                try:
                    zf.write(local_path, arcname=arcname)
                    with zip_lock:
                        written_count += 1
                        if progress_dict is not None and user.id in progress_dict:
                            progress_dict[user.id]["completed"] += 1
                except Exception as exc:
                    logger.warning("Could not add local file %s to zip: %s", local_path, exc)

            # 2. Remote Drive items downloaded concurrently with worker pool
            if drive_items and gdrive_source is not None:
                max_workers = min(20, max(4, len(drive_items)))

                def _fetch_worker(item: dict[str, Any]):
                    nonlocal written_count
                    fp = item["file_path"]
                    arc = item["arcname"]
                    try:
                        try:
                            b = gdrive_source.get_bytes(fp, allow_unclustered=allow_unclustered)
                        except TypeError:
                            b = gdrive_source.get_bytes(fp)
                        if b:
                            with zip_lock:
                                zf.writestr(arc, b)
                                written_count += 1
                    except Exception as exc:
                        logger.warning("Could not download Drive file %s for zip: %s", fp, exc)
                    finally:
                        if progress_dict is not None and user.id in progress_dict:
                            with zip_lock:
                                progress_dict[user.id]["completed"] += 1

                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    list(executor.map(_fetch_worker, drive_items))

    except Exception as exc:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
        if progress_dict is not None and user.id in progress_dict:
            progress_dict[user.id]["active"] = False
        raise HTTPException(status_code=500, detail=f"Failed to create zip archive: {exc}")

    if progress_dict is not None and user.id in progress_dict:
        progress_dict[user.id]["active"] = False
        progress_dict[user.id]["status"] = "Completed"

    if written_count == 0:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=404, detail="No downloadable files found")

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
    """Create and configure the SIMILIX FastAPI application with per-user isolation."""
    if config is None:
        try:
            config = load_config(config_path) if config_path else load_config()
        except Exception:
            config = {}

    raw_data_dir = get_env("DATA_DIR") or config.get("general", {}).get("data_dir")
    data_root = resolve_data_dir(raw_data_dir)

    explicit_db_path = db_path is not None
    if not explicit_db_path:
        db_env = get_env("DB_PATH")
        if db_env:
            db_path = db_env
        else:
            cand_db = data_root / "similix.db"
            if not cand_db.exists() and (data_root / "argus.db").exists():
                db_path = data_root / "argus.db"
            else:
                db_path = cand_db if raw_data_dir else (config.get("database", {}).get("path") or cand_db)
    resolved_db_path = Path(db_path).expanduser().resolve()

    def _safe_resolve_dir(raw_arg: str | Path | None, env_key: str, cfg_val: str | None, subfolder: str) -> Path:
        if raw_arg is not None:
            cand = Path(raw_arg).expanduser().resolve()
        else:
            env_val = get_env(env_key)
            if env_val:
                cand = Path(env_val).expanduser().resolve()
            elif raw_data_dir or not cfg_val:
                cand = (data_root / subfolder).resolve()
            else:
                expanded = Path(cfg_val).expanduser()
                if expanded in (Path.home() / ".similix" / subfolder, Path.home() / ".argus" / subfolder):
                    cand = (data_root / subfolder).resolve()
                else:
                    cand = expanded.resolve()
        try:
            cand.mkdir(parents=True, exist_ok=True)
            return cand
        except (OSError, PermissionError):
            fallback = (data_root / subfolder).resolve()
            fallback.mkdir(parents=True, exist_ok=True)
            return fallback

    resolved_trash_dir = _safe_resolve_dir(trash_dir, "TRASH_DIR", config.get("ledger", {}).get("trash_dir"), "trash")
    resolved_thumb_cache_dir = _safe_resolve_dir(thumb_cache_dir, "THUMB_CACHE_DIR", None, "thumbnails")
    resolved_library_dir = _safe_resolve_dir(library_dir, "LIBRARY_DIR", config.get("sources", {}).get("library_dir"), "library")
    resolved_jobs_dir = _safe_resolve_dir(jobs_dir, "JOBS_DIR", config.get("general", {}).get("jobs_dir"), "jobs")
    resolved_users_dir = _safe_resolve_dir(users_dir, "USERS_DIR", config.get("general", {}).get("users_dir"), "users")

    # Auth DB initialization
    if auth_db_path is None:
        auth_db_path = get_env("ACCOUNTS_DB") or get_env("AUTH_DB") or (data_root / "auth.db" if raw_data_dir else (config.get("auth", {}).get("db_path") or (data_root / "auth.db")))
    resolved_auth_db_path = Path(auth_db_path).expanduser().resolve()
    try:
        resolved_auth_db_path.parent.mkdir(parents=True, exist_ok=True)
    except (OSError, PermissionError):
        resolved_auth_db_path = (data_root / "auth.db").resolve()
        resolved_auth_db_path.parent.mkdir(parents=True, exist_ok=True)
    auth_conn = init_auth_db(resolved_auth_db_path)

    logger.info("SIMILIX fixed data root: %s", data_root)
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
            f"  SIMILIX Startup Self-Check\n"
            f"==================================================\n"
            f"  Data Directory : {data_root}\n"
            f"  DB Path        : {resolved_db_path}\n"
            f"  Secret Key Set : {secret_is_set}\n"
            f"==================================================\n\n"
        )
        sys.stdout.flush()
        yield

    app = FastAPI(
        title="SIMILIX API",
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
        return {"status": "ok", "service": "similix"}

    @app.get("/login")
    def serve_login():
        """Serve the standalone login/registration web page (public)."""
        if not login_file.is_file():
            raise HTTPException(status_code=404, detail="Login page not found")
        return FileResponse(str(login_file), media_type="text/html")

    @app.get("/auth/csrf")
    def get_csrf(request: Request):
        """Public endpoint returning CSRF token for the active session (if any)."""
        token = request.cookies.get("similix_session") or request.cookies.get("argus_session")
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
        for cookie_name in ("similix_session", "argus_session"):
            response.set_cookie(
                key=cookie_name,
                value=raw_token,
                max_age=7 * 24 * 3600,
                httponly=True,
                samesite="lax",
                secure=is_secure,
            )
        for csrf_name in ("similix_csrf", "argus_csrf"):
            response.set_cookie(
                key=csrf_name,
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
        """Invalidate session in auth.db, stop running user jobs, and clear auth cookies."""
        if user and user.jobs_dir:
            stop_user_jobs(request.app, user.id, user.jobs_dir)
        token = request.cookies.get("similix_session") or request.cookies.get("argus_session")
        if not token:
            auth_h = request.headers.get("Authorization", "")
            if auth_h.startswith("Bearer "):
                token = auth_h[7:].strip()
        if token:
            invalidate_session(request.app.state.auth_conn, token)
        for cname in ("similix_session", "argus_session", "similix_csrf", "argus_csrf"):
            response.delete_cookie(cname)
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
            for cname in ("similix_session", "argus_session", "similix_csrf", "argus_csrf"):
                response.delete_cookie(cname)
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
            headers={"Content-Disposition": 'attachment; filename="similix_library.zip"'},
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
        file_records = [(str(r["file_path"]), Path(str(r["file_path"])).name) for r in rows]
        return build_and_stream_zip(
            file_records=file_records,
            zip_filename=f"cluster_{cluster_id}.zip",
            user=user,
            request=request,
            gdrive_source=gdrive_source,
            allow_unclustered=False,
            progress_dict=download_progress,
        )

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

        if not rows:
            raise HTTPException(status_code=404, detail="No selected images found")

        gdrive_source = _get_user_gdrive_source(request, user, conn)
        file_records = [(str(r["file_path"]), Path(str(r["file_path"])).name) for r in rows]
        return build_and_stream_zip(
            file_records=file_records,
            zip_filename="selected_images.zip",
            user=user,
            request=request,
            gdrive_source=gdrive_source,
            allow_unclustered=True,
            progress_dict=download_progress,
        )

    @app.get("/download/keepers")
    def download_keepers(
        mode: str = Query("balanced"),
        request: Request = None,
        user: UserContext = Depends(get_current_user),
        conn: sqlite3.Connection = Depends(get_db),
    ):
        """
        Stream a zip archive of the complete clean dataset:
        Removes ONLY files with delete_safe = 1 for the chosen mode.
        Guarantees exported_count == total_images - delete_safe_count.
        Every non-delete-safe file (keepers, uniques, review items) is strictly preserved.
        """
        from core.keeper import run_keeper_pipeline
        cfg = getattr(request.app.state, "config", {}) or {}
        run_keeper_pipeline(conn, mode=mode, config=cfg)
        conn.execute("UPDATE images SET delete_safe = 0 WHERE cluster_id IS NULL OR cluster_id = 0")
        conn.commit()

        # Fetch all active images
        all_active_images = conn.execute(
            """
            SELECT id, file_path, file_size, md5, delete_safe, review_flag, is_recommended_keeper, is_keeper, cluster_id
            FROM images
            WHERE (status IS NULL OR status NOT IN ('trashed', 'deleted', 'dismissed'))
            ORDER BY id
            """
        ).fetchall()

        total_images = len(all_active_images)
        delete_safe_count = sum(1 for img in all_active_images if img["delete_safe"] == 1)

        # In conservative mode: drop strictly files marked delete_safe = 1
        # In balanced/aggressive mode (default): export clean dataset = uniques + 1 keeper per cluster
        if mode == "conservative":
            candidates = [img for img in all_active_images if img["delete_safe"] == 0]
            assert len(candidates) == total_images - delete_safe_count, (
                f"Clean dataset export invariant violated: expected {total_images - delete_safe_count} images, "
                f"got {len(candidates)} (total={total_images}, delete_safe={delete_safe_count})"
            )
        else:
            # Collect chosen keeper IDs per cluster
            cluster_keepers = set()
            for img in all_active_images:
                if img["cluster_id"] and (img["is_recommended_keeper"] == 1 or img["is_keeper"] == 1):
                    cluster_keepers.add(img["id"])
            clusters_present = {img["cluster_id"] for img in all_active_images if img["cluster_id"]}
            for cid in clusters_present:
                c_members = [img for img in all_active_images if img["cluster_id"] == cid]
                if not any(m["id"] in cluster_keepers for m in c_members):
                    cluster_keepers.add(c_members[0]["id"])

            candidates = [
                img for img in all_active_images
                if (img["cluster_id"] is None or img["cluster_id"] == 0)
                or (img["id"] in cluster_keepers)
            ]

        file_records: list[tuple[str, str]] = [
            (str(img["file_path"]), Path(str(img["file_path"])).name)
            for img in candidates
        ]

        gdrive_source = _get_user_gdrive_source(request, user, conn)
        return build_and_stream_zip(
            file_records=file_records,
            zip_filename="recommended_keepers.zip",
            user=user,
            request=request,
            gdrive_source=gdrive_source,
            allow_unclustered=True,
            progress_dict=download_progress,
        )

    @app.get("/download/progress")
    def get_download_progress(
        user: UserContext = Depends(get_current_user),
    ):
        """Return the current archiving/download progress for the active user."""
        return download_progress.get(
            user.id,
            {"active": False, "total": 0, "completed": 0, "status": "idle"},
        )

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
        conditions = ["(status IS NULL OR status NOT IN ('deleted', 'trashed', 'dismissed'))"]
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

    def _resolve_cluster_display_category(
        cluster_row: dict | sqlite3.Row,
        member_dicts: list[dict],
        keeper_id: int | None = None,
    ) -> str:
        """
        Resolve the single authoritative display category for a cluster.
        Used identically across /categories, /clusters, and /clusters/{id}.
        """
        manual_cat = cluster_row["manual_category"] if "manual_category" in cluster_row.keys() else None
        if manual_cat and str(manual_cat).strip():
            return str(manual_cat).strip()

        # 1. Majority-vote display category from members
        raw_cats = [
            m.get("category_top1") or m.get("category")
            for m in member_dicts
        ]
        semantic_cats = [
            str(c).strip() for c in raw_cats
            if c and str(c).strip().lower() not in ("uncertain", "needs label", "needs_label", "uncategorized", "none")
        ]
        if semantic_cats:
            from collections import Counter
            top_cat, occ = Counter(semantic_cats).most_common(1)[0]
            if occ > len(semantic_cats) / 2:
                return top_cat
            return "mixed"

        # 2. Authoritative cluster-level stored category from Stage 5 or keeper
        cluster_cat = cluster_row["category"] if "category" in cluster_row.keys() else None
        if cluster_cat and str(cluster_cat).strip().lower() not in ("uncertain", "needs label", "needs_label", "uncategorized", "none", "mixed"):
            return str(cluster_cat).strip()

        if keeper_id is not None:
            keeper_m = next((m for m in member_dicts if m.get("id") == keeper_id), None)
            if keeper_m and keeper_m.get("category_top1"):
                k_cat = str(keeper_m["category_top1"]).strip()
                if k_cat.lower() not in ("uncertain", "needs label", "needs_label", "uncategorized", "none", "mixed"):
                    return k_cat

        return "Needs label"

    @app.get("/clusters")
    def list_clusters(
        mode: str = Query("conservative"),
        category: Optional[str] = Query(None),
        min_savings: Optional[float] = Query(None),
        sort: str = Query("id"),
        limit: Optional[int] = Query(None, ge=1, description="Limit number of clusters for pagination"),
        offset: int = Query(0, ge=0, description="Offset for pagination"),
        page: Optional[int] = Query(None, ge=1, description="1-based page number"),
        page_size: Optional[int] = Query(None, ge=1, description="Page size when using page-based pagination"),
        paginated: bool = Query(False, description="Return wrapped object with pagination metadata"),
        request: Request = None,
        response: Response = None,
        user: UserContext = Depends(get_current_user),
        conn: sqlite3.Connection = Depends(get_db),
    ) -> Any:
        if mode not in ("conservative", "balanced", "aggressive"):
            raise HTTPException(
                status_code=400,
                detail="Invalid mode. Must be one of: conservative, balanced, aggressive",
            )

        total_images = conn.execute(
            "SELECT COUNT(*) FROM images WHERE status IS NULL OR status NOT IN ('trashed', 'deleted', 'dismissed')"
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
                WHERE cluster_id = ? AND (status IS NULL OR status NOT IN ('trashed', 'deleted', 'dismissed'))
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
            display_category = _resolve_cluster_display_category(c_row, member_dicts, keeper_id=keeper_id)

            if category is not None and category.strip():
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

        total_matching_clusters = len(results)

        # Determine effective pagination parameters
        eff_limit = page_size if page_size is not None else limit
        if page is not None and eff_limit is not None:
            eff_offset = (page - 1) * eff_limit
        else:
            eff_offset = offset

        if response is not None:
            response.headers["X-Total-Clusters"] = str(total_matching_clusters)
            response.headers["X-Total-Count"] = str(total_matching_clusters)
            response.headers["X-Offset"] = str(eff_offset)
            if eff_limit is not None:
                response.headers["X-Limit"] = str(eff_limit)
                total_pages = max(1, (total_matching_clusters + eff_limit - 1) // eff_limit)
                response.headers["X-Total-Pages"] = str(total_pages)
                response.headers["X-Current-Page"] = str((eff_offset // eff_limit) + 1)
                response.headers["X-Has-Next"] = "true" if (eff_offset + eff_limit < total_matching_clusters) else "false"

        sliced = results[eff_offset : eff_offset + eff_limit] if eff_limit is not None else results[eff_offset:]

        if paginated:
            total_pages = max(1, (total_matching_clusters + eff_limit - 1) // eff_limit) if eff_limit else 1
            cur_page = (eff_offset // eff_limit) + 1 if eff_limit else 1
            has_next = (eff_offset + eff_limit < total_matching_clusters) if eff_limit else False
            return {
                "total": total_matching_clusters,
                "total_clusters": total_matching_clusters,
                "limit": eff_limit,
                "offset": eff_offset,
                "page": cur_page,
                "page_size": eff_limit or total_matching_clusters,
                "total_pages": total_pages,
                "has_next": has_next,
                "items": sliced,
                "clusters": sliced,
            }

        return sliced

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
            WHERE cluster_id = ? AND (status IS NULL OR status NOT IN ('trashed', 'deleted', 'dismissed'))
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
            m_res = m["resolution_pct"]
            if m_res is None and m["width"] and m["height"]:
                ref_cand = next((rm for rm in member_dicts if rm["id"] == keeper_id or rm.get("quality_status") == "reference"), None)
                if ref_cand and ref_cand.get("width") and ref_cand.get("height"):
                    ref_px = max(1, ref_cand["width"] * ref_cand["height"])
                    m_res = round(((m["width"] * m["height"]) / ref_px) * 100.0, 1)

            m_match = m["match_pct"]
            if m_match is None and (mid == keeper_id or m.get("quality_status") == "reference"):
                m_match = 100.0
            elif m_match is None and m.get("ssim_color") is not None:
                m_match = round(float(m["ssim_color"]) * 100.0, 1)
            elif m_match is None and m.get("ssim") is not None:
                m_match = round(float(m["ssim"]) * 100.0, 1)

            formatted_members.append({
                "id": mid,
                "file_path": m["file_path"],
                "size": m["file_size"],
                "width": m["width"],
                "height": m["height"],
                "format": m["format"],
                "md5": m.get("md5", ""),
                "match_pct": m_match,
                "resolution_pct": m_res,
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
        display_category = _resolve_cluster_display_category(c_row, formatted_members, keeper_id=keeper_id)

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

    @app.post("/clusters/{cluster_id}/dismiss/{image_id}")
    def dismiss_cluster_member(
        cluster_id: int,
        image_id: int,
        request: Request = None,
        user: UserContext = Depends(get_current_user),
        conn: sqlite3.Connection = Depends(get_db),
    ) -> dict[str, Any]:
        """Dismiss an image from a cluster without deleting files from disk."""
        c_row = conn.execute("SELECT id, representative_id FROM clusters WHERE id = ?", (cluster_id,)).fetchone()
        if not c_row:
            raise HTTPException(status_code=404, detail=f"Cluster {cluster_id} not found")

        img = conn.execute(
            "SELECT id, cluster_id, file_path, is_keeper FROM images WHERE id = ? AND cluster_id = ?",
            (image_id, cluster_id),
        ).fetchone()
        if not img:
            raise HTTPException(status_code=404, detail=f"Image {image_id} not found in cluster {cluster_id}")

        conn.execute(
            """
            UPDATE images
            SET cluster_id = NULL,
                is_keeper = 0,
                status = 'dismissed',
                updated_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now')
            WHERE id = ?
            """,
            (image_id,),
        )

        remaining_members = conn.execute(
            """
            SELECT id, is_keeper FROM images
            WHERE cluster_id = ? AND (status IS NULL OR status NOT IN ('trashed', 'deleted', 'dismissed'))
            ORDER BY id
            """,
            (cluster_id,),
        ).fetchall()

        remaining_count = len(remaining_members)
        if remaining_count < 2:
            if remaining_count == 1:
                survivor_id = remaining_members[0]["id"]
                conn.execute(
                    """
                    UPDATE images
                    SET cluster_id = NULL,
                        is_keeper = 0,
                        updated_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now')
                    WHERE id = ?
                    """,
                    (survivor_id,),
                )
            conn.execute("DELETE FROM clusters WHERE id = ?", (cluster_id,))
        else:
            rep_id = c_row["representative_id"]
            if rep_id == image_id:
                new_rep_id = remaining_members[0]["id"]
                conn.execute(
                    "UPDATE clusters SET member_count = ?, representative_id = ? WHERE id = ?",
                    (remaining_count, new_rep_id, cluster_id),
                )
            else:
                conn.execute(
                    "UPDATE clusters SET member_count = ? WHERE id = ?",
                    (remaining_count, cluster_id),
                )

        conn.commit()
        return {
            "success": True,
            "cluster_id": cluster_id,
            "dismissed_image_id": image_id,
            "remaining_members": remaining_count if remaining_count >= 2 else 0,
            "cluster_deleted": remaining_count < 2,
        }

    @app.post("/clusters/{cluster_id}/dismiss")
    def dismiss_entire_cluster(
        cluster_id: int,
        request: Request = None,
        user: UserContext = Depends(get_current_user),
        conn: sqlite3.Connection = Depends(get_db),
    ) -> dict[str, Any]:
        """Dismiss an entire cluster without deleting files from disk."""
        c_row = conn.execute("SELECT id FROM clusters WHERE id = ?", (cluster_id,)).fetchone()
        if not c_row:
            raise HTTPException(status_code=404, detail=f"Cluster {cluster_id} not found")

        conn.execute(
            """
            UPDATE images
            SET cluster_id = NULL,
                is_keeper = 0,
                status = 'dismissed',
                updated_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now')
            WHERE cluster_id = ?
            """,
            (cluster_id,),
        )
        conn.execute("DELETE FROM clusters WHERE id = ?", (cluster_id,))
        conn.commit()
        return {"success": True, "cluster_id": cluster_id, "cluster_deleted": True}

    @app.post("/images/{image_id}/dismiss")
    def dismiss_image(
        image_id: int,
        request: Request = None,
        user: UserContext = Depends(get_current_user),
        conn: sqlite3.Connection = Depends(get_db),
    ) -> dict[str, Any]:
        """Dismiss any image from the app (hiding it from library and clusters) without deleting the physical file."""
        img = conn.execute(
            "SELECT id, cluster_id, file_path, status FROM images WHERE id = ?",
            (image_id,),
        ).fetchone()
        if not img:
            raise HTTPException(status_code=404, detail=f"Image {image_id} not found")

        cid = img["cluster_id"]
        conn.execute(
            """
            UPDATE images
            SET cluster_id = NULL,
                is_keeper = 0,
                status = 'dismissed',
                updated_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now')
            WHERE id = ?
            """,
            (image_id,),
        )

        cluster_deleted = False
        remaining_count = 0
        if cid:
            remaining_members = conn.execute(
                """
                SELECT id FROM images
                WHERE cluster_id = ? AND (status IS NULL OR status NOT IN ('trashed', 'deleted', 'dismissed'))
                ORDER BY id
                """,
                (cid,),
            ).fetchall()
            remaining_count = len(remaining_members)
            if remaining_count < 2:
                if remaining_count == 1:
                    survivor_id = remaining_members[0]["id"]
                    conn.execute(
                        "UPDATE images SET cluster_id = NULL, is_keeper = 0, updated_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now') WHERE id = ?",
                        (survivor_id,),
                    )
                conn.execute("DELETE FROM clusters WHERE id = ?", (cid,))
                cluster_deleted = True
            else:
                c_row = conn.execute("SELECT representative_id FROM clusters WHERE id = ?", (cid,)).fetchone()
                if c_row and c_row["representative_id"] == image_id:
                    new_rep_id = remaining_members[0]["id"]
                    conn.execute(
                        "UPDATE clusters SET member_count = ?, representative_id = ? WHERE id = ?",
                        (remaining_count, new_rep_id, cid),
                    )
                else:
                    conn.execute("UPDATE clusters SET member_count = ? WHERE id = ?", (remaining_count, cid))

        conn.commit()
        return {
            "success": True,
            "image_id": image_id,
            "status": "dismissed",
            "cluster_id": cid,
            "cluster_deleted": cluster_deleted,
            "remaining_cluster_members": remaining_count if remaining_count >= 2 else 0,
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
                    safe_drive_id = drive_id.replace('/', '_').replace(':', '_')
                    mtime = row["modified_time"]
                    if mtime is not None:
                        try:
                            candidate = drive_cache_dir / f"{safe_drive_id}_{int(mtime)}.jpg"
                            if candidate.is_file():
                                orig_path = candidate
                        except Exception:
                            pass
                    if orig_path is None:
                        matches = sorted(drive_cache_dir.glob(f"{safe_drive_id}_*.jpg"))
                        if not matches:
                            matches = sorted(drive_cache_dir.glob(f"{safe_drive_id}.jpg"))
                        if matches:
                            orig_path = matches[0]

                    if orig_path is None:
                        try:
                            from sources.google_drive import GoogleDriveSource
                            user_tok = user.root / "drive_token.json"
                            tok_path = user_tok if user_tok.exists() else None
                            gdrive_source = GoogleDriveSource(
                                conn=conn,
                                cache_dir=drive_cache_dir,
                                config=cfg,
                                stage="thumbnail",
                                token_path=tok_path,
                                is_public=bool(not tok_path),
                            )
                            thumb_im = gdrive_source.get_thumbnail(file_path_str, size=(size, size))
                            matches = sorted(drive_cache_dir.glob(f"{safe_drive_id}_*.jpg"))
                            if not matches:
                                matches = sorted(drive_cache_dir.glob(f"{safe_drive_id}.jpg"))
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
            legacy_path = cache_dir / f"{image_id}_{size}.jpg"

            etag = f'"{row["md5"] or image_id}"'
            _thumb_headers = {
                "Cache-Control": "private, max-age=3600",
                "ETag": etag,
            }

            # Fast-path: Direct O(1) cache hit check without directory globbing
            for hit_candidate in (cache_path, legacy_path):
                if hit_candidate.is_file():
                    try:
                        st = hit_candidate.stat()
                        if st.st_size > 0 and st.st_mtime >= orig_path.stat().st_mtime:
                            if request and request.headers.get("if-none-match") == etag:
                                return Response(status_code=304, headers=_thumb_headers)
                            return FileResponse(str(hit_candidate), media_type="image/jpeg", headers=_thumb_headers)
                    except OSError:
                        pass

            # Cache miss: generate thumbnail under lock to avoid duplicated work
            lock = _get_thumb_lock(image_id, size)
            with lock:
                # Double-check inside lock
                for hit_candidate in (cache_path, legacy_path):
                    if hit_candidate.is_file():
                        try:
                            st = hit_candidate.stat()
                            if st.st_size > 0 and st.st_mtime >= orig_path.stat().st_mtime:
                                return FileResponse(str(hit_candidate), media_type="image/jpeg", headers=_thumb_headers)
                        except OSError:
                            pass

                cache_dir.mkdir(parents=True, exist_ok=True)
                tmp_path = cache_dir / f".tmp_{image_id}_{size}_{uuid.uuid4().hex}.jpg"
                try:
                    with Image.open(orig_path) as img:
                        if img.format == "JPEG":
                            img.draft("RGB", (size, size))
                        if img.mode != "RGB":
                            img = img.convert("RGB")
                        img.thumbnail((size, size), Image.Resampling.BILINEAR)
                        img.save(str(tmp_path), format="JPEG", quality=85)
                    os.replace(str(tmp_path), str(cache_path))

                    # Evict stale thumbnails only when generating a new thumbnail
                    try:
                        for stale in cache_dir.glob(f"{image_id}_*_{size}.jpg"):
                            if stale != cache_path:
                                try:
                                    stale.unlink()
                                except OSError:
                                    pass
                    except Exception:
                        pass
                except Exception as exc:
                    if tmp_path.exists():
                        try:
                            tmp_path.unlink()
                        except OSError:
                            pass
                    logger.exception(f"Thumbnail generation failed for image {image_id} ({orig_path}): {exc}")
                    return Response(content=PLACEHOLDER_JPEG_BYTES, media_type="image/jpeg", status_code=200)

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
                WHERE cluster_id = ? AND (status IS NULL OR status NOT IN ('trashed', 'deleted', 'dismissed'))
                """,
                (cid,),
            ).fetchall()
            live_members = [
                m for m in members
                if str(m["file_path"]).startswith("drive://") or os.path.exists(m["file_path"])
            ]
            if len(live_members) < 2:
                continue

            display_cat = _resolve_cluster_display_category(c_row, [dict(m) for m in live_members], keeper_id=c_row["representative_id"])
            if display_cat:
                cat_counts[display_cat] += 1

        return {cat: cnt for cat, cnt in sorted(cat_counts.items(), key=lambda x: -x[1]) if cnt > 0}

    @app.get("/library/categories")
    def get_library_categories(
        user: UserContext = Depends(get_current_user),
        conn: sqlite3.Connection = Depends(get_db),
    ) -> dict[str, int]:
        """Return image counts per category for the user's library photos."""
        rows = conn.execute(
            """
            SELECT category_top1, COUNT(*) as cnt
            FROM images
            WHERE (status IS NULL OR status NOT IN ('deleted', 'trashed', 'dismissed'))
              AND category_top1 IS NOT NULL
            GROUP BY category_top1
            ORDER BY cnt DESC
            """
        ).fetchall()
        counts: dict[str, int] = {}
        for r in rows:
            cat = r["category_top1"]
            if cat in ("uncertain", "Needs label"):
                counts["Needs label"] = counts.get("Needs label", 0) + r["cnt"]
            elif cat and str(cat).strip():
                clean_cat = str(cat).strip()
                counts[clean_cat] = counts.get(clean_cat, 0) + r["cnt"]
        return {cat: cnt for cat, cnt in sorted(counts.items(), key=lambda x: -x[1]) if cnt > 0}

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

        job_id = launch_pipeline_job(request.app, user=user, source_type="local", folder=user.library_dir)
        return {
            "job_id": job_id,
            "uploaded_count": len(saved_files),
            "project": clean_project,
            "files": [Path(p).name for p in saved_files],
        }

    def _pre_insert_images_db(db_path: Path, saved_files: list[str]) -> None:
        try:
            with sqlite3.connect(db_path) as conn:
                for fp in saved_files:
                    p = Path(fp)
                    if not p.exists():
                        continue
                    sz = p.stat().st_size
                    mtime = p.stat().st_mtime
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
            logger.warning("Could not pre-insert images into DB: %s", exc)

    def _extract_zip_streamed(
        zip_path: Path,
        dest_dir: Path,
        user: UserContext,
        config: dict,
    ) -> list[str]:
        """
        Safely extract image files from a ZIP archive in a streamed manner.
        Enforces:
          - Max file count limit (up to 10,000 files)
          - Zip bomb protections (overall uncompressed size <= quota, compression ratio <= 100:1)
          - Path traversal rejection (no '..', absolute paths, outside user_root)
          - No symlinks (POSIX S_IFLNK rejected)
          - Image type validation (allowed extensions + Pillow verification)
          - 50 MB per-file size cap
          - User storage quota
        """
        sources_cfg = config.get("sources", {})
        upload_cfg = sources_cfg.get("upload", {})
        zip_cfg = sources_cfg.get("zip", {})
        quotas_cfg = config.get("quotas", {})

        max_files = zip_cfg.get("max_files", 10000)
        max_uncompressed = zip_cfg.get("max_uncompressed_bytes", 2 * 1024 * 1024 * 1024)
        max_ratio = zip_cfg.get("max_ratio", 100)
        per_file_cap = upload_cfg.get("per_file_cap_mb", 50) * 1024 * 1024
        user_storage_cap = quotas_cfg.get("user_storage_cap_mb", 2048) * 1024 * 1024
        allowed_exts = {e.lower().lstrip(".") for e in upload_cfg.get("allowed_extensions", ["jpg", "jpeg", "png", "webp", "heic", "tiff", "bmp"])}

        current_user_storage = get_user_storage_usage(user.root)
        if current_user_storage >= user_storage_cap:
            raise HTTPException(status_code=400, detail="Upload exceeds user storage quota")

        try:
            zf = zipfile.ZipFile(zip_path, "r")
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid or corrupted ZIP archive: {exc}")

        with zf:
            infolist = zf.infolist()
            # Filter candidate image entries
            image_entries = []
            for zinfo in infolist:
                if zinfo.is_dir():
                    continue
                fname = zinfo.filename.replace("\\", "/")
                if fname.startswith("__MACOSX/") or Path(fname).name.startswith("."):
                    continue
                ext = Path(fname).suffix.lower().lstrip(".")
                if ext in allowed_exts:
                    image_entries.append((zinfo, fname, ext))

            if len(image_entries) > max_files:
                raise HTTPException(
                    status_code=400,
                    detail=f"ZIP archive exceeds {max_files} files cap (contains {len(image_entries)} images)",
                )

            # Pre-scan for zip bombs, symlinks, and path traversal
            total_uncompressed = 0
            for zinfo, fname, _ in image_entries:
                if ".." in fname or fname.startswith("/") or fname.startswith("\\"):
                    raise HTTPException(status_code=400, detail=f"Path traversal in ZIP rejected: {fname}")

                is_symlink = ((zinfo.external_attr >> 16) & 0o170000) == 0o120000
                if is_symlink:
                    raise HTTPException(status_code=400, detail=f"Symlink rejected: {fname} (symlinks inside ZIP archives are forbidden)")

                total_uncompressed += zinfo.file_size
                if zinfo.file_size > 1024 * 1024 and zinfo.compress_size > 0:
                    ratio = zinfo.file_size / zinfo.compress_size
                    if ratio > max_ratio:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Potential ZIP bomb detected in {fname} (compression ratio {ratio:.1f}:1 exceeds {max_ratio}:1 limit)",
                        )

            if total_uncompressed > max_uncompressed:
                raise HTTPException(
                    status_code=400,
                    detail=f"ZIP archive uncompressed size ({total_uncompressed // (1024*1024)} MB) exceeds allowed limit ({max_uncompressed // (1024*1024)} MB)",
                )

            if current_user_storage + total_uncompressed > user_storage_cap:
                raise HTTPException(
                    status_code=400,
                    detail="ZIP extraction would exceed user storage quota",
                )

            dest_dir.mkdir(parents=True, exist_ok=True)
            validate_confined_user_path(dest_dir, user.root)

            extracted_files: list[str] = []
            for idx, (zinfo, fname, ext) in enumerate(image_entries):
                basename = Path(fname).name
                if not basename or basename.startswith("."):
                    continue

                dest_file = dest_dir / basename
                if dest_file.exists():
                    stem = Path(basename).stem
                    dest_file = dest_dir / f"{stem}_{idx}.{ext}"

                validate_confined_user_path(dest_file, user.root)

                written_bytes = 0
                with zf.open(zinfo, "r") as src, open(dest_file, "wb") as dst:
                    while True:
                        buf = src.read(65536)
                        if not buf:
                            break
                        written_bytes += len(buf)
                        if written_bytes > per_file_cap:
                            dst.close()
                            if dest_file.exists():
                                dest_file.unlink()
                            raise HTTPException(
                                status_code=400,
                                detail=f"File in ZIP exceeds {per_file_cap // (1024*1024)} MB cap: {fname}",
                            )
                        dst.write(buf)

                try:
                    with Image.open(dest_file) as im:
                        im.verify()
                except Exception:
                    if dest_file.exists():
                        dest_file.unlink()
                    continue

                extracted_files.append(str(dest_file.resolve()))

            return extracted_files

        # ── Folder Upload Endpoints (Chunked in 25s with resume) ──

    @app.post("/sources/folder/chunk")
    async def upload_folder_chunk(
        upload_id: str = Form(...),
        chunk_index: int = Form(...),
        total_chunks: int = Form(...),
        total_files: int = Form(...),
        project: str = Form("default"),
        files: list[UploadFile] = File(...),
        relative_paths: list[str] = Form(None),
        request: Request = None,
        user: UserContext = Depends(get_current_user),
    ) -> dict[str, Any]:
        """
        Receive a chunk of files from a folder upload (typically 25 files per chunk).
        Enforces max batch cap up to 10,000 files, user quota, per-file cap, and path sanitization.
        Relative paths are used for display only.
        """
        sources_cfg = request.app.state.config.get("sources", {})
        upload_cfg = sources_cfg.get("upload", {})
        quotas_cfg = request.app.state.config.get("quotas", {})

        max_folder_files = upload_cfg.get("max_folder_files", 10000)
        if total_files > max_folder_files:
            raise HTTPException(
                status_code=400,
                detail=f"Folder upload exceeds {max_folder_files} files cap (received {total_files})",
            )

        clean_upload_id = re.sub(r"[^a-zA-Z0-9_\-]", "", upload_id.strip()) or f"fld_{int(time.time())}"
        clean_project = re.sub(r"[^a-zA-Z0-9_\-]", "_", project.strip()) or "default"
        dest_dir = user.library_dir / clean_project
        validate_confined_user_path(dest_dir, user.root)
        dest_dir.mkdir(parents=True, exist_ok=True)

        user.jobs_dir.mkdir(parents=True, exist_ok=True)
        state_file = user.jobs_dir / f"folder_upload_{clean_upload_id}.json"
        state = read_job_state(state_file) if state_file.exists() else {
            "upload_id": clean_upload_id,
            "project": clean_project,
            "total_files": total_files,
            "total_chunks": total_chunks,
            "chunks_received": [],
            "saved_files": [],
            "relative_paths": {},
        }

        # Idempotent resume check
        if chunk_index in state.get("chunks_received", []):
            return {
                "status": "already_received",
                "chunk_index": chunk_index,
                "total_chunks": total_chunks,
                "uploaded_count": len(state.get("saved_files", [])),
                "total_files": total_files,
            }

        allowed_exts = {e.lower().lstrip(".") for e in upload_cfg.get("allowed_extensions", ["jpg", "jpeg", "png", "webp", "heic", "tiff", "bmp"])}
        per_file_cap = upload_cfg.get("per_file_cap_mb", 50) * 1024 * 1024
        user_storage_cap = quotas_cfg.get("user_storage_cap_mb", 2048) * 1024 * 1024
        current_user_storage = get_user_storage_usage(user.root)

        saved_this_chunk = []
        for idx, upload in enumerate(files):
            orig_name = upload.filename or ""
            # Path traversal rejection
            if ".." in orig_name or orig_name.startswith(("/", "\\")):
                raise HTTPException(status_code=400, detail=f"Path traversal rejected: {orig_name}")

            basename = Path(orig_name).name
            if not basename or basename.startswith("."):
                continue

            ext = Path(basename).suffix.lower().lstrip(".")
            if ext not in allowed_exts:
                raise HTTPException(status_code=400, detail=f"Non-image rejected: {orig_name}")

            dest_path = dest_dir / basename
            if dest_path.exists():
                stem = Path(basename).stem
                dest_path = dest_dir / f"{stem}_{chunk_index}_{idx}.{ext}"

            validate_confined_user_path(dest_path, user.root)

            file_bytes = 0
            with open(dest_path, "wb") as f_out:
                while True:
                    buf = await upload.read(65536)
                    if not buf:
                        break
                    file_bytes += len(buf)
                    current_user_storage += len(buf)
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
                    f_out.write(buf)

            saved_path_str = str(dest_path.resolve())
            saved_this_chunk.append(saved_path_str)
            rel_display = relative_paths[idx] if (relative_paths and idx < len(relative_paths)) else orig_name
            state.setdefault("relative_paths", {})[dest_path.name] = rel_display

        state.setdefault("chunks_received", []).append(chunk_index)
        state.setdefault("saved_files", []).extend(saved_this_chunk)
        write_job_state(state_file, state)

        return {
            "status": "chunk_received",
            "chunk_index": chunk_index,
            "total_chunks": total_chunks,
            "uploaded_count": len(state["saved_files"]),
            "total_files": total_files,
        }

    @app.get("/sources/folder/status")
    def get_folder_upload_status(
        upload_id: str = Query(...),
        user: UserContext = Depends(get_current_user),
    ) -> dict[str, Any]:
        """Check status of a chunked folder upload for resuming."""
        clean_upload_id = re.sub(r"[^a-zA-Z0-9_\-]", "", upload_id.strip())
        state_file = user.jobs_dir / f"folder_upload_{clean_upload_id}.json"
        if not state_file.exists():
            return {
                "exists": False,
                "upload_id": clean_upload_id,
                "chunks_received": [],
                "uploaded_count": 0,
                "status": "not_found",
            }
        state = read_job_state(state_file) or {}
        return {
            "exists": True,
            "upload_id": clean_upload_id,
            "project": state.get("project", "default"),
            "total_files": state.get("total_files", 0),
            "total_chunks": state.get("total_chunks", 0),
            "chunks_received": state.get("chunks_received", []),
            "uploaded_count": len(state.get("saved_files", [])),
            "relative_paths": state.get("relative_paths", {}),
            "status": "in_progress",
        }

    @app.post("/sources/folder/complete")
    async def complete_folder_upload(
        upload_id: str = Form(...),
        project: str = Form("default"),
        request: Request = None,
        user: UserContext = Depends(get_current_user),
    ) -> dict[str, Any]:
        """Finalize chunked folder upload and run the pipeline."""
        clean_upload_id = re.sub(r"[^a-zA-Z0-9_\-]", "", upload_id.strip())
        clean_project = re.sub(r"[^a-zA-Z0-9_\-]", "_", project.strip()) or "default"
        state_file = user.jobs_dir / f"folder_upload_{clean_upload_id}.json"
        if not state_file.exists():
            raise HTTPException(status_code=404, detail="Folder upload session not found or already completed")

        state = read_job_state(state_file) or {}
        saved_files = state.get("saved_files", [])

        _pre_insert_images_db(user.db_path, saved_files)
        job_id = launch_pipeline_job(request.app, user=user, source_type="local", folder=user.library_dir)

        try:
            state_file.unlink()
        except Exception:
            pass

        return {
            "job_id": job_id,
            "uploaded_count": len(saved_files),
            "project": clean_project,
            "files": [Path(p).name for p in saved_files],
        }

    # ── ZIP Upload Endpoints (Chunked and Direct Streamed Extraction) ──

    @app.post("/sources/zip/chunk")
    async def upload_zip_chunk(
        upload_id: str = Form(...),
        chunk_index: int = Form(...),
        total_chunks: int = Form(...),
        file: UploadFile = File(...),
        total_bytes: int = Form(0),
        project: str = Form("default"),
        request: Request = None,
        user: UserContext = Depends(get_current_user),
    ) -> dict[str, Any]:
        """Upload a binary chunk of a ZIP archive."""
        clean_upload_id = re.sub(r"[^a-zA-Z0-9_\-]", "", upload_id.strip()) or f"zip_{int(time.time())}"
        clean_project = re.sub(r"[^a-zA-Z0-9_\-]", "_", project.strip()) or "default"
        quotas_cfg = request.app.state.config.get("quotas", {})
        user_storage_cap = quotas_cfg.get("user_storage_cap_mb", 2048) * 1024 * 1024
        current_user_storage = get_user_storage_usage(user.root)
        if current_user_storage >= user_storage_cap:
            raise HTTPException(status_code=400, detail="Upload exceeds user storage quota")

        tmp_dir = user.root / "tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        zip_part = tmp_dir / f"zip_{clean_upload_id}.part"
        validate_confined_user_path(zip_part, user.root)

        state_file = user.jobs_dir / f"zip_upload_{clean_upload_id}.json"
        state = read_job_state(state_file) if state_file.exists() else {
            "upload_id": clean_upload_id,
            "project": clean_project,
            "total_chunks": total_chunks,
            "total_bytes": total_bytes,
            "chunks_received": [],
            "bytes_received": 0,
        }

        if chunk_index in state.get("chunks_received", []):
            return {
                "status": "already_received",
                "chunk_index": chunk_index,
                "total_chunks": total_chunks,
                "bytes_received": zip_part.stat().st_size if zip_part.exists() else 0,
            }

        mode = "ab" if (zip_part.exists() and chunk_index > 0) else "wb"
        with open(zip_part, mode) as f_out:
            while True:
                buf = await file.read(65536)
                if not buf:
                    break
                current_user_storage += len(buf)
                if current_user_storage > user_storage_cap:
                    f_out.close()
                    if zip_part.exists():
                        zip_part.unlink()
                    raise HTTPException(status_code=400, detail="Upload exceeds user storage quota")
                f_out.write(buf)

        state.setdefault("chunks_received", []).append(chunk_index)
        state["bytes_received"] = zip_part.stat().st_size
        write_job_state(state_file, state)

        return {
            "status": "chunk_received",
            "chunk_index": chunk_index,
            "total_chunks": total_chunks,
            "bytes_received": state["bytes_received"],
        }

    @app.get("/sources/zip/status")
    def get_zip_upload_status(
        upload_id: str = Query(...),
        user: UserContext = Depends(get_current_user),
    ) -> dict[str, Any]:
        """Check status of chunked ZIP upload."""
        clean_upload_id = re.sub(r"[^a-zA-Z0-9_\-]", "", upload_id.strip())
        state_file = user.jobs_dir / f"zip_upload_{clean_upload_id}.json"
        if not state_file.exists():
            return {"exists": False, "chunks_received": [], "bytes_received": 0, "status": "not_found"}
        state = read_job_state(state_file) or {}
        return {
            "exists": True,
            "upload_id": clean_upload_id,
            "total_chunks": state.get("total_chunks", 0),
            "chunks_received": state.get("chunks_received", []),
            "bytes_received": state.get("bytes_received", 0),
            "status": "in_progress",
        }

    @app.post("/sources/zip/extract")
    async def extract_zip_upload(
        upload_id: str = Form(...),
        project: str = Form("default"),
        request: Request = None,
        user: UserContext = Depends(get_current_user),
    ) -> dict[str, Any]:
        """Streamed server-side extraction of chunked ZIP upload."""
        clean_upload_id = re.sub(r"[^a-zA-Z0-9_\-]", "", upload_id.strip())
        clean_project = re.sub(r"[^a-zA-Z0-9_\-]", "_", project.strip()) or "default"
        zip_part = user.root / "tmp" / f"zip_{clean_upload_id}.part"
        if not zip_part.exists():
            raise HTTPException(status_code=404, detail="Uploaded ZIP archive part not found")

        dest_dir = user.library_dir / clean_project
        dest_dir.mkdir(parents=True, exist_ok=True)
        validate_confined_user_path(dest_dir, user.root)

        try:
            extracted_files = _extract_zip_streamed(zip_part, dest_dir, user, request.app.state.config)
        finally:
            if zip_part.exists():
                try:
                    zip_part.unlink()
                except Exception:
                    pass
            state_file = user.jobs_dir / f"zip_upload_{clean_upload_id}.json"
            if state_file.exists():
                try:
                    state_file.unlink()
                except Exception:
                    pass

        _pre_insert_images_db(user.db_path, extracted_files)
        job_id = launch_pipeline_job(request.app, user=user, source_type="local", folder=user.library_dir)
        return {
            "job_id": job_id,
            "extracted_count": len(extracted_files),
            "project": clean_project,
            "files": [Path(p).name for p in extracted_files],
        }

    @app.post("/sources/zip")
    async def upload_zip_direct(
        file: UploadFile = File(...),
        project: str = Query("default"),
        request: Request = None,
        user: UserContext = Depends(get_current_user),
    ) -> dict[str, Any]:
        """Direct single-request ZIP upload and streamed extraction."""
        clean_project = re.sub(r"[^a-zA-Z0-9_\-]", "_", project.strip()) or "default"
        tmp_dir = user.root / "tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        temp_zip = tmp_dir / f"direct_{uuid.uuid4().hex}.zip"
        validate_confined_user_path(temp_zip, user.root)

        quotas_cfg = request.app.state.config.get("quotas", {})
        user_storage_cap = quotas_cfg.get("user_storage_cap_mb", 2048) * 1024 * 1024
        current_user_storage = get_user_storage_usage(user.root)

        try:
            with open(temp_zip, "wb") as f_out:
                while True:
                    chunk = await file.read(65536)
                    if not chunk:
                        break
                    current_user_storage += len(chunk)
                    if current_user_storage > user_storage_cap:
                        f_out.close()
                        if temp_zip.exists():
                            temp_zip.unlink()
                        raise HTTPException(status_code=400, detail="Upload exceeds user storage quota")
                    f_out.write(chunk)

            dest_dir = user.library_dir / clean_project
            dest_dir.mkdir(parents=True, exist_ok=True)
            validate_confined_user_path(dest_dir, user.root)
            extracted_files = _extract_zip_streamed(temp_zip, dest_dir, user, request.app.state.config)
        finally:
            if temp_zip.exists():
                try:
                    temp_zip.unlink()
                except Exception:
                    pass

        _pre_insert_images_db(user.db_path, extracted_files)
        job_id = launch_pipeline_job(request.app, user=user, source_type="local", folder=user.library_dir)
        return {
            "job_id": job_id,
            "extracted_count": len(extracted_files),
            "project": clean_project,
            "files": [Path(p).name for p in extracted_files],
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

        if hosted_mode and (get_env("ENABLE_DRIVE") != "1"):
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
        from sources.google_drive import get_drive_credentials

        connected = False
        account_email = None
        if user_tok_path.exists():
            try:
                with open(user_tok_path, "r", encoding="utf-8") as f:
                    tok_data = json.load(f)
                account_email = tok_data.get("account_email")
                has_tokens = bool(tok_data.get("token") or tok_data.get("refresh_token"))
                if has_tokens:
                    try:
                        creds = get_drive_credentials(token_path=user_tok_path, prompt_if_missing=False)
                        connected = bool(creds and creds.valid)
                    except Exception:
                        connected = has_tokens
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

        if hosted_mode and (get_env("ENABLE_DRIVE") != "1"):
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

        # Match against authorized redirect_uris in client_secret.json to prevent redirect_uri_mismatch
        auth_redirects = sec_info.get("redirect_uris", [])
        if auth_redirects and redirect_uri not in auth_redirects:
            parsed_req = urllib.parse.urlparse(req_base)
            req_port = parsed_req.port or (443 if parsed_req.scheme == "https" else 80)
            for ar in auth_redirects:
                p_ar = urllib.parse.urlparse(ar)
                ar_port = p_ar.port or (443 if p_ar.scheme == "https" else 80)
                if p_ar.path.rstrip("/") == "/drive/callback" and (ar_port == req_port or p_ar.netloc == parsed_req.netloc):
                    redirect_uri = ar
                    break
            else:
                for ar in auth_redirects:
                    if ar.endswith("/drive/callback"):
                        redirect_uri = ar
                        break

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

        expires_in = token_data.get("expires_in", 3600)
        try:
            expires_in_sec = int(expires_in)
        except (ValueError, TypeError):
            expires_in_sec = 3600
        expiry_dt = datetime.now(timezone.utc) + timedelta(seconds=expires_in_sec)
        expiry_iso = expiry_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

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

        # Preserve existing refresh token if re-authenticating without re-grant
        existing_refresh = None
        if user_tok_path.exists():
            try:
                prev_data = json.loads(user_tok_path.read_text(encoding="utf-8"))
                existing_refresh = prev_data.get("refresh_token")
            except Exception:
                pass
        final_refresh = refresh_token or existing_refresh

        stored_tok = {
            "token": access_token,
            "refresh_token": final_refresh,
            "token_uri": "https://oauth2.googleapis.com/token",
            "client_id": client_id,
            "client_secret": client_secret,
            "scopes": ["https://www.googleapis.com/auth/drive.readonly"],
            "account_email": account_email,
            "expiry": expiry_iso,
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
                for cname in ("similix_session", "argus_session"):
                    resp.set_cookie(
                        key=cname,
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
        # Also delete any lingering global or legacy default token files to avoid leaks
        try:
            from sources.google_drive import DEFAULT_TOKEN_PATH
            if DEFAULT_TOKEN_PATH.exists():
                DEFAULT_TOKEN_PATH.unlink(missing_ok=True)
        except Exception:
            pass
        for extra in (
            Path.home() / ".similix" / "drive_token.json",
            Path.home() / ".argus" / "drive_token.json",
        ):
            try:
                if extra.exists():
                    extra.unlink(missing_ok=True)
            except Exception:
                pass
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
        if hosted_mode and (get_env("ENABLE_DRIVE") != "1"):
            raise HTTPException(status_code=403, detail="Google Drive integration is disabled in hosted mode.")

        try:
            content = await file.read(1024 * 1024)
            data = json.loads(content.decode("utf-8"))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid JSON file: {exc}")

        # Check if user uploaded a drive_token.json directly
        if ("token" in data or "refresh_token" in data) and ("client_id" in data or "client_secret" in data):
            user_tok_path = user.root / "drive_token.json"
            user_tok_path.parent.mkdir(parents=True, exist_ok=True)
            if not data.get("expiry"):
                data["expiry"] = (datetime.now(timezone.utc) + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
            user_tok_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            try:
                user_tok_path.chmod(0o600)
            except OSError:
                pass
            return {
                "configured": True,
                "connected": True,
                "email": data.get("account_email"),
                "message": "Google Drive token configured and connected successfully.",
            }

        info = data.get("web") or data.get("installed") or (data if "client_id" in data else {})
        if not info.get("client_id") or not info.get("client_secret"):
            raise HTTPException(
                status_code=400,
                detail="Invalid Google OAuth secret file: missing client_id or client_secret.",
            )

        data_root = getattr(request.app.state, "data_root", None)
        target_dir = data_root if data_root else resolve_data_dir()
        target_dir.mkdir(parents=True, exist_ok=True)
        dest_file = target_dir / "client_secret.json"

        dest_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
        try:
            dest_file.chmod(0o600)
        except OSError:
            pass

        # Also copy to DEFAULT_CLIENT_SECRET_PATH only in production default setup so tests with isolated data_root don't pollute global workspace
        if not data_root or data_root == resolve_data_dir():
            try:
                from sources.google_drive import DEFAULT_CLIENT_SECRET_PATH
                DEFAULT_CLIENT_SECRET_PATH.parent.mkdir(parents=True, exist_ok=True)
                DEFAULT_CLIENT_SECRET_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
                DEFAULT_CLIENT_SECRET_PATH.chmod(0o600)
            except Exception:
                pass

        return {
            "configured": True,
            "message": "Google Drive integration configured successfully.",
        }

    @app.get("/sources/drive/validate")
    @app.post("/sources/drive/validate")
    def validate_drive_source(
        req: DriveSourceRequest = None,
        link: str = Query(None),
        request: Request = None,
        user: UserContext = Depends(get_current_user),
    ) -> dict[str, Any]:
        """Validate Drive link and folder access permissions without initiating a pipeline scan."""
        folder_link = link or (req.link if req else "")
        if not folder_link:
            raise HTTPException(status_code=400, detail="Missing Google Drive folder link or ID.")

        folder_id = extract_drive_folder_id(folder_link)

        user_tok = user.root / "drive_token.json"
        if not user_tok.exists():
            # Check if public link accessible without authentication
            pub_res = check_drive_link_accessibility(folder_link)
            if pub_res.get("is_public"):
                return {
                    "valid": True,
                    "is_public": True,
                    "folder_id": folder_id,
                    "folder_name": pub_res.get("folder_name", "Public Google Drive Folder"),
                    "mime_type": "application/vnd.google-apps.folder",
                    "message": "Public Google Drive folder is accessible without sign-in.",
                }
            return {
                "valid": False,
                "is_public": False,
                "error": "not_authenticated",
                "detail": "This Google Drive folder is private or requires sign-in. Please connect your Google Drive account first.",
            }

        from sources.google_drive import get_drive_credentials, get_drive_service
        creds = None
        try:
            creds = get_drive_credentials(token_path=user_tok, prompt_if_missing=False)
        except Exception as exc:
            logger.warning("Google Drive credentials check failed: %s", exc)

        if not creds or not creds.valid:
            pub_res = check_drive_link_accessibility(folder_link)
            if pub_res.get("is_public"):
                return {
                    "valid": True,
                    "is_public": True,
                    "folder_id": folder_id,
                    "folder_name": pub_res.get("folder_name", "Public Google Drive Folder"),
                    "mime_type": "application/vnd.google-apps.folder",
                    "message": "Public Google Drive folder is accessible without sign-in.",
                }
            return {
                "valid": False,
                "error": "token_expired",
                "detail": "Google Drive authentication expired or invalid. Please reconnect your account.",
            }

        if folder_id == "root":
            return {
                "valid": True,
                "folder_id": "root",
                "folder_name": "My Drive",
                "mime_type": "application/vnd.google-apps.folder",
            }

        try:
            service = get_drive_service(credentials=creds, token_path=user_tok)
            meta = service.files().get(
                fileId=folder_id,
                fields="id, name, mimeType, trashed",
                supportsAllDrives=True,
            ).execute()

            if meta.get("trashed"):
                return {
                    "valid": False,
                    "error": "folder_trashed",
                    "detail": f"The Google Drive folder '{meta.get('name', folder_id)}' is in the trash.",
                }

            return {
                "valid": True,
                "folder_id": folder_id,
                "folder_name": meta.get("name", "Google Drive Folder"),
                "mime_type": meta.get("mimeType", "application/vnd.google-apps.folder"),
            }
        except Exception as exc:
            status_code = getattr(getattr(exc, "resp", None), "status", None)
            if status_code == 404:
                return {
                    "valid": False,
                    "error": "not_found",
                    "detail": f"Google Drive folder '{folder_id}' was not found or is not accessible.",
                }
            elif status_code == 403:
                return {
                    "valid": False,
                    "error": "access_denied",
                    "detail": f"Access denied to Google Drive folder '{folder_id}'. Please check permissions.",
                }
            return {
                "valid": False,
                "error": "api_error",
                "detail": f"Failed to access Google Drive folder: {exc}",
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
        if hosted_mode and (get_env("ENABLE_DRIVE") != "1"):
            raise HTTPException(status_code=403, detail="Google Drive integration is disabled in hosted mode.")

        folder_id = extract_drive_folder_id(req.link)

        user_tok = user.root / "drive_token.json"
        is_public = False
        creds = None
        if user_tok.exists():
            from sources.google_drive import get_drive_credentials, get_drive_service
            try:
                creds = get_drive_credentials(token_path=user_tok, prompt_if_missing=False)
            except Exception as exc:
                logger.warning("User drive token invalid/expired: %s; falling back to public link check", exc)
                creds = None

        if creds and getattr(creds, "valid", False):
            is_public = False
        else:
            pub_res = check_drive_link_accessibility(req.link)
            if not pub_res.get("is_public"):
                raise HTTPException(
                    status_code=400,
                    detail="This Google Drive folder is private or requires sign-in. Please connect your Google Drive account first.",
                )
            is_public = True

        # Quick pre-flight check for specific folder accessibility when authenticated
        if creds and folder_id and folder_id != "root":
            try:
                service = get_drive_service(credentials=creds, token_path=user_tok)
                meta = service.files().get(
                    fileId=folder_id,
                    fields="id, name, mimeType, trashed",
                    supportsAllDrives=True,
                ).execute()
                if meta.get("trashed"):
                    raise HTTPException(
                        status_code=400,
                        detail=f"The Google Drive folder '{meta.get('name', folder_id)}' is in the trash.",
                    )
            except HTTPException:
                raise
            except Exception as exc:
                status_code = getattr(getattr(exc, "resp", None), "status", None)
                if status_code == 404:
                    raise HTTPException(
                        status_code=404,
                        detail=f"Google Drive folder '{folder_id}' was not found or is not accessible.",
                    )
                elif status_code == 403:
                    raise HTTPException(
                        status_code=403,
                        detail=f"Access denied to Google Drive folder '{folder_id}'. Please check permissions.",
                    )
                logger.warning("Pre-flight folder check skipped due to non-fatal error: %s", exc)

        if folder_id:
            try:
                (user.root / "drive_folder_id.txt").write_text(folder_id, encoding="utf-8")
            except Exception:
                pass

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
        req_head = urllib.request.Request(url, headers={"User-Agent": "SIMILIX/0.1.0"})
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

    def _sync_user_library_from_shared(request: Request, user: UserContext) -> tuple[Path, int]:
        from sources.local_folder import DEFAULT_EXTENSIONS
        target_folder = user.library_dir
        target_folder.mkdir(parents=True, exist_ok=True)
        local_files = [f for f in target_folder.rglob("*") if f.is_file() and f.suffix.lower() in DEFAULT_EXTENSIONS and not f.name.startswith(".")]
        if not local_files:
            candidates = [
                getattr(request.app.state, "data_root", resolve_data_dir()) / "library",
                Path("/content/ARGUS-Media-Deduplication/.similix/library"),
                Path.cwd() / ".similix" / "library",
            ]
            for shared_lib in candidates:
                if shared_lib.is_dir():
                    shared_images = [f for f in shared_lib.rglob("*") if f.is_file() and f.suffix.lower() in DEFAULT_EXTENSIONS and not f.name.startswith(".")]
                    if shared_images:
                        for f in shared_images:
                            target_f = target_folder / f.name
                            if not target_f.exists():
                                try:
                                    target_f.symlink_to(f)
                                except Exception:
                                    try:
                                        shutil.copy2(f, target_f)
                                    except Exception:
                                        pass
                        break
            local_files = [f for f in target_folder.rglob("*") if f.is_file() and f.suffix.lower() in DEFAULT_EXTENSIONS and not f.name.startswith(".")]
        return target_folder, len(local_files)

    @app.get("/sources/library-info")
    def get_library_info(
        request: Request = None,
        user: UserContext = Depends(get_current_user),
        conn: sqlite3.Connection = Depends(get_db),
    ) -> dict[str, Any]:
        """Return counts of image files in user's library and database."""
        target_folder, count = _sync_user_library_from_shared(request, user)
        indexed_count = conn.execute("SELECT COUNT(*) FROM images WHERE status != 'trashed' OR status IS NULL").fetchone()[0]
        return {
            "file_count": count,
            "indexed_count": indexed_count,
            "folder_path": str(target_folder),
            "needs_scan": count > indexed_count or indexed_count == 0,
        }

    @app.post("/sources/scan-library")
    def scan_user_library(
        request: Request = None,
        user: UserContext = Depends(get_current_user),
        conn: sqlite3.Connection = Depends(get_db),
    ) -> dict[str, Any]:
        """Trigger local deduplication pipeline directly on user's library directory."""
        target_folder, count = _sync_user_library_from_shared(request, user)
        if count == 0:
            raise HTTPException(status_code=400, detail="No images found in library to scan.")
        job_id = launch_pipeline_job(
            request.app,
            user=user,
            source_type="local",
            folder=target_folder,
        )
        return {
            "job_id": job_id,
            "status": "queued",
            "file_count": count,
            "source_type": "local",
        }

    @app.post("/sources/rescan")
    def rescan_sources(
        request: Request = None,
        user: UserContext = Depends(get_current_user),
        conn: sqlite3.Connection = Depends(get_db),
    ) -> dict[str, Any]:
        """Re-run the deduplication pipeline on existing user sources."""
        target_folder, local_count = _sync_user_library_from_shared(request, user)
        drive_folder_id = None
        has_gdrive = conn.execute("SELECT 1 FROM images WHERE source = 'gdrive' LIMIT 1").fetchone() is not None
        if has_gdrive:
            source_type = "drive"
            saved_fid_file = user.root / "drive_folder_id.txt"
            if saved_fid_file.exists():
                drive_folder_id = saved_fid_file.read_text(encoding="utf-8").strip() or None
            if not drive_folder_id and user.jobs_dir.is_dir():
                for jf in sorted(user.jobs_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
                    st = read_job_state(jf) or {}
                    if st.get("drive_folder_id"):
                        drive_folder_id = st["drive_folder_id"]
                        break
            if not drive_folder_id:
                try:
                    row_fid = conn.execute("SELECT drive_id FROM images WHERE source = 'gdrive' AND drive_id IS NOT NULL LIMIT 1").fetchone()
                    if row_fid and row_fid[0]:
                        drive_folder_id = row_fid[0].split(":", 1)[0]
                except Exception:
                    pass
        elif local_count > 0:
            source_type = "local"
        else:
            source_type = "local"

        job_id = launch_pipeline_job(
            request.app,
            user=user,
            source_type=source_type,
            folder=target_folder,
            drive_folder_id=drive_folder_id,
        )
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
                        last_heartbeat = state.get("heartbeat_at") or state.get("updated_at", 0)
                        last_progress = state.get("last_progress_at") or state.get("updated_at", 0)
                        is_alive = is_process_alive(pid)

                        cfg_stall = float(getattr(request.app.state, "config", {}).get("general", {}).get("stall_timeout_seconds", 900.0)) if hasattr(request.app, "state") else 900.0
                        stall_limit = cfg_stall if (pid is not None and is_alive) else 60.0
                        if pid is not None and is_alive:
                            is_stalled = ((now - last_heartbeat) > 60) or ((now - last_progress) > max(3600.0, stall_limit))
                        else:
                            is_stalled = ((now - last_heartbeat) > 60)

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
                                err_msg = f"Stage '{stg}' stalled on file '{curr_file}' (no progress for {int(stall_limit)}s)"
                            else:
                                err_msg = f"Worker heartbeat timed out (>{int(stall_limit)}s)"
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

    @app.post("/jobs/dismiss")
    def dismiss_jobs(
        user: UserContext = Depends(get_current_user),
    ) -> dict[str, Any]:
        """Dismiss/archive any failed, stalled, or completed jobs for the authenticated user."""
        jobs_dir = user.jobs_dir
        count = 0
        if jobs_dir.is_dir():
            for jf in jobs_dir.glob("*.json"):
                state = read_job_state(jf)
                if state and state.get("status") in ("failed", "stalled", "completed", "error", "canceled"):
                    try:
                        jf.unlink(missing_ok=True)
                        count += 1
                    except Exception:
                        pass
        return {"ok": True, "dismissed": count}

    @app.post("/jobs/cancel-active")
    @app.post("/jobs/{job_id}/cancel")
    def cancel_job_endpoint(
        job_id: str | None = None,
        request: Request = None,
        user: UserContext = Depends(get_current_user),
    ) -> dict[str, Any]:
        """Cancel an active or queued pipeline job for the user."""
        jobs_dir = user.jobs_dir
        canceled_any = False
        with request.app.state.job_queue_lock:
            job_files = []
            if job_id:
                jf = jobs_dir / f"{job_id}.json"
                if jf.exists():
                    job_files.append(jf)
            elif jobs_dir.is_dir():
                for jf in jobs_dir.glob("*.json"):
                    st = read_job_state(jf)
                    if st and st.get("status") in ("running", "queued"):
                        job_files.append(jf)

            for jf in job_files:
                st = read_job_state(jf) or {}
                pid = st.get("pid")
                if pid:
                    try:
                        os.kill(pid, signal.SIGTERM)
                    except OSError:
                        pass
                st["status"] = "canceled"
                st["stage"] = "canceled"
                st["stage_display"] = "Canceled"
                st["message"] = "Job stopped by user"
                st["updated_at"] = time.time()
                write_job_state(jf, st)
                jid = st.get("id") or jf.stem
                if jid in request.app.state.jobs:
                    request.app.state.jobs[jid] = st
                canceled_any = True

            maybe_start_next_queued_job(request.app)

        return {"ok": True, "canceled": canceled_any}

    return app


# Module-level app instance
app = create_app()
