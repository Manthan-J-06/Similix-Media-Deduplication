"""
core/auth.py — Authentication, session management, and per-user isolation for Argus.

Features:
- Separate auth.db for user credentials, sessions, and rate-limiting audit log.
- Bcrypt password hashing with constant-time verification.
- Min 10-character password policy.
- Login rate-limiting: 5 failures per 15 minutes per email+IP (returns 429).
- Secure sessions: random 256-bit token stored hashed (SHA-256) in SQLite,
  7-day expiry, rotated on login, HttpOnly + SameSite=Lax cookie.
- CSRF protection: 256-bit token generated per session, required on every POST.
- Per-user directory isolation: ~/.argus/users/<user_id>/ with own DB, library, trash, thumbs, jobs.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
import secrets
import shutil
import sqlite3
import threading
import time
from typing import Any, Callable
import uuid

try:
    import bcrypt
except ImportError:
    bcrypt = None
from fastapi import HTTPException

# Thread lock for synchronizing operations on shared sqlite3 auth connection
_auth_lock = threading.RLock()

# Default rate-limiting parameters: 5 failures within 15 minutes (900 seconds)
DEFAULT_MAX_FAILURES = 5
DEFAULT_FAILURE_WINDOW = 900
DEFAULT_SESSION_EXPIRY_DAYS = 7
MIN_PASSWORD_LENGTH = 10


def init_auth_db(db_path: Path | str) -> sqlite3.Connection:
    """Initialize the auth.db schema for users, sessions, and login audit."""
    path = Path(db_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=10.0, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout=10000;")

    with conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL COLLATE NOCASE,
                password_hash TEXT NOT NULL,
                created_at REAL NOT NULL,
                is_active INTEGER NOT NULL DEFAULT 1
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                token_hash TEXT UNIQUE NOT NULL,
                csrf_token TEXT NOT NULL,
                created_at REAL NOT NULL,
                expires_at REAL NOT NULL,
                last_active_at REAL NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS login_attempts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT NOT NULL COLLATE NOCASE,
                ip TEXT NOT NULL,
                attempted_at REAL NOT NULL,
                success INTEGER NOT NULL
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_token_hash ON sessions(token_hash);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_login_attempts ON login_attempts(email, ip, attempted_at);")

    return conn


def hash_password(password: str) -> str:
    """Hash password using bcrypt (or standard PBKDF2 fallback if bcrypt unavailable)."""
    if len(password) < MIN_PASSWORD_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Password must be at least {MIN_PASSWORD_LENGTH} characters long",
        )
    if bcrypt is not None:
        salt = bcrypt.gensalt(rounds=12)
        return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")
    salt = secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 100_000)
    return f"pbkdf2_sha256${salt}${dk.hex()}"


def verify_password(password: str, hashed_password: str) -> bool:
    """Constant-time password verification supporting bcrypt and PBKDF2."""
    try:
        if hashed_password.startswith("pbkdf2_sha256$"):
            parts = hashed_password.split("$")
            if len(parts) == 3:
                salt, expected_hex = parts[1], parts[2]
                dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 100_000)
                return secrets.compare_digest(dk.hex(), expected_hex)
        if bcrypt is not None:
            return bcrypt.checkpw(password.encode("utf-8"), hashed_password.encode("utf-8"))
    except Exception:
        return False
    return False


def hash_token(raw_token: str) -> str:
    """Return SHA-256 hash of a session token for storage and lookup."""
    return hashlib.sha256(raw_token.encode("utf-8")).hexdigest()


def check_login_rate_limit(
    conn: sqlite3.Connection,
    email: str,
    ip: str,
    window_seconds: int = DEFAULT_FAILURE_WINDOW,
    max_failures: int = DEFAULT_MAX_FAILURES,
) -> bool:
    """Return True if login attempts are within allowed rate limit, False if rate limited."""
    cutoff = time.time() - window_seconds
    clean_email = email.strip().lower()
    clean_ip = ip.strip()

    with _auth_lock:
        row = conn.execute(
            """
            SELECT COUNT(*) FROM login_attempts
            WHERE email = ? AND ip = ? AND attempted_at >= ? AND success = 0
            """,
            (clean_email, clean_ip, cutoff),
        ).fetchone()

    failure_count = row[0] if row else 0
    return failure_count < max_failures


def record_login_attempt(
    conn: sqlite3.Connection,
    email: str,
    ip: str,
    success: bool,
) -> None:
    """Record an attempted login in login_attempts audit table."""
    clean_email = email.strip().lower()
    clean_ip = ip.strip()
    with _auth_lock:
        with conn:
            conn.execute(
                """
                INSERT INTO login_attempts (email, ip, attempted_at, success)
                VALUES (?, ?, ?, ?)
                """,
                (clean_email, clean_ip, time.time(), 1 if success else 0),
            )


def register_user(
    conn: sqlite3.Connection,
    email: str,
    password: str,
    allow_registration: bool = True,
) -> dict[str, Any]:
    """Register a new user in auth.db. Rejects duplicate emails and passwords under 10 chars."""
    if not allow_registration:
        raise HTTPException(
            status_code=403,
            detail="Registration is disabled (invite-only mode)",
        )

    clean_email = email.strip().lower()
    if not clean_email or "@" not in clean_email or len(clean_email) > 255:
        raise HTTPException(status_code=400, detail="Invalid email address format")

    if len(password) < MIN_PASSWORD_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Password must be at least {MIN_PASSWORD_LENGTH} characters long",
        )

    with _auth_lock:
        # Check for existing email
        existing = conn.execute("SELECT id FROM users WHERE email = ?", (clean_email,)).fetchone()
        if existing:
            raise HTTPException(status_code=400, detail="Email is already registered")

        pw_hash = hash_password(password)
        now = time.time()

        with conn:
            cursor = conn.execute(
                """
                INSERT INTO users (email, password_hash, created_at, is_active)
                VALUES (?, ?, ?, 1)
                """,
                (clean_email, pw_hash, now),
            )
            user_id = cursor.lastrowid

    return {
        "id": user_id,
        "email": clean_email,
        "created_at": now,
        "is_active": True,
    }


def authenticate_user(
    conn: sqlite3.Connection,
    email: str,
    password: str,
    ip: str,
) -> dict[str, Any]:
    """Verify email/password against auth.db, enforcing rate-limiting (5 failures / 15m)."""
    clean_email = email.strip().lower()
    clean_ip = ip.strip()

    if not check_login_rate_limit(conn, clean_email, clean_ip):
        raise HTTPException(
            status_code=429,
            detail="Too many failed login attempts. Please try again after 15 minutes.",
        )

    with _auth_lock:
        row = conn.execute(
            "SELECT id, email, password_hash, is_active, created_at FROM users WHERE email = ?",
            (clean_email,),
        ).fetchone()

    if not row:
        record_login_attempt(conn, clean_email, clean_ip, success=False)
        raise HTTPException(status_code=401, detail="Invalid email or password")

    if not row["is_active"]:
        record_login_attempt(conn, clean_email, clean_ip, success=False)
        raise HTTPException(status_code=401, detail="User account is deactivated")

    if not verify_password(password, row["password_hash"]):
        record_login_attempt(conn, clean_email, clean_ip, success=False)
        raise HTTPException(status_code=401, detail="Invalid email or password")

    # Success: record attempt and return user data
    record_login_attempt(conn, clean_email, clean_ip, success=True)
    return {
        "id": row["id"],
        "email": row["email"],
        "created_at": row["created_at"],
        "is_active": bool(row["is_active"]),
    }


def create_session(
    conn: sqlite3.Connection,
    user_id: int,
    expiry_days: int = DEFAULT_SESSION_EXPIRY_DAYS,
) -> tuple[str, str]:
    """Create a new 256-bit session token and CSRF token, rotating out old sessions for the user."""
    raw_token = secrets.token_hex(32)  # 256-bit random hex string
    token_h = hash_token(raw_token)
    csrf_tok = secrets.token_hex(32)   # 256-bit random CSRF token
    now = time.time()
    expires_at = now + (expiry_days * 24 * 3600)

    with _auth_lock:
        with conn:
            # Rotate on login: invalidate all previous sessions for this user
            conn.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
            conn.execute(
                """
                INSERT INTO sessions (user_id, token_hash, csrf_token, created_at, expires_at, last_active_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (user_id, token_h, csrf_tok, now, expires_at, now),
            )

    return raw_token, csrf_tok


def get_session_user(
    conn: sqlite3.Connection,
    raw_token: str,
) -> dict[str, Any] | None:
    """Validate raw session token and return user+session details if valid and unexpired."""
    if not raw_token or not isinstance(raw_token, str):
        return None

    token_h = hash_token(raw_token)
    now = time.time()

    with _auth_lock:
        row = conn.execute(
            """
            SELECT s.id AS session_id, s.user_id, s.csrf_token, s.expires_at, s.last_active_at,
                   u.email, u.is_active, u.created_at AS user_created_at
            FROM sessions s
            JOIN users u ON u.id = s.user_id
            WHERE s.token_hash = ?
            """,
            (token_h,),
        ).fetchone()

        if not row:
            return None

        # Check expiration
        if row["expires_at"] <= now or not row["is_active"]:
            # Clean up expired session
            try:
                with conn:
                    conn.execute("DELETE FROM sessions WHERE id = ?", (row["session_id"],))
            except sqlite3.OperationalError:
                pass
            return None

        # Update last_active_at (throttled to at most once per 60 seconds to avoid write contention)
        last_active = row["last_active_at"] if "last_active_at" in row.keys() else 0
        if now - last_active >= 60:
            try:
                with conn:
                    conn.execute("UPDATE sessions SET last_active_at = ? WHERE id = ?", (now, row["session_id"]))
            except sqlite3.OperationalError:
                pass

    return {
        "user_id": row["user_id"],
        "email": row["email"],
        "csrf_token": row["csrf_token"],
        "expires_at": row["expires_at"],
        "created_at": row["user_created_at"],
    }


def invalidate_session(
    conn: sqlite3.Connection,
    raw_token: str,
) -> None:
    """Invalidate a session on logout."""
    if not raw_token:
        return
    token_h = hash_token(raw_token)
    with _auth_lock:
        with conn:
            conn.execute("DELETE FROM sessions WHERE token_hash = ?", (token_h,))


def get_user_dirs(
    base_users_dir: Path,
    user_id: int,
    custom_overrides: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """
    Resolve and initialize per-user isolated directory tree:
    ~/.similix/users/<user_id>/
      ├── similix.db (or legacy argus.db)
      ├── library/
      ├── trash/
      ├── thumbs/
      ├── jobs/
      └── drive_cache/
    """
    user_root = (base_users_dir / str(user_id)).expanduser().resolve()

    def _resolve_user_db(target_root: Path, override: str | None = None) -> Path:
        if override:
            return Path(override).expanduser().resolve()
        similix_db = target_root / "similix.db"
        argus_db = target_root / "argus.db"
        if argus_db.exists() and not similix_db.exists():
            return argus_db.resolve()
        if argus_db.exists() and similix_db.exists():
            try:
                with sqlite3.connect(argus_db) as a_conn:
                    a_cnt = a_conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]
                if a_cnt > 0:
                    with sqlite3.connect(similix_db) as s_conn:
                        s_cnt = s_conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]
                    if s_cnt == 0:
                        return argus_db.resolve()
            except Exception:
                pass
        if similix_db.exists():
            return similix_db.resolve()
        if argus_db.exists():
            return argus_db.resolve()
        return similix_db.resolve()

    if user_id == 1 and custom_overrides:
        root = Path(custom_overrides.get("root") or user_root).expanduser().resolve()
        db_path = _resolve_user_db(user_root, custom_overrides.get("db_path"))
        library_dir = Path(custom_overrides.get("library_dir") or (user_root / "library")).expanduser().resolve()
        trash_dir = Path(custom_overrides.get("trash_dir") or (user_root / "trash")).expanduser().resolve()
        thumb_cache_dir = Path(custom_overrides.get("thumb_cache_dir") or (user_root / "thumbs")).expanduser().resolve()
        jobs_dir = Path(custom_overrides.get("jobs_dir") or (user_root / "jobs")).expanduser().resolve()
        drive_cache_dir = Path(custom_overrides.get("drive_cache_dir") or (user_root / "drive_cache")).expanduser().resolve()
    else:
        root = user_root
        db_path = _resolve_user_db(user_root)
        library_dir = user_root / "library"
        trash_dir = user_root / "trash"
        thumb_cache_dir = user_root / "thumbs"
        jobs_dir = user_root / "jobs"
        drive_cache_dir = user_root / "drive_cache"

    for d in (root, library_dir, trash_dir, thumb_cache_dir, jobs_dir, drive_cache_dir):
        d.mkdir(parents=True, exist_ok=True)

    return {
        "root": root,
        "db_path": db_path,
        "library_dir": library_dir,
        "trash_dir": trash_dir,
        "thumb_cache_dir": thumb_cache_dir,
        "jobs_dir": jobs_dir,
        "drive_cache_dir": drive_cache_dir,
    }


def validate_confined_user_path(
    target_path: str | Path,
    user_root: Path,
    allow_symlinks: bool = False,
) -> Path:
    """
    Ensure target_path resolves strictly inside user_root.
    Rejects '..' path traversal, symlinks pointing outside, and unconfined targets.
    """
    raw_str = str(target_path)
    if ".." in raw_str:
        raise HTTPException(
            status_code=400,
            detail="Security violation: path traversal ('..') is not permitted",
        )

    p = Path(target_path)
    if not allow_symlinks:
        if p.is_symlink():
            raise HTTPException(
                status_code=400,
                detail="Security violation: symlinks are not permitted",
            )
        curr = p
        while curr != curr.parent:
            if curr.is_symlink():
                raise HTTPException(
                    status_code=400,
                    detail="Security violation: symlinks are not permitted",
                )
            if curr == user_root:
                break
            curr = curr.parent

    resolved = p.resolve()
    user_root_resolved = user_root.resolve()

    try:
        resolved.relative_to(user_root_resolved)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Security violation: path '{target_path}' resolves outside user boundary",
        )

    return resolved


def delete_user_account(
    conn: sqlite3.Connection,
    user_id: int,
    password: str,
    users_dir: Path | str,
    on_before_delete: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Delete a user account, verifying password first, wiping the user's isolated directory and auth DB rows."""
    with _auth_lock:
        row = conn.execute(
            "SELECT id, email, password_hash FROM users WHERE id = ?",
            (user_id,),
        ).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="User not found")

        if not verify_password(password, row["password_hash"]):
            raise HTTPException(status_code=400, detail="Incorrect password confirmation")

        # If password is correct, execute pre-deletion hook (e.g. stop running/queued jobs)
        if on_before_delete is not None:
            on_before_delete()

        # Invalidate ALL sessions for this user and delete rows in auth.db immediately
        with conn:
            conn.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
            conn.execute("DELETE FROM login_attempts WHERE email = ?", (row["email"],))
            conn.execute("DELETE FROM users WHERE id = ?", (user_id,))

    # Instant atomic move out of users_dir + asynchronous purge outside of _auth_lock
    user_path = Path(users_dir).expanduser().resolve() / str(user_id)
    if user_path.exists():
        purge_dir = user_path.parent.parent / "purge"
        target_to_remove = user_path
        try:
            purge_dir.mkdir(parents=True, exist_ok=True)
            purge_target = purge_dir / f"user_{user_id}_{uuid.uuid4().hex}"
            os.replace(str(user_path), str(purge_target))
            target_to_remove = purge_target
        except OSError:
            pass

        def _bg_purge(path_to_clean: Path):
            try:
                if path_to_clean.exists():
                    shutil.rmtree(str(path_to_clean), ignore_errors=True)
            except Exception:
                pass

        threading.Thread(target=_bg_purge, args=(target_to_remove,), daemon=True).start()

    return {"status": "deleted", "user_id": user_id, "email": row["email"]}


def admin_delete_user_account(
    conn: sqlite3.Connection,
    user_identifier: str | int,
    users_dir: Path | str,
) -> dict[str, Any]:
    """Administrative deletion of a user account by email or user ID without requiring password."""
    with _auth_lock:
        if isinstance(user_identifier, int) or str(user_identifier).isdigit():
            row = conn.execute("SELECT id, email FROM users WHERE id = ?", (int(user_identifier),)).fetchone()
        else:
            row = conn.execute("SELECT id, email FROM users WHERE email = ?", (str(user_identifier).strip().lower(),)).fetchone()

        if not row:
            raise ValueError(f"User '{user_identifier}' not found in auth.db")

        user_id = row["id"]
        email = row["email"]

        with conn:
            conn.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
            conn.execute("DELETE FROM login_attempts WHERE email = ?", (email,))
            conn.execute("DELETE FROM users WHERE id = ?", (user_id,))

    user_path = Path(users_dir).expanduser().resolve() / str(user_id)
    if user_path.exists():
        purge_dir = user_path.parent.parent / "purge"
        target_to_remove = user_path
        try:
            purge_dir.mkdir(parents=True, exist_ok=True)
            purge_target = purge_dir / f"user_{user_id}_{uuid.uuid4().hex}"
            os.replace(str(user_path), str(purge_target))
            target_to_remove = purge_target
        except OSError:
            pass

        def _bg_purge(path_to_clean: Path):
            try:
                if path_to_clean.exists():
                    shutil.rmtree(str(path_to_clean), ignore_errors=True)
            except Exception:
                pass

        threading.Thread(target=_bg_purge, args=(target_to_remove,), daemon=True).start()

    return {"status": "deleted", "user_id": user_id, "email": email}
