"""
sources/google_drive.py — Google Drive implementation of ImageSource.

READ-ONLY integration:
- Scopes strictly limited to https://www.googleapis.com/auth/drive.readonly
- No trash, delete, or write operations to Google Drive.
- Stage 0 exact duplicate detection from metadata only (0 downloads).
- Stage 1-2 perceptual hashing and embeddings via =s512 thumbnail links,
  cached locally under ~/.argus/drive_cache/.
- Stage 3-4 full file download strictly for clustered members only, streamed to
  temporary storage and immediately removed after use.
- Rate limiting with exponential backoff on 403, 429, and 5xx errors.
- Bandwidth accounting tracked in SQLite drive_bandwidth table.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import io
import json
import logging
import os
import random
import re
import sqlite3
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

from PIL import Image

from sources.base import ImageMetadata, ImageSource
from sources.local_folder import DEFAULT_EXTENSIONS

logger = logging.getLogger("argus.gdrive")

DRIVE_READONLY_SCOPE = "https://www.googleapis.com/auth/drive.readonly"
SCOPES = [DRIVE_READONLY_SCOPE]

DEFAULT_CLIENT_SECRET_PATH = Path(os.environ.get("ARGUS_DATA_DIR", "~/.argus")).expanduser() / "client_secret.json"
DEFAULT_TOKEN_PATH = Path(os.environ.get("ARGUS_DATA_DIR", "~/.argus")).expanduser() / "drive_token.json"
DEFAULT_CACHE_DIR = Path(os.environ.get("ARGUS_DATA_DIR", "~/.argus")).expanduser() / "drive_cache"
DEFAULT_MAX_CACHE_BYTES = 2 * 1024 * 1024 * 1024  # 2 GB

DRIVE_LIST_FIELDS = (
    "nextPageToken, files("
    "id, name, mimeType, size, md5Checksum, modifiedTime, "
    "imageMediaMetadata(width,height), thumbnailLink, parents, trashed"
    ")"
)


def build_drive_uri(file_id: str, name: str) -> str:
    """Construct virtual path drive://<id>/<name>."""
    clean_name = name.replace("/", "_")
    return f"drive://{file_id}/{clean_name}"


def parse_drive_uri(uri: str) -> tuple[str, str]:
    """Parse drive://<id>/<name> into (file_id, name)."""
    if uri.startswith("drive://"):
        remainder = uri[len("drive://"):]
        if "/" in remainder:
            file_id, name = remainder.split("/", 1)
            return file_id, name
        return remainder, ""
    return uri, ""


def retry_with_backoff(
    fn: Callable[..., Any],
    *args: Any,
    max_retries: int = 5,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    **kwargs: Any,
) -> Any:
    """
    Execute callable with exponential backoff on retryable HTTP errors (403, 429, 5xx).
    """
    attempt = 0
    while True:
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            attempt += 1
            if attempt > max_retries:
                raise

            retryable = False
            status_code = None

            # Handle googleapiclient.errors.HttpError
            if hasattr(exc, "resp") and hasattr(exc.resp, "status"):
                status_code = getattr(exc.resp, "status")
            elif hasattr(exc, "code"):
                status_code = getattr(exc, "code")

            if status_code in (403, 429, 500, 502, 503, 504):
                retryable = True
            elif isinstance(exc, (urllib.error.URLError, ConnectionError, TimeoutError)):
                retryable = True

            if not retryable:
                raise

            delay = initial_delay * (backoff_factor ** (attempt - 1)) + random.uniform(0, 0.5)
            logger.warning(
                "Transient error %s (status=%s). Retrying in %.2fs (attempt %d/%d)...",
                exc,
                status_code,
                delay,
                attempt,
                max_retries,
            )
            time.sleep(delay)


def get_drive_credentials(
    client_secret_path: str | Path | None = None,
    token_path: str | Path | None = None,
    prompt_if_missing: bool = True,
) -> Any:
    """
    Retrieve or initialize OAuth 2.0 credentials for Google Drive read-only access.
    Stores the token with secure 0o600 permissions.
    """
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow

    sec_path = Path(client_secret_path or DEFAULT_CLIENT_SECRET_PATH).expanduser()
    tok_path = Path(token_path or DEFAULT_TOKEN_PATH).expanduser()

    creds = None
    if tok_path.exists():
        try:
            creds = Credentials.from_authorized_user_file(str(tok_path), SCOPES)
        except Exception as exc:
            logger.warning("Could not read credentials from %s: %s", tok_path, exc)
            creds = None

    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            # Save refreshed credentials
            tok_path.parent.mkdir(parents=True, exist_ok=True)
            tok_path.write_text(creds.to_json())
            tok_path.chmod(0o600)
            return creds
        except Exception as exc:
            logger.warning("Could not refresh token: %s", exc)
            creds = None

    if not creds or not creds.valid:
        if not prompt_if_missing:
            raise FileNotFoundError(
                f"Valid Google Drive credentials not found at {tok_path}. "
                f"Run `argus drive-auth` to authorize."
            )

        if not sec_path.exists():
            raise FileNotFoundError(
                f"OAuth client secrets file not found at {sec_path}.\n"
                f"Please download your OAuth client credentials JSON from Google Cloud Console "
                f"and save to {sec_path}."
            )

        flow = InstalledAppFlow.from_client_secrets_file(str(sec_path), SCOPES)
        creds = flow.run_local_server(port=0)

        # Save the token with strict 0o600 permissions
        tok_path.parent.mkdir(parents=True, exist_ok=True)
        tok_path.write_text(creds.to_json())
        tok_path.chmod(0o600)

    return creds


def get_drive_service(
    credentials: Any = None,
    client_secret_path: str | Path | None = None,
    token_path: str | Path | None = None,
) -> Any:
    """Build and return an authorized Google Drive v3 Resource."""
    from googleapiclient.discovery import build

    if credentials is None:
        credentials = get_drive_credentials(
            client_secret_path=client_secret_path,
            token_path=token_path,
            prompt_if_missing=False,
        )
    return build("drive", "v3", credentials=credentials, cache_discovery=False)


_db_lock = threading.RLock()


def record_bandwidth(
    conn: sqlite3.Connection | None,
    stage: str,
    file_id: str,
    bytes_count: int,
    item_type: str,
) -> None:
    """Record network bytes downloaded in the drive_bandwidth table."""
    if conn is None or bytes_count <= 0:
        return
    try:
        with _db_lock:
            conn.execute(
                """
                INSERT INTO drive_bandwidth (stage, file_id, bytes_downloaded, item_type)
                VALUES (?, ?, ?, ?)
                """,
                (stage, file_id, bytes_count, item_type),
            )
            conn.commit()
    except Exception as exc:
        logger.debug("Failed to record drive bandwidth: %s", exc)


def _format_bytes(b: int) -> str:
    """Format byte counts nicely."""
    if b < 1024:
        return f"{b} B"
    elif b < 1024 * 1024:
        return f"{b / 1024:.1f} KB"
    elif b < 1024 * 1024 * 1024:
        return f"{b / (1024 * 1024):.2f} MB"
    else:
        return f"{b / (1024 * 1024 * 1024):.2f} GB"


def print_bandwidth_summary(conn: sqlite3.Connection) -> None:
    """Print comprehensive summary of Google Drive bandwidth usage vs full downloads."""
    cursor = conn.execute(
        """
        SELECT stage, item_type, SUM(bytes_downloaded), COUNT(*)
        FROM drive_bandwidth
        GROUP BY stage, item_type
        ORDER BY stage
        """
    )
    rows = cursor.fetchall()
    total_downloaded = sum(r[2] for r in rows) if rows else 0

    total_lib_size_row = conn.execute(
        "SELECT SUM(file_size), COUNT(*) FROM images WHERE source = 'gdrive'"
    ).fetchone()
    total_lib_bytes = (total_lib_size_row[0] or 0) if total_lib_size_row else 0
    total_images = (total_lib_size_row[1] or 0) if total_lib_size_row else 0

    print("==================================================")
    print("  Google Drive Bandwidth Accounting Summary")
    print("==================================================")
    print(f"Total Google Drive images indexed : {total_images}")
    print(f"Full library uncompressed size    : {_format_bytes(total_lib_bytes)}")
    print()
    print("Downloads by stage:")
    if not rows:
        print("  - Stage 0 (metadata only)        :       0 B (0 network downloads)")
    else:
        for stage, item_type, b, count in rows:
            print(f"  - {stage.upper():<8} ({item_type:<10})      : {_format_bytes(b):>10} ({count} items)")
    print()
    print(f"Total network bytes consumed      : {_format_bytes(total_downloaded)}")
    if total_lib_bytes > 0:
        saved_bytes = max(0, total_lib_bytes - total_downloaded)
        saved_pct = (saved_bytes / total_lib_bytes) * 100.0
        print(f"Bandwidth saved vs full download  : {_format_bytes(saved_bytes)} ({saved_pct:.1f}% saved)")
    print("==================================================")


class GoogleDriveSource(ImageSource):
    """
    ImageSource implementation for Google Drive.
    Operates strictly with read-only permissions and bandwidth minimization.
    """

    def __init__(
        self,
        folder_id: str | None = None,
        all_drive: bool = False,
        service: Any = None,
        credentials: Any = None,
        cache_dir: str | Path | None = None,
        temp_dir: str | Path | None = None,
        conn: sqlite3.Connection | None = None,
        concurrency: int = 4,
        stage: str = "stage0",
        max_file_download_size: int = 100 * 1024 * 1024,
        max_cache_bytes: int = DEFAULT_MAX_CACHE_BYTES,
        config: dict | None = None,
        token_path: str | Path | None = None,
    ):
        self.folder_id = folder_id
        self.all_drive = all_drive
        self._service = service
        self._credentials = credentials
        self.token_path = Path(token_path).expanduser() if token_path else None
        self.config = config or {}
        cfg_cache = self.config.get("sources", {}).get("google_drive", {}).get("cache_dir")
        self.cache_dir = Path(cache_dir or cfg_cache or DEFAULT_CACHE_DIR).expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir = Path(temp_dir) if temp_dir else None
        self.conn = conn
        self.concurrency = max(1, concurrency)
        self.stage = stage
        self.max_file_download_size = max_file_download_size
        self.max_cache_bytes = max_cache_bytes

        self.skipped_files: list[tuple[str, str]] = []
        self._metadata_cache: dict[str, ImageMetadata] = {}
        self._drive_item_cache: dict[str, dict[str, Any]] = {}
        self._cache_lock = threading.RLock()
        self._api_lock = threading.RLock()
        self._db_lock = _db_lock

    def _enforce_cache_cap(self, incoming_bytes: int = 0) -> None:
        """
        Enforce max_cache_bytes limit on cache_dir using LRU eviction based on access time (st_atime).
        """
        with self._cache_lock:
            if not self.cache_dir.exists():
                return

            entries: list[tuple[float, int, Path]] = []
            total_size = 0
            try:
                for p in self.cache_dir.iterdir():
                    if p.is_file() and not p.name.startswith("."):
                        stat = p.stat()
                        entries.append((stat.st_atime, stat.st_size, p))
                        total_size += stat.st_size
            except OSError:
                return

            target_size = self.max_cache_bytes - incoming_bytes
            if total_size <= target_size:
                return

            # Sort by access time ascending (oldest accessed first = LRU)
            entries.sort(key=lambda x: x[0])

            for atime, size, path in entries:
                try:
                    path.unlink(missing_ok=True)
                    total_size -= size
                except OSError:
                    pass
                if total_size <= target_size:
                    break

    @property
    def service(self) -> Any:
        if self._service is None:
            self._service = get_drive_service(credentials=self.credentials, token_path=self.token_path)
        return self._service

    @property
    def credentials(self) -> Any:
        if self._credentials is None:
            self._credentials = get_drive_credentials(token_path=self.token_path, prompt_if_missing=False)
        return self._credentials

    def parse_file_id(self, file_id_or_uri: str) -> tuple[str, str]:
        """Normalize file_id or virtual URI into (drive_id, name)."""
        return parse_drive_uri(file_id_or_uri)

    def list_files(self) -> Iterator[str]:
        """
        Recursively query Google Drive for image files.
        Paginated listing requesting only necessary metadata fields.
        Skips files without md5Checksum.
        """
        self.skipped_files.clear()
        folder_queue: list[str | None] = []

        if self.all_drive:
            folder_queue.append(None)
        elif self.folder_id:
            folder_queue.append(self.folder_id)
        else:
            folder_queue.append("root")

        while folder_queue:
            curr_folder = folder_queue.pop(0)
            page_token = None

            while True:
                # Query construction
                conditions = ["trashed = false"]
                if curr_folder is not None:
                    conditions.append(f"'{curr_folder}' in parents")

                query_str = " and ".join(conditions)

                req = self.service.files().list(
                    q=query_str,
                    pageSize=1000,
                    fields=DRIVE_LIST_FIELDS,
                    pageToken=page_token,
                    supportsAllDrives=True,
                    includeItemsFromAllDrives=True,
                )
                res = retry_with_backoff(lambda: req.execute())
                res_bytes = len(json.dumps(res).encode("utf-8"))
                record_bandwidth(self.conn, self.stage, curr_folder or "drive_root", res_bytes, "metadata")

                files = res.get("files", [])
                for item in files:
                    if item.get("trashed"):
                        continue

                    mime_type = item.get("mimeType", "")
                    # Discover child folders if scoped to folder
                    if mime_type == "application/vnd.google-apps.folder":
                        if not self.all_drive:
                            folder_queue.append(item["id"])
                        continue

                    name = item.get("name", "")
                    ext = Path(name).suffix.lower()

                    # Filter for image files
                    is_image = mime_type.startswith("image/") or ext in DEFAULT_EXTENSIONS
                    if not is_image:
                        continue

                    drive_id = item["id"]
                    virtual_path = build_drive_uri(drive_id, name)

                    # Exact duplicates require md5Checksum
                    md5 = item.get("md5Checksum")
                    if not md5:
                        self.skipped_files.append((virtual_path, "missing md5Checksum"))
                        continue

                    # Parse modifiedTime
                    mtime_raw = item.get("modifiedTime")
                    mtime_ts = 0.0
                    if mtime_raw:
                        try:
                            dt = datetime.fromisoformat(mtime_raw.replace("Z", "+00:00"))
                            mtime_ts = dt.timestamp()
                        except Exception:
                            mtime_ts = time.time()

                    # Image dimensions
                    imm = item.get("imageMediaMetadata") or {}
                    w = imm.get("width")
                    h = imm.get("height")

                    # Image format
                    fmt = None
                    if "/" in mime_type:
                        fmt = mime_type.split("/")[1].upper()
                    elif ext:
                        fmt = ext.lstrip(".").upper()

                    size = int(item.get("size", 0))

                    meta = ImageMetadata(
                        file_id=virtual_path,
                        file_size=size,
                        modified_time=mtime_ts,
                        width=w,
                        height=h,
                        format=fmt,
                        md5=md5,
                        source="gdrive",
                        drive_id=drive_id,
                    )

                    self._metadata_cache[virtual_path] = meta
                    self._metadata_cache[drive_id] = meta
                    self._drive_item_cache[drive_id] = item

                    yield virtual_path

                page_token = res.get("nextPageToken")
                if not page_token:
                    break

    def get_metadata(self, file_id: str) -> ImageMetadata:
        """Return ImageMetadata from cache or directly via Google Drive API."""
        with self._cache_lock:
            if file_id in self._metadata_cache:
                return self._metadata_cache[file_id]

            drive_id, name = self.parse_file_id(file_id)
            if drive_id in self._metadata_cache:
                return self._metadata_cache[drive_id]

        # Fetch metadata from API
        with self._api_lock:
            req = self.service.files().get(
                fileId=drive_id,
                fields="id, name, mimeType, size, md5Checksum, modifiedTime, imageMediaMetadata(width,height), thumbnailLink, parents, trashed",
                supportsAllDrives=True,
            )
            item = retry_with_backoff(lambda: req.execute())
        res_bytes = len(json.dumps(item).encode("utf-8"))
        record_bandwidth(self.conn, self.stage, drive_id, res_bytes, "metadata")
        with self._cache_lock:
            self._drive_item_cache[drive_id] = item

        mtime_raw = item.get("modifiedTime")
        mtime_ts = 0.0
        if mtime_raw:
            try:
                dt = datetime.fromisoformat(mtime_raw.replace("Z", "+00:00"))
                mtime_ts = dt.timestamp()
            except Exception:
                mtime_ts = time.time()

        imm = item.get("imageMediaMetadata") or {}
        w = imm.get("width")
        h = imm.get("height")
        mime_type = item.get("mimeType", "")
        item_name = item.get("name", name)
        ext = Path(item_name).suffix.lower()

        fmt = None
        if "/" in mime_type:
            fmt = mime_type.split("/")[1].upper()
        elif ext:
            fmt = ext.lstrip(".").upper()

        virtual_path = build_drive_uri(drive_id, item_name)
        meta = ImageMetadata(
            file_id=virtual_path,
            file_size=int(item.get("size", 0)),
            modified_time=mtime_ts,
            width=w,
            height=h,
            format=fmt,
            md5=item.get("md5Checksum"),
            source="gdrive",
            drive_id=drive_id,
        )
        with self._cache_lock:
            self._metadata_cache[virtual_path] = meta
            self._metadata_cache[drive_id] = meta
        return meta

    def _fetch_fresh_thumbnail_link(self, drive_id: str) -> str | None:
        """Request fresh thumbnailLink when existing link expired."""
        with self._api_lock:
            req = self.service.files().get(
                fileId=drive_id,
                fields="thumbnailLink",
                supportsAllDrives=True,
            )
            item = retry_with_backoff(lambda: req.execute())
        res_bytes = len(json.dumps(item).encode("utf-8"))
        record_bandwidth(self.conn, self.stage, drive_id, res_bytes, "metadata")
        link = item.get("thumbnailLink")
        with self._cache_lock:
            if drive_id in self._drive_item_cache and link:
                self._drive_item_cache[drive_id]["thumbnailLink"] = link
        return link

    def _download_thumbnail_bytes(
        self,
        link: str,
        timeout: float = 20.0,
        max_retries: int = 3,
    ) -> bytes | None:
        """Download thumbnail bytes from URL with per-request 20s timeout and 3 retries with backoff."""
        headers = {"User-Agent": "Argus/1.0"}
        try:
            creds = self.credentials
            if creds and getattr(creds, "token", None):
                headers["Authorization"] = f"Bearer {creds.token}"
        except Exception:
            pass

        attempt = 0
        while True:
            try:
                req = urllib.request.Request(link, headers=headers)
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    if resp.status == 200:
                        return resp.read()
                return None
            except urllib.error.HTTPError as exc:
                if exc.code in (401, 403, 404, 410):
                    # Expired or unauthorized link
                    return None
                if exc.code in (429, 500, 502, 503, 504) and attempt < max_retries:
                    attempt += 1
                    delay = 0.5 * (2 ** (attempt - 1)) + random.uniform(0, 0.25)
                    logger.warning(
                        "HTTP error %d downloading thumbnail, retrying in %.2fs (attempt %d/%d)...",
                        exc.code,
                        delay,
                        attempt,
                        max_retries,
                    )
                    time.sleep(delay)
                    continue
                logger.debug("HTTP error downloading thumbnail: %s", exc)
                return None
            except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as exc:
                if attempt < max_retries:
                    attempt += 1
                    delay = 0.5 * (2 ** (attempt - 1)) + random.uniform(0, 0.25)
                    logger.warning(
                        "Network error %s downloading thumbnail, retrying in %.2fs (attempt %d/%d)...",
                        exc,
                        delay,
                        attempt,
                        max_retries,
                    )
                    time.sleep(delay)
                    continue
                logger.debug("Error downloading thumbnail: %s", exc)
                return None
            except Exception as exc:
                logger.debug("Unexpected error downloading thumbnail: %s", exc)
                return None

    def fetch_thumbnails(
        self,
        file_ids: list[str],
        max_workers: int = 8,
        progress_callback: Callable[[int, int, str], None] | None = None,
        size: tuple[int, int] = (256, 256),
    ) -> dict[str, Image.Image]:
        """
        Fetch Drive thumbnails with a thread pool (8 workers), per-request timeout 20s,
        3 retries with backoff, and report "N of M" after every file.
        Cache by file id + modifiedTime; skip files already processed.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        total = len(file_ids)
        if total == 0:
            return {}

        # Preload metadata / mtime for the requested files to avoid concurrent DB queries
        if self.conn is not None:
            with self._cache_lock:
                missing_paths = [
                    fid for fid in file_ids
                    if fid not in self._metadata_cache and self.parse_file_id(fid)[0] not in self._metadata_cache
                ]
            if missing_paths:
                with self._db_lock:
                    try:
                        for i in range(0, len(missing_paths), 500):
                            chunk = missing_paths[i : i + 500]
                            placeholders = ",".join("?" for _ in chunk)
                            cursor = self.conn.execute(
                                f"SELECT file_path, drive_id, modified_time FROM images WHERE file_path IN ({placeholders})",
                                chunk,
                            )
                            with self._cache_lock:
                                for r_path, r_did, r_mtime in cursor.fetchall():
                                    m_ts = float(r_mtime or 0.0)
                                    meta = ImageMetadata(
                                        file_id=r_path,
                                        file_size=0,
                                        modified_time=m_ts,
                                        drive_id=r_did,
                                        source="gdrive",
                                    )
                                    self._metadata_cache[r_path] = meta
                                    if r_did:
                                        self._metadata_cache[r_did] = meta
                    except Exception as exc:
                        logger.debug("Failed to preload metadata: %s", exc)

        results: dict[str, Image.Image] = {}
        completed = 0
        lock = threading.Lock()

        def _fetch_single(fid: str) -> tuple[str, Image.Image | None]:
            nonlocal completed
            img = None
            try:
                img = self.get_thumbnail(fid, size=size)
            except Exception as exc:
                logger.warning("Failed to fetch thumbnail for %s: %s", fid, exc)
            with lock:
                completed += 1
                cur = completed
                if progress_callback is not None:
                    try:
                        progress_callback(cur, total, fid)
                    except Exception:
                        pass
            return fid, img

        workers = min(max(1, max_workers), max(1, total))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_fetch_single, fid) for fid in file_ids]
            for fut in as_completed(futures):
                try:
                    fid, img = fut.result()
                    if img is not None:
                        results[fid] = img
                except Exception:
                    pass

        return results

    def get_thumbnail(self, file_id: str, size: tuple[int, int] = (256, 256)) -> Image.Image:
        """
        Retrieve thumbnail from cache or fetch via =s512 link.
        Caches thumbnails under ~/.argus/drive_cache/ keyed by file id + modifiedTime.
        """
        drive_id, _ = self.parse_file_id(file_id)

        # Determine mtime
        mtime = 0
        with self._cache_lock:
            meta = self._metadata_cache.get(drive_id) or self._metadata_cache.get(file_id)
        if meta and meta.modified_time:
            mtime = int(meta.modified_time)
        elif self.conn is not None:
            with self._db_lock:
                try:
                    row = self.conn.execute(
                        "SELECT modified_time FROM images WHERE drive_id = ? OR file_path = ? LIMIT 1",
                        (drive_id, file_id),
                    ).fetchone()
                    if row and row[0]:
                        mtime = int(row[0])
                except Exception as exc:
                    logger.debug("Failed to fetch mtime for %s: %s", file_id, exc)

        cache_path = self.cache_dir / f"{drive_id}_{mtime}.jpg"

        # Check local cache
        if cache_path.exists():
            try:
                if cache_path.stat().st_size > 0:
                    with Image.open(cache_path) as img:
                        thumb = img.convert("RGB")
                        thumb.thumbnail(size, Image.Resampling.LANCZOS)
                        try:
                            now = time.time()
                            os.utime(cache_path, (now, now))
                        except OSError:
                            pass
                        return thumb
            except Exception:
                try:
                    cache_path.unlink(missing_ok=True)
                except OSError:
                    pass

        # Retrieve thumbnailLink
        link = None
        with self._cache_lock:
            if drive_id in self._drive_item_cache:
                link = self._drive_item_cache[drive_id].get("thumbnailLink")

        raw_bytes = None
        if link:
            # Upgrade link to =s512
            if "=s" in link:
                s512_link = re.sub(r"=s\d+.*$", "=s512", link)
            else:
                s512_link = link + "=s512"
            raw_bytes = self._download_thumbnail_bytes(s512_link)

        # If expired or missing, fetch fresh link
        if raw_bytes is None:
            fresh_link = self._fetch_fresh_thumbnail_link(drive_id)
            if not fresh_link:
                raise OSError(f"Google Drive has no thumbnail available for file {drive_id}")

            if "=s" in fresh_link:
                s512_link = re.sub(r"=s\d+.*$", "=s512", fresh_link)
            else:
                s512_link = fresh_link + "=s512"

            raw_bytes = self._download_thumbnail_bytes(s512_link)
            if raw_bytes is None:
                raise OSError(f"Failed to download fresh thumbnail for file {drive_id}")

        # Decode and verify image bytes before caching
        try:
            with Image.open(io.BytesIO(raw_bytes)) as img:
                thumb = img.convert("RGB")
                thumb.thumbnail(size, Image.Resampling.LANCZOS)
        except Exception as exc:
            raise OSError(f"Failed to decode thumbnail bytes for {drive_id}: {exc}") from exc

        # Record bandwidth
        record_bandwidth(self.conn, self.stage, drive_id, len(raw_bytes), "thumbnail")

        # Enforce LRU cache cap before saving
        self._enforce_cache_cap(incoming_bytes=len(raw_bytes))

        # Save to cache atomically
        tmp_cache = self.cache_dir / f".tmp_{drive_id}_{os.getpid()}_{threading.get_ident()}_{time.time_ns()}.jpg"
        try:
            tmp_cache.write_bytes(raw_bytes)
            os.replace(tmp_cache, cache_path)
            try:
                now = time.time()
                os.utime(cache_path, (now, now))
            except OSError:
                pass
        except Exception as exc:
            logger.debug("Failed saving cache for %s: %s", drive_id, exc)
            if tmp_cache.exists():
                try:
                    tmp_cache.unlink(missing_ok=True)
                except OSError:
                    pass

        return thumb

    def get_bytes(self, file_id: str) -> bytes:
        """
        Download full file bytes ONLY for members of candidate clusters.
        Streams to temporary storage, checks size cap, and immediately cleans up.
        Guarantees that unclustered files are NEVER downloaded.
        """
        drive_id, _ = self.parse_file_id(file_id)

        # Security check: must be a member of a cluster
        if self.conn is not None:
            with self._db_lock:
                row = self.conn.execute(
                    "SELECT cluster_id FROM images WHERE (drive_id = ? OR file_path = ?) LIMIT 1",
                    (drive_id, file_id),
                ).fetchone()
            if not row or row[0] is None:
                raise PermissionError(
                    f"Security Invariant: Cannot download full file for unclustered image: {file_id}"
                )
        else:
            raise PermissionError(
                f"Security Invariant: Database connection required to verify cluster membership for {file_id}"
            )

        # Check metadata size cap before downloading
        with self._cache_lock:
            meta = self._metadata_cache.get(drive_id) or self._metadata_cache.get(file_id)
        if meta and meta.file_size > self.max_file_download_size:
            raise ValueError(
                f"File {file_id} size ({meta.file_size} bytes) exceeds download cap ({self.max_file_download_size} bytes)"
            )

        from googleapiclient.http import MediaIoBaseDownload

        tmp_fd, tmp_path_str = tempfile.mkstemp(
            prefix=f"argus_stream_{drive_id}_",
            dir=str(self.temp_dir) if self.temp_dir else None,
        )
        os.close(tmp_fd)
        tmp_path = Path(tmp_path_str)

        try:
            with self._api_lock:
                req = self.service.files().get_media(fileId=drive_id, supportsAllDrives=True)
                with open(tmp_path, "wb") as fh:
                    downloader = MediaIoBaseDownload(fh, req, chunksize=1024 * 1024)
                    done = False
                    while not done:
                        status, done = retry_with_backoff(lambda: downloader.next_chunk())
                        if status and status.resumable_progress > self.max_file_download_size:
                            raise ValueError(
                                f"Streaming download exceeded size cap ({self.max_file_download_size} bytes)"
                            )

            data = tmp_path.read_bytes()
        finally:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

        record_bandwidth(self.conn, self.stage, drive_id, len(data), "full_file")
        return data


@dataclass
class MemberConfirmationRecord:
    member_id: int
    file_path: str
    cluster_id: int
    thumb_match_pct: float | None
    full_match_pct: float | None
    thumb_ssim: float | None
    full_ssim: float | None
    bytes_downloaded: int
    is_delete_safe: bool = False
    reason: str | None = None


@dataclass
class ConfirmDriveResult:
    confirmed_members: list[MemberConfirmationRecord] = field(default_factory=list)
    total_confirmed_bytes: int = 0
    total_library_bytes: int = 0
    bandwidth_ratio_pct: float = 0.0


def run_confirm_drive(
    conn: sqlite3.Connection,
    config: dict | None = None,
    mode: str = "balanced",
    source: GoogleDriveSource | None = None,
) -> ConfirmDriveResult:
    """
    Confirm Drive candidate delete_safe images at full resolution.
    For every Drive member that WOULD be delete_safe in the current mode:
    1. Streams the full file (size cap, temp dir, deleted after use).
    2. Recomputes ssim/ssim_color/match_pct at 1024px against the reference's full file.
    3. Sets quality_basis='full'.
    4. Re-runs existing invariants via run_keeper_pipeline.
    Logs all bytes in drive_bandwidth table under stage='confirm'.
    """
    from core.keeper import choose_keeper, run_keeper_pipeline
    from core.quality import align_and_compute_metrics, ensure_quality_columns

    ensure_quality_columns(conn)

    # Initialize GoogleDriveSource if not provided
    if source is None:
        source = GoogleDriveSource(conn=conn, config=config, stage="confirm")
    else:
        source.stage = "confirm"
        if source.conn is None:
            source.conn = conn

    initial_confirmed_bytes_row = conn.execute(
        "SELECT COALESCE(SUM(bytes_downloaded), 0) FROM drive_bandwidth WHERE stage = 'confirm'"
    ).fetchone()
    initial_confirmed_bytes = int(initial_confirmed_bytes_row[0]) if initial_confirmed_bytes_row else 0

    clusters = conn.execute(
        "SELECT id, representative_id FROM clusters ORDER BY id"
    ).fetchall()

    confirmed_records: list[MemberConfirmationRecord] = []

    for cluster in clusters:
        cid = cluster["id"]
        members_rows = conn.execute(
            "SELECT * FROM images WHERE cluster_id = ? ORDER BY id",
            (cid,),
        ).fetchall()
        if not members_rows or len(members_rows) < 2:
            continue

        members = [dict(r) for r in members_rows]

        # Determine which members WOULD be delete_safe in current mode:
        hypo_members = []
        for m in members:
            m_copy = dict(m)
            m_copy["quality_basis"] = "full"
            hypo_members.append(m_copy)

        hypo_plan = choose_keeper(hypo_members, mode=mode, config=config)
        hypo_delete_safe_ids = set(hypo_plan.get("delete_safe_ids", []))
        keeper_id = hypo_plan.get("keeper_id")
        keeper_m = next((x for x in members if x["id"] == keeper_id), None)
        keeper_md5 = keeper_m.get("md5") if keeper_m else None

        # Identify Drive members that need full-resolution confirmation:
        candidates_to_confirm = []
        for m in members:
            mid = m["id"]
            if mid not in hypo_delete_safe_ids:
                continue
            is_drive = m.get("source") == "gdrive" or str(m.get("file_path", "")).startswith("drive://")
            if not is_drive:
                continue
            is_keeper_copy = bool(keeper_md5 and m.get("md5") and m.get("md5") == keeper_md5)
            if is_keeper_copy:
                continue
            if m.get("quality_basis") == "full":
                continue
            candidates_to_confirm.append(m)

        if not candidates_to_confirm:
            continue

        # Find cluster reference member (representative_id or first member)
        ref_id = cluster["representative_id"]
        ref_m = next((x for x in members if x["id"] == ref_id), members[0])

        # Load reference full file
        ref_path = ref_m["file_path"]
        ref_bytes = source.get_bytes(ref_path)
        ref_img_full = Image.open(io.BytesIO(ref_bytes)).convert("RGB")
        ref_orig_dims = (ref_m.get("width") or ref_img_full.size[0], ref_m.get("height") or ref_img_full.size[1])
        ref_sz = ref_m.get("file_size") or len(ref_bytes)

        # For each candidate, stream full file and recompute at 1024px against ref
        for cand in candidates_to_confirm:
            cand_id = cand["id"]
            cand_path = cand["file_path"]
            thumb_match_pct = cand.get("match_pct")
            thumb_ssim = cand.get("ssim")

            bytes_before_row = conn.execute(
                "SELECT COALESCE(SUM(bytes_downloaded), 0) FROM drive_bandwidth WHERE stage = 'confirm'"
            ).fetchone()
            bytes_before = int(bytes_before_row[0]) if bytes_before_row else 0

            cand_bytes = source.get_bytes(cand_path)
            cand_img_full = Image.open(io.BytesIO(cand_bytes)).convert("RGB")
            cand_orig_dims = (cand.get("width") or cand_img_full.size[0], cand.get("height") or cand_img_full.size[1])
            cand_sz = cand.get("file_size") or len(cand_bytes)

            metrics = align_and_compute_metrics(
                member_path=cand_img_full,
                reference_path=ref_img_full,
                member_id=cand_id,
                reference_id=ref_m["id"],
                member_orig_dims=cand_orig_dims,
                reference_orig_dims=ref_orig_dims,
                member_file_size=cand_sz,
                reference_file_size=ref_sz,
                conn=conn,
                config=config,
                source=None,
            )

            is_exact = bool(cand.get("md5") and cand.get("md5") == ref_m.get("md5"))
            if is_exact:
                match_pct = 100.0
                resolution_pct = 100.0
            else:
                match_pct = round(metrics.ssim_color * 100.0, 1) if metrics.ssim_color is not None else None
                resolution_pct = round(metrics.pixel_ratio * 100.0, 1) if metrics.pixel_ratio is not None else None

            # Update candidate in DB with quality_basis='full'
            conn.execute(
                """
                UPDATE images
                SET ssim = ?,
                    ssim_color = ?,
                    sharpness_ratio = ?,
                    pixel_ratio = ?,
                    bytes_ratio = ?,
                    quality_status = ?,
                    match_pct = ?,
                    resolution_pct = ?,
                    quality_basis = 'full',
                    updated_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now')
                WHERE id = ?
                """,
                (
                    metrics.ssim,
                    metrics.ssim_color,
                    metrics.sharpness_ratio,
                    metrics.pixel_ratio,
                    metrics.bytes_ratio,
                    metrics.quality_status,
                    match_pct,
                    resolution_pct,
                    cand_id,
                ),
            )

            # Update reference in DB with quality_basis='full'
            conn.execute(
                "UPDATE images SET quality_basis = 'full' WHERE id = ?",
                (ref_m["id"],),
            )
            conn.commit()

            bytes_after_row = conn.execute(
                "SELECT COALESCE(SUM(bytes_downloaded), 0) FROM drive_bandwidth WHERE stage = 'confirm'"
            ).fetchone()
            bytes_after = int(bytes_after_row[0]) if bytes_after_row else 0

            confirmed_records.append(
                MemberConfirmationRecord(
                    member_id=cand_id,
                    file_path=cand_path,
                    cluster_id=cid,
                    thumb_match_pct=thumb_match_pct,
                    full_match_pct=match_pct,
                    thumb_ssim=thumb_ssim,
                    full_ssim=metrics.ssim,
                    bytes_downloaded=bytes_after - bytes_before,
                )
            )

    # Re-run existing invariants via run_keeper_pipeline
    run_keeper_pipeline(conn, config=config, mode=mode)

    # Populate final delete_safe / reason status for confirmed members
    for rec in confirmed_records:
        row = conn.execute(
            "SELECT delete_safe, reason FROM images WHERE id = ?",
            (rec.member_id,),
        ).fetchone()
        if row:
            rec.is_delete_safe = bool(row["delete_safe"])
            rec.reason = row["reason"]

    # Bandwidth accounting totals
    final_confirmed_bytes_row = conn.execute(
        "SELECT COALESCE(SUM(bytes_downloaded), 0) FROM drive_bandwidth WHERE stage = 'confirm'"
    ).fetchone()
    total_confirmed = (int(final_confirmed_bytes_row[0]) if final_confirmed_bytes_row else 0) - initial_confirmed_bytes

    lib_row = conn.execute("SELECT COALESCE(SUM(file_size), 0) FROM images").fetchone()
    total_library = int(lib_row[0]) if lib_row else 0

    ratio_pct = (total_confirmed / total_library * 100.0) if total_library > 0 else 0.0

    return ConfirmDriveResult(
        confirmed_members=confirmed_records,
        total_confirmed_bytes=total_confirmed,
        total_library_bytes=total_library,
        bandwidth_ratio_pct=round(ratio_pct, 2),
    )
