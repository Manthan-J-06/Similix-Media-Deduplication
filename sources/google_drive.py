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

import io
import json
import logging
import os
import random
import re
import sqlite3
import sys
import tempfile
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

DEFAULT_CLIENT_SECRET_PATH = Path.home() / ".argus" / "client_secret.json"
DEFAULT_TOKEN_PATH = Path.home() / ".argus" / "drive_token.json"
DEFAULT_CACHE_DIR = Path.home() / ".argus" / "drive_cache"

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
    ):
        self.folder_id = folder_id
        self.all_drive = all_drive
        self._service = service
        self._credentials = credentials
        self.cache_dir = Path(cache_dir or DEFAULT_CACHE_DIR).expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir = Path(temp_dir) if temp_dir else None
        self.conn = conn
        self.concurrency = max(1, concurrency)
        self.stage = stage
        self.max_file_download_size = max_file_download_size

        self.skipped_files: list[tuple[str, str]] = []
        self._metadata_cache: dict[str, ImageMetadata] = {}
        self._drive_item_cache: dict[str, dict[str, Any]] = {}

    @property
    def service(self) -> Any:
        if self._service is None:
            self._service = get_drive_service(credentials=self._credentials)
        return self._service

    @property
    def credentials(self) -> Any:
        if self._credentials is None:
            self._credentials = get_drive_credentials(prompt_if_missing=False)
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
        if file_id in self._metadata_cache:
            return self._metadata_cache[file_id]

        drive_id, name = self.parse_file_id(file_id)
        if drive_id in self._metadata_cache:
            return self._metadata_cache[drive_id]

        # Fetch metadata from API
        req = self.service.files().get(
            fileId=drive_id,
            fields="id, name, mimeType, size, md5Checksum, modifiedTime, imageMediaMetadata(width,height), thumbnailLink, parents, trashed",
            supportsAllDrives=True,
        )
        item = retry_with_backoff(lambda: req.execute())
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
        self._metadata_cache[virtual_path] = meta
        self._metadata_cache[drive_id] = meta
        return meta

    def _fetch_fresh_thumbnail_link(self, drive_id: str) -> str | None:
        """Request fresh thumbnailLink when existing link expired."""
        req = self.service.files().get(
            fileId=drive_id,
            fields="thumbnailLink",
            supportsAllDrives=True,
        )
        item = retry_with_backoff(lambda: req.execute())
        link = item.get("thumbnailLink")
        if drive_id in self._drive_item_cache and link:
            self._drive_item_cache[drive_id]["thumbnailLink"] = link
        return link

    def _download_thumbnail_bytes(self, link: str) -> bytes | None:
        """Download thumbnail bytes from URL."""
        headers = {"User-Agent": "Argus/1.0"}
        try:
            creds = self.credentials
            if creds and getattr(creds, "token", None):
                headers["Authorization"] = f"Bearer {creds.token}"
        except Exception:
            pass

        req = urllib.request.Request(link, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                if resp.status == 200:
                    return resp.read()
        except urllib.error.HTTPError as exc:
            if exc.code in (401, 403, 404, 410):
                # Expired or unauthorized link
                return None
            logger.debug("HTTP error downloading thumbnail: %s", exc)
            return None
        except Exception as exc:
            logger.debug("Error downloading thumbnail: %s", exc)
            return None
        return None

    def get_thumbnail(self, file_id: str, size: tuple[int, int] = (256, 256)) -> Image.Image:
        """
        Retrieve thumbnail from cache or fetch via =s512 link.
        Caches thumbnails under ~/.argus/drive_cache/ keyed by file id + modifiedTime.
        """
        drive_id, _ = self.parse_file_id(file_id)

        # Determine mtime
        mtime = 0
        if drive_id in self._metadata_cache:
            mtime = int(self._metadata_cache[drive_id].modified_time)
        elif self.conn is not None:
            row = self.conn.execute(
                "SELECT modified_time FROM images WHERE drive_id = ? OR file_path = ? LIMIT 1",
                (drive_id, file_id),
            ).fetchone()
            if row and row[0]:
                mtime = int(row[0])

        cache_path = self.cache_dir / f"{drive_id}_{mtime}.jpg"

        # Check local cache
        if cache_path.exists() and cache_path.stat().st_size > 0:
            try:
                with Image.open(cache_path) as img:
                    thumb = img.convert("RGB")
                    thumb.thumbnail(size, Image.Resampling.LANCZOS)
                    return thumb
            except Exception:
                cache_path.unlink(missing_ok=True)

        # Retrieve thumbnailLink
        link = None
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

        # Record bandwidth
        record_bandwidth(self.conn, self.stage, drive_id, len(raw_bytes), "thumbnail")

        # Save to cache atomically
        tmp_cache = self.cache_dir / f".tmp_{drive_id}_{os.getpid()}_{time.time_ns()}.jpg"
        try:
            tmp_cache.write_bytes(raw_bytes)
            os.replace(tmp_cache, cache_path)
        except Exception:
            if tmp_cache.exists():
                tmp_cache.unlink(missing_ok=True)

        with Image.open(io.BytesIO(raw_bytes)) as img:
            thumb = img.convert("RGB")
            thumb.thumbnail(size, Image.Resampling.LANCZOS)
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
        meta = self._metadata_cache.get(drive_id) or self._metadata_cache.get(file_id)
        if meta and meta.file_size > self.max_file_download_size:
            raise ValueError(
                f"File {file_id} size ({meta.file_size} bytes) exceeds download cap ({self.max_file_download_size} bytes)"
            )

        from googleapiclient.http import MediaIoBaseDownload

        req = self.service.files().get_media(fileId=drive_id, supportsAllDrives=True)
        tmp_fd, tmp_path_str = tempfile.mkstemp(
            prefix=f"argus_stream_{drive_id}_",
            dir=str(self.temp_dir) if self.temp_dir else None,
        )
        os.close(tmp_fd)
        tmp_path = Path(tmp_path_str)

        try:
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
