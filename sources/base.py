"""
sources/base.py — Generic ImageSource interface.

Defines the contract that any storage backend (local folder, Google Drive,
S3, etc.) must implement. This keeps the core pipeline completely decoupled
from the underlying filesystem or remote storage.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Iterator
from PIL import Image


@dataclass
class ImageMetadata:
    """Metadata for an image file."""
    file_id: str             # Unique identifier (file path or remote ID)
    file_size: int           # Size in bytes
    modified_time: float     # Last modified POSIX timestamp
    width: int | None = None
    height: int | None = None
    format: str | None = None # e.g. "JPEG", "PNG", "WEBP", "HEIC"
    md5: str | None = None   # Hex MD5 checksum if available from source metadata
    source: str = "local"    # 'local' | 'gdrive'
    drive_id: str | None = None # Stable Google Drive file ID if remote


class ImageSource(ABC):
    """Abstract interface for image backends."""

    @abstractmethod
    def list_files(self) -> Iterator[str]:
        """Yield unique file IDs / paths for all eligible image files."""
        ...

    @abstractmethod
    def get_metadata(self, file_id: str) -> ImageMetadata:
        """
        Extract basic metadata (size, mtime, width, height, format).
        Should avoid decoding full pixel data whenever possible.
        """
        ...

    @abstractmethod
    def get_bytes(self, file_id: str) -> bytes:
        """Return the raw bytes of the file."""
        ...

    @abstractmethod
    def get_thumbnail(self, file_id: str, size: tuple[int, int] = (256, 256)) -> Image.Image:
        """
        Generate and return a PIL Image thumbnail.
        Must use fast decoders (like JPEG draft mode) to avoid full image loads.
        """
        ...

    def fetch_thumbnails(
        self,
        file_ids: list[str],
        max_workers: int = 8,
        progress_callback: Callable[[int, int, str], None] | None = None,
        size: tuple[int, int] = (256, 256),
    ) -> dict[str, Image.Image]:
        """
        Fetch thumbnails for multiple file IDs concurrently using a thread pool (default 8 workers).
        Invokes progress_callback(cur, total, file_id) after every file.
        """
        import threading
        from concurrent.futures import ThreadPoolExecutor, as_completed

        total = len(file_ids)
        if total == 0:
            return {}

        results: dict[str, Image.Image] = {}
        completed = 0
        lock = threading.Lock()

        def _fetch_one(fid: str) -> tuple[str, Image.Image | None]:
            nonlocal completed
            try:
                img = self.get_thumbnail(fid, size=size)
            except Exception:
                img = None
            with lock:
                completed += 1
                cur = completed
            if progress_callback is not None:
                try:
                    progress_callback(cur, total, fid)
                except Exception:
                    pass
            return fid, img

        workers = min(max_workers, max(1, total))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_fetch_one, fid) for fid in file_ids]
            for fut in as_completed(futures):
                try:
                    fid, img = fut.result()
                    if img is not None:
                        results[fid] = img
                except Exception:
                    pass

        return results
