"""
sources/base.py — Generic ImageSource interface.

Defines the contract that any storage backend (local folder, Google Drive,
S3, etc.) must implement. This keeps the core pipeline completely decoupled
from the underlying filesystem or remote storage.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator
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
