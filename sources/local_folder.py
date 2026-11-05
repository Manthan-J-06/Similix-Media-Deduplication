"""
sources/local_folder.py — Local filesystem implementation of ImageSource.

Adheres to memory discipline:
- `get_metadata`: uses Pillow's lazy header inspection (does not read pixel data).
- `get_thumbnail`: uses JPEG draft mode (`img.draft("RGB", size)`) to downscale
  at decode time in libjpeg, never loading full 24MP+ images into memory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from PIL import Image

try:
    import pillow_heif
    # Register HEIC/HEIF opener
    pillow_heif.register_heif_opener()
except ImportError:
    pass

from sources.base import ImageMetadata, ImageSource

RAW_EXTENSIONS = frozenset({
    ".arw", ".cr2", ".cr3", ".nef", ".orf",
    ".rw2", ".dng", ".raf", ".pef",
})

DEFAULT_EXTENSIONS = frozenset({
    ".jpg", ".jpeg", ".png", ".webp", ".gif",
    ".bmp", ".tiff", ".tif", ".heic", ".heif",
}) | RAW_EXTENSIONS


class LocalFolderSource(ImageSource):
    """ImageSource representing a local directory tree."""

    def __init__(self, root: str | Path, extensions: frozenset[str] = DEFAULT_EXTENSIONS):
        self.root = Path(root).resolve()
        if not self.root.exists() or not self.root.is_dir():
            raise ValueError(f"Directory does not exist or is not a directory: {self.root}")
        self.extensions = frozenset(e.lower() if e.startswith(".") else f".{e.lower()}" for e in extensions)

    def list_files(self) -> Iterator[str]:
        """Recursively yield all matching image paths under root."""
        for path in self.root.rglob("*"):
            if path.is_file() and path.suffix.lower() in self.extensions:
                yield str(path.resolve())

    def get_metadata(self, file_id: str) -> ImageMetadata:
        """
        Inspect header without loading pixel data into RAM.
        """
        path = Path(file_id)
        stat = path.stat()
        file_size = stat.st_size
        modified_time = stat.st_mtime

        if file_size == 0:
            raise OSError(f"Empty file (0 bytes): {path.name}")

        width = None
        height = None
        fmt = None

        try:
            with Image.open(path) as img:
                width, height = img.size
                fmt = img.format
        except Exception as exc:
            # For RAW files, format falls back to file suffix
            if path.suffix.lower() in RAW_EXTENSIONS:
                fmt = path.suffix.lstrip(".").upper()
            else:
                raise OSError(f"Cannot identify or decode image: {exc}") from exc

        return ImageMetadata(
            file_id=str(path.resolve()),
            file_size=file_size,
            modified_time=modified_time,
            width=width,
            height=height,
            format=fmt,
        )

    def get_bytes(self, file_id: str) -> bytes:
        """Return raw file bytes."""
        return Path(file_id).read_bytes()

    def get_thumbnail(self, file_id: str, size: tuple[int, int] = (256, 256)) -> Image.Image:
        """
        Generate a thumbnail with memory discipline.
        Uses libjpeg draft mode for JPEG images to downscale during decode.
        """
        path = Path(file_id)
        with Image.open(path) as img:
            if img.format == "JPEG":
                # Draft mode instructs libjpeg to decode only an approximation
                # at or near the target dimensions.
                img.draft("RGB", size)

            thumb = img.copy()

        # Handle color modes
        if thumb.mode not in ("RGB", "L"):
            if thumb.mode == "RGBA":
                bg = Image.new("RGB", thumb.size, (255, 255, 255))
                bg.paste(thumb, mask=thumb.split()[3])
                thumb = bg
            else:
                thumb = thumb.convert("RGB")

        thumb.thumbnail(size, Image.Resampling.LANCZOS)
        return thumb
