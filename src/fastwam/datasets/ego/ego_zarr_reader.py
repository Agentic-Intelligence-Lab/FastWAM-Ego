"""Lightweight EgoVerse Zarr v3 reader for JPEG image arrays.

Reads ``images.front_1`` (variable-length JPEG shards) without requiring zarr v3,
which needs Python >= 3.11. Numeric arrays are not required for video-only training.
"""

from __future__ import annotations

import io
import json
import re
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

_JPEG_START = b"\xff\xd8\xff"


def _load_episode_metadata(episode_path: Path) -> dict[str, Any]:
    zarr_json = episode_path / "zarr.json"
    if not zarr_json.exists():
        raise FileNotFoundError(f"Missing episode metadata: {zarr_json}")
    with zarr_json.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    attrs = payload.get("attributes", payload)
    if "total_frames" not in attrs:
        raise KeyError(f"`total_frames` missing in {zarr_json}")
    return dict(attrs)


def _build_jpeg_offsets(shard_bytes: bytes) -> list[int]:
    return [match.start() for match in re.finditer(_JPEG_START, shard_bytes)]


def _decode_jpeg_slice(shard_bytes: bytes, start: int, end: int) -> np.ndarray:
    jpeg = shard_bytes[start:end]
    if not jpeg.startswith(_JPEG_START):
        local = jpeg.find(_JPEG_START)
        if local < 0:
            raise ValueError("JPEG start marker not found in shard slice.")
        jpeg = jpeg[local:]
    with Image.open(io.BytesIO(jpeg)) as img:
        rgb = img.convert("RGB")
    return np.asarray(rgb, dtype=np.uint8)


class EgoZarrEpisode:
    """Read RGB frames from one EgoVerse zarr episode directory."""

    __slots__ = (
        "_image_key",
        "_jpeg_offsets",
        "_lock",
        "_metadata",
        "_path",
        "_shard_bytes",
        "_shard_paths",
    )

    def __init__(self, episode_path: str | Path, image_key: str = "images.front_1"):
        self._path = Path(episode_path)
        self._image_key = image_key
        self._metadata = _load_episode_metadata(self._path)
        self._shard_paths = self._discover_image_shards()
        self._shard_bytes: list[bytes] | None = None
        self._jpeg_offsets: list[list[int]] | None = None
        self._lock = threading.Lock()

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    @property
    def total_frames(self) -> int:
        return int(self._metadata["total_frames"])

    @property
    def fps(self) -> int:
        return int(self._metadata.get("fps", 30))

    def _discover_image_shards(self) -> list[Path]:
        image_dir = self._path / self._image_key / "c"
        if not image_dir.is_dir():
            raise FileNotFoundError(f"Missing image shard directory: {image_dir}")
        shard_paths = sorted(
            (path for path in image_dir.iterdir() if path.is_file()),
            key=lambda path: int(path.name) if path.name.isdigit() else path.name,
        )
        if not shard_paths:
            raise FileNotFoundError(f"No image shards found under {image_dir}")
        return shard_paths

    def _ensure_loaded(self) -> None:
        if self._shard_bytes is not None and self._jpeg_offsets is not None:
            return
        with self._lock:
            if self._shard_bytes is not None and self._jpeg_offsets is not None:
                return
            shard_bytes: list[bytes] = []
            jpeg_offsets: list[list[int]] = []
            for shard_path in self._shard_paths:
                data = shard_path.read_bytes()
                shard_bytes.append(data)
                jpeg_offsets.append(_build_jpeg_offsets(data))
            self._shard_bytes = shard_bytes
            self._jpeg_offsets = jpeg_offsets

    def _global_frame_location(self, frame_idx: int) -> tuple[int, int]:
        self._ensure_loaded()
        assert self._jpeg_offsets is not None
        remaining = frame_idx
        for shard_idx, offsets in enumerate(self._jpeg_offsets):
            if remaining < len(offsets):
                return shard_idx, remaining
            remaining -= len(offsets)
        raise IndexError(f"Frame index {frame_idx} out of range for episode {self._path.name}")

    def read_rgb_frame(self, frame_idx: int) -> np.ndarray:
        if frame_idx < 0 or frame_idx >= self.total_frames:
            raise IndexError(f"Frame index {frame_idx} out of range [0, {self.total_frames})")
        self._ensure_loaded()
        assert self._shard_bytes is not None
        assert self._jpeg_offsets is not None

        shard_idx, local_idx = self._global_frame_location(frame_idx)
        shard = self._shard_bytes[shard_idx]
        offsets = self._jpeg_offsets[shard_idx]
        start = offsets[local_idx]
        end = offsets[local_idx + 1] if local_idx + 1 < len(offsets) else len(shard)
        return _decode_jpeg_slice(shard, start, end)

    def read_rgb_frames(self, start: int, end: int, stride: int = 1) -> np.ndarray:
        if start < 0 or end <= start:
            raise ValueError(f"Invalid frame range [{start}, {end})")
        indices = list(range(start, end, stride))
        frames = [self.read_rgb_frame(idx) for idx in indices]
        return np.stack(frames, axis=0)


@lru_cache(maxsize=256)
def get_ego_episode(episode_path: str, image_key: str = "images.front_1") -> EgoZarrEpisode:
    return EgoZarrEpisode(episode_path=episode_path, image_key=image_key)
