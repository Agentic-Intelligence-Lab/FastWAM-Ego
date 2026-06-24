#!/usr/bin/env python3
"""Validate a local LeRobot v2.x dataset directory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq


REQUIRED_META_FILES = ("info.json", "tasks.jsonl", "episodes.jsonl")
OPTIONAL_META_FILES = ("episodes_stats.jsonl", "stats.json")
EXPECTED_VIDEO_KEYS = (
    "observation.images.head",
    "observation.images.left_wrist",
    "observation.images.right_wrist",
    "observation.images.front_view",
)


def _read_parquet(path: Path):
    return pq.read_table(path)


def validate_dataset(dataset_root: Path, strict_videos: bool = True) -> int:
    errors: list[str] = []
    warnings: list[str] = []

    if not dataset_root.is_dir():
        errors.append(f"Dataset root does not exist: {dataset_root}")
        _report(errors, warnings)
        return 1

    meta_dir = dataset_root / "meta"
    data_dir = dataset_root / "data"
    videos_dir = dataset_root / "videos"

    for name in REQUIRED_META_FILES:
        if not (meta_dir / name).exists():
            errors.append(f"Missing required meta file: meta/{name}")

    parquet_files = sorted(data_dir.glob("chunk-*/episode_*.parquet"))
    if not parquet_files:
        errors.append(f"No parquet episodes found under {data_dir}")

    valid_episodes: list[tuple[int, int]] = []
    corrupt_episodes: list[str] = []
    for parquet_path in parquet_files:
        episode_index = int(parquet_path.stem.split("_")[-1])
        try:
            table = _read_parquet(parquet_path)
        except Exception as exc:
            corrupt_episodes.append(f"{parquet_path.name}: {exc}")
            continue
        valid_episodes.append((episode_index, table.num_rows))

    if corrupt_episodes:
        for msg in corrupt_episodes:
            warnings.append(f"Unreadable parquet {msg}")
    if not valid_episodes:
        errors.append("No readable parquet episodes found.")

    if meta_dir.joinpath("info.json").exists():
        info = json.loads((meta_dir / "info.json").read_text(encoding="utf-8"))
        declared_eps = int(info.get("total_episodes", -1))
        if declared_eps != len(valid_episodes):
            warnings.append(
                f"info.json total_episodes={declared_eps} but found {len(valid_episodes)} readable parquet files"
            )
        fps = info.get("fps")
        if fps is not None and valid_episodes:
            sample = _read_parquet(parquet_files[0])
            if "timestamp" in sample.column_names and sample.num_rows > 1:
                import numpy as np

                ts = np.array(sample["timestamp"].to_numpy(zero_copy_only=False), dtype=float)
                est_fps = 1.0 / np.median(np.diff(ts))
                if abs(est_fps - float(fps)) > 2.0:
                    warnings.append(f"Declared fps={fps} but episode_000000 median fps≈{est_fps:.2f}")

    mp4_files = list(videos_dir.glob("**/*.mp4")) if videos_dir.exists() else []
    if strict_videos:
        if not mp4_files:
            errors.append("No .mp4 video files found under videos/")
        else:
            for video_key in EXPECTED_VIDEO_KEYS:
                key_matches = [p for p in mp4_files if f"/{video_key}/" in str(p).replace("\\", "/")]
                if not key_matches:
                    warnings.append(f"No mp4 files found for camera key: {video_key}")
            expected_count = len(valid_episodes) * len(EXPECTED_VIDEO_KEYS)
            if len(mp4_files) < expected_count:
                warnings.append(
                    f"Found {len(mp4_files)} mp4 files, expected about {expected_count} "
                    f"({len(valid_episodes)} episodes x {len(EXPECTED_VIDEO_KEYS)} cameras)"
                )
    elif not mp4_files:
        warnings.append("No .mp4 video files found; training will fail until videos are exported.")

    if valid_episodes:
        sample = _read_parquet(parquet_files[0])
        required_cols = {"action", "observation.state", "timestamp", "frame_index", "episode_index", "task_index"}
        missing = sorted(required_cols - set(sample.column_names))
        if missing:
            errors.append(f"Parquet missing required columns: {missing}")
        action_type = sample["action"].type
        state_type = sample["observation.state"].type
        action_dim = getattr(action_type, "list_size", None)
        state_dim = getattr(state_type, "list_size", None)
        if action_dim != 14 or state_dim != 14:
            warnings.append(
                "Expected 14-dim action/state (bimanual Piper); "
                f"got action_dim={action_dim} state_dim={state_dim}"
            )

    _report(errors, warnings, dataset_root=dataset_root, episodes=valid_episodes, mp4_count=len(mp4_files))
    return 1 if errors else 0


def _report(errors, warnings, dataset_root: Path | None = None, episodes=None, mp4_count: int = 0):
    if dataset_root is not None:
        print(f"Dataset: {dataset_root}")
    if episodes:
        total_frames = sum(length for _, length in episodes)
        print(f"Readable episodes: {len(episodes)} | total frames: {total_frames}")
    print(f"Video mp4 files: {mp4_count}")
    for msg in warnings:
        print(f"[warn] {msg}")
    for msg in errors:
        print(f"[error] {msg}", file=sys.stderr)
    if not errors and not warnings:
        print("Validation passed.")


def main():
    parser = argparse.ArgumentParser(description="Validate a local LeRobot dataset.")
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument(
        "--allow-missing-videos",
        action="store_true",
        help="Treat missing mp4 files as warnings instead of errors.",
    )
    args = parser.parse_args()
    code = validate_dataset(args.dataset_root, strict_videos=not args.allow_missing_videos)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
