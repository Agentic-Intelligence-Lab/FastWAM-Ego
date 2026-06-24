#!/usr/bin/env python3
"""Generate missing LeRobot meta files from local parquet episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

DEFAULT_TASK = "fold the T-shirt"
REFERENCE_INFO = Path("/mnt/data/yuanmingqi/datasets/realworld_piper/cook_bread_v2/meta/info.json")


def _feature_stats(array: np.ndarray) -> dict:
    return {
        "min": array.min(axis=0).tolist(),
        "max": array.max(axis=0).tolist(),
        "mean": array.mean(axis=0).tolist(),
        "std": array.std(axis=0).tolist(),
        "count": [int(array.shape[0])],
    }


def _scalar_stats(array: np.ndarray) -> dict:
    flat = array.reshape(-1)
    return {
        "min": [float(flat.min())],
        "max": [float(flat.max())],
        "mean": [float(flat.mean())],
        "std": [float(flat.std())],
        "count": [int(flat.shape[0])],
    }


def _placeholder_image_stats() -> dict:
    return {
        "min": [[[0.0]], [[0.0]], [[0.0]]],
        "max": [[[1.0]], [[1.0]], [[1.0]]],
        "mean": [[[0.5]], [[0.5]], [[0.5]]],
        "std": [[[0.25]], [[0.25]], [[0.25]]],
        "count": [1],
    }


def scan_episodes(dataset_root: Path) -> list[dict]:
    episodes = []
    for parquet_path in sorted(dataset_root.glob("data/chunk-*/episode_*.parquet")):
        episode_index = int(parquet_path.stem.split("_")[-1])
        try:
            table = pq.read_table(parquet_path)
        except Exception as exc:
            print(f"[skip] unreadable {parquet_path}: {exc}")
            continue
        episodes.append(
            {
                "episode_index": episode_index,
                "parquet_path": parquet_path,
                "table": table,
                "length": int(table.num_rows),
            }
        )
    return episodes


def build_episodes_jsonl(episodes: list[dict], task: str) -> list[dict]:
    rows = []
    for item in episodes:
        table = item["table"]
        ts = np.asarray(table["timestamp"].to_numpy(zero_copy_only=False), dtype=np.float64)
        obs_ts = np.asarray(table["real_observation_timestamp_s"].to_numpy(zero_copy_only=False), dtype=np.float64)
        act_ts = np.asarray(table["real_action_timestamp_s"].to_numpy(zero_copy_only=False), dtype=np.float64)
        obs_wall = np.asarray(table["real_observation_wall_time_ns"].to_numpy(zero_copy_only=False), dtype=np.int64)
        act_wall = np.asarray(table["real_action_wall_time_ns"].to_numpy(zero_copy_only=False), dtype=np.int64)
        rows.append(
            {
                "episode_index": item["episode_index"],
                "tasks": [task],
                "length": item["length"],
                "real_duration_s": float(ts[-1] - ts[0]) if len(ts) > 1 else 0.0,
                "actual_fps": float(1.0 / np.median(np.diff(ts))) if len(ts) > 1 else 30.0,
                "first_observation_time_s": float(obs_ts[0]),
                "last_action_time_s": float(act_ts[-1]),
                "first_observation_wall_time_ns": int(obs_wall[0]),
                "last_action_wall_time_ns": int(act_wall[-1]),
            }
        )
    return rows


def build_episode_stats(table) -> dict:
    stats: dict = {}
    for col in table.column_names:
        values = np.asarray(table[col].to_numpy(zero_copy_only=False))
        if values.dtype == object:
            continue
        if values.ndim == 1:
            stats[col] = _scalar_stats(values)
        elif values.ndim == 2:
            stats[col] = _feature_stats(values)
    for video_key in (
        "observation.images.head",
        "observation.images.left_wrist",
        "observation.images.right_wrist",
        "observation.images.front_view",
    ):
        stats[video_key] = _placeholder_image_stats()
    return stats


def build_info(episodes: list[dict], task: str, robot_type: str) -> dict:
    if not REFERENCE_INFO.exists():
        raise FileNotFoundError(f"Reference info.json not found: {REFERENCE_INFO}")
    info = json.loads(REFERENCE_INFO.read_text(encoding="utf-8"))
    total_frames = sum(item["length"] for item in episodes)
    info.update(
        {
            "robot_type": robot_type,
            "total_episodes": len(episodes),
            "total_frames": total_frames,
            "total_tasks": 1,
            "total_videos": len(episodes) * 4,
            "total_chunks": 1,
            "chunks_size": 1000,
            "fps": 30,
            "splits": {"train": f"0:{len(episodes)}"},
        }
    )
    return info


def write_jsonl(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def repair_dataset(dataset_root: Path, task: str, robot_type: str, dry_run: bool = False):
    episodes = scan_episodes(dataset_root)
    if not episodes:
        raise RuntimeError(f"No readable parquet episodes found under {dataset_root}")

    meta_dir = dataset_root / "meta"
    tasks_rows = [{"task_index": 0, "task": task}]
    episodes_rows = build_episodes_jsonl(episodes, task=task)
    stats_rows = [
        {"episode_index": item["episode_index"], "stats": build_episode_stats(item["table"])}
        for item in episodes
    ]
    info = build_info(episodes, task=task, robot_type=robot_type)

    print(f"Dataset: {dataset_root}")
    print(f"Readable episodes: {len(episodes)}")
    print(f"Total frames: {sum(item['length'] for item in episodes)}")
    print(f"Task: {task}")

    if dry_run:
        print("Dry run only; no files written.")
        return

    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / "info.json").write_text(json.dumps(info, indent=4, ensure_ascii=True) + "\n", encoding="utf-8")
    write_jsonl(meta_dir / "tasks.jsonl", tasks_rows)
    write_jsonl(meta_dir / "episodes.jsonl", episodes_rows)
    write_jsonl(meta_dir / "episodes_stats.jsonl", stats_rows)
    print(f"Wrote meta files to {meta_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate LeRobot meta files from parquet episodes.")
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--robot-type", default="piper")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    repair_dataset(args.dataset_root, task=args.task, robot_type=args.robot_type, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
