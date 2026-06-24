"""EgoVerse zarr video dataset for video-only FastWAM augmentation."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from fastwam.datasets.dataset_utils import CenterCrop, Normalize, ResizeSmallestSideAspectPreserving
from fastwam.utils.logging_config import get_logger

DEFAULT_PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"

from .ego_zarr_reader import get_ego_episode

logger = get_logger(__name__)


def _episode_dir(episode: dict, dataset_root: Path) -> Path:
    source_path = episode.get("source_path")
    if source_path:
        return Path(source_path)
    return dataset_root / str(episode["episode_hash"])


def _enrich_episode_task_fields(episode: dict, dataset_root: Path) -> dict:
    episode = dict(episode)
    if episode.get("task_description") or episode.get("task_name"):
        return episode

    zarr_json = _episode_dir(episode, dataset_root) / "zarr.json"
    if not zarr_json.exists():
        return episode

    with zarr_json.open("r", encoding="utf-8") as f:
        attrs = json.load(f).get("attributes", {})
    if attrs.get("task_description"):
        episode["task_description"] = str(attrs["task_description"])
    if attrs.get("task_name"):
        episode["task_name"] = str(attrs["task_name"])
    return episode


def load_ego_episodes(dataset_root: str | Path, manifest_path: Optional[str] = None) -> list[dict]:
    root = Path(dataset_root)
    manifest_file = Path(manifest_path) if manifest_path else root / "manifest.json"
    if manifest_file.exists():
        with manifest_file.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        kept = payload.get("kept", [])
        if kept:
            return [_enrich_episode_task_fields(episode, root) for episode in kept]

    episodes = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        zarr_json = child / "zarr.json"
        if not zarr_json.exists():
            continue
        with zarr_json.open("r", encoding="utf-8") as f:
            attrs = json.load(f).get("attributes", {})
        episodes.append(
            {
                "episode_hash": child.name,
                "source_path": str(child.resolve()),
                "total_frames": int(attrs["total_frames"]),
                "task_description": str(attrs.get("task_description", attrs.get("task_name", "perform a task"))),
                "task_name": str(attrs.get("task_name", "")),
            }
        )
    return episodes


def _load_manifest_episodes(dataset_root: Path, manifest_path: Optional[str]) -> list[dict]:
    return load_ego_episodes(dataset_root, manifest_path=manifest_path)


class EgoVerseVideoDataset(torch.utils.data.Dataset):
    """Video-only ego dataset compatible with FastWAM ``training_loss``."""

    def __init__(
        self,
        dataset_root: str,
        num_frames: int = 33,
        action_video_freq_ratio: int = 4,
        video_size: list[int] | tuple[int, int] = (224, 448),
        action_dim: int = 7,
        proprio_dim: int = 8,
        image_key: str = "images.front_1",
        obs_stride: int = 1,
        text_embedding_cache_dir: Optional[str] = None,
        context_len: int = 128,
        manifest_path: Optional[str] = None,
        val_set_proportion: float = 0.0,
        is_training_set: bool = True,
        seed: int = 42,
        task_prompt_template: str = DEFAULT_PROMPT,
    ):
        self.dataset_root = Path(dataset_root)
        self.num_frames = int(num_frames)
        self.action_video_freq_ratio = int(action_video_freq_ratio)
        self.video_size = list(video_size)
        self.action_dim = int(action_dim)
        self.proprio_dim = int(proprio_dim)
        self.image_key = image_key
        self.obs_stride = int(obs_stride)
        self.text_embedding_cache_dir = text_embedding_cache_dir
        self.context_len = int(context_len)
        self.task_prompt_template = task_prompt_template

        if self.num_frames <= 1:
            raise ValueError(f"`num_frames` must be > 1, got {self.num_frames}")
        if (self.num_frames - 1) % self.action_video_freq_ratio != 0:
            raise ValueError(
                f"`num_frames - 1` must be divisible by `action_video_freq_ratio`, "
                f"got {self.num_frames - 1} and {self.action_video_freq_ratio}"
            )
        video_frames = (self.num_frames - 1) // self.action_video_freq_ratio + 1
        if (video_frames - 1) % 4 != 0:
            raise ValueError(f"video frame count must satisfy T % 4 == 1, got {video_frames}")

        self.video_sample_indices = list(range(0, self.num_frames, self.action_video_freq_ratio))
        self.action_horizon = self.num_frames - 1

        episodes = _load_manifest_episodes(self.dataset_root, manifest_path)
        if not episodes:
            raise ValueError(f"No EgoVerse episodes found under {self.dataset_root}")

        rng = np.random.default_rng(seed)
        indices = np.arange(len(episodes))
        rng.shuffle(indices)
        split_idx = int(len(episodes) * (1.0 - float(val_set_proportion)))
        if val_set_proportion <= 1e-6:
            selected = indices
        elif is_training_set:
            selected = indices[:split_idx]
        else:
            selected = indices[split_idx:]

        self.episodes: list[dict] = [episodes[i] for i in selected]
        self.sample_index: list[tuple[int, int]] = []
        window = (self.num_frames - 1) * self.obs_stride + 1
        for episode_idx, episode in enumerate(self.episodes):
            total_frames = int(episode["total_frames"])
            max_start = total_frames - window
            if max_start < 0:
                continue
            for start in range(0, max_start + 1):
                self.sample_index.append((episode_idx, start))

        if not self.sample_index:
            raise ValueError("No valid ego training windows could be constructed from the selected episodes.")

        self.resize_transform = ResizeSmallestSideAspectPreserving(
            args={"img_w": self.video_size[1], "img_h": self.video_size[0]},
        )
        self.crop_transform = CenterCrop(
            args={"img_w": self.video_size[1], "img_h": self.video_size[0]},
        )
        self.normalize_transform = Normalize(args={"mean": 0.5, "std": 0.5})

        logger.info(
            "EgoVerseVideoDataset: episodes=%d samples=%d image_key=%s obs_stride=%d",
            len(self.episodes),
            len(self.sample_index),
            self.image_key,
            self.obs_stride,
        )

    def __len__(self) -> int:
        return len(self.sample_index)

    def _episode_path(self, episode_idx: int) -> Path:
        episode = self.episodes[episode_idx]
        source_path = episode.get("source_path")
        if source_path:
            return Path(source_path)
        episode_hash = episode["episode_hash"]
        return self.dataset_root / episode_hash

    def _instruction_for_episode(self, episode_idx: int) -> str:
        episode = self.episodes[episode_idx]
        task = str(episode.get("task_description") or episode.get("task_name") or "perform a task")
        return self.task_prompt_template.format(task=task)

    def _get_cached_text_context(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        if self.text_embedding_cache_dir is None:
            raise ValueError("text_embedding_cache_dir is not set.")
        cache_dir = self.text_embedding_cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        cache_path = os.path.join(cache_dir, f"{hashed}.t5_len{self.context_len}.wan22ti2v5b.pt")
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"Missing text embedding cache: {cache_path}. "
                "Run scripts/precompute_text_embeds.py with the ego-augment data config first."
            )
        payload = torch.load(cache_path, map_location="cpu")
        context = payload["context"]
        context_mask = payload["mask"].bool()
        if context.shape[0] != self.context_len or context_mask.shape[0] != self.context_len:
            raise ValueError(f"Cached context length mismatch in {cache_path}")
        return context, context_mask

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str | bool]:
        episode_idx, start = self.sample_index[idx]
        episode_path = self._episode_path(episode_idx)
        episode = get_ego_episode(str(episode_path), image_key=self.image_key)

        end = start + (self.num_frames - 1) * self.obs_stride + 1
        rgb = episode.read_rgb_frames(start=start, end=end, stride=self.obs_stride)  # [T, H, W, 3]
        video = torch.from_numpy(rgb).permute(0, 3, 1, 2).contiguous()  # [T, C, H, W]
        video = video[self.video_sample_indices]
        video = self.resize_transform(video)
        video = self.crop_transform(video)
        video = self.normalize_transform(video)
        video = video.permute(1, 0, 2, 3).contiguous()  # [C, T_video, H, W]

        instruction = self._instruction_for_episode(episode_idx)
        context, context_mask = self._get_cached_text_context(instruction)
        context = context.clone()
        context_mask = context_mask.clone()
        context[~context_mask] = 0.0
        context_mask = torch.ones_like(context_mask)

        action = torch.zeros(self.action_horizon, self.action_dim, dtype=torch.float32)
        proprio = torch.zeros(self.action_horizon, self.proprio_dim, dtype=torch.float32)
        return {
            "video": video,
            "action": action,
            "proprio": proprio,
            "prompt": instruction,
            "context": context,
            "context_mask": context_mask,
            "image_is_pad": torch.zeros(len(self.video_sample_indices), dtype=torch.bool),
            "action_is_pad": torch.ones(self.action_horizon, dtype=torch.bool),
            "proprio_is_pad": torch.ones(self.action_horizon, dtype=torch.bool),
            "has_action": False,
        }
