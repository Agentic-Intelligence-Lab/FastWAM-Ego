#!/usr/bin/env python3
"""Compute dataset_stats.json for LeRobot robot training."""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from fastwam.datasets.lerobot.base_lerobot_dataset import BaseLerobotDataset
from fastwam.datasets.lerobot.utils.normalizer import save_dataset_stats_to_json
from fastwam.utils.config_resolvers import register_default_resolvers
from fastwam.utils.logging_config import get_logger, setup_logging

register_default_resolvers()
logger = get_logger(__name__)


def _resolve_robot_train_cfg(data_cfg: DictConfig) -> tuple[DictConfig, Path]:
    if "dataset_stats_path" not in data_cfg:
        raise ValueError("`data.dataset_stats_path` is required.")
    out_path = Path(str(data_cfg.dataset_stats_path)).expanduser()
    train_cfg = data_cfg.train
    target = str(train_cfg.get("_target_", ""))
    if target.endswith("MixedTrainDataset"):
        return train_cfg.robot, out_path
    return train_cfg, out_path


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    setup_logging(log_level=logging.INFO)
    if cfg.data is None:
        raise ValueError("`cfg.data` is required.")

    robot_cfg, out_path = _resolve_robot_train_cfg(cfg.data)
    robot_cfg = OmegaConf.create(OmegaConf.to_container(robot_cfg, resolve=True))
    processor = instantiate(robot_cfg.processor)

    lerobot = BaseLerobotDataset(
        dataset_dirs=[str(path) for path in robot_cfg.dataset_dirs],
        shape_meta=OmegaConf.to_container(robot_cfg.shape_meta, resolve=True),
        obs_size=int(robot_cfg.num_frames),
        action_size=int(robot_cfg.num_frames) - 1,
        val_set_proportion=float(robot_cfg.val_set_proportion),
        is_training_set=True,
        global_sample_stride=int(robot_cfg.get("global_sample_stride", 1)),
    )

    logger.info(
        "Computing normalization stats for %d episodes (%d frames)...",
        lerobot.multi_dataset.num_episodes,
        lerobot.multi_dataset.num_frames,
    )
    stats = lerobot.get_dataset_stats(processor)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_dataset_stats_to_json(stats, str(out_path))
    logger.info("Wrote dataset stats to %s", out_path)


if __name__ == "__main__":
    main()
