"""Mix robot demonstrations with ego video-only augmentation samples."""

from __future__ import annotations

import torch

from fastwam.utils.logging_config import get_logger

logger = get_logger(__name__)


class MixedTrainDataset(torch.utils.data.Dataset):
    """Concatenate robot and ego datasets while keeping the robot path unchanged."""

    def __init__(
        self,
        robot,
        ego=None,
        ego_robot_ratio: float | None = None,
        ego_repeat_factor: float | None = None,
    ):
        self.robot_dataset = robot
        self.ego_dataset = ego
        self.ego_robot_ratio = None if ego_robot_ratio is None else float(ego_robot_ratio)
        self.ego_repeat_factor = None if ego_repeat_factor is None else float(ego_repeat_factor)

        robot_len = len(self.robot_dataset)
        if self.ego_dataset is None:
            self._ego_virtual_len = 0
        elif self.ego_robot_ratio is not None:
            if self.ego_robot_ratio <= 0.0:
                self._ego_virtual_len = 0
            else:
                self._ego_virtual_len = int(round(robot_len * self.ego_robot_ratio))
        elif self.ego_repeat_factor is not None and self.ego_repeat_factor > 0.0:
            self._ego_virtual_len = int(round(len(self.ego_dataset) * self.ego_repeat_factor))
        else:
            self._ego_virtual_len = 0

        if self._ego_virtual_len <= 0 and self.ego_dataset is not None and len(self.ego_dataset) > 0:
            if (self.ego_robot_ratio is not None and self.ego_robot_ratio > 0.0) or (
                self.ego_repeat_factor is not None and self.ego_repeat_factor > 0.0
            ):
                self._ego_virtual_len = 1

        ego_fraction = (self._ego_virtual_len / (robot_len + self._ego_virtual_len)) if (robot_len + self._ego_virtual_len) > 0 else 0.0
        logger.info(
            "MixedTrainDataset: robot=%d ego_pool=%d ego_virtual=%d ego_robot_ratio=%s ego_repeat_factor=%s epoch_ego_fraction=%.1f%%",
            robot_len,
            0 if self.ego_dataset is None else len(self.ego_dataset),
            self._ego_virtual_len,
            self.ego_robot_ratio,
            self.ego_repeat_factor,
            100.0 * ego_fraction,
        )

    @property
    def lerobot_dataset(self):
        return self.robot_dataset.lerobot_dataset

    def __len__(self) -> int:
        return len(self.robot_dataset) + self._ego_virtual_len

    def __getitem__(self, idx: int):
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds for MixedTrainDataset of length {len(self)}")

        robot_len = len(self.robot_dataset)
        if idx < robot_len:
            sample = self.robot_dataset[idx]
            if isinstance(sample, dict) and "has_action" not in sample:
                sample = dict(sample)
                sample["has_action"] = True
            return sample

        if self.ego_dataset is None or self._ego_virtual_len <= 0:
            raise IndexError(f"Ego index requested but ego dataset is disabled: idx={idx}")

        ego_idx = (idx - robot_len) % len(self.ego_dataset)
        return self.ego_dataset[ego_idx]
