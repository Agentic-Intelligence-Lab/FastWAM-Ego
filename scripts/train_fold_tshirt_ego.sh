#!/usr/bin/env bash
set -euo pipefail

# Real robot fold-T-shirt data + EgoVerse ego video augmentation.
#
# Usage:
#   bash scripts/train_fold_tshirt_ego.sh <nproc_per_node>
#
# Example:
#   bash scripts/train_fold_tshirt_ego.sh 8

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

NPROC_PER_NODE="${1:?Usage: bash scripts/train_fold_tshirt_ego.sh <nproc_per_node>}"
shift || true

ROBOT_DATASET="/mnt/data/yuanmingqi/datasets/realworld_piper/fold_tshirt_20260616"
TASK_CONFIG="fold_tshirt_joint_3cam_1e-4_ego"

echo "[fold_tshirt] Step 1/5: ensure LeRobot meta exists"
if [[ ! -f "${ROBOT_DATASET}/meta/info.json" ]]; then
  python scripts/repair_lerobot_meta.py "${ROBOT_DATASET}" --task "fold up the orange T-shirt."
else
  echo "[fold_tshirt] meta/info.json already exists, skip repair"
fi

echo "[fold_tshirt] Step 2/5: validate robot dataset"
python scripts/validate_lerobot_dataset.py "${ROBOT_DATASET}" --allow-missing-videos || {
  echo "[fold_tshirt] Validation failed." >&2
  exit 1
}

echo "[fold_tshirt] Step 3/5: ensure dataset_stats.json"
if [[ ! -f "${ROBOT_DATASET}/dataset_stats.json" ]]; then
  python scripts/compute_dataset_stats.py task="${TASK_CONFIG}"
else
  echo "[fold_tshirt] dataset_stats.json already exists, skip compute"
fi

echo "[fold_tshirt] Step 4/5: precompute text embeddings"
python scripts/precompute_text_embeds.py task="${TASK_CONFIG}" \
  data.text_embeds_cache="${ROBOT_DATASET}/text_embeds_cache"

VIDEO_COUNT="$(find "${ROBOT_DATASET}/videos" -name '*.mp4' 2>/dev/null | wc -l | tr -d ' ')"
if [[ "${VIDEO_COUNT}" == "0" ]]; then
  echo "[fold_tshirt] ERROR: no robot mp4 videos found under ${ROBOT_DATASET}/videos" >&2
  echo "[fold_tshirt] Export/sync episode videos before launching training." >&2
  exit 1
fi

echo "[fold_tshirt] Step 5/5: launch training (${NPROC_PER_NODE} GPUs)"
bash scripts/train_zero1.sh "${NPROC_PER_NODE}" task="${TASK_CONFIG}" "$@"
