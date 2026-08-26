#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Structured-TOML launch for action_policy_robotwin_nano CLEAN SUBSET — Cosmos3-Nano
# RoboTwin 2.0 action-policy SFT (HSDP, full SFT, clean episodes only).
# Drives cosmos_framework.scripts.train against
# examples/toml/sft_config/action_policy_robotwin_nano_clean.toml.
#
# Trains on the 50 non-randomized ("clean") episodes of each task block
# (episode_index % 550 < 50 -> 2,500 total). All other recipe knobs match the
# whole-scene recipe (chunk 64, vsub4, ColorJitter aug, state conditioning).
#
# WHY 2 NODES — the load-bearing knob, not a resource convenience. Clean is ~9%
# of the data (2,500 / 27,500 episodes), so the 8-node GBS 2048 shape would run
# ~63 epochs over 390k windows in 25k iters and overfit. 2 nodes give
# GBS 512 x 10k = 5.12M samples ~= 13 epochs over clean, matching the reference
# clean run's sample budget rather than just its step count.
#
# Required env vars:
#   ROBOTWIN_ROOT         local RoboTwin LeRobot dataset dir (no default)
# Optional env vars (defaults below; override to relocate data/checkpoints):
#   BASE_CHECKPOINT_PATH  default: examples/checkpoints/Cosmos3-Nano
#   WAN_VAE_PATH          default: examples/checkpoints/wan22_vae/Wan2.2_VAE.pth
#   HF_TOKEN              if any tokenizer download requires gated HF access
#   OUTPUT_ROOT           default: outputs/train
#
# Usage (HSDP 4x2 reference: 2 nodes x 4 GPUs; set NNODES/NODE_RANK/MASTER_ADDR):
#   ROBOTWIN_ROOT=<dir> NPROC_PER_NODE=4 NNODES=2 NODE_RANK=$RANK \
#     MASTER_ADDR=<host> bash examples/launch_sft_action_policy_robotwin_nano_clean.sh
#
# Usage (single 8-GPU node, same GBS 512 via shard=8 x replicate=1):
#   ROBOTWIN_ROOT=<dir> EXTRA_TAIL_OVERRIDES="\
#     model.parallelism.data_parallel_shard_degree=8 \
#     model.parallelism.data_parallel_replicate_degree=1" \
#     bash examples/launch_sft_action_policy_robotwin_nano_clean.sh

TOML_FILE="examples/toml/sft_config/action_policy_robotwin_nano_clean.toml"
: "${BASE_CHECKPOINT_PATH:=examples/checkpoints/Cosmos3-Nano}"

export ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-}"

EXTRA_DATASET_CHECK='[[ -f "$ROBOTWIN_ROOT/meta/info.json" ]] || { echo "ERROR: ROBOTWIN_ROOT must be a local LeRobot dir containing meta/info.json (got: '\''$ROBOTWIN_ROOT'\''). See examples/robotwin/README.md for dataset setup." >&2; exit 1; }'

# Inject the clean scene filter. This is the only difference vs the whole-scene
# launch: robotwin_scene=clean tells RoboTwinLeRobotDataset._filter_valid_episodes
# to keep only episode_index % 550 < 50 (the 50 non-randomized episodes per task).
TAIL_OVERRIDES=(
    "dataloader_train.dataloader.datasets.robotwin.dataset.robotwin_scene=clean"
    ${EXTRA_TAIL_OVERRIDES:-}
)

source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"
