#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Structured-TOML launch for action_policy_robotwin_nano CLEAN+RANDOM — Cosmos3-Nano
# RoboTwin 2.0 action-policy SFT (HSDP, full SFT, all 27,500 episodes).
# Drives cosmos_framework.scripts.train against
# examples/toml/sft_config/action_policy_robotwin_nano_clean_plus_random.toml.
#
# Point ROBOTWIN_ROOT at a RoboTwin 2.0 LeRobot v3.0 root (3 cameras at 640x480,
# 30 fps, 14-D absolute dual-arm joint actions). The recipe trains on subsampled
# video (video_subsample_factor=4, 7.5 Hz) with ColorJitter augmentation for
# 25000 iters. The default shape is HSDP 8x4 (global batch 2048);
# set NNODES/NODE_RANK/MASTER_ADDR per node.
# See examples/robotwin/README.md.
#
# Required env vars:
#   ROBOTWIN_ROOT         local RoboTwin LeRobot dataset dir (no default)
# Optional env vars (defaults below; override to relocate data/checkpoints):
#   BASE_CHECKPOINT_PATH  default: examples/checkpoints/Cosmos3-Nano
#   WAN_VAE_PATH          default: examples/checkpoints/wan22_vae/Wan2.2_VAE.pth
#   HF_TOKEN              if any tokenizer download requires gated HF access
#   OUTPUT_ROOT           default: outputs/train
#
# Usage (single 8-GPU node — override the 32-rank reference shape):
#   ROBOTWIN_ROOT=<dir> EXTRA_TAIL_OVERRIDES="\
#     model.parallelism.data_parallel_shard_degree=8 \
#     model.parallelism.data_parallel_replicate_degree=1 \
#     trainer.grad_accum_iter=4" \
#     bash examples/launch_sft_action_policy_robotwin_nano_clean_plus_random.sh
#
# Usage (HSDP 8x4 reference; set NNODES/NODE_RANK/MASTER_ADDR per node):
#   ROBOTWIN_ROOT=<dir> NPROC_PER_NODE=4 NNODES=8 NODE_RANK=$RANK \
#     MASTER_ADDR=<host> bash examples/launch_sft_action_policy_robotwin_nano_clean_plus_random.sh

TOML_FILE="examples/toml/sft_config/action_policy_robotwin_nano_clean_plus_random.toml"
: "${BASE_CHECKPOINT_PATH:=examples/checkpoints/Cosmos3-Nano}"

# RoboTwinLeRobotDataset reads ${oc.env:ROBOTWIN_ROOT} directly (a LOCAL LeRobot dir);
# export it so torchrun (launched in this shell) inherits it.
export ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-}"

EXTRA_DATASET_CHECK='[[ -f "$ROBOTWIN_ROOT/meta/info.json" ]] || { echo "ERROR: ROBOTWIN_ROOT must be a local LeRobot dir containing meta/info.json (got: '\''$ROBOTWIN_ROOT'\''). See examples/robotwin/README.md for dataset setup." >&2; exit 1; }'

# Extra Hydra overrides from the environment: a space-separated string word-split into
# the TAIL_OVERRIDES array. An exported string survives `bash <wrapper>` (a child
# process), unlike a TAIL_OVERRIDES array set in your shell. Use it for smoke runs,
# e.g. EXTRA_TAIL_OVERRIDES="trainer.max_iter=5 job.wandb_mode=offline".
TAIL_OVERRIDES=(
    ${EXTRA_TAIL_OVERRIDES:-}
)

source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"
