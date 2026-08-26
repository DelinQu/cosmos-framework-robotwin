#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
#
# Evaluate a trained RoboTwin policy.  Host side: validate inputs, then
# `docker run` the image YOU built (see examples/robotwin/docker/) with weights
# and an output dir bind-mounted.  The actual work happens in run_task.sh.
#
# Two reported protocols — pick one via TASK_CONFIG:
#
#   # in-distribution: non-randomized scenes, unseen instructions, 100 ep/task
#   TASK_CONFIG=demo_clean_large \
#   CHECKPOINT_PATH=<iter_*> WAN_VAE_PATH=<Wan2.2_VAE.pth> \
#     bash examples/robotwin/eval/run_eval.sh
#
#   # randomized: domain-randomized scenes, unseen instructions, 100 ep/task
#   TASK_CONFIG=demo_randomized_large \
#   CHECKPOINT_PATH=<iter_*> WAN_VAE_PATH=<Wan2.2_VAE.pth> \
#     bash examples/robotwin/eval/run_eval.sh
#
# Both use SHARDS=5 x TEST_NUM=20 = 100 episodes per task by default.
# Run both sequentially (different OUTPUT_DIR) to fill the full result table.
#
# Smoke test (2 tasks, 5 episodes, no shards):
#   TASK_CONFIG=demo_clean_large SHARDS=1 TEST_NUM=5 \
#   CHECKPOINT_PATH=... WAN_VAE_PATH=... \
#     bash examples/robotwin/eval/run_eval.sh place_empty_cup adjust_bottle
#
# Summarize after each run:
#   python examples/robotwin/eval/collect_results.py <run_name> \
#     --results-dir <OUTPUT_DIR> --ckpt <CHECKPOINT_PATH>
#
# HARDWARE: A100 (80 GB) is the suggested configuration. The CuRobo wheel in
# the image is compiled for Ampere; Hopper GPUs need to recompile CuRobo from
# source before the expert-check planner will run. A 24 GB card is likely to
# OOM with the policy server and SAPIEN resident on one GPU.
set -euo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# --- inputs -------------------------------------------------------------------
: "${CHECKPOINT_PATH:?set CHECKPOINT_PATH to a trained checkpoint dir}"
: "${WAN_VAE_PATH:?set WAN_VAE_PATH to Wan2.2_VAE.pth}"
image=${IMAGE:-cosmos-robotwin:0.4.0}
out_dir=${OUTPUT_DIR:-$PWD/outputs/robotwin_eval}
gpus=${GPUS:-all}

# Full RoboTwin 2.0 suite; positional args override for a smoke run.
tasks="adjust_bottle beat_block_hammer blocks_ranking_rgb blocks_ranking_size \
click_alarmclock click_bell dump_bin_bigbin grab_roller handover_block handover_mic \
hanging_mug lift_pot move_can_pot move_pillbottle_pad move_playingcard_away move_stapler_pad \
open_laptop open_microwave pick_diverse_bottles pick_dual_bottles place_a2b_left place_a2b_right \
place_bread_basket place_bread_skillet place_burger_fries place_can_basket place_cans_plasticbox \
place_container_plate place_dual_shoes place_empty_cup place_fan place_mouse_pad place_object_basket \
place_object_scale place_object_stand place_phone_stand place_shoe press_stapler put_bottles_dustbin \
put_object_cabinet rotate_qrcode scan_object shake_bottle shake_bottle_horizontally stack_blocks_three \
stack_blocks_two stack_bowls_three stack_bowls_two stamp_seal turn_switch"
tasks=${*:-$tasks}

[ -e "$CHECKPOINT_PATH" ] || { echo "ERROR: CHECKPOINT_PATH not found: $CHECKPOINT_PATH" >&2; exit 1; }
[ -f "$WAN_VAE_PATH" ]    || { echo "ERROR: WAN_VAE_PATH not found: $WAN_VAE_PATH" >&2; exit 1; }
docker image inspect "$image" >/dev/null 2>&1 || {
    echo "ERROR: image '$image' not found locally. Build it first:" >&2
    echo "  docker build -f examples/robotwin/docker/Dockerfile -t $image ." >&2
    exit 1
}

mkdir -p "$out_dir"
ckpt_abs=$(cd "$(dirname "$CHECKPOINT_PATH")" && pwd)/$(basename "$CHECKPOINT_PATH")
vae_abs=$(cd "$(dirname "$WAN_VAE_PATH")" && pwd)/$(basename "$WAN_VAE_PATH")
out_abs=$(cd "$out_dir" && pwd)

echo "=== image      : $image"
echo "=== checkpoint : $ckpt_abs"
echo "=== vae        : $vae_abs"
echo "=== output     : $out_abs"
echo "=== protocol   : ${TASK_CONFIG:-demo_randomized_large} / ${INSTRUCTION_TYPE:-unseen}"
echo "=== episodes   : ${SHARDS:-5} shards x ${TEST_NUM:-20} = $((${SHARDS:-5} * ${TEST_NUM:-20})) per task"
echo "=== tasks      : $(echo "$tasks" | wc -w)"

# --- run ----------------------------------------------------------------------
# --shm-size: SAPIEN and the dataloaders need far more than Docker's 64 MB default.
# Weights are mounted read-only; eval scripts are mounted so edits need no rebuild.
exec docker run --rm -it \
    --gpus "$gpus" \
    --shm-size=32g \
    -v "$ckpt_abs":/ckpt/policy:ro \
    -v "$vae_abs":/ckpt/vae.pth:ro \
    -v "$out_abs":/out \
    -v "$here":/eval:ro \
    -e TEST_NUM="${TEST_NUM:-20}" \
    -e SEED="${SEED:-0}" \
    -e SHARDS="${SHARDS:-5}" \
    -e TASK_CONFIG="${TASK_CONFIG:-demo_randomized_large}" \
    -e SERVER_CONFIG="${SERVER_CONFIG:-packages/vla-eval-cosmos3/configs/robotwin.yaml}" \
    -e INSTRUCTION_TYPE="${INSTRUCTION_TYPE:-unseen}" \
    -e RECORD_VIDEO="${RECORD_VIDEO:-false}" \
    -e VLA_EVAL_WATCHDOG_TIMEOUT_S="${VLA_EVAL_WATCHDOG_TIMEOUT_S:-4800}" \
    "$image" \
    bash /eval/run_task.sh $tasks
