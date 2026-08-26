#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
#
# In-container RoboTwin eval driver. Not meant to be run directly — launched by
# run_eval.sh, which mounts this file at /eval and provides the bind mounts.
#
# Two isolated processes on one GPU, talking over ws://localhost:8000:
#   server    — cosmos venv:      python -m vla_eval_cosmos3
#   benchmark — robotwin conda:   vla-eval run  (SAPIEN sim + CuRobo expert check)
#
# The server is started ONCE and reused across every task: loading the policy
# costs minutes, and there is nothing task-specific about it. Each task still
# gets its own `vla-eval run` process, which also bounds SAPIEN's per-process
# env-creation leak (the renderer's denoiser exhausts GPU memory somewhere above
# ~50 env creations, so keep TEST_NUM well under that).
#
# Mounts (from run_eval.sh): /ckpt/policy, /ckpt/vae.pth, /out, /eval
# Env: TEST_NUM SEED SHARDS TASK_CONFIG INSTRUCTION_TYPE RECORD_VIDEO
#      SERVER_CONFIG VLA_EVAL_WATCHDOG_TIMEOUT_S
set -euo pipefail

framework=/opt/cosmos-framework
venv=$framework/.venv
test_num=${TEST_NUM:-20}
seed=${SEED:-0}
# RoboTwin derives its scene seeds as 100000*(1+seed), so distinct SEED values give
# DISJOINT episode sets. SHARDS>1 runs each task once per shard (seed, seed+1, ...)
# into <task>-s<N>/, which collect_results.py merges weighted by episode count.
# This is how you get the official 100 episodes/task without one process making
# 100 env creations — see the note on the SAPIEN leak above.
shards=${SHARDS:-5}
task_config=${TASK_CONFIG:-demo_randomized_large}
instruction_type=${INSTRUCTION_TYPE:-unseen}
record_video=${RECORD_VIDEO:-false}
# Serving recipe. The shipped robotwin.yaml matches the reference recipe
# (video_subsample_factor=4). Point SERVER_CONFIG at your own copy if your
# checkpoint trained with different settings — a mismatch does not error, it
# just scores lower.
server_config=${SERVER_CONFIG:-packages/vla-eval-cosmos3/configs/robotwin.yaml}

mkdir -p /out/logs
exec > >(tee -a /out/logs/eval.log) 2>&1

# Reap the policy server on exit. It is a multi-GB CUDA process that would
# otherwise outlive the script, and because `exec > >(tee ...)` leaks the pipe's
# write end to every child, `tee` would never see EOF either. Every step is
# `|| true`: under `set -e` a kill on an already-dead server would abort the trap
# before `exit $rc` and report a good eval as failed.
cleanup() {
    rc=$?
    kill "${server_pid:-}" 2>/dev/null || true
    for _ in 1 2 3 4 5; do
        kill -0 "${server_pid:-}" 2>/dev/null || break
        sleep 1
    done
    kill -9 "${server_pid:-}" 2>/dev/null || true
    exit "$rc"
}
trap cleanup EXIT

# --- 1. preflight -------------------------------------------------------------
# A broken CuRobo import does not abort the benchmark — it yields tens of
# thousands of cascading "no attribute 'left_planner'" warnings until the
# watchdog fires. Two seconds here beats twenty minutes there.
conda run -n robotwin python -c "
import warp, warp.torch, numpy, curobo, mplib, sapien
assert not warp.__version__.startswith('1.14'), 'warp 1.14 has no warp.torch — rebuild the image'
print('sim env OK: warp', warp.__version__, 'numpy', numpy.__version__)"

# demo_*_large = the stock config with Large_D435 (640x480) instead of D435
# (320x240). Abort rather than fall back: 320x240 upscaled to the model canvas
# took grasp-and-place from ~90% to 0% and reads as a weak policy, not a bug.
rt_cfg=/app/RoboTwin/task_config/$task_config.yml
[ -f "$rt_cfg" ] || { echo "MISSING $rt_cfg — rebuild the image (see examples/robotwin/docker/)"; exit 1; }
echo "=== camera config ($task_config):"; grep -A4 '^camera:' "$rt_cfg"

# --- 2. server ----------------------------------------------------------------
# cwd must be the framework root: the model build opens config assets by relative
# path, and configs/robotwin.yaml names action_stats_path relative to it.
# /workspace/src is the base image's vla_eval harness (the sim-side benchmark
# modules), which the venv's wheel lacks.
cd "$framework"
env -u LD_LIBRARY_PATH PYTHONPATH=/workspace/src \
  "$venv/bin/python" -m vla_eval_cosmos3 \
  --config "$server_config" --port 8000 \
  > /out/logs/server.log 2>&1 &
server_pid=$!

echo "=== server config: $server_config"
echo "=== loading policy (this takes a few minutes)..."
SECONDS=0
until grep -q "Starting server on ws" /out/logs/server.log; do
  kill -0 "$server_pid" 2>/dev/null || { echo "SERVER DIED:"; tail -60 /out/logs/server.log; exit 1; }
  [ "$SECONDS" -lt 1800 ] || { echo "SERVER TIMEOUT (1800s):"; tail -60 /out/logs/server.log; exit 1; }
  sleep 5
done
echo "=== server up after ${SECONDS}s"

# --- 3. benchmark, one task at a time -----------------------------------------
export VLA_EVAL_WATCHDOG_TIMEOUT_S=${VLA_EVAL_WATCHDOG_TIMEOUT_S:-4800}
echo "=== watchdog: ${VLA_EVAL_WATCHDOG_TIMEOUT_S}s with no episode progress"

failed=""
for task in "$@"; do
    # RoboTwin budgets steps PER TASK (400-1700; envs/_base_task.py reads this
    # file), but the harness ignores it and hardcodes max_steps=400 for
    # everything. That under-budgets 21 of the 50 tasks by up to 3x —
    # blocks_ranking_* get 400 of their official 1200 and score ~10% sitting at
    # the cap, which measures the config, not the policy.
    step_lim=$(awk -v t="$task:" '$1 == t {print $2}' \
        /app/RoboTwin/task_config/_eval_step_limit.yml 2>/dev/null || true)
    : "${step_lim:=400}"

    for s in $(seq 0 $((shards - 1))); do
        shard_seed=$((seed + s))
        if [ "$shards" -gt 1 ]; then
            out=/out/${task}-s${s}
            label="$task shard $((s + 1))/$shards (seed=$shard_seed)"
        else
            out=/out/$task
            label="$task (seed=$shard_seed)"
        fi
        mkdir -p "$out"

        echo ""
        echo "=== $label — steps=$step_lim, episodes=$test_num, config=$task_config"

        cat > /tmp/eval.yaml <<EOF
server: {url: "ws://localhost:8000"}
output_dir: "$out"
benchmarks:
  - benchmark: "vla_eval.benchmarks.robotwin.benchmark:RoboTwinBenchmark"
    episodes_per_task: 1
    action_dim: 14
    max_steps: $step_lim
    recording: {record_video: $record_video, record_step: true, video_fps: 20}
    params:
      task_name: $task
      task_config: $task_config
      seed: $shard_seed
      instruction_type: $instruction_type
      test_num: $test_num
      skip_expert_check: false
EOF

        # No wall-clock cap: measured cost is 3.8-11.4 min/episode, so a fixed
        # budget cannot know how long a task needs. A failing shard is recorded and
        # the sweep continues — one bad task should not cost the other 49.
        set +e
        conda run -n robotwin vla-eval run --no-docker -c /tmp/eval.yaml
        bench_rc=$?
        set -e
        if [ "$bench_rc" -eq 124 ]; then
            echo "=== WATCHDOG PANIC on $label: no episode progress within ${VLA_EVAL_WATCHDOG_TIMEOUT_S}s"
            echo "    progress: $(cat "$out/RoboTwinBenchmark.progress" 2>/dev/null || echo unknown)"
            echo "    results are PARTIAL; no *_aggregate.json will exist."
            failed="$failed $(basename "$out")"
        elif [ "$bench_rc" -ne 0 ]; then
            echo "=== $label FAILED rc=$bench_rc — results may be partial"
            failed="$failed $(basename "$out")"
        fi
    done
done

echo ""
echo "=== done -> /out"
[ -z "$failed" ] || echo "=== tasks with a non-zero exit:$failed"
