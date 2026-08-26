<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: OpenMDW-1.1 -->

# RoboTwin 2.0 — Cosmos3-Nano action policy

Post-train Cosmos3-Nano on [RoboTwin 2.0](https://github.com/RoboTwin-Platform/RoboTwin)
(dual-arm ALOHA, 14-D absolute joint positions) and evaluate on the 50-task suite.

## Results

All 50 tasks, 100 episodes each, unseen instruction split:

| recipe | protocol | macro success |
| --- | --- | ---: |
| whole scene (`_clean_plus_random`, iter-025000) | `demo_clean_large` | **95.24%** |
| whole scene (`_clean_plus_random`, iter-025000) | `demo_randomized_large` | **94.24%** |
| clean2random (`_clean`, iter-010000) | `demo_randomized_large` (OOD) | **53.44%** |

<details>
<summary>Per-task breakdown</summary>

`ok=50`, `errored=0`, `truncated=0`, `rejected=0` for all three runs.

| task | ws clean | ws random | clean→random (OOD) |
| --- | ---: | ---: | ---: |
| adjust_bottle | 100% | 98% | 80% |
| beat_block_hammer | 100% | 98% | 28% |
| blocks_ranking_rgb | 98% | 100% | 56% |
| blocks_ranking_size | 90% | 88% | 30% |
| click_alarmclock | 100% | 100% | 68% |
| click_bell | 100% | 100% | 90% |
| dump_bin_bigbin | 96% | 98% | 86% |
| grab_roller | 100% | 100% | 60% |
| handover_block | 94% | 92% | 14% |
| handover_mic | 80% | 92% | 30% |
| hanging_mug | 88% | 80% | 40% |
| lift_pot | 100% | 100% | 24% |
| move_can_pot | 96% | 96% | 12% |
| move_pillbottle_pad | 100% | 100% | 50% |
| move_playingcard_away | 100% | 100% | 20% |
| move_stapler_pad | 88% | 88% | 10% |
| open_laptop | 100% | 98% | 54% |
| open_microwave | 94% | 98% | 12% |
| pick_diverse_bottles | 92% | 94% | 60% |
| pick_dual_bottles | 100% | 100% | 70% |
| place_a2b_left | 98% | 88% | 48% |
| place_a2b_right | 92% | 92% | 38% |
| place_bread_basket | 84% | 88% | 62% |
| place_bread_skillet | 94% | 92% | 50% |
| place_burger_fries | 100% | 100% | 76% |
| place_can_basket | 90% | 68% | 46% |
| place_cans_plasticbox | 100% | 100% | 82% |
| place_container_plate | 100% | 100% | 78% |
| place_dual_shoes | 92% | 98% | 22% |
| place_empty_cup | 100% | 100% | 86% |
| place_fan | 96% | 88% | 46% |
| place_mouse_pad | 96% | 90% | 56% |
| place_object_basket | 86% | 94% | 46% |
| place_object_scale | 100% | 94% | 42% |
| place_object_stand | 100% | 96% | 74% |
| place_phone_stand | 98% | 98% | 64% |
| place_shoe | 100% | 100% | 78% |
| press_stapler | 96% | 100% | 90% |
| put_bottles_dustbin | 92% | 98% | 78% |
| put_object_cabinet | 86% | 80% | 26% |
| rotate_qrcode | 98% | 88% | 2% |
| scan_object | 94% | 100% | 16% |
| shake_bottle | 100% | 96% | 98% |
| shake_bottle_horizontally | 100% | 98% | 98% |
| stack_blocks_three | 100% | 96% | 58% |
| stack_blocks_two | 100% | 100% | 80% |
| stack_bowls_three | 82% | 86% | 78% |
| stack_bowls_two | 94% | 96% | 84% |
| stamp_seal | 92% | 92% | 38% |
| turn_switch | 86% | 76% | 38% |
| **MACRO** | **95.24%** | **94.24%** | **53.44%** |

</details>

## Dataset

The training data is available on Hugging Face:
[`lerobot/robotwin_unified`](https://huggingface.co/datasets/lerobot/robotwin_unified)
— a LeRobot v3.0 dataset with 50 tasks × 550 episodes each (50 clean + 500
domain-randomized), three cameras at 640×480 (30 fps), 14-D absolute dual-arm
joint actions.

The three cameras are tiled into one `concat_view` frame (what the model sees):

```
┌───────────────────────┐
│       cam_high        │  third-person, full width
├───────────┬───────────┤
│ left_wrist│right_wrist│  each half-width
└───────────┴───────────┘
```

```bash
# download (requires git-lfs)
git clone https://huggingface.co/datasets/lerobot/robotwin_unified $ROBOTWIN_ROOT
```

Set `ROBOTWIN_ROOT` to the cloned directory. The loader expects
`$ROBOTWIN_ROOT/meta/info.json`.

## Training

### 1. Base checkpoint

Convert `nvidia/Cosmos3-Nano` to DCP and fetch the Wan2.2 VAE (see
[`docs/training.md`](../../docs/training.md)), then:

```bash
export BASE_CHECKPOINT_PATH=examples/checkpoints/Cosmos3-Nano
export WAN_VAE_PATH=examples/checkpoints/wan22_vae/Wan2.2_VAE.pth
```

### 2. Run

Two recipes sharing the same experiment config and dataset code:

| Param | `_clean_plus_random` | `_clean` |
| --- | --- | --- |
| **Data** | all 27,500 eps (50 clean + 500 random per task) | 2,500 eps (50 clean per task) |
| **Action chunk** | 64 | 64 |
| **Video subsample** | 4 (7.5 Hz → 17 frames) | 4 (7.5 Hz → 17 frames) |
| **Image aug** | ColorJitter (brightness=0.3, contrast=0.4, saturation=0.5) | ColorJitter (same) |
| **State conditioning** | `use_state=True` (joint state as action row 0) | `use_state=True` |
| **Action norm** | quantile (q01/q99 → [−1, 1]) | quantile |
| **Shape** | HSDP 8×4 (32 ranks) | HSDP 4×2 (8 ranks) |
| **Global batch** | 2048 | 512 |
| **Learning rate** | 5e-5 | 5e-5 |
| **Iterations** | 25k | 10k |

```bash
# whole scene (reference: 8 nodes × 4 GPUs)
ROBOTWIN_ROOT=<dir> bash examples/launch_sft_action_policy_robotwin_nano_clean_plus_random.sh

# clean subset (reference: 2 nodes × 4 GPUs)
ROBOTWIN_ROOT=<dir> bash examples/launch_sft_action_policy_robotwin_nano_clean.sh
```

Single-node smoke test (8 GPUs):

```bash
ROBOTWIN_ROOT=<dir> EXTRA_TAIL_OVERRIDES="\
  model.parallelism.data_parallel_shard_degree=8 \
  model.parallelism.data_parallel_replicate_degree=1 \
  trainer.grad_accum_iter=4 \
  trainer.max_iter=5 job.wandb_mode=offline" \
  bash examples/launch_sft_action_policy_robotwin_nano_clean_plus_random.sh
```

## Serving

Two entry points — both read `packages/vla-eval-cosmos3/configs/robotwin.yaml`:

```bash
# vla-eval WebSocket server (used by run_eval.sh)
.venv/bin/python -m vla_eval_cosmos3 \
  --config packages/vla-eval-cosmos3/configs/robotwin.yaml --port 8000

# plain HTTP server (ad-hoc / debugging)
python -m cosmos_framework.scripts.action_policy_server_robotwin \
  --experiment action_policy_robotwin_nano \
  --experiment-overrides "model.config.tokenizer.vae_path=$WAN_VAE_PATH" \
  --checkpoint-path <run>/checkpoints/iter_000025000 \
  --action-normalization quantile \
  --action-stats-path cosmos_framework/data/generator/action/normalizer_stats/robotwin_stats.json \
  --raw-action-dim 14 --action-chunk-size 64 --video-subsample-factor 4 \
  --fps 30 --port 8000
```

`POST /predict` payload for the plain server:

```jsonc
{
  "image": "<base64 PNG of the concat_view composite>",
  "prompt": "<task description>",
  "domain_name": "robotwin",
  "image_size": 720,
  "state": [/* 14 floats: current joints in raw units */]
}
```

## Evaluation

Two isolated processes share one GPU inside the Docker container, connected over
a local WebSocket:

```
┌─────────────────────────────────────────────────────────┐
│  Docker container (cosmos-robotwin:0.4.0)               │
│                                                         │
│  cosmos venv                 robotwin conda             │
│  ┌──────────────────┐        ┌──────────────────────┐   │
│  │  vla_eval_cosmos3│◄──────►│  vla-eval benchmark  │   │
│  │  (policy server) │ ws://  │  (SAPIEN sim +       │   │
│  │                  │ :8000  │   CuRobo expert)     │   │
│  └──────────────────┘        └──────────────────────┘   │
│   loads once, reused                                    │
│   across all 50 tasks                                   │
└─────────────────────────────────────────────────────────┘
```

```bash
# 1. build the image once (~30–60 min)
docker build -f examples/robotwin/docker/Dockerfile -t cosmos-robotwin:0.4.0 .

# 2. smoke test: 2 tasks, 5 episodes
TASK_CONFIG=demo_clean_large SHARDS=1 TEST_NUM=5 \
CHECKPOINT_PATH=<run>/checkpoints/iter_000025000 WAN_VAE_PATH=<...>/Wan2.2_VAE.pth \
  bash examples/robotwin/eval/run_eval.sh place_empty_cup adjust_bottle

# 3a. in-distribution (clean scenes, unseen instructions, 100 ep/task)
TASK_CONFIG=demo_clean_large \
CHECKPOINT_PATH=<...> WAN_VAE_PATH=<...> OUTPUT_DIR=outputs/robotwin_eval_clean \
  bash examples/robotwin/eval/run_eval.sh

# 3b. randomized (domain-randomized scenes, unseen instructions, 100 ep/task)
TASK_CONFIG=demo_randomized_large \
CHECKPOINT_PATH=<...> WAN_VAE_PATH=<...> OUTPUT_DIR=outputs/robotwin_eval_random \
  bash examples/robotwin/eval/run_eval.sh

# 4. summarize
python examples/robotwin/eval/collect_results.py clean_run \
  --results-dir outputs/robotwin_eval_clean --ckpt <...>/iter_000025000
python examples/robotwin/eval/collect_results.py random_run \
  --results-dir outputs/robotwin_eval_random --ckpt <...>/iter_000025000
```

Step 3 is slow (3.8–11.4 min per episode). Always run step 2 first.

### Hardware

A100 (80 GB) is the suggested GPU. The CuRobo wheel in the image is compiled for
Ampere; Hopper GPUs (H100, H200) need to recompile CuRobo from source before the
expert-check planner will run. A 24 GB card is likely to OOM with the policy
server and SAPIEN on the same GPU.

## Inference configuration

A mismatch in any of the following produces plausible-looking numbers rather than
an error:

| Setting | Must match | Why |
| --- | --- | --- |
| `use_state` | `true` | State row dropped or noised → proprioception lost |
| `action_normalization` | `quantile` | Outputs decoded in the wrong space |
| `action_chunk_size` | `64` | Wrong conditioning layout |
| `video_subsample_factor` | `4` | 65 frames sent instead of 17; scores far lower |
| `image_size` | `720` | 640×480 cameras → 720×640 composite, 720 = no pre-resize |
| `fps` | `30` | Wrong temporal conditioning |
| Camera resolution | 640×480 (`Large_D435`) | 320×240 took grasp-and-place from ~90% to 0% |
| `max_batch_size` | `1` (with `use_state`) | Batched path skips state row |

`collect_results.py` reports `ok`, `errored`, `truncated`, and `rejected` counts
separately — the macro covers `ok` tasks only, so nothing is silently dropped.
