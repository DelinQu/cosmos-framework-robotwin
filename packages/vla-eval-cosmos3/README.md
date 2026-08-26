<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: OpenMDW-1.1 -->

# vla-eval-cosmos3

A [vla-evaluation-harness](https://github.com/allenai/vla-evaluation-harness)
model server that serves **Cosmos3 action policies**.

It wraps cosmos-framework's `RoboTwinModelService`
(`cosmos_framework/scripts/action_policy_server_robotwin.py`) **in-process** — no
second HTTP server, no extra network hop — and exposes it through vla-eval's
`PredictModelServer` interface:

| vla-eval route  | Cosmos3 call                            |
| --------------- | --------------------------------------- |
| `predict()`     | `ActionModelService.predict_policy()`      |
| `predict_batch()` | `ActionModelService.predict_policy_batch()` |

Both routes reach the same diffusion sampler, so guidance (CFG) and `num_steps`
behave identically single vs batched.

## Install

Installs into the cosmos-framework venv, which supplies `cosmos_framework`:

```bash
uv pip install vla-eval ./packages/vla-eval-cosmos3
```

## Run

```bash
.venv/bin/python -m vla_eval_cosmos3 \
  --config packages/vla-eval-cosmos3/configs/robotwin.yaml --port 8000
```

Serves on `ws://localhost:8000`. The benchmark process connects to it from the
simulator's own environment — a separate interpreter, fully isolated. Override
any config field on the CLI with `--args.<key>=<value>`, e.g.
`--args.checkpoint_path=/ckpt/policy`.

Run it from the **framework root**: `configs/robotwin.yaml` names
`action_stats_path` relative to that root, and the model build opens config
assets by relative path.

## Configs

`configs/robotwin.yaml` — RoboTwin 2.0 (dual-arm ALOHA, 14-D absolute
`joint_pos`). The policy consumes one `concat_view` image: `head_camera`
full-width on top, `left_camera` + `right_camera` (each downscaled 2x)
side-by-side below. `domain_name: robotwin` (domain id 17).

### The config must match the checkpoint

These fields are not preferences — they define the space the policy was trained
in, and a mismatch **silently mangles the motion rather than erroring**:

| Field                  | Reference RoboTwin value | Why it matters                                               |
| ---------------------- | ------------------------ | ------------------------------------------------------------ |
| `use_state`            | `true`                   | Prepends the current 14-D joint state as action row 0         |
| `action_normalization` | `quantile`               | q01/q99 → `[-1,1]`, applied to the state row too              |
| `action_stats_path`    | `robotwin_stats.json`    | The stats the policy's outputs are expressed in               |
| `action_chunk_size`    | `64`                     | = training `chunk_length`                                     |
| `video_subsample_factor`| `4`                     | Observation frames per chunk = `chunk // factor + 1` (= 17)   |
| `fps`                  | `30`                     | = training fps                                                |
| `image_size`           | `720`                    | Coupled to the sim camera type — see below                    |
| `format_prompt_as_json`| `true`                   | Training wraps the task string in the action JSON prompt      |

`image_size` is coupled to the **simulator camera type**, not chosen freely.
640x480 cameras (`Large_D435`) composite to 720x640, and `720` means "no
pre-resize", matching training's single 720→640x640 resize. With 320x240
cameras it would have to be 360.

`max_batch_size` must stay `1` whenever `use_state` is true: the batched path
does not prepend the state row. The server raises on that combination rather
than serving unconditioned actions.

`video_subsample_factor: 4` matches the shipped recipe, which trains on
subsampled video (7.5 Hz video against 30 Hz actions → 17 observation frames for
chunk 64). Serving such a checkpoint at factor 1 does not error — it sends 65
frames instead of 17 and quietly scores far lower. A dense checkpoint needs it
set back to 1.

## Adding another embodiment

Copy `configs/robotwin.yaml`, then set `domain_name` (must exist in
`EMBODIMENT_TO_DOMAIN_ID`), `raw_action_dim`, and `camera_layout`
(`robotwin_concat`, or `single` for one camera). A layout that training does not
produce needs a new branch in `CosmosPolicyModelServer._compose_image`.
