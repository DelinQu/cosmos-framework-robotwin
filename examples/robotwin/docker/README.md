<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: OpenMDW-1.1 -->

# RoboTwin evaluation image

Files baked into `cosmos-robotwin` by [`Dockerfile`](./Dockerfile). Build it from
the repo root:

```bash
docker build -f examples/robotwin/docker/Dockerfile -t cosmos-robotwin:0.4.0 .
```

## Task configs

`demo_clean_large.yml` and `demo_randomized_large.yml` are the RoboTwin harness's
`demo_clean.yml` / `demo_randomized.yml` with **one field changed**: the head and
wrist camera type goes from `D435` (320x240) to `Large_D435` (640x480). Same
`fovy` (37), so the field of view is identical — only the pixel count differs,
and 640x480 is what the training videos were rendered at.

Serving the policy 320x240 frames, which the server then upscales to the 640x640
model canvas, costs 4x the pixels and is silent: press/shake tasks tolerate the
blur, grasp-and-place does not. Measured on an iter-7000 checkpoint, 10 episodes:

| task | D435 320x240 | Large_D435 640x480 |
|---|---:|---:|
| adjust_bottle | 0% | 100% |
| stack_blocks_two | 0% | 100% |
| place_empty_cup | 0% | 90% |
| place_a2b_left | 0% | 60% |
| shake_bottle (control) | 80% | 100% |

Which to evaluate on:
- **`demo_clean_large`** — in-distribution. The training data is clean-domain, so
  this measures fit.
- **`demo_randomized_large`** — adds random background, cluttered table, random
  lighting and ±0.03 table height. Camera pose is *not* randomized
  (`random_head_camera_dis: 0`). This measures generalization, and on the
  reference checkpoint costs only 0.36 points against `demo_clean_large`
  (92.84% vs 93.20%) — so it is currently measuring very little.

Select it with `TASK_CONFIG` in `examples/robotwin/eval/run_eval.sh` (which maps
to the benchmark's `task_config` parameter).

## Env patches

`envs/{open_laptop,place_object_scale,put_object_cabinet}.py` are full-file forks
of RoboTwin `c3ddfa8`. Each moves the `check_success()` state derivation out of
`play_once()` (the *expert* script) into `_init_eval_state()`, called at the end
of `setup_demo()`.

Policy evaluation runs the expert on a different env object than the rollout, so
without this the rollout env raises `AttributeError` on its first step and these
three tasks score 0% regardless of policy quality. Timing matters: `origin_z` is
the object's *settled* resting height, compared against a 7 mm threshold, so the
derivation has to happen after `check_stable()` has stepped the sim.

The Dockerfile AST-asserts all three patches at build time. If you bump
`BASE_IMAGE` to an image shipping a newer RoboTwin, **re-fork these three files
first** — nothing detects the drift, and the `COPY` would silently revert
upstream work.

## Regenerating after a harness upgrade

The task configs are copies, so a RoboTwin upgrade that changes the base configs
will not reach them. Re-derive both from the image's own copies:

```bash
for cfg in demo_clean demo_randomized; do
  docker run --rm cosmos-robotwin:0.4.0 \
    cat /app/RoboTwin/task_config/$cfg.yml \
    | sed -e 's/head_camera_type: D435/head_camera_type: Large_D435/' \
          -e 's/wrist_camera_type: D435/wrist_camera_type: Large_D435/' \
    > examples/robotwin/docker/${cfg}_large.yml
done
```

The harness also ships `create_task_config.sh`, the upstream-sanctioned way to
author variants — prefer it if it can set the camera type directly.
