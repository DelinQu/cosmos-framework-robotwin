# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Compute action normalization stats for the RoboTwin (dual-arm ALOHA) dataset.

Reads the raw 14-D absolute ``joint_pos`` ``action`` column straight from the
LeRobot data parquets (the RoboTwinLeRobotDataset feeds actions through
un-transformed, so parquet stats == training-time action stats) and writes the
per-dim ``mean/std/min/max/q01/q99`` JSON that
``BaseActionLeRobotDataset._load_norm_stats`` expects under the ``"global"`` key.

Write it to the path ``RoboTwinLeRobotDataset._normalizer_path()`` resolves:
``cosmos_framework/data/generator/action/normalizer_stats/robotwin_stats.json``.
The bundled file was produced this way; regenerate it only if you retrain on a
different RoboTwin dataset build, since the stats and the checkpoint must agree.

The reference recipe consumes these via ``action_normalization="quantile"``
(q01/q99 -> [-1, 1]).

Usage:
    python examples/robotwin/tools/compute_robotwin_action_stats.py \
        --data-root <robotwin LeRobot root> \
        --out cosmos_framework/data/generator/action/normalizer_stats/robotwin_stats.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

_ACTION_FEATURE = "action"
_EXPECTED_DIM = 14


def _load_actions(data_root: Path) -> np.ndarray:
    files = sorted(data_root.glob("data/chunk-*/file-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No data parquets under {data_root}/data/chunk-*/file-*.parquet")
    chunks = []
    for f in files:
        col = pq.read_table(f, columns=[_ACTION_FEATURE]).column(_ACTION_FEATURE)
        # LeRobot stores `action` as a fixed-size / variable list column -> stack to [N, D].
        arr = np.asarray(col.to_pylist(), dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != _EXPECTED_DIM:
            raise ValueError(f"{f}: expected action shape [N,{_EXPECTED_DIM}], got {arr.shape}")
        chunks.append(arr)
        print(f"  {f.name}: {arr.shape[0]} frames")
    actions = np.concatenate(chunks, axis=0)
    print(f"Total: {actions.shape[0]} frames x {actions.shape[1]} dims")
    return actions


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, required=True, help="RoboTwin LeRobotDataset root (has meta/, data/).")
    ap.add_argument("--out", type=Path, required=True, help="Output stats JSON path.")
    args = ap.parse_args()

    actions = _load_actions(args.data_root.expanduser())
    stats = {
        "mean": np.mean(actions, axis=0),
        "std": np.std(actions, axis=0),
        "min": np.min(actions, axis=0),
        "max": np.max(actions, axis=0),
        "q01": np.quantile(actions, 0.01, axis=0),
        "q99": np.quantile(actions, 0.99, axis=0),
    }
    payload = {"global": {k: v.astype(np.float64).round(8).tolist() for k, v in stats.items()}}

    out = args.out.expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {out}")
    for k, v in stats.items():
        print(f"  {k}: {np.round(v, 4).tolist()}")


if __name__ == "__main__":
    main()
