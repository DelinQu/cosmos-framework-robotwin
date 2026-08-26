#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Collect one RoboTwin eval run into a self-describing, committable directory.

Reads the output dir produced by ``run_eval.sh`` (one subdirectory per task) and
writes::

    examples/robotwin/results/<run_name>/
    ├── eval_config.yaml    what the harness actually ran, read back from results
    ├── server_config.yaml  normalization / camera layout / image_size
    ├── results.csv         one row per task
    └── summary.md          macro + exclusions

Every task is classified, because a bare success rate hides two failure modes
that have each produced wrong numbers here:

    ok         aggregate written, no real env errors
    errored    num_errors > 0 — the env raised, so 0% means "never ran", not
               "policy failed"
    truncated  no aggregate, only .progress — the watchdog fired mid-run. These
               are the LONGEST tasks, so dropping them silently inflates the macro
    missing    neither file — the task never produced output

Episodes RoboTwin refused to start (``UnStableError``: the scene never settled)
are dropped from the denominator rather than scored 0 or counted against the
task -- see ``_SCENE_REJECT``. The ``rejected`` column reports how many.

The macro is computed over ``ok`` only, and everything else is listed explicitly.

Usage:
    python examples/robotwin/eval/collect_results.py randomized_large_iter025000 \\
        --results-dir outputs/robotwin_eval \\
        --ckpt /path/to/iter_000025000
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

AGGREGATE = "RoboTwinBenchmark_aggregate.json"
PROGRESS = "RoboTwinBenchmark.progress"
COLUMNS = [
    "run",
    "task",
    "status",
    "shards",
    "episodes",
    "completed",
    "num_errors",
    "rejected",
    "success_pct",
    "avg_steps",
    "failure_reason",
]

# run_eval.sh writes shards of one task as "<task>-s<N>" (SHARDS>1).
SHARD_RE = re.compile(r"-s(\d+)$")

# RoboTwin refuses to start an episode whose scene never settles, raising
# UnStableError from _init_task_env_ before step 0. That is the benchmark
# declining to pose the question, not the env breaking and not the policy
# failing -- and it is DETERMINISTIC: the same seed rejects on every retry, so
# it cannot be cleared by resubmitting. Counting it as a failure would penalise
# the policy for a scene it never saw; counting the whole task as `errored`
# would discard a task outright over one bad seed. Measured: dump_bin_bigbin
# lost 1 of 20 episodes to seed 300002 and was dropped from the macro despite
# going 19/19 on the episodes that actually ran.
_SCENE_REJECT = "UnStableError"


def _is_scene_reject(ep: dict[str, Any]) -> bool:
    return _SCENE_REJECT in (ep.get("failure_detail") or "")


def row_from_aggregate(run: str, task: str, doc: dict[str, Any]) -> dict[str, Any]:
    t = doc["tasks"][0]
    eps = t.get("episodes", [])
    rejected = [e for e in eps if _is_scene_reject(e)]
    # Real errors only: the env actually raised. Scene rejections leave the
    # denominator instead, so they neither fail the policy nor kill the task.
    n_err = max(0, t.get("num_errors", 0) - len(rejected))
    valid = [e for e in eps if not _is_scene_reject(e)]
    reasons = collections.Counter(e.get("failure_reason") for e in valid if e.get("failure_reason"))
    if rejected:
        reasons[f"scene-rejected({_SCENE_REJECT})"] = len(rejected)
    # Recompute rather than trust mean_success/avg_steps: the aggregate scores a
    # rejected episode as a 0-step failure.
    n_valid = len(valid) or t["num_episodes"]
    if rejected and valid:
        succ = sum(1 for e in valid if e.get("metrics", {}).get("success"))
        success_pct = round(succ / len(valid) * 100, 1)
        steps = [e.get("steps") or 0 for e in valid]
        avg_steps = round(sum(steps) / len(steps), 1)
    else:
        success_pct = round(t["mean_success"] * 100, 1)
        avg_steps = round(t["avg_steps"], 1)
    return {
        "run": run,
        "task": task,
        "status": "errored" if n_err else "ok",
        "episodes": n_valid,
        "completed": n_valid,
        "num_errors": n_err,
        "rejected": len(rejected),
        "success_pct": success_pct,
        "avg_steps": avg_steps,
        "failure_reason": ";".join(f"{k}x{v}" for k, v in reasons.most_common()),
        "shards": 1,
    }


def row_from_progress(run: str, task: str, doc: dict[str, Any]) -> dict[str, Any]:
    return {
        "run": run,
        "task": task,
        "status": "truncated",
        "episodes": doc.get("total", ""),
        "completed": doc.get("completed", ""),
        "num_errors": doc.get("errors", ""),
        "rejected": 0,
        "success_pct": "",
        "avg_steps": "",
        "failure_reason": "watchdog timeout (no aggregate written)",
        "shards": 1,
    }


def merge(parts: list[dict[str, Any]]) -> dict[str, Any]:
    """Fold shards of one task into one row.

    Success must be weighted by episode count — averaging the shards' percentages
    would be wrong the moment they differ in size (e.g. a shard that died early).
    A truncated shard has no success_pct, so any truncated part makes the whole
    task truncated rather than silently reporting a partial macro as complete.
    """
    if len(parts) == 1:
        return parts[0]
    if any(p["status"] == "truncated" for p in parts):
        done = sum(int(p["completed"] or 0) for p in parts)
        total = sum(int(p["episodes"] or 0) for p in parts)
        head = dict(parts[0])
        head.update(
            status="truncated",
            shards=len(parts),
            episodes=total,
            completed=done,
            rejected=sum(int(p.get("rejected") or 0) for p in parts),
            success_pct="",
            avg_steps="",
            failure_reason=f"{sum(1 for p in parts if p['status'] == 'truncated')} of "
            f"{len(parts)} shards hit the watchdog timeout",
        )
        return head
    eps = sum(p["episodes"] for p in parts)
    successes = sum(p["success_pct"] / 100.0 * p["episodes"] for p in parts)
    steps = sum(p["avg_steps"] * p["episodes"] for p in parts) / eps if eps else 0.0
    errs = sum(p["num_errors"] for p in parts)
    rej = sum(int(p.get("rejected") or 0) for p in parts)
    reasons = ";".join(sorted({r for p in parts for r in p["failure_reason"].split(";") if r}))
    head = dict(parts[0])
    head.update(
        status="errored" if errs else "ok",
        episodes=eps,
        completed=eps,
        num_errors=errs,
        rejected=rej,
        success_pct=round(successes / eps * 100, 1) if eps else 0.0,
        avg_steps=round(steps, 1),
        failure_reason=reasons,
        shards=len(parts),
    )
    return head


def collect(results_dir: Path, run: str, skip: set[Path] = frozenset()) -> tuple[list[dict], dict | None]:
    """One row per task subdirectory of ``results_dir``.

    ``skip`` holds resolved paths that are not tasks — notably the output
    directory itself, which a caller may legitimately place inside the results
    dir and which would otherwise be reported as a ``missing`` task.
    """
    sample: dict[str, Any] | None = None
    parts: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)

    for task_dir in sorted(p for p in results_dir.iterdir() if p.is_dir()):
        if task_dir.name == "logs" or task_dir.resolve() in skip:
            continue
        task = SHARD_RE.sub("", task_dir.name)  # fold "<task>-s<N>" into "<task>"
        agg, prog = task_dir / AGGREGATE, task_dir / PROGRESS
        if agg.exists():
            doc = json.loads(agg.read_text())
            sample = sample or doc
            parts[task].append(row_from_aggregate(run, task, doc))
        elif prog.exists():
            parts[task].append(row_from_progress(run, task, json.loads(prog.read_text())))
        else:
            parts[task].append(
                {
                    "run": run,
                    "task": task,
                    "status": "missing",
                    "episodes": "",
                    "completed": "",
                    "num_errors": "",
                    "rejected": 0,
                    "success_pct": "",
                    "avg_steps": "",
                    "shards": 1,
                    "failure_reason": "no aggregate and no progress file",
                }
            )

    rows = [merge([p for p in ps if p["status"] != "missing"] or ps) for ps in parts.values()]
    rows.sort(key=lambda r: r["task"])
    return rows, sample


def write_eval_config(out: Path, run: str, ckpt: str, doc: dict[str, Any]) -> None:
    cfg = doc.get("config", {})
    params = cfg.get("params", {})
    obs = (doc.get("server_info") or {}).get("observation_spec", {})
    out.write_text(
        f"""# Read back from the run's own aggregate — what the harness actually ran,
# not what we believe we submitted.
run: {run}
checkpoint: {ckpt or "unknown"}
harness_version: {doc.get("harness_version", "?")}
benchmark: {cfg.get("benchmark", "?")}
mode: {cfg.get("mode", "?")}
episodes_per_task: {cfg.get("episodes_per_task", "?")}
recording: {cfg.get("recording", {})}
params:
  task_config: {params.get("task_config", "?")}
  test_num: {params.get("test_num", "?")}
  seed: {params.get("seed", "?")}
  instruction_type: {params.get("instruction_type", "?")}
  skip_expert_check: {params.get("skip_expert_check", "?")}
server:
  state_conditioned: {"state" in obs}
  action_spec: {(doc.get("server_info") or {}).get("action_spec", {})}
"""
    )


def write_summary(out: Path, run: str, ckpt: str, rows: list[dict]) -> None:
    ok = [r for r in rows if r["status"] == "ok"]
    macro = sum(r["success_pct"] for r in ok) / len(ok) if ok else float("nan")

    L = [f"# {run}", ""]
    if ckpt:
        L += [f"Checkpoint: `{ckpt}`", ""]
    L += [
        "Macro covers `ok` tasks only. Errored tasks (the env raised) and "
        "truncated tasks (watchdog timeout) are excluded and listed below: "
        "scoring them 0% understates the policy, and dropping them without "
        "saying so overstates it, since the truncated ones are the longest tasks.",
        "",
        "| ok | errored | truncated | missing | rejected eps | macro (ok) |",
        "|---:|---:|---:|---:|---:|---:|",
        f"| {len(ok)} | {sum(1 for r in rows if r['status'] == 'errored')} "
        f"| {sum(1 for r in rows if r['status'] == 'truncated')} "
        f"| {sum(1 for r in rows if r['status'] == 'missing')} "
        f"| {sum(int(r.get('rejected') or 0) for r in rows)} | {macro:.2f}% |",
        "",
    ]

    if ok:
        L += ["## Per task", "", "| task | success | avg steps |", "|---|---:|---:|"]
        L += [
            f"| {r['task']} | {r['success_pct']:.0f}% | {r['avg_steps']:.0f} |"
            for r in sorted(ok, key=lambda r: -r["success_pct"])
        ]
        L.append("")

    excl = [r for r in rows if r["status"] != "ok"]
    if excl:
        L += ["## Excluded", "", "| task | status | detail |", "|---|---|---|"]
        for r in sorted(excl, key=lambda r: r["task"]):
            detail = r["failure_reason"]
            if r["status"] == "truncated":
                detail = f"{r['completed']}/{r['episodes']} episodes — {detail}"
            L.append(f"| {r['task']} | {r['status']} | {detail} |")
        L.append("")

    out.write_text("\n".join(L))


def main() -> int:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_name", help="directory name, e.g. randomized_large_iter025000")
    ap.add_argument(
        "--results-dir", type=Path, required=True, help="run_eval.sh OUTPUT_DIR (holds one subdir per task)"
    )
    ap.add_argument("--ckpt", default="", help="checkpoint dir; its config.yaml is copied in as train_config.yaml")
    ap.add_argument("-o", "--out", type=Path, default=None, help="default: ../results/<run_name>")
    args = ap.parse_args()

    results_dir = args.results_dir.expanduser()
    if not results_dir.is_dir():
        print(f"results dir not found: {results_dir}", file=sys.stderr)
        return 1

    out = args.out or (here.parent / "results" / args.run_name)
    out.mkdir(parents=True, exist_ok=True)

    rows, sample = collect(results_dir, args.run_name, skip={out.resolve()})
    if not rows:
        print(f"no task directories under {results_dir}", file=sys.stderr)
        return 1

    with (out / "results.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)

    if sample:
        write_eval_config(out / "eval_config.yaml", args.run_name, args.ckpt, sample)
    write_summary(out / "summary.md", args.run_name, args.ckpt, rows)

    # The training config is the only record of what produced these numbers.
    if args.ckpt:
        src = Path(args.ckpt).expanduser() / "config.yaml"
        if src.exists():
            shutil.copyfile(src, out / "train_config.yaml")
        else:
            print(f"WARNING: no config.yaml under {args.ckpt}", file=sys.stderr)

    # The server config pins normalization, camera layout and image_size — all of
    # which change results without changing the checkpoint.
    server_cfg = here.parents[2] / "packages/vla-eval-cosmos3/configs/robotwin.yaml"
    if server_cfg.exists():
        shutil.copyfile(server_cfg, out / "server_config.yaml")

    ok = [r for r in rows if r["status"] == "ok"]
    macro = sum(r["success_pct"] for r in ok) / len(ok) if ok else float("nan")
    print(f"{len(rows)} rows -> {out}")
    print(
        f"  ok={len(ok)} "
        f"errored={sum(1 for r in rows if r['status'] == 'errored')} "
        f"truncated={sum(1 for r in rows if r['status'] == 'truncated')} "
        f"missing={sum(1 for r in rows if r['status'] == 'missing')} "
        f"rejected_eps={sum(int(r.get('rejected') or 0) for r in rows)} "
        f"macro={macro:.2f}%"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
