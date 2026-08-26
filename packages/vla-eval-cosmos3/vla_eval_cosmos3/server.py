# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""NVIDIA Cosmos3 action-policy model server for vla-evaluation-harness.

Bridges a trained **Cosmos3** action policy (e.g. Cosmos3-Nano post-trained on
RoboTwin / LIBERO / DROID) into vla-eval.  It wraps cosmos-framework's
``RoboTwinModelService`` (from ``cosmos_framework.scripts.action_policy_server_robotwin``)
**in-process** — no second HTTP server, no extra network hop — and exposes it via
vla-eval's ``PredictModelServer`` interface:

* ``predict()``       -> ``RoboTwinModelService.predict_policy()``       (single obs)
* ``predict_batch()`` -> ``RoboTwinModelService.predict_policy_batch()`` (GPU-batched)

Both routes call the same diffusion sampler (``generate_samples_from_batch``), so
guidance (CFG) / ``num_steps`` behave identically single vs batched.

Launch (in cosmos-framework's venv, with cosmos_framework + vla-eval installed):

    .venv/bin/python -m vla_eval_cosmos3 \
        --config packages/vla-eval-cosmos3/configs/robotwin.yaml --port 8000

    The benchmark runs in the RoboTwin conda env and connects over
    ``ws://localhost:8000`` — a separate process, fully isolated from this env.

RoboTwin observation contract (see ``configs/robotwin.yaml`` for the full recipe):
    The policy consumes a single ``concat_view`` image: ``head_camera`` full-width
    on top, ``left_camera`` + ``right_camera`` (each downscaled 2x) side-by-side on
    the bottom row.  ``domain_name="robotwin"`` (id 17), 14-D absolute joint_pos
    output.  Normalization must match the checkpoint: the reference RoboTwin recipe
    trains with ``use_state=True`` + quantile stats, so the config sets both.  A
    checkpoint trained without them needs ``use_state=false`` +
    ``action_stats_path=null``; a mismatch silently mangles the motion.
"""

from __future__ import annotations

import base64
import io
import logging
from typing import Any

import numpy as np
from vla_eval.model_servers.base import SessionContext
from vla_eval.model_servers.predict import PredictModelServer
from vla_eval.specs import IMAGE_RGB, LANGUAGE, DimSpec
from vla_eval.types import Action, Observation

logger = logging.getLogger(__name__)


def _to_uint8_rgb(arr: Any) -> np.ndarray:
    """Coerce a camera frame to a contiguous HxWx3 uint8 RGB array."""
    a = np.asarray(arr)
    if a.dtype != np.uint8:
        # Floats in [0,1] -> [0,255]; otherwise assume already 0..255.
        a = (a * 255.0).clip(0, 255).astype(np.uint8) if a.max() <= 1.0 else a.clip(0, 255).astype(np.uint8)
    if a.ndim == 2:
        a = np.stack([a] * 3, axis=-1)
    if a.shape[-1] == 4:  # drop alpha
        a = a[..., :3]
    return np.ascontiguousarray(a)


class CosmosPolicyModelServer(PredictModelServer):
    """vla-eval model server wrapping a Cosmos3 action policy."""

    def __init__(
        self,
        *,
        checkpoint_path: str,
        experiment: str,
        vae_path: str,
        domain_name: str,
        action_stats_path: str | None = None,  # None -> server skips denorm (raw joints)
        raw_action_dim: int,
        action_chunk_size: int,
        image_size: int,
        fps: int,
        video_subsample_factor: int = 1,
        camera_layout: str = "robotwin_concat",
        use_state: bool = False,  # True for checkpoints trained with use_state=True
        action_normalization: str = "minmax",  # only used if action_stats_path is set
        max_action_dim: int | None = None,
        format_prompt_as_json: bool | None = None,
        guidance: float = 1.0,
        num_steps: int = 30,
        seed: int = 0,
        sampler: str = "unipc",
        output_dir: str | None = None,
        experiment_overrides: list[str] | None = None,
        dump_dir: str | None = None,
        dump_every: int = 1,
        max_batch_size: int = 1,
        max_wait_time: float = 0.01,
        **kwargs: Any,
    ) -> None:
        """
        Args mirror the RoboTwin serving recipe (`configs/robotwin.yaml`):

            checkpoint_path: Trained DCP checkpoint dir WITH action heads
                (an ``iter_*`` dir), or an exported HF/safetensors dir.
            experiment: Hydra experiment supplying the model config for a DCP
                checkpoint (e.g. ``action_policy_robotwin_nano``).
            vae_path: Path to the Wan2.2 VAE weights; injected as
                ``model.config.tokenizer.vae_path`` override.
            domain_name: Cosmos embodiment domain sent per request
                (``robotwin`` -> id 17, ``libero`` -> 5, ``droid_lerobot`` -> 8).
            action_stats_path: JSON with de-normalisation stats. Must match what the
                checkpoint trained with; null -> server skips denorm.
            raw_action_dim: Unpadded action width (RoboTwin dual-arm = 14).
            action_chunk_size: Actions per inference == training ``chunk_length``.
                Buffered by ``PredictModelServer`` and served one per env step.
            video_subsample_factor: Action steps per observation frame, matching
                the dataset's ``video_subsample_factor``. 1 = dense video; 2 = the
                ``vsub2`` recipe (15 Hz video, 30 Hz actions) -> 17 frames for
                chunk 32, and the matching ``encode_exact_durations``. A mismatch
                does not error: the model just sees a vision sequence of the wrong
                length and quietly scores lower.
            image_size: Target image height sent to the policy. Coupled to the sim
                camera type: RoboTwin at 640x480 (Large_D435) composites to 720x640,
                so 720 means "no pre-resize" and matches training.
            fps: Conditioning fps == training fps (RoboTwin = 30).
            camera_layout: How to compose obs cameras into the policy image.
                ``robotwin_concat`` = head on top, left|right wrists on the
                bottom row.  ``single`` = first camera only.
            action_normalization: Normalisation to invert; only used when
                action_stats_path is set (RoboTwin: ``quantile``).
            guidance: Diffusion CFG scale (1.0 = no CFG, the RoboTwin default).
        """
        # `ActionModelService.predict_policy_batch` ignores the `state` field —
        # only the single-observation path prepends it as action row 0. Batching a
        # state-conditioned policy would therefore drop proprioception on every
        # request and quietly degrade the score instead of failing. Refuse it.
        if use_state and max_batch_size > 1:
            raise ValueError(
                f"use_state=True requires max_batch_size=1 (got {max_batch_size}): the batched "
                "path does not prepend the state row, so conditioning would be silently dropped."
            )
        super().__init__(
            chunk_size=action_chunk_size,
            max_batch_size=max_batch_size,
            max_wait_time=max_wait_time,
            **kwargs,
        )
        self.domain_name = domain_name
        self.image_size = int(image_size)
        self.camera_layout = camera_layout
        self.use_state = use_state

        # Heavy imports deferred to construction (they pull in torch + the full
        # Cosmos inference stack).  Per the base ABC, the model must be fully
        # loaded before __init__ returns.
        from pathlib import Path as _P

        from cosmos_framework.inference.common.args import CheckpointOverrides
        from cosmos_framework.scripts.action_policy_server_robotwin import (
            RoboTwinModelService,
            RoboTwinServerArgs,
        )

        # Guardrails are a text/video generation-safety model (Cosmos-Guardrail1)
        # irrelevant to action-policy inference; leaving them on makes OmniInference
        # shell out to `uvx` to download the guardrail weights and consumes GPU
        # memory. Disable them by forcing setup_overrides.guardrails=False.
        class _NoGuardrailArgs(RoboTwinServerArgs):
            def build_setup_overrides(self):  # type: ignore[override]
                base = super().build_setup_overrides()
                base.guardrails = False
                return base

        overrides = list(experiment_overrides or [])
        vae_override = f"model.config.tokenizer.vae_path={vae_path}"
        if vae_override not in overrides:
            overrides.append(vae_override)

        args = _NoGuardrailArgs(
            checkpoint=CheckpointOverrides(
                checkpoint_path=checkpoint_path,
                experiment=experiment,
                experiment_overrides=overrides,
            ),
            output_dir=_P(output_dir) if output_dir else None,
            sampler=sampler,  # type: ignore[arg-type]
            seed=seed,
            guidance=guidance,
            num_steps=num_steps,
            fps=fps,
            action_chunk_size=action_chunk_size,
            video_subsample_factor=video_subsample_factor,
            max_action_dim=max_action_dim,
            raw_action_dim=raw_action_dim,
            action_stats_path=_P(action_stats_path) if action_stats_path else None,
            action_normalization=action_normalization,  # type: ignore[arg-type]
            format_prompt_as_json=format_prompt_as_json,
            dump_dir=_P(dump_dir) if dump_dir else None,
            dump_every=dump_every,
        )
        logger.info(
            "Loading Cosmos policy: experiment=%s domain=%s checkpoint=%s",
            experiment,
            domain_name,
            checkpoint_path,
        )
        self._svc = RoboTwinModelService(args)

        # Warmup: the first forward JIT-compiles CUDA kernels (~25s). Without this
        # the first *real* request (worse when batched) blows the benchmark's 30s
        # act-timeout. Do a dummy forward now at the batch size.
        try:
            _hw = (480, 640, 3)
            if camera_layout == "robotwin_concat":
                _dummy = {
                    "images": {k: np.zeros(_hw, np.uint8) for k in ("head_camera", "left_camera", "right_camera")},
                    "task_description": "warmup",
                }
            else:
                _dummy = {"images": {"cam": np.zeros(_hw, np.uint8)}, "task_description": "warmup"}
            _req = self._to_req(_dummy)
            _n = max(1, max_batch_size)
            self._svc.predict_policy_batch([_req] * _n) if _n > 1 else self._svc.predict_policy(_req)
            logger.info("Warmup forward done (batch=%d).", _n)
        except Exception as e:  # never fail startup on warmup
            logger.warning("Warmup forward failed (non-fatal): %s", e)

        logger.info("Cosmos policy loaded (chunk=%d, raw_action_dim=%d).", action_chunk_size, raw_action_dim)

    # -- specs ---------------------------------------------------------------

    def get_action_spec(self) -> dict[str, DimSpec]:
        # 14-D dual-arm absolute joint positions — matches RoboTwinBenchmark.
        return {"joints": DimSpec("joints", 14, "joint_positions")}

    def get_observation_spec(self) -> dict[str, DimSpec]:
        if self.camera_layout == "robotwin_concat":
            spec = {
                "head_camera": IMAGE_RGB,
                "left_camera": IMAGE_RGB,
                "right_camera": IMAGE_RGB,
                "language": LANGUAGE,
            }
        else:
            spec = {"image": IMAGE_RGB, "language": LANGUAGE}
        if self.use_state:
            # 14-D current joint positions, same layout as the action.
            spec["state"] = DimSpec("state", 14, "joint_positions")
        return spec

    # -- image composition ---------------------------------------------------

    def _compose_image(self, obs: Observation) -> np.ndarray:
        """Build the policy input image (HxWx3 uint8) from the obs cameras."""
        images = obs.get("images", {})
        if not isinstance(images, dict):  # single ndarray obs
            return _to_uint8_rgb(images)

        if self.camera_layout == "single":
            return _to_uint8_rgb(next(iter(images.values())))

        # robotwin_concat: head full-width on top; left|right wrists (each 2x
        # downscaled) on the bottom row -> (3H/2, W, 3).  Mirrors
        # RoboTwinLeRobotDataset._compose_concat_view.
        from PIL import Image as PILImage

        head = _to_uint8_rgb(images["head_camera"])
        left = _to_uint8_rgb(images["left_camera"])
        right = _to_uint8_rgb(images["right_camera"])
        h, w = head.shape[:2]
        half_h, half_w = h // 2, w // 2

        def _resize(a: np.ndarray) -> np.ndarray:
            return np.asarray(PILImage.fromarray(a).resize((half_w, half_h), resample=PILImage.Resampling.BILINEAR))

        bottom = np.concatenate([_resize(left), _resize(right)], axis=1)  # (H/2, W, 3)
        return np.concatenate([head, bottom], axis=0)  # (3H/2, W, 3)

    def _to_req(self, obs: Observation) -> dict[str, Any]:
        from PIL import Image as PILImage

        img = self._compose_image(obs)
        buf = io.BytesIO()
        PILImage.fromarray(img).save(buf, format="PNG")
        req = {
            "image": base64.b64encode(buf.getvalue()).decode("ascii"),
            "prompt": obs.get("task_description", ""),
            "domain_name": self.domain_name,
            "image_size": self.image_size,
        }
        # State conditioning: checkpoints trained with `use_state=True` expect the
        # current 14-D joint state prepended as action row 0. The server normalizes
        # and prepends it; omit for vision-only checkpoints (use_state=False).
        if self.use_state:
            state = obs.get("joint_state")
            if state is not None:
                req["state"] = np.asarray(state, dtype=np.float32).flatten().tolist()
        return req

    # -- inference -----------------------------------------------------------

    def predict(self, obs: Observation, ctx: SessionContext) -> Action:
        out = self._svc.predict_policy(self._to_req(obs))
        return {"actions": np.asarray(out["action"], dtype=np.float32)}  # [T, raw_action_dim]

    def predict_batch(self, obs_batch: list[Observation], ctx_batch: list[SessionContext]) -> list[Action]:
        reqs = [self._to_req(o) for o in obs_batch]
        out = self._svc.predict_policy_batch(reqs)
        return [{"actions": np.asarray(a, dtype=np.float32)} for a in out["actions"]]


def main() -> None:
    """Console/`python -m vla_eval_cosmos3` entry point."""
    from vla_eval.model_servers.serve import run_server

    run_server(CosmosPolicyModelServer)


if __name__ == "__main__":
    main()
