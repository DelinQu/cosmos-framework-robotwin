# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Cosmos3-Nano RoboTwin 2.0 policy HTTP server.

Extends the LIBERO action server with three RoboTwin-specific additions:

  1. **Subsampled video** (``--video-subsample-factor 4``) — sends
     ``chunk // factor + 1 = 17`` observation frames and auto-applies
     ``encode_exact_durations=[17]`` before model construction, matching the
     training recipe.  The LIBERO server always sends ``chunk + 1`` dense frames;
     serving a subsampled checkpoint with the dense server gives a sequence 4x
     longer than training — nothing errors, the score just drops.

  2. **State conditioning** — when the request carries a ``"state"`` key (a
     14-element list of current joint positions), it is forward-normalized and
     prepended as action row 0 before the diffusion forward pass.  The predicted
     state row is stripped from the output so the caller always receives exactly
     ``action_chunk_size`` action steps.  Matches ``use_state=True`` in training.

  3. **RoboTwin concat_view prompt** — injects
     ``ADDITIONAL_VIEW_DESCRIPTIONS["robotwin"]`` into the JSON prompt so the model
     receives the same spatial grounding sentence it saw during training.  Omitting
     it is a silent divergence that degrades left/right placement accuracy.

Usage::

    CHECKPOINT_PATH=<...>/checkpoints/iter_000025000 \\
    WAN_VAE_PATH=<...>/Wan2.2_VAE.pth \\
    python cosmos_framework/scripts/action_policy_server_robotwin.py \\
      --experiment action_policy_robotwin_nano \\
      --action-chunk-size 64 \\
      --video-subsample-factor 4 \\
      --action-normalization quantile \\
      --action-stats-path cosmos_framework/data/generator/action/normalizer_stats/robotwin_stats.json

See ``examples/robotwin/README.md`` for the full configuration reference.
"""

from __future__ import annotations

import base64
import io
import json
import time
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np
import torch
import tyro
from PIL import Image

from cosmos_framework.data.generator.action.utils.action_processing import (
    ActionProcessingRecord,
    make_batched_action_processing_fields,
)
from cosmos_framework.data.generator.action.utils.domain_utils import get_domain_id
from cosmos_framework.data.generator.action.utils.transforms import (
    build_sequence_plan_from_mode,
    find_closest_target_size,
    reflection_pad_to_target,
    remove_reflection_padding,
)
from cosmos_framework.data.generator.action.utils.viewpoint_utils import ADDITIONAL_VIEW_DESCRIPTIONS
from cosmos_framework.inference.args import OmniSetupOverrides
from cosmos_framework.scripts.action_policy_server_libero import (
    ActionModelService,
    ActionServerArgs,
    _ActionHandler,
    _LIBERO_JSON_VIEWPOINT,
    _augment_prompt_with_metadata,
    _decode_base64_png_to_rgb_uint8,
    _save_policy_request_dump,
    _video_tensor_to_pil_images,
)
from cosmos_framework.scripts.action_policy_server_utils import get_local_ip
from cosmos_framework.utils import log
from cosmos_framework.utils.generator.data_utils import get_vision_data_resolution


# ---------------------------------------------------------------------------
# Args — extend with video_subsample_factor
# ---------------------------------------------------------------------------


class RoboTwinServerArgs(ActionServerArgs):
    """CLI arguments for the RoboTwin server.  Inherits all LIBERO server args
    and adds ``video_subsample_factor`` for subsampled-video checkpoints."""

    video_subsample_factor: int = 4
    """Observation frames per action step (matches ``dataset.video_subsample_factor``).
    The server sends ``action_chunk_size // factor + 1`` frames and applies
    ``encode_exact_durations=[that count]`` before constructing the model.
    Must match the training recipe: factor 4 → 17 frames for chunk 64."""

    def build_setup_overrides(self) -> OmniSetupOverrides:
        overrides = super().build_setup_overrides()
        if self.video_subsample_factor > 1 and self.action_chunk_size is not None:
            factor = self.video_subsample_factor
            chunk = int(self.action_chunk_size)
            if chunk % factor:
                raise ValueError(
                    f"action_chunk_size ({chunk}) must be divisible by "
                    f"video_subsample_factor ({factor})"
                )
            durations = [chunk // factor + 1]
            overrides.experiment_overrides = [
                *overrides.experiment_overrides,
                f"model.config.tokenizer.encode_exact_durations={durations}",
            ]
            log.info(
                f"[action-server] video_subsample_factor={factor}: "
                f"sending {durations[0]} observation frames, "
                f"encode_exact_durations={durations}"
            )
        elif self.video_subsample_factor > 1 and self.action_chunk_size is None:
            log.warning(
                f"[action-server] video_subsample_factor={self.video_subsample_factor} but "
                "--action-chunk-size not set; encode_exact_durations not applied. "
                "Pass --action-chunk-size or add encode_exact_durations to experiment_overrides."
            )
        return overrides


# ---------------------------------------------------------------------------
# Service — RoboTwin-specific overrides
# ---------------------------------------------------------------------------


class RoboTwinModelService(ActionModelService):
    """ActionModelService extended for RoboTwin 2.0 checkpoints."""

    def __init__(self, args: RoboTwinServerArgs) -> None:  # type: ignore[override]
        super().__init__(args)
        self._video_subsample_factor: int = max(1, int(args.video_subsample_factor))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _obs_frames(self) -> int:
        """Observation frames per chunk: ``chunk // video_subsample_factor + 1``."""
        return self.cfg.action_chunk_size // self._video_subsample_factor + 1

    def _normalize_action(self, action: torch.Tensor) -> torch.Tensor:
        """Forward-normalize raw joint values into the model's action space.

        Used to normalize the state row before prepending it to the action tensor.
        Only correct when the state shares the action's channel layout (RoboTwin's
        14-D dual-arm joints do).  Inverse of ``_denormalize_action``.
        """
        if self.action_normalization == "meanstd":
            if self.action_mean is None or self.action_std is None:
                return action
            d = self.action_mean.shape[0]
            mean = self.action_mean.to(action.device)
            std = self.action_std.to(action.device)
            return (action[..., :d] - mean) / std

        if self.action_min is None or self.action_range is None:
            return action
        d = self.action_min.shape[0]
        amin = self.action_min.to(action.device)
        arange = self.action_range.to(action.device)
        return (action[..., :d] - amin) / arange * 2.0 - 1.0

    def _build_json_prompt(
        self,
        prompt: str,
        *,
        video: torch.Tensor,
        image_size: torch.Tensor,
        domain_name: str = "",
    ) -> str:
        """JSON prompt builder with RoboTwin ``additional_view_description`` injection."""
        data_dict: dict[str, Any] = {
            "ai_caption": prompt,
            "viewpoint": _LIBERO_JSON_VIEWPOINT,
            "video": video,
            "image_size": image_size,
            "conditioning_fps": torch.tensor(self.cfg.fps, dtype=torch.long),
            "mode": "wam",
            "action": torch.zeros(
                (self.cfg.action_chunk_size, self.cfg.max_action_dim), dtype=torch.float32
            ),
            "idle_frames": torch.tensor(0, dtype=torch.long),
        }
        view_desc = ADDITIONAL_VIEW_DESCRIPTIONS.get(domain_name)
        if view_desc is not None:
            data_dict["additional_view_description"] = view_desc
        formatted = self._prompt_json_formatter(data_dict)["ai_caption"]
        return json.dumps(formatted) if isinstance(formatted, dict) else str(formatted)

    # ------------------------------------------------------------------
    # Predict path
    # ------------------------------------------------------------------

    def _prep_policy_item(self, req: dict[str, Any]) -> dict[str, Any]:
        """Prep with subsampled-video frame count and state-aware sequence plan."""
        image_b64 = req.get("image")
        if not isinstance(image_b64, str):
            raise ValueError("'image' must be a base64 string")
        prompt = req.get("prompt")
        if not isinstance(prompt, str):
            raise ValueError("'prompt' must be a string")
        domain_name = req.get("domain_name")
        if not isinstance(domain_name, str):
            raise ValueError("'domain_name' must be a string")
        image_size = req.get("image_size")
        if not isinstance(image_size, int) or image_size <= 0:
            raise ValueError("'image_size' must be a positive integer")

        img_chw_uint8 = _decode_base64_png_to_rgb_uint8(image_b64)
        img_h, img_w = img_chw_uint8.shape[-2:]
        if img_h != image_size:
            scale = image_size / img_h
            new_w = int(round(img_w * scale))
            hwc = img_chw_uint8.permute(1, 2, 0).cpu().numpy()
            resized = Image.fromarray(hwc).resize((new_w, image_size), resample=Image.Resampling.BILINEAR)
            arr = np.asarray(resized, dtype=np.uint8).copy()
            img_chw_uint8 = torch.from_numpy(arr).permute(2, 0, 1).contiguous()

        t_frames = self._obs_frames()
        _, final_h, final_w = img_chw_uint8.shape
        video_c_t_h_w_uint8 = img_chw_uint8.unsqueeze(1).repeat(1, t_frames, 1, 1)
        resolution = get_vision_data_resolution((final_h, final_w))
        target_w, target_h = find_closest_target_size(final_h, final_w, resolution)
        pad_dict: dict[str, Any] = {"video": video_c_t_h_w_uint8}
        reflection_pad_to_target(pad_dict, ["video"], True, target_w, target_h)

        use_state = req.get("state") is not None
        n_action_rows = self.cfg.action_chunk_size + (1 if use_state else 0)
        sequence_plan = build_sequence_plan_from_mode(
            mode="wam",
            video_length=t_frames,
            action_length=n_action_rows,
            has_text=True,
        )

        if self._prompt_json_formatter is not None:
            augmented_prompt = self._build_json_prompt(
                prompt,
                video=pad_dict["video"],
                image_size=pad_dict["image_size"],
                domain_name=domain_name,
            )
        else:
            augmented_prompt = _augment_prompt_with_metadata(
                prompt,
                t_frames=t_frames,
                fps=self.cfg.fps,
                height=final_h,
                width=final_w,
                append_duration_fps=self.append_duration_fps,
                append_resolution_info=self.append_resolution_info,
            )

        return {
            "img_chw_uint8": img_chw_uint8,
            "video_padded": pad_dict["video"],
            "padded_image_size": pad_dict["image_size"],
            "augmented_prompt": augmented_prompt,
            "sequence_plan": sequence_plan,
            "domain_name": domain_name,
            "image_size": image_size,
            "use_state": use_state,
        }

    def predict_policy_batch(self, reqs: list[dict[str, Any]]) -> dict[str, Any]:
        """Batched inference.  State conditioning is single-item only — reject here."""
        if any(r.get("state") is not None for r in reqs):
            raise ValueError(
                "batched inference does not support state conditioning; "
                "use the single-observation /predict endpoint for state-conditioned checkpoints"
            )
        return super().predict_policy_batch(reqs)

    def predict_policy(self, req: dict[str, Any]) -> dict[str, Any]:
        """Single-step policy inference with state conditioning and subsampled video."""
        t0 = time.monotonic()

        injected_id = req.get("request_id", None)
        if isinstance(injected_id, int) and injected_id > 0:
            request_id = int(injected_id)
        else:
            with self._req_id_lock:
                self._req_id += 1
                request_id = int(self._req_id)

        t_decode0 = time.monotonic()
        prep = self._prep_policy_item(req)
        t_decode1 = time.monotonic()

        img_chw_uint8 = prep["img_chw_uint8"]
        video_padded = prep["video_padded"]
        padded_image_size = prep["padded_image_size"]
        augmented_prompt = prep["augmented_prompt"]
        sequence_plan = prep["sequence_plan"]
        domain_name = prep["domain_name"]
        image_size = prep["image_size"]
        use_state = prep["use_state"]

        # Build the action tensor.  With state conditioning prepend the current
        # joints (normalized) as row 0; the model returns chunk+1 rows, and we
        # strip row 0 from the output so the caller always gets chunk rows.
        n_rows = self.cfg.action_chunk_size + (1 if use_state else 0)
        action_t_d = torch.zeros((n_rows, self.cfg.max_action_dim), dtype=torch.float32)
        if use_state:
            state_raw = req["state"]
            s = torch.as_tensor(state_raw, dtype=torch.float32).flatten()[: self.raw_action_dim]
            action_t_d[0, : s.numel()] = self._normalize_action(s)

        input_video_key = self._input_video_key()
        batch: dict[str, Any] = {
            input_video_key: [[video_padded]],
            **make_batched_action_processing_fields(
                ActionProcessingRecord(raw_action_dim=self.raw_action_dim, action_normalizer=None),
                batch_size=1,
            ),
            "action": [[action_t_d]],
            "mode": ["wam"],
            "ai_caption": [augmented_prompt],
            "prompt": [augmented_prompt],
            "conditioning_fps": [torch.tensor(self.cfg.fps, dtype=torch.long)],
            "image_size": padded_image_size.unsqueeze(0).to(device="cuda"),
            "domain_id": [torch.tensor(get_domain_id(domain_name), dtype=torch.long)],
            "sequence_plan": [sequence_plan],
        }

        if getattr(self.model, "training", False):
            log.warning(f"[action-server] request_id={request_id} WARNING: model.training=True")

        log.info(
            f"[action-server] request_id={request_id} mode=policy "
            f"prompt={augmented_prompt!r} domain_name={domain_name!r} image_size={image_size} "
            f"img={tuple(img_chw_uint8.shape)} steps={self.cfg.num_steps} guidance={self.cfg.guidance} "
            f"use_state={use_state}"
        )

        t_inf0 = time.monotonic()
        with self._lock:
            with torch.inference_mode():
                samples = self.model.generate_samples_from_batch(
                    batch,
                    guidance=self.cfg.guidance,
                    seed=[self.cfg.seed],
                    num_steps=self.cfg.num_steps,
                    has_negative_prompt=False,
                )
                pred_action = samples["action"][0]
                pred_video_c_t_h_w = self.model.decode(samples["vision"][0]).squeeze(0)
                pred_video_c_t_h_w = remove_reflection_padding(pred_video_c_t_h_w, padded_image_size)
        t_inf1 = time.monotonic()

        pred_action = pred_action.float().squeeze(0)
        if use_state:
            pred_action = pred_action[1:]  # drop the state row
        pred_action = self._denormalize_action(pred_action)
        pred_action_list = pred_action.detach().cpu().numpy().tolist()

        pred_video_frames = _video_tensor_to_pil_images(pred_video_c_t_h_w)
        pred_video_b64: list[str] = []
        for frame in pred_video_frames:
            buf = io.BytesIO()
            frame.save(buf, format="PNG")
            pred_video_b64.append(base64.b64encode(buf.getvalue()).decode("ascii"))

        if self._should_dump(request_id):
            dump_dir = self.cfg.dump_dir
            assert dump_dir is not None
            dump_root = Path(dump_dir)
            dump_root.mkdir(parents=True, exist_ok=True)
            try:
                _save_policy_request_dump(
                    dump_root=dump_root,
                    request_id=request_id,
                    request_json=req,
                    obs_chw_uint8=img_chw_uint8,
                    pred_action=pred_action_list,
                    pred_video_c_t_h_w=pred_video_c_t_h_w,
                    fps=int(self.cfg.fps),
                )
            except Exception as e:
                log.error(f"[action-server] dump failed for request_id={request_id}: {e}")

        dt_total_ms = (time.monotonic() - t0) * 1000.0
        dt_decode_ms = (t_decode1 - t_decode0) * 1000.0
        dt_inf_ms = (t_inf1 - t_inf0) * 1000.0
        log.info(
            f"[action-server] request_id={request_id} done action_steps={len(pred_action_list)} "
            f"video_frames={len(pred_video_b64)} "
            f"ms_total={dt_total_ms:.1f} ms_decode={dt_decode_ms:.1f} ms_infer={dt_inf_ms:.1f}"
        )
        return {"action": pred_action_list, "video": pred_video_b64}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def serve(args: RoboTwinServerArgs) -> None:
    if args.dump_dir is not None:
        Path(args.dump_dir).resolve().mkdir(parents=True, exist_ok=True)
        log.info(f"[action-server] dump_root={args.dump_dir} dump_every={args.dump_every}")

    service = RoboTwinModelService(args)

    local_ip = get_local_ip()
    log.info(
        f"[action-server] starting host={args.host} port={int(args.port)} "
        f"experiment_name={service.cfg.experiment_name!r} "
        f"steps={service.cfg.num_steps} guidance={service.cfg.guidance} fps={service.cfg.fps} "
        f"action_chunk_size={service.cfg.action_chunk_size} "
        f"video_subsample_factor={service._video_subsample_factor} "
        f"obs_frames={service._obs_frames()} "
        f"max_action_dim={service.cfg.max_action_dim}"
    )
    log.info(f"[action-server] Server accessible at: http://{local_ip}:{int(args.port)}/")
    log.info("[action-server] Endpoints:")
    log.info("  - GET  /            : Health check")
    log.info("  - GET  /info        : Model info")
    log.info("  - POST /predict     : Policy inference (supports 'state' key for state conditioning)")
    log.info("  - POST /predict_batch: Batched inference (no state conditioning)")

    httpd: ThreadingHTTPServer = ThreadingHTTPServer((args.host, int(args.port)), _ActionHandler)
    setattr(httpd, "service", service)
    httpd.serve_forever()


def main() -> None:
    args = tyro.cli(
        RoboTwinServerArgs,
        description=__doc__,
        config=(
            tyro.conf.OmitArgPrefixes,
            tyro.conf.CascadeSubcommandArgs,
            tyro.conf.OmitSubcommandPrefixes,
        ),
    )
    serve(args)


if __name__ == "__main__":
    main()
