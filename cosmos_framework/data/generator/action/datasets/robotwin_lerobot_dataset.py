# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""RoboTwin 2.0 (dual-arm ALOHA) LeRobot dataset — absolute ``joint_pos`` action policy.

Structurally the DROID ``joint_pos`` recipe (a ``BaseActionLeRobotDataset``
subclass: lazy per-shard ``LeRobotDataset`` readers, ``delta_timestamps`` frame
windowing, ``concat_view`` video), but simpler on the action side — RoboTwin
stores a single 14-D ``action`` feature that is *already* the absolute dual-arm
joint command, so there is NO pose/rot6d conversion and NO separate gripper
feature to splice.

Action layout (14D, from ``meta/info.json`` ``names.motors``)::

    [ left_joint_0..5, left_gripper, right_joint_0..5, right_gripper ]
      \\____ 6 joints ____/  \\__1__/  \\____ 6 joints ____/  \\__1__/

i.e. per arm = 6 joints + 1 gripper. The chunk is read directly off the stored
feature -> action tensor ``[chunk_length, 14]``.

Cameras (3x 480x640 video): ``cam_high`` (top-mounted third-person),
``cam_left_wrist``, ``cam_right_wrist``. ``concat_view`` tiles them as::

    ┌───────────────────────┐
    │       cam_high        │   (H, W)   full width, third-person
    ├───────────┬───────────┤
    │ left_wrist│right_wrist│   (H/2, W/2) each -> (H/2, W)
    └───────────┴───────────┘

The composite is ``[T, C, 3H/2, W]``; ``ActionTransformPipeline`` handles the
final spatial resize/pad to the requested ``resolution``.

Normalization defaults to ``None`` (raw absolute joints, like DROID
``joint_pos``). When a method is requested the base loads the bundled
``normalizer_stats/robotwin_stats.json``; with ``use_state=True`` the state row
is prepended *before* normalization, so it is normalized with the same
per-channel stats as the action it shares a layout with.

RoboTwin 2.0 scene layout
--------------------------
The canonical release is 50 tasks in alphabetical order, each contributing a
contiguous block of 550 episodes: the first 50 are the non-randomized ("clean")
scenes, the remaining 500 are domain-randomized ("random"). Scene membership is
pure index arithmetic on ``episode_index`` — no separate directory, no copy::

    clean   <=>  episode_index % 550 <  50        (2,500 total)
    random  <=>  episode_index % 550 >= 50        (25,000 total)

``robotwin_scene="all"`` (default) is every episode; ``"clean"`` or ``"random"``
filters positionally via ``_filter_valid_episodes``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as T

from cosmos_framework.utils import log
from cosmos_framework.data.generator.action.datasets.cosmos3_action_lerobot import (
    ActionNormalization,
    ActionSpec,
    BaseActionLeRobotDataset,
    Gripper,
    Joint,
    build_action_spec,
)
from cosmos_framework.data.generator.action.utils.pose_utils import PoseConvention
from cosmos_framework.data.generator.action.utils.viewpoint_utils import (
    ADDITIONAL_VIEW_DESCRIPTIONS,
    Viewpoint,
)

# Bundled stats live one directory up, alongside every other action dataset's
# (the base convention would look in ``datasets/normalizers``, which does not
# exist in this tree).
_NORMALIZER_PATH = Path(__file__).parent.parent / "normalizer_stats" / "robotwin_stats.json"

# LeRobot feature keys (see meta/info.json).
_ACTION_FEATURE = "action"
_STATE_FEATURE = "observation.state"
_CAM_HIGH = "observation.images.cam_high"
_CAM_LEFT_WRIST = "observation.images.cam_left_wrist"
_CAM_RIGHT_WRIST = "observation.images.cam_right_wrist"

_ACTION_DIM = 14  # dual-arm ALOHA: (6 joints + 1 gripper) x 2

# ---- RoboTwin 2.0 episode layout ---------------------------------------------
# 50 tasks x 550 episodes each = 27,500 total.
# First 50 of each block = clean (non-randomized), rest = random.
_ROBOTWIN_CLEAN_PER_TASK = 50
_ROBOTWIN_RANDOM_PER_TASK = 500
_ROBOTWIN_EPISODES_PER_TASK = _ROBOTWIN_CLEAN_PER_TASK + _ROBOTWIN_RANDOM_PER_TASK  # 550
_ROBOTWIN_SCENES = ("clean", "random")


class RoboTwinLeRobotDataset(BaseActionLeRobotDataset):
    """RoboTwin dual-arm ALOHA action-policy dataset (14D absolute ``joint_pos``).

    Reads the raw 14-D ``action`` feature directly (no pose conversion),
    windows by ``delta_timestamps`` at native fps, and composes a three-camera
    ``concat_view`` (``cam_high`` on top; ``cam_left_wrist`` + ``cam_right_wrist``
    tiled on the bottom row).
    """

    EMBODIMENT_TYPE: str = "robotwin"

    def _normalizer_path(self) -> Path:
        """Bundled RoboTwin q01/q99 stats (see ``examples/robotwin/tools/compute_robotwin_action_stats.py``)."""
        return _NORMALIZER_PATH

    def __init__(
        self,
        root: str,
        fps: float = 30.0,
        chunk_length: int = 64,
        split_seed: int = 42,
        split_val_ratio: float = 0.01,
        split: str = "train",
        mode: str = "wam",
        # Actions are ABSOLUTE dual-arm joint_pos read raw (no delta/pose op), so
        # there is no framewise convention. None => honest naming and skips the
        # unused idle-frame captioning (``_compute_idle_frames`` only supports
        # ``backward_framewise``). The value never transforms the action.
        pose_convention: PoseConvention | None = None,
        action_normalization: ActionNormalization | None = None,
        tolerance_s: float = 2e-4,
        viewpoint: Viewpoint = "concat_view",
        use_state: bool = False,
        # ColorJitter image augmentation (brightness/contrast/saturation), applied
        # consistently across all cameras and all frames of the clip. Off by default;
        # the reference runs enable it to regularize against the clean-domain training
        # distribution (``use_image_augmentation=True``).
        use_image_augmentation: bool = False,
        # Action steps per observation frame. 1 = dense video (one frame per action
        # step, 30 Hz here). 4 = the reference recipe: 7.5 Hz video against 30 Hz
        # actions, so the observation stream is chunk/4 + 1 = 17 frames while the
        # action stream stays dense. Inference MUST be told the same factor -- see
        # ``ActionServerArgs.video_subsample_factor``.
        video_subsample_factor: int = 4,
        # Scene subset of the RoboTwin 2.0 release. "all" (default) is every
        # episode (27,500); "clean" keeps only the 50 non-randomized episodes of
        # each task block (2,500 total); "random" keeps only the 25,000
        # domain-randomized ones. Selection is purely positional on episode_index.
        robotwin_scene: str = "all",
        enable_fast_init: bool = False,
    ) -> None:
        if viewpoint not in ("concat_view", "third_person_view"):
            raise ValueError(
                f"RoboTwinLeRobotDataset supports viewpoint in ('concat_view', 'third_person_view'), got {viewpoint!r}."
            )
        if robotwin_scene not in (*_ROBOTWIN_SCENES, "all"):
            raise ValueError(
                f"robotwin_scene must be one of {(*_ROBOTWIN_SCENES, 'all')}, got {robotwin_scene!r}"
            )
        self._robotwin_scene = robotwin_scene
        super().__init__(
            fps=fps,
            chunk_length=chunk_length,
            split_seed=split_seed,
            split_val_ratio=split_val_ratio,
            split=split,
            mode=mode,
            embodiment_type=self.EMBODIMENT_TYPE,
            viewpoint=viewpoint,
            pose_convention=pose_convention,
            rotation_format=None,  # joint-space: no rotation encoding
            action_normalization=action_normalization,
            tolerance_s=tolerance_s,
            enable_fast_init=enable_fast_init,
        )

        self._use_state = use_state
        # Optional image augmentation. Built lazily on first use so worker forks
        # share nothing (avoids multiprocessing seed conflicts with T.RandomXxx).
        self._use_image_augmentation = use_image_augmentation
        self._image_augmentor: T.Compose | None = None

        # Single local LeRobot root (v3.0). Registered eagerly (metadata-only);
        # heavy per-shard video readers stay lazy behind the base LRU cache, just
        # like DROIDLeRobotDataset.
        self._all_shard_roots = [root]

        if video_subsample_factor < 1:
            raise ValueError(f"video_subsample_factor must be >= 1, got {video_subsample_factor}")
        if chunk_length % video_subsample_factor:
            raise ValueError(
                f"chunk_length ({chunk_length}) must be divisible by video_subsample_factor ({video_subsample_factor})"
            )
        self._video_subsample_factor = video_subsample_factor

        # ``observation`` frames span [0, chunk_length] (chunk_length + 1 frames);
        # ``action`` spans [0, chunk_length) (chunk_length frames). Matches the
        # DROID joint_pos windowing and the experiment's encode_exact_durations.
        #
        # With video_subsample_factor=N the video is sampled every N action steps
        # (factor 2 -> 15 Hz video from 30 Hz data), giving chunk/N + 1 observation
        # frames against an unchanged dense action stream. ``encode_exact_durations``
        # must be set to that frame count, and so must the inference server.
        n_video_frames = self._chunk_length // video_subsample_factor + 1
        observation_ts = [i * self._dt * video_subsample_factor for i in range(n_video_frames)]
        action_ts = [i * self._dt for i in range(0, self._chunk_length)]
        self._delta_timestamps: dict[str, list[float]] = {_ACTION_FEATURE: action_ts}
        if self._use_state:
            # Opt-in proprioceptive conditioning: window the 14-D observation.state
            # at the SAME timestamps as the action chunk. The frame-0 (t=0, current)
            # state is prepended as row 0 of the action tensor in __getitem__ — the
            # DROID joint_pos ``use_state`` contract.
            self._delta_timestamps[_STATE_FEATURE] = action_ts
        # Always fetch cam_high (third-person / top of concat); add the two wrist
        # cameras only for the concat_view layout.
        self._delta_timestamps[_CAM_HIGH] = observation_ts
        if self._viewpoint == "concat_view":
            self._delta_timestamps[_CAM_LEFT_WRIST] = observation_ts
            self._delta_timestamps[_CAM_RIGHT_WRIST] = observation_ts

        self._register_sources()

    # ---- scene filtering ---------------------------------------------------

    def _filter_valid_episodes(self, meta, episode_ids):
        """Restrict the episode set to ``robotwin_scene`` (the base calls this from
        ``_append_index_records`` when the attribute exists, before span building).

        Selection is positional on ``episode_index`` -- see the ``_ROBOTWIN_*``
        constants. Input order is preserved (``split_episode_ids`` hands us a
        seeded-shuffled list and downstream span order follows it).
        """
        if self._robotwin_scene == "all":
            return episode_ids

        total = int(meta.total_episodes)
        if total % _ROBOTWIN_EPISODES_PER_TASK != 0:
            raise ValueError(
                f"robotwin_scene={self._robotwin_scene!r} assumes the canonical "
                f"{_ROBOTWIN_EPISODES_PER_TASK}-episodes-per-task layout "
                f"({_ROBOTWIN_CLEAN_PER_TASK} clean + {_ROBOTWIN_RANDOM_PER_TASK} random), "
                f"but this dataset reports total_episodes={total}, which is not a multiple "
                f"of {_ROBOTWIN_EPISODES_PER_TASK}. Refusing to guess the split."
            )

        want_clean = self._robotwin_scene == "clean"
        kept = [
            eid
            for eid in episode_ids
            if ((int(eid) % _ROBOTWIN_EPISODES_PER_TASK) < _ROBOTWIN_CLEAN_PER_TASK) == want_clean
        ]
        log.info(
            f"{self.__class__.__name__}: robotwin_scene={self._robotwin_scene!r} -> "
            f"{len(kept)}/{len(episode_ids)} episodes kept "
            f"({total // _ROBOTWIN_EPISODES_PER_TASK} task blocks)"
        )
        if not kept:
            raise ValueError(
                f"robotwin_scene={self._robotwin_scene!r} selected 0 of {len(episode_ids)} episodes."
            )
        return kept

    # ---- spec / dims -------------------------------------------------------

    @property
    def action_dim(self) -> int:
        return _ACTION_DIM

    def _build_action_spec(self) -> ActionSpec:
        """14D dual-arm ALOHA: ``[L joint(6), L gripper, R joint(6), R gripper]``."""
        return build_action_spec(
            Joint(n=6, label="joint", prefix="left"),
            Gripper(prefix="left"),
            Joint(n=6, label="joint", prefix="right"),
            Gripper(prefix="right"),
        )

    # ---- video -------------------------------------------------------------

    def _compose_concat_view(self, sample: dict[str, Any]) -> torch.Tensor:
        """Tile the three cameras into one frame.

        Layout (per frame): ``cam_high`` full-width on top; ``cam_left_wrist``
        (left) + ``cam_right_wrist`` (right), each downscaled 2x, on the bottom
        row -> ``[T, C, 3H/2, W]``.

        When ``use_image_augmentation=True``, applies ColorJitter consistently
        across all cameras and all frames of the clip (temporally + cross-view
        consistent) by concatenating on the time axis, augmenting jointly, then
        splitting back. Augmentor is built lazily on first use so worker forks do
        not share RNG state.
        """
        top = sample[_CAM_HIGH]  # [T,C,H,W]
        left = sample[_CAM_LEFT_WRIST]  # [T,C,H_l,W_l]
        right = sample[_CAM_RIGHT_WRIST]  # [T,C,H_r,W_r]

        if self._use_image_augmentation:
            if self._image_augmentor is None:
                _, _, h_a, w_a = top.shape
                self._image_augmentor = T.Compose(
                    [
                        T.Resize((h_a, w_a), antialias=True),
                        T.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5),
                    ]
                )
            n, m = top.shape[0], top.shape[0] + left.shape[0]
            combined = self._image_augmentor(torch.cat([top, left, right], dim=0))
            top, left, right = combined[:n], combined[n:m], combined[m:]

        _, _, h, w = top.shape
        half_h, half_w = h // 2, w // 2
        left = F.interpolate(left, size=(half_h, half_w), mode="bilinear", align_corners=False)  # [T,C,H/2,W/2]
        right = F.interpolate(right, size=(half_h, half_w), mode="bilinear", align_corners=False)  # [T,C,H/2,W/2]
        bottom = torch.cat([left, right], dim=-1)  # [T,C,H/2,W]
        return torch.cat([top, bottom], dim=-2)  # [T,C,3H/2,W]

    # ---- sample build ------------------------------------------------------

    def __getitem__(self, idx: int) -> dict[str, Any]:
        mode, _, _, sample = self._fetch_sample(idx)

        # RoboTwin has a single language task string per episode.
        ai_caption = sample["task"]

        if self._skip_video_loading:
            video = None
        elif self._viewpoint == "concat_view":
            video = self._compose_concat_view(sample)  # [T,C,3H/2,W]
        else:
            video = sample[_CAM_HIGH]  # [T,C,H,W]

        # Absolute dual-arm joint_pos: the stored 14-D action IS the chunk target.
        action = sample[_ACTION_FEATURE].float()  # [chunk_length, 14]

        if self._use_state:
            # DROID joint_pos ``use_state`` contract: prepend the current (t=0)
            # proprioceptive state as row 0 of the action tensor -> [chunk_length + 1, 14].
            # There is NO separate extras key — the model reads the state off the action
            # tensor's leading row. RoboTwin's observation.state shares the action's 14-D
            # layout ([L joint(6), L gripper, R joint(6), R gripper]), so the whole vector
            # is used as-is; unlike DROID it needs no joint/gripper splice (DROID stores
            # joint state and gripper state as two separate features).
            #
            # Prepended BEFORE _build_result, so the base normalizes the state row with
            # the same per-channel stats as the action — which is what the inference
            # server reproduces via ActionModelService._normalize_action.
            initial_state = sample[_STATE_FEATURE][0].float()  # [14] — state at t=0
            action = torch.cat([initial_state.unsqueeze(0), action], dim=0)  # [chunk_length + 1, 14]

        extras: dict[str, Any] = {}
        if self._viewpoint == "concat_view":
            # Shared with the inference server via ADDITIONAL_VIEW_DESCRIPTIONS so the
            # prompt cannot drift between training and eval.
            extras["additional_view_description"] = ADDITIONAL_VIEW_DESCRIPTIONS["robotwin"]

        return self._build_result(
            mode=mode,
            video=video,
            action=action,
            ai_caption=ai_caption,
            **extras,
        )
