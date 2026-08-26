# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""``action_policy_robotwin_nano`` — Cosmos3-Nano RoboTwin 2.0 action policy SFT recipe.

Mirrors the vision SFT stack (PackingDataLoader + RankPartitionedDataLoader),
but feeds the RoboTwin (dual-arm ALOHA) action dataset (14D absolute
``joint_pos``, quantile-normalized, state-conditioned) through
``ActionTransformPipeline``, and trains the generation + action heads from the
public ``nvidia/Cosmos3-Nano`` base.

Trains on subsampled video: ``video_subsample_factor=4`` gives 7.5 Hz video
against 30 Hz actions, so the observation stream is ``64 // 4 + 1 = 17`` frames
(5 VAE latents) while the action stream stays dense. The vision token budget is
identical to the previous vsub2 recipe (both yield 17 frames); only the action
sequence grows from 33 to 65 tokens (chunk + state row). ``encode_exact_durations``
is pinned to [17] to match, and the inference server must be told the same factor
(see the serving config).

Two shipped recipes (each has its own TOML + launch script):
  - ``_clean_plus_random`` whole scene (all 27,500 episodes), 8n, GBS 2048, 25k iters
  - ``_clean``             clean subset (~9%, 2,500 episodes), 2n, GBS 512, 10k iters

Structurally identical to ``action_policy_droid_nano`` (joint_pos +
action_normalization=None + concat_view), swapping the DROID dataset for the
RoboTwin one; the checkpoint/optimizer keys are embodiment-agnostic.

Usage (single 8-GPU node — see launch scripts for the reference HSDP shape)::

    ROBOTWIN_ROOT=/path/to/lerobot/robotwin_unified \\
    BASE_CHECKPOINT_PATH=<Cosmos3-Nano DCP dir> \\
    WAN_VAE_PATH=<Wan2.2_VAE.pth> \\
    bash examples/launch_sft_action_policy_robotwin_nano_clean_plus_random.sh
"""

import copy

from hydra.core.config_store import ConfigStore

from cosmos_framework.configs.base.experiment.sft.models.nano_model_config import NANO_MODEL_CONFIG
from cosmos_framework.data.generator.action.datasets.action_sft_dataset import get_action_robotwin_sft_dataset
from cosmos_framework.data.generator.joint_dataloader import (
    PackingDataLoader,
    RankPartitionedDataLoader,
)
from cosmos_framework.utils.lazy_config import LazyCall as L
from cosmos_framework.utils.lazy_config import LazyDict

cs = ConfigStore.instance()


action_policy_robotwin_nano = LazyDict(
    dict(
        defaults=[
            {"override /model": "mot_fsdp"},
            {"override /data_train": None},
            {"override /data_val": None},
            # FusedAdam with fp32 master_weights + eps 1e-8 (bf16 params + eps 1e-6
            # diverged on the action loss).
            {"override /optimizer": "fusedadamw"},
            {"override /scheduler": "lambdalinear"},  # linear LR decay
            {"override /checkpoint": "s3"},
            {
                "override /callbacks": [
                    "basic",
                    "optimization",
                    "job_monitor",
                ]
            },
            {"override /ema": "power"},
            {"override /tokenizer": "wan2pt2_tokenizer"},
            {"override /sound_tokenizer": None},
            {"override /vlm_config": None},
            {"override /ckpt_type": "dcp"},
            "_self_",
        ],
        job=dict(
            project="cosmos3",
            group="action_sft",
            name="action_policy_robotwin_nano",
            wandb_mode="disabled",
        ),
        model=dict(
            config=copy.deepcopy(NANO_MODEL_CONFIG),  # action_gen=True, max_action_dim=64
        ),
        optimizer=dict(
            betas=[0.9, 0.99],
            eps=1.0e-08,
            fused=True,  # popped by build_optimizer for FusedAdam (fused by construction)
            # Train the generation + action heads.
            keys_to_select=[
                "moe_gen",
                "time_embedder",
                "vae2llm",
                "llm2vae",
                "action2llm",
                "llm2action",
                "action_modality_embed",
            ],
            lr=2.0e-04,  # for the 8192 global batch
            lr_multipliers={
                "action2llm": 5.0,
                "llm2action": 5.0,
                "action_modality_embed": 5.0,
            },
            optimizer_type="FusedAdam",
            weight_decay=0.05,
        ),
        scheduler=dict(
            lr_scheduler_type="LambdaLinear",
            cycle_lengths=[100],  # smoke: 100 iters (real run sets via TOML)
            # Aligned with the LIBERO recipe and with what the reference run
            # actually used; the old [0.4]/[0.0]/[0.0] was a warm-start artifact.
            f_max=[1.0],
            f_min=[0.1],  # floor at 10% of lr rather than annealing to 0
            f_start=[1.0e-06],
            verbosity_interval=0,
            warm_up_steps=[0],
        ),
        trainer=dict(
            distributed_parallelism="fsdp",
            grad_accum_iter=1,
            logging_iter=1,
            max_iter=100,  # smoke
            max_val_iter=None,
            run_validation=False,
            run_validation_on_start=False,
            save_zero_checkpoint=False,
            seed=42,
            timeout_period=999999999,
            validation_iter=100,
            compile_config=dict(recompile_limit=8, use_duck_shape=False),
            cudnn=dict(benchmark=True, deterministic=False),
            ddp=dict(broadcast_buffers=True, find_unused_parameters=False, static_graph=True),
            grad_scaler_args=dict(enabled=False),
            callbacks=dict(
                dataloader_speed=dict(every_n=100, save_s3=False, step_size=1),
                device_monitor=dict(
                    every_n=200, log_memory_detail=True, save_s3=False, step_size=1, upload_every_n_mul=5
                ),
                grad_clip=dict(clip_norm=1.0, force_finite=True),
                heart_beat=dict(every_n=200, save_s3=False, step_size=1, update_interval_in_minute=20),
                iter_speed=dict(every_n=1, hit_thres=50, save_s3=False, save_s3_every_log_n=500),
                low_precision=dict(update_iter=1),
                manual_gc=dict(every_n=5, gc_level=1, warm_up=1),
                param_count=dict(save_s3=False),
                skip_nan_step=dict(max_consecutive_nan=100),
                training_stats=dict(log_freq=100),
            ),
        ),
        checkpoint=dict(
            broadcast_via_filesystem=False,
            dcp_async_mode_enabled=False,
            enable_gcs_patch_in_boto3=True,
            keys_not_to_resume=[],
            # Skip net_ema. (EMA warm-starts from net, see dcp.py) and the action
            # heads, so they init fresh from the base (the base has no RoboTwin-trained
            # action heads).
            keys_to_skip_loading=[
                "net_ema.",
                "action2llm",
                "llm2action",
                "action_modality_embed",
                "action_pos_embed",
            ],
            load_ema_to_reg=False,
            load_path="???",  # Cosmos3-Nano DCP dir; supply via TOML/env
            load_training_state=False,
            only_load_scheduler_state=False,
            save_iter=100,
            strict_resume=False,  # base init: tolerate key set differences
            verbose=True,
            hf_export=dict(
                enabled=False,
                export_every_n=1,
                hf_repo_id=None,
                upload_to_object_store=dict(bucket="", credentials="", enabled=False),
            ),
            jit=dict(device="cuda", dtype="bfloat16", enabled=False, input_shape=None, strict=True),
            load_from_object_store=dict(bucket="", credentials="", enabled=False),
            save_to_object_store=dict(bucket="", credentials="", enabled=False),
        ),
        dataloader_train=L(PackingDataLoader)(
            audio_sample_rate=48000,
            dataset_name="action_robotwin",
            max_samples_per_batch=64,  # per rank; the run TOML sets the value that hits the target global batch
            max_sequence_length=None,  # None disables token packing (TOML can't express null)
            patch_spatial=2,
            sound_latent_fps=0,
            tokenizer_spatial_compression_factor=16,
            tokenizer_temporal_compression_factor=4,
            dataloader=L(RankPartitionedDataLoader)(
                batch_size=1,
                in_order=False,
                num_workers=12,  # 3-cam AV1 480p decode is CPU-heavy; more workers keep the GPU fed
                persistent_workers=True,
                pin_memory=True,
                prefetch_factor=6,
                sampler=None,
                # Shuffling is handled by the dataset (iterable_shuffle=True below):
                # ActionIterableShuffleDataset streams rank x worker-sharded, episode-order-
                # shuffled, sequential-within-episode. The map-style dataset has no internal
                # shuffle, so a SequentialSampler would feed every rank the SAME consecutive
                # overlapping windows -> global batch ~1 episode -> unstable grad-norm; a plain
                # RandomSampler decorrelates but does random-access I/O -> slow + OOM. The
                # iterable gives decorrelation with sequential reads.
                datasets=dict(
                    robotwin=dict(
                        ratio=1,
                        dataset=L(get_action_robotwin_sft_dataset)(
                            root="${oc.env:ROBOTWIN_ROOT}",
                            fps=30.0,
                            chunk_length=64,
                            # Policy-only task mode ("wam": jointly denoise video + actions
                            # from the first clean frame). "joint" would randomly pick
                            # forward_dynamics/inverse_dynamics/wam per sample (multi-task),
                            # which dilutes each per-task loss by ~1/3.
                            mode="wam",
                            # Proprioceptive state conditioning ON: the RoboTwin loader prepends
                            # the current 14-D observation.state as action row 0 (DROID joint_pos
                            # contract), making the action tensor [chunk_length + 1, 14]. Because
                            # state shares the action's 14-D joint layout, the normalizer below
                            # normalizes the state row with the same per-channel stats as the action.
                            use_state=True,
                            # ColorJitter augmentation (brightness/contrast/saturation), applied
                            # consistently across all cameras and frames of each clip.
                            use_image_augmentation=True,
                            # Quantile (q01/q99) normalization from normalizer_stats/robotwin_stats.json,
                            # applied to the real action channels (state row 0 included) -> [-1, 1].
                            action_normalization="quantile",
                            viewpoint="concat_view",  # cam_high (top) + L/R wrist (bottom)
                            resolution="480",  # 640x480 data @ 480p
                            max_action_dim="${model.config.max_action_dim}",
                            cfg_dropout_rate=0.1,
                            tokenizer_config="${model.config.vlm_config.tokenizer}",
                            # Match DROID GA: format the action prompt as JSON via
                            # ActionPromptJsonFormatter instead of plain text.
                            format_prompt_as_json=True,
                            iterable_shuffle=True,  # rank x worker episode-shuffle stream
                            episode_shuffle_seed=42,
                            # 100% of episodes train. run_validation is off and
                            # dataloader_val is None, so a holdout would just be
                            # data withheld from training for nothing.
                            val_ratio=0.0,
                            # 7.5 Hz video against 30 Hz actions: the observation
                            # stream is chunk//4 + 1 = 17 frames while actions stay
                            # dense. encode_exact_durations below is pinned to match,
                            # and the serving config must carry the same factor --
                            # a checkpoint trained on 17 frames but served 33 does not
                            # error, it just scores far lower.
                            video_subsample_factor=4,
                            # Scene subset: "all" trains on the full 27,500-episode
                            # dataset. The clean recipe overrides this to "clean"
                            # (2,500 episodes) via the launch script's TAIL_OVERRIDES.
                            robotwin_scene="all",
                        ),
                    ),
                ),
            ),
        ),
        dataloader_val=None,
        upload_reproducible_setup=False,
    ),
    flags={"allow_objects": True},
)


# chunk_length=64 with video_subsample_factor=4 -> 64//4 + 1 = 17 observation
# frames; pin the VAE encode duration to match. Set post-construction so it lands
# on the deep-copied NANO_MODEL_CONFIG.tokenizer. Same count as the previous
# vsub2 recipe (32//2 + 1 = 17); the action sequence grows from 33 to 65 tokens.
action_policy_robotwin_nano["model"]["config"]["tokenizer"]["encode_exact_durations"] = [17]


# Uncap the packed-sequence length. The NANO default (45056) caps the packed sequence,
# truncating long windows to ~1/4 of their natural length; -1 (uncapped) processes
# the full vision sequence per step. Does not change the per-token loss; widens the
# effective vision context per step.
action_policy_robotwin_nano["model"]["config"]["max_num_tokens_after_packing"] = -1


# Weight the vision flow-matching loss 10x in the total loss (the NANO default is 1.0).
# loss_scale multiplies only the vision term, balancing it against the action loss
# (action_loss_weight=10) so both heads train at comparable gradient magnitude.
action_policy_robotwin_nano["model"]["config"]["rectified_flow_training_config"]["loss_scale"] = 10.0


for _item in [action_policy_robotwin_nano]:
    _name = [k for k, v in globals().items() if v is _item][0]
    cs.store(group="experiment", package="_global_", name=_name, node=_item)
