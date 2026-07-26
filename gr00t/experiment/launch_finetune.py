# Launch finetuning for N1.6 on "single node".
# This script tries to provide a similar user experience as current OSS.

import json
import os
from pathlib import Path

import tyro

from gr00t.configs.base_config import get_default_config
from gr00t.configs.finetune_config import FinetuneConfig
from gr00t.experiment.experiment import run


# Make sure the user provided modality config is registered.
def load_modality_config(modality_config_path: str):
    import importlib
    import sys

    path = Path(modality_config_path)
    if path.exists() and path.suffix == ".py":
        sys.path.append(str(path.parent))
        importlib.import_module(path.stem)
        print(f"Loaded modality config: {path}")
    else:
        raise FileNotFoundError(f"Modality config path does not exist: {modality_config_path}")


if __name__ == "__main__":
    # Set LOGURU_LEVEL environment variable if not already set (default: INFO)
    if "LOGURU_LEVEL" not in os.environ:
        os.environ["LOGURU_LEVEL"] = "INFO"
    # Use tyro for clean CLI
    ft_config = tyro.cli(FinetuneConfig, description=__doc__)
    embodiment_tag = ft_config.embodiment_tag.value

    # all rank workers should register for the modality config
    if ft_config.modality_config_path is not None:
        load_modality_config(ft_config.modality_config_path)

    config = get_default_config().load_dict(
        {
            "data": {
                "download_cache": False,
                "datasets": [
                    {
                        "dataset_paths": [ft_config.dataset_path],
                        "mix_ratio": 1.0,
                        "embodiment_tag": embodiment_tag,
                    }
                ],
            }
        }
    )
    config.load_config_path = None

    # overwrite with finetune config supplied by the user
    config.model.tune_llm = ft_config.tune_llm
    config.model.tune_visual = ft_config.tune_visual
    config.model.tune_projector = ft_config.tune_projector
    config.model.tune_diffusion_model = ft_config.tune_diffusion_model
    config.model.state_dropout_prob = ft_config.state_dropout_prob
    config.model.random_rotation_angle = ft_config.random_rotation_angle
    config.model.color_jitter_params = ft_config.color_jitter_params
    if ft_config.extra_augmentation_config:
        config.model.extra_augmentation_config = json.loads(ft_config.extra_augmentation_config)
    else:
        config.model.extra_augmentation_config = None

    config.model.load_bf16 = False
    config.model.reproject_vision = False
    config.model.eagle_collator = True
    config.model.model_name = "nvidia/Eagle-Block2A-2B-v2"
    config.model.backbone_trainable_params_fp32 = True
    config.model.use_relative_action = True

    config.training.experiment_name = ft_config.experiment_name
    config.training.start_from_checkpoint = ft_config.base_model_path
    config.training.optim = "adamw_torch"
    config.training.global_batch_size = ft_config.global_batch_size
    config.training.dataloader_num_workers = ft_config.dataloader_num_workers
    config.training.learning_rate = ft_config.learning_rate
    config.training.max_grad_norm = ft_config.max_grad_norm
    config.training.gradient_accumulation_steps = ft_config.gradient_accumulation_steps
    config.training.output_dir = ft_config.output_dir
    config.training.save_steps = ft_config.save_steps
    config.training.save_total_limit = ft_config.save_total_limit
    config.training.num_gpus = ft_config.num_gpus
    config.training.use_wandb = ft_config.use_wandb
    config.training.max_steps = ft_config.max_steps
    config.training.weight_decay = ft_config.weight_decay
    config.training.warmup_ratio = ft_config.warmup_ratio
    config.training.wandb_project = ft_config.wandb_project

    config.data.shard_size = ft_config.shard_size
    config.data.episode_sampling_rate = ft_config.episode_sampling_rate
    config.data.num_shards_per_epoch = ft_config.num_shards_per_epoch

    config.training.skip_weight_loading = ft_config.skip_weight_loading

    # HAMLET configuration
    config.model.hamlet_mode = ft_config.hamlet_mode
    config.model.n_moment_tokens = ft_config.n_moment_tokens
    config.model.memory_window = ft_config.memory_window
    config.model.memory_num_layers = ft_config.memory_num_layers
    config.model.memory_arch = ft_config.memory_arch
    config.model.memory_hidden = ft_config.memory_hidden
    config.model.memory_state_dim = ft_config.memory_state_dim
    # Multi-scale memory (V1/V2)
    config.model.memory_scales = ft_config.memory_scales
    config.model.memory_scales_uniform = ft_config.memory_scales_uniform
    config.model.memory_scales_dog = ft_config.memory_scales_dog
    config.model.memory_comm = ft_config.memory_comm
    config.model.memory_aux_lambda = ft_config.memory_aux_lambda
    config.model.memory_aux_horizons = ft_config.memory_aux_horizons
    config.model.memory_aux_warmup_steps = ft_config.memory_aux_warmup_steps
    config.model.memory_aux_ema = ft_config.memory_aux_ema
    config.model.memory_aux_detach_moment = ft_config.memory_aux_detach_moment
    config.model.memory_stride = ft_config.memory_stride
    config.model.mem_cond_type = ft_config.mem_cond_type
    config.model.mem_fs_inject = ft_config.mem_fs_inject
    config.model.memory_type = ft_config.memory_type
    config.model.mem_film_layers = ft_config.mem_film_layers
    config.model.mem_source = ft_config.mem_source
    config.model.mem_framesamp_budget = ft_config.mem_framesamp_budget
    config.model.mem_framesamp_frames = ft_config.mem_framesamp_frames
    # TokenDrop-style diff keyframes: stamp the model config so the SAVED CHECKPOINT
    # defaults to the matching inference selector (mem_fs_select='diff').
    config.model.mem_fs_select = ft_config.mem_fs_select
    config.model.mem_fs_diff_stride = ft_config.mem_fs_diff_stride
    config.model.mem_fs_attn_layer = ft_config.mem_fs_attn_layer
    config.model.mem_fs_diff_share = ft_config.mem_fs_diff_share
    config.model.mem_fs_tail_share = ft_config.mem_fs_tail_share
    config.model.mem_fs_pos_rope = ft_config.mem_fs_pos_rope
    # VAR-pyramid multi-coarseness memory (mem_fs_select='var_pyramid'): stamp all knobs onto
    # the model config so the action head AND the saved checkpoint see them (getattr defaults
    # would otherwise silently train with RANDOM VAR weights).
    config.model.mem_varp_ckpt = ft_config.mem_varp_ckpt
    config.model.mem_varp_res = ft_config.mem_varp_res
    config.model.mem_varp_budget = ft_config.mem_varp_budget
    config.model.mem_varp_gate_hard = ft_config.mem_varp_gate_hard
    config.model.mem_varp_budget_lambda = ft_config.mem_varp_budget_lambda
    config.model.mem_varp_target_frac = ft_config.mem_varp_target_frac
    config.model.mem_varp_gist_scales = ft_config.mem_varp_gist_scales
    config.model.mem_varp_view = ft_config.mem_varp_view
    config.model.mem_image_side = ft_config.mem_image_side
    config.model.mem_window_mode = ft_config.mem_window_mode
    if (
        ft_config.hamlet_mode == "finetune"
        and ft_config.freeze_moment_tokens
        and not ft_config.load_moment_tokens_from
    ):
        print(
            "[HAMLET][WARN] freeze_moment_tokens=True but no --load-moment-tokens-from "
            "was given: randomly initialized moment tokens would stay frozen for the "
            "whole run. Load TCL-pretrained tokens or pass --no-freeze-moment-tokens."
        )
    config.model.freeze_moment_tokens = ft_config.freeze_moment_tokens
    config.model.tcl_tau = ft_config.tcl_tau
    config.model.tcl_no_projection_head = ft_config.tcl_no_projection_head
    # Trainability is now an explicit per-stage knob (Stage-1 sets 0, Stage-2 keeps 4),
    # not an inline mode override. See run_scripts/train_tcl_stage1.sh vs train_hamlet_stage2.sh.
    config.model.tune_top_llm_layers = ft_config.tune_top_llm_layers
    config.training.load_moment_tokens_from = ft_config.load_moment_tokens_from

    # HAMLET — override video delta_indices on the registered modality configs.
    if ft_config.hamlet_mode == "finetune" and ft_config.memory_window > 1:
        from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
        stride = ft_config.memory_stride
        K = ft_config.memory_window
        new_indices = [-(K - 1 - i) * stride for i in range(K)]
        # mem_source='framesamp': in ADDITION to the K-step memory-window frames, ask the
        # loader to append `mem_framesamp_frames` EPISODE-SPANNING linspace frames to the same
        # video stream (handled in extract_step_data via the `framesamp_frames` attribute on
        # the video ModalityConfig). Backward-compatible: when mem_source='moment' the
        # attribute stays 0 and the loader behaves byte-for-byte as before.
        # mem_cond_type='dual' ALSO needs the episode-spanning framesamp frames (its h_spatial
        # channel is built from their raw patch tokens), independent of mem_source.
        fs_frames = (
            int(ft_config.mem_framesamp_frames)
            if (ft_config.mem_source == "framesamp" or ft_config.mem_cond_type == "dual")
            else 0
        )
        for tag in MODALITY_CONFIGS:
            if "video" in MODALITY_CONFIGS[tag]:
                MODALITY_CONFIGS[tag]["video"].delta_indices = new_indices
                MODALITY_CONFIGS[tag]["video"].framesamp_frames = fs_frames
                MODALITY_CONFIGS[tag]["video"].mem_window_mode = ft_config.mem_window_mode
                # CAUSAL training keyframes instead of acausal whole-episode linspace, so the
                # training candidate pool matches the causal inference selector. 'diff'
                # (TokenDrop) and 'patch_union' each map to their own causal frame indexer
                # (the latter excludes the anchor to match its history<t read); everything
                # else (fifo/exp-d, linspace baselines) stays acausal linspace.
                MODALITY_CONFIGS[tag]["video"].framesamp_select = {
                    "diff": "diff",
                    "patch_union": "patch_union",
                    # var_pyramid: CAUSAL diff-keyframes incl. the anchor as last frame —
                    # matches its inference pixel FIFO (write-then-read, current frame in
                    # memory). Acausal linspace here was the patch_union-v1 root cause.
                    "var_pyramid": "diff",
                }.get(ft_config.mem_fs_select, "linspace")
                MODALITY_CONFIGS[tag]["video"].framesamp_diff_stride = (
                    ft_config.mem_fs_diff_stride
                )
        config.data.allow_padding = True
        print(f"[HAMLET] K-step batching: stride={stride} window={(K-1)*stride} delta_indices={new_indices}")
        if ft_config.mem_window_mode == "linspace":
            print(
                f"[HAMLET][episode-span] mem_window_mode=linspace: the K={K} moment-window frames "
                f"are CAUSAL linspace(0, anchor, K) over the history-so-far (delta_indices ignored)."
            )
        if fs_frames > 0:
            print(
                f"[HAMLET][framesamp] appending {fs_frames} episode-spanning linspace frames "
                f"per video stream (mem_framesamp_budget={ft_config.mem_framesamp_budget})"
            )
    elif ft_config.hamlet_mode == "tcl":
        from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
        new_indices = [0, -999]
        for tag in MODALITY_CONFIGS:
            if "video" in MODALITY_CONFIGS[tag]:
                MODALITY_CONFIGS[tag]["video"].delta_indices = new_indices
        config.data.allow_padding = True
        # Stage-specific numerical/trainability knobs (LR, max_grad_norm, tune_top_llm_layers,
        # tcl_no_projection_head) come from the CLI / the Stage-1 config script — NOT hardcoded
        # here — so Stage-1 vs Stage-2 are fully described by their own run_scripts.
        print(f"[HAMLET-TCL] video delta_indices = {new_indices} | lr={ft_config.learning_rate} | "
              f"max_grad_norm={ft_config.max_grad_norm} | tune_top_llm_layers={ft_config.tune_top_llm_layers} | "
              f"no_projection_head={ft_config.tcl_no_projection_head}")

    run(config)
