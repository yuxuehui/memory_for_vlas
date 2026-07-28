from dataclasses import MISSING, asdict, dataclass, field, is_dataclass
from enum import Enum
import json
from pathlib import Path

import torch
from transformers import PretrainedConfig

from . import register_model_config


@dataclass
class Gr00tN1d6Config(PretrainedConfig):
    """Unified configuration for Gr00tN1d6 model with backbone and action head."""

    # Model identification
    model_type: str = "Gr00tN1d6"
    model_dtype: str = "bfloat16"  # Use bfloat16 for Flash Attention compatibility

    # backbone configuration
    model_name: str = "nvidia/Eagle-Block2A-2B-v2"
    backbone_model_type: str = "eagle"
    model_revision: str | None = None
    tune_top_llm_layers: int = 4  # Number of top LLM layers to tune
    backbone_embedding_dim: int = 2048  # project_to_dim
    tune_llm: bool = False
    tune_visual: bool = False
    select_layer: int = 16
    reproject_vision: bool = False
    use_flash_attention: bool = True
    load_bf16: bool = True  # Enable BF16 loading
    collator_overwrite_image_inputs: bool = False  # Deprecated; use eagle_collator.
    eagle_collator: bool = (
        False  # this allows model to change image size in collator, needed for eagle any-res
    )
    backbone_trainable_params_fp32: bool = True

    ### Processing parameters
    image_crop_size: tuple[int, int] | None = None
    image_target_size: tuple[int, int] | None = None

    shortest_image_edge: int | None = 256
    crop_fraction: float | None = 0.95

    random_rotation_angle: int | None = None
    color_jitter_params: dict[str, float] | None = None
    use_albumentations_transforms: bool = True
    # Extra augmentation config (mask-based and others).
    extra_augmentation_config: dict | None = None
    formalize_language: bool = True
    apply_sincos_state_encoding: bool = (
        False  # Global flag to enable per-embodiment sin/cos encoding
    )
    use_relative_action: bool = False

    # Action head configuration parameters
    max_state_dim: int = 29  # Default from state_shape
    max_action_dim: int = 29  # Default from action_shape
    action_horizon: int = 16
    hidden_size: int = 1024
    input_embedding_dim: int = 1536

    # Global parameters from YAML
    add_pos_embed: bool = True
    attn_dropout: float = 0.2
    use_vlln: bool = True
    max_seq_len: int = 1024
    # Diffusion model type selection
    use_alternate_vl_dit: bool = True  # True for AlternateVLDiT, False for DiT
    attend_text_every_n_blocks: int = 2

    # Diffusion model configuration with 32 layers (main difference from N15)
    diffusion_model_cfg: dict = field(
        default_factory=lambda: {
            "positional_embeddings": None,
            "num_layers": 32,  # 32 layers instead of 16
            "num_attention_heads": 32,
            "attention_head_dim": 48,
            "norm_type": "ada_norm",
            "dropout": 0.2,
            "final_dropout": True,
            "output_dim": 1024,
            "interleave_self_attention": True,
        }
    )

    # Flow matching parameters
    num_inference_timesteps: int = 4
    noise_beta_alpha: float = 1.5
    noise_beta_beta: float = 1.0
    noise_s: float = 0.999
    num_timestep_buckets: int = 1000

    # Training parameters
    tune_projector: bool = True
    tune_diffusion_model: bool = True
    tune_vlln: bool = True

    # State Augmentation parameters
    state_dropout_prob: float = 0.0  # State dropout probability
    state_additive_noise_scale: float = 0.0  # Scale for additive Gaussian noise on state features

    # Multi-embodiment parameters
    max_num_embodiments: int = 32

    # --- HAMLET ---
    # hamlet_mode in {"off", "tcl", "finetune"}.
    hamlet_mode: str = "finetune"
    n_moment_tokens: int = 4
    memory_window: int = 4
    memory_num_layers: int = 2
    # HAMLET memory aggregator architecture over the K-step moment-token history:
    #   "transformer" (default) = block-causal MemoryTransformer (attention);
    #   "gru" | "ssm" (S4D) | "mamba" (selective SSM) = explicit recurrent/state-space memory
    #   (drop-in SequenceMemory) that accumulates "what events have happened" as a running state.
    memory_arch: str = "transformer"
    memory_hidden: int = 512      # SequenceMemory bottleneck width (gru/ssm/mamba)
    memory_state_dim: int = 64    # SSM state size per layer (ssm/mamba)
    # MULTI-SCALE memory (V1/V2, Markdown/10_multiscale_temporal_memory.md). memory_scales =
    # comma-separated ascending window lengths ("2,8,16"; max MUST equal memory_window; "" = off).
    # One aggregator per scale over suffix windows; all current read-outs concat into the KV tail.
    memory_scales: str = ""
    memory_scales_uniform: bool = False   # + per-scale uniform-mean tokens (guaranteed sinc response)
    memory_scales_dog: bool = False       # + adjacent-scale difference tokens (band-pass)
    memory_comm: bool = False             # V2 coarse->fine zero-init cross-attn communication
    memory_aux_lambda: float = 0.0        # V2 per-scale BYOL predictive-loss weight (0 = off)
    memory_aux_horizons: str = ""         # V2 per-scale horizons; "" = auto (skip K_l == K)
    memory_aux_warmup_steps: int = 2000   # V2 aux-lambda linear warmup (per training forward)
    memory_aux_ema: float = 0.996         # V2 fp32 EMA momentum for the BYOL target encoder
    memory_aux_detach_moment: bool = False  # V2 detach moment tokens for the online aux branch
    # Env steps between cached memory snapshots; persisted to the checkpoint
    # config so evaluation can enforce n_action_steps == memory_stride.
    memory_stride: int = 16
    freeze_moment_tokens: bool = False
    # memory-to-action conditioning:
    #   "cross_attn" (default): memory-aggregated moment tokens replace the backbone
    #       moment-token tail and enter the DiT as cross-attention KV.
    #   "adaln": the pooled memory vector is added (zero-init) to the DiT timestep
    #       embedding and the moment-token tail is sliced off the KV (memory enters
    #       only via AdaLN).
    mem_cond_type: str = "cross_attn"
    # What flows through the memory module: {"moment_token", "vision_feature"}.
    memory_type: str = "moment_token"
    # mem_cond_type=="modul" only. Injection-DEPTH control: which DiT blocks get a
    # MemoryFiLM. "all" (default) = every block (current behavior); "mid" = a mid-deep
    # band; "8,10,12" = explicit indices; "8-20"/"8:20" = a (start, end) range.
    mem_film_layers: str = "all"
    # mem_cond_type=="modul" only. Memory-RICHNESS control: what the FiLM cross-attends.
    # "moment" (default) = compressed K*n_q moment tokens (current behavior); "framesamp" =
    # many raw per-frame vision tokens (linspace sub-sample) capped at mem_framesamp_budget.
    mem_source: str = "moment"
    # Moment-memory window sampling. "recent" (default) = recent-stride K-window (current
    # behavior); "linspace" = CAUSAL episode-spanning K-window (linspace over [0, current_step],
    # past only) so train & inference match. Used at inference to switch the rolling cache from
    # recent-FIFO to a full-buffer + linspace subsample. Orthogonal to mem_source.
    mem_window_mode: str = "recent"
    mem_framesamp_budget: int = 512
    # mem_source=="framesamp" only. Number of EPISODE-SPANNING video frames the loader
    # even-linspace-samples across the WHOLE episode (independent of memory_window/stride),
    # appended after the K-step memory-window frames in the same video stream so the model
    # can slice them off the backbone rows and build the framesamp mem_seq from raw patch
    # tokens. Ignored unless mem_source=="framesamp".
    mem_framesamp_frames: int = 8
    # mem_source=="framesamp", INFERENCE-only. Frame selection for the rolling cache:
    # "fifo" (default) = recent-F window (current behavior); "diff" = TokenDrop-style
    # pixel-difference keyframes (RoboMME mem_buffer.py, frame-level): frame 0 + running
    # top-(F-2) by mean |pixel diff| at mem_fs_diff_stride + current frame. Fixes the
    # recent-FIFO losing demo/early events on long episodes (train frames are
    # episode-spanning linspace; see Markdown/vlm_keyframe_labels probe). Training is
    # unaffected — the loader path never reads this.
    mem_fs_select: str = "fifo"
    # Scoring cadence for mem_fs_select=="diff"/"patch_union", in policy calls.
    mem_fs_diff_stride: int = 8
    # mem_fs_select=="patch_union" only: DiT cross-attn layer for the attention
    # channel, and the diff channel share of the patch budget.
    mem_fs_attn_layer: int = 13
    mem_fs_diff_share: float = 0.5
    # patch_union 3rd channel: of the NON-novelty budget, the share for tail_L15
    # (patch . frame-summary dot product). 0.0 = 2-way nov∪act; 0.5 = 3-way equal act/tail.
    mem_fs_tail_share: float = 0.0
    # patch_union: PPE-style 3D RoPE (Δt,y,x) on stored memory keys in DiT cross-attn (note-23).
    mem_fs_pos_rope: bool = False
    # note-24 LEARNED selection: a PatchScoreHead scores every candidate patch and a budgeted
    # Soft-TopK turns the scores into alpha (sum alpha == budget); alpha rides the DiT
    # cross-attn as an additive log(alpha) bias on the memory keys, so alpha=0 is exactly a
    # deleted key and the action loss can finally reach the selector. Replaces the
    # novelty/act/tail channels when True (mem_fs_diff_share etc. are then unused).
    mem_fs_learned_select: bool = False
    mem_fs_score_hidden: int = 0            # 0 -> backbone_embedding_dim // 8
    mem_fs_gate_tau_hi: float = 1.0         # Soft-TopK temperature, annealed hi -> lo so the
    mem_fs_gate_tau_lo: float = 0.1         #   train forward converges to the hard deployment
    mem_fs_gumbel_hi: float = 1.0           # exploration noise: without it only ALREADY
    mem_fs_gumbel_lo: float = 0.0           #   selected patches receive gradient
    mem_fs_anneal_steps: int = 20000
    # rank by the EXISTING heuristic (z novelty + z act) plus a zero-init correction, so a
    # warm-started run starts from today's selection instead of a random subset.
    mem_fs_score_residual: bool = False
    # mem_cond_type=="cross_attn" only. Cross-attn ROUTING for the memory tokens that ride
    # the action-head KV. False (default) = memory tagged image_mask=False (TEXT pathway,
    # current behavior); True = memory tagged image_mask=True (IMAGE pathway). Only the
    # cross_attn paths are affected (modul/adaln slice the moment tail off the KV).
    mem_image_side: bool = False
    # mem_cond_type=="dual" only (v2b, Markdown/16). Inject per-frame memory state into the
    # framesamp h_spatial tokens: "none" (default, byte-identical v1 dual) | "te" (sinusoidal
    # frame-index, ordering-only ablation) | "moment" (per-frame raw tails contextualized in
    # episode order by a dedicated block-causal MemoryTransformer; zero-init projection).
    mem_fs_inject: str = "none"
    # If None, defaults to `action_horizon` at runtime.
    tcl_tau: float = 0.07
    # Stage-1 (TCL) only: drop the moment_to_repr projection head (InfoNCE on the pooled
    # moment-token repr directly), so the moment tokens are the only trainable path.
    tcl_no_projection_head: bool = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            # PATCH: Backward compatibility for legacy argument "collator_overwrite_image_inputs"
            if key == "collator_overwrite_image_inputs":
                setattr(self, "eagle_collator", value)
            # /PATCH
            setattr(self, key, value)

        # Ensures that all dataclass defaults (including those using default_factory)
        # are explicitly assigned to the instance, even if dataclasses initialization or subclassing
        # (PretrainedConfig) interferes with normal default injection.
        for f in self.__dataclass_fields__.values():
            if not hasattr(self, f.name):
                if f.default is not MISSING:
                    setattr(self, f.name, f.default)
                elif getattr(f, "default_factory", MISSING) is not MISSING:
                    setattr(self, f.name, f.default_factory())

    def to_filtered_dict(self, exclude_augment: bool = True) -> dict:
        """Return a dictionary representation of this config, optionally excluding augmentation keys."""
        if is_dataclass(self):
            cfg = asdict(self)
        else:
            cfg = dict(self.__dict__)

        if exclude_augment:
            exclude_keys = {
                "random_rotation_angle",
                "color_jitter_params",
                "use_albumentations_transforms",
                "formalize_language",
                "image_crop_size",
                "image_target_size",
                "shortest_image_edge",
                "crop_fraction",
            }
            cfg = {k: v for k, v in cfg.items() if k not in exclude_keys}

        return cfg

    def to_filtered_json(self, exclude_augment: bool = True, **kwargs) -> str:
        """Return a JSON string of this config, optionally excluding augmentation keys."""

        def default(o):
            if isinstance(o, (Path, torch.dtype, torch.device)):
                return str(o)
            if isinstance(o, Enum):
                return o.value
            return str(o)

        return json.dumps(
            self.to_filtered_dict(exclude_augment),
            indent=2,
            default=default,
            **kwargs,
        )


register_model_config("GrootN1d6", Gr00tN1d6Config)
