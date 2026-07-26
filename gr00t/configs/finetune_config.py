# Finetune config used for single node post-training.
from dataclasses import dataclass
from typing import Literal

from gr00t.data.embodiment_tags import EmbodimentTag


@dataclass
class FinetuneConfig:
    """
    Configuration for fine-tuning a Vision-Language-Action (VLA) model.

    This dataclass defines all parameters needed to launch a fine-tuning job
    on a pretrained base model using a custom dataset and embodiment-specific
    modality configuration. It controls model tuning options, data augmentation,
    and training hyperparameters.
    """

    # --- Data and Model Paths ---
    base_model_path: str
    """Path to the pretrained base model checkpoint (e.g., Hugging Face model hub or local directory)."""

    dataset_path: str
    """Path to the dataset root directory containing trajectory data for fine-tuning."""

    embodiment_tag: EmbodimentTag
    """Identifier specifying which embodiment (robot configuration) this fine-tuning run targets."""

    modality_config_path: str | None = None
    """
    Path to a Python file defining the modality configuration for the given embodiment. 
    If None, use the pre-registered modality config in `gr00t/configs/data/embodiment_configs.py`. 
    """

    # --- Model Tuning Flags ---
    tune_llm: bool = False
    """If True, fine-tune the language model (LLM) backbone during training."""

    tune_visual: bool = False
    """If True, fine-tune the visual encoder (e.g., ViT or CNN backbone)."""

    tune_top_llm_layers: int = 4
    """Number of top Eagle-LLM layers to keep trainable (base recipe: 4 = layers 12-15,
    trained alongside the action head; lower LLM + vision tower stay frozen).
    Stage-1 (TCL) MUST set this to 0: otherwise the (discarded) top LLM layers absorb the
    contrastive task and the moment tokens barely move from init. Stage-2 keeps 4."""

    tune_projector: bool = True
    """If True, fine-tune the multimodal projector layers that map vision/language features to a shared space."""

    tune_diffusion_model: bool = True
    """If True, fine-tune the diffusion-based action decoder (if present in the model)."""

    state_dropout_prob: float = 0.0
    """
    Dropout probability applied to state inputs for regularization during training.
    """

    # --- Data Augmentation ---
    random_rotation_angle: int | None = None
    """Maximum rotation angle (in degrees) for random rotation augmentation of input images."""

    color_jitter_params: dict[str, float] | None = None
    """
    Parameters for color jitter augmentation on images.

    Expected keys include:
      - "brightness": float
      - "contrast": float
      - "saturation": float
      - "hue": float
    Example: {"brightness": 0.4, "contrast": 0.4, "saturation": 0.4, "hue": 0.1}

    If None, applying the default color jitter augmentation from the pretrained model.
    """
    extra_augmentation_config: str | None = None
    """
    JSON string for extra image augmentations (mask-based and others).

    Expected keys include:
      - "background_noise_transforms": list of dicts for noise on mask regions
          - "target_mask_values": list of int (e.g., [0])
          - "p": float (probability of applying)
      - "masked_region_transforms": list of dicts for color tint on mask regions
          - "target_mask_values": list of int (e.g., [4] or [5])
          - "p": float (probability of applying)
          - "alpha_range": [min, max] for random_tint intensity

    Example: {"background_noise_transforms": [{"target_mask_values": [0], "p": 0.9}],
              "masked_region_transforms": [{"target_mask_values": [4], "p": 1.0, "alpha_range": [0, 1]}]}

    If None, no extra augmentations are applied.
    """

    # --- Training Configuration ---
    global_batch_size: int = 64
    """Total effective batch size across all GPUs and accumulation steps."""

    dataloader_num_workers: int = 2
    """Number of parallel worker processes used for data loading."""

    learning_rate: float = 1e-4
    """Initial learning rate for optimizer. Stage-1 (TCL): 2e-5; Stage-2: 1e-4."""

    max_grad_norm: float = 1.0
    """Gradient-clipping max norm. Stage-1 (TCL) uses 0.5 for InfoNCE stability."""

    gradient_accumulation_steps: int = 1
    """Number of forward passes to accumulate before performing a backward/update step."""

    output_dir: str = "./outputs"
    """Directory where model checkpoints, logs, and outputs are saved."""

    experiment_name: str | None = None
    """Optional experiment name used as the W&B run name. Defaults to the output directory basename."""

    wandb_project: str = "finetune-gr00t-n1d6"
    """W&B project name to log runs to."""

    save_steps: int = 1000
    """Frequency (in training steps) at which to save checkpoints."""

    save_total_limit: int = 5
    """Maximum number of checkpoints to keep before older ones are deleted."""

    num_gpus: int = 1
    """Number of GPUs available for distributed or single-node training."""

    use_wandb: bool = False
    """
    If True, log metrics and artifacts to Weights & Biases (wandb).
    The project is `finetune-gr00t-n1d6`.
    You need to login to wandb to view the logs.
    """

    max_steps: int = 10000
    """Total number of training steps to run before stopping."""

    weight_decay: float = 1e-5
    """Weight decay coefficient for optimizer (L2 regularization)."""

    warmup_ratio: float = 0.05
    """Proportion of total training steps used for learning rate warm-up."""

    shard_size: int = 2**10
    """Size of the shard to use for the dataset during preloading."""

    episode_sampling_rate: float = 0.1
    """Sampling rate for the episodes."""

    num_shards_per_epoch: int = int(1e5)
    """Number of shards to use for the dataset. reduce this number if vram is limited."""

    skip_weight_loading: bool = False
    """If True, skip loading model weights from base_model_path (architecture only).
    Useful for CI/testing to skip the slow checkpoint shard loading."""

    # --- HAMLET (History-Aware Memory with Learned Tokens) ---
    hamlet_mode: Literal["off", "tcl", "finetune"] = "finetune"
    """HAMLET training mode.
    - "off": vanilla GR00T N1.6 finetune (no HAMLET).
    - "tcl": Stage 1 — time-contrastive pretraining of moment tokens.
    - "finetune": Stage 2 — HAMLET end-to-end fine-tune (memory module + action head).
    """

    n_moment_tokens: int = 4
    """Number of learnable moment tokens (n_q) appended to the VLM input."""

    memory_window: int = 4
    """History window length T — number of past moment-token sets fed to the memory transformer."""

    memory_stride: int = 16
    """Stride (in env steps) between consecutive past snapshots in the HAMLET memory window.
    Must equal `n_action_steps` (the inference replanning interval) so the cache, which is
    updated once per policy call, naturally holds snapshots at [t-(K-1)S, ..., t-S, t]."""

    memory_num_layers: int = 2
    """Depth of the HAMLET memory transformer (paper default: 2)."""

    memory_arch: Literal["transformer", "gru", "ssm", "mamba"] = "transformer"
    """Memory aggregator over the K-step moment-token history. "transformer" (default) =
    block-causal attention MemoryTransformer; "gru"/"ssm"(S4D)/"mamba"(selective SSM) =
    recurrent/state-space SequenceMemory that accumulates "what events happened" as a state."""
    memory_hidden: int = 512
    """SequenceMemory bottleneck width (gru/ssm/mamba only)."""
    memory_state_dim: int = 64
    """SSM state size per layer (ssm/mamba only)."""

    memory_scales: str = ""
    """MULTI-SCALE memory (V1, Markdown/10_multiscale_temporal_memory.md): comma-separated window
    lengths, ascending (e.g. "2,8,16"). Runs one `memory_arch` aggregator per scale over the SUFFIX
    of the K-step moment history and concatenates all scales' current read-outs into the KV tail
    (masks grown like the framesamp M!=n_q path). max(scales) MUST equal memory_window (the loader
    provides exactly K snapshots; shorter scales slice suffixes — no loader change). "" = off
    (single-scale, byte-identical to the existing path)."""
    memory_scales_uniform: bool = False
    """Multi-scale only — append per-scale PARAMETER-FREE uniform-mean tokens (the one read-out with
    a guaranteed Dirichlet/sinc response, cutoff ~ 1/K_l): +len(scales)*n_q KV tokens."""
    memory_scales_dog: bool = False
    """Multi-scale only — append adjacent-scale DIFFERENCE tokens (m_short - m_long). Zero DC gain
    -> the only genuinely band-pass channel a simplex read-out bank can emit: +(L-1)*n_q tokens."""
    memory_comm: bool = False
    """Multi-scale V2 — coarse->fine communication (HKSL-style): each finer scale's read-out
    cross-attends the next-coarser scale's read-out (zero-init output proj -> exact no-op at init)."""
    memory_aux_lambda: float = 0.0
    """Multi-scale V2 — weight of the per-scale BYOL predictive loss (0 = off). Each scale l
    predicts the momentum-EMA read-out at t from its online read-out at t-h_l (short scale ->
    near-term change, long scale -> slow state): the explicit signal that forces scales to
    SPECIALIZE (window length alone does not pin the realized softmax spread)."""
    memory_aux_horizons: str = ""
    """Multi-scale V2 — comma-separated per-scale prediction horizons h_l (steps ahead). "" = auto:
    h_l = min(max(1, K_l//2), K - K_l); scales with no room (K_l == K) are skipped (horizon
    starvation — give the longest scale headroom by raising memory_window if you want it in)."""
    memory_aux_warmup_steps: int = 2000
    """Multi-scale V2 — linear warmup steps for the aux-loss weight (matches the zero-init
    warm-start convention). Counted per training forward; not checkpointed (resume re-ramps)."""
    memory_aux_ema: float = 0.996
    """Multi-scale V2 — momentum of the fp32 EMA target encoder (BYOL non-collapse requires
    EMA + stop-grad + online predictor, all three)."""
    memory_aux_detach_moment: bool = False
    """Multi-scale V2 — detach moment tokens for the ONLINE aux branch too, so the predictive loss
    shapes only the memory modules (not the shared backbone moment tokens). Fallback knob if the
    action loss regresses."""

    mem_cond_type: Literal["cross_attn", "adaln", "modul", "dual"] = "cross_attn"
    # v2b (note 16): dual-only framesamp injection — "none" | "te" | "moment".
    mem_fs_inject: Literal["none", "te", "moment"] = "none"
    """How memory conditions the action head.
    - "cross_attn" (default): memory-aggregated moment tokens replace the backbone
      moment-token tail and enter the DiT as cross-attention KV.
    - "adaln": the pooled memory vector goes through a zero-init Linear and is added to
      the DiT timestep embedding; the moment-token tail is sliced off the KV.
    - "modul" (MME-VLA style): the action tokens cross-attend to the full memory SEQUENCE
      and FiLM-modulate per DiT block (zero-init -> identity at start); tail sliced off the KV.
      Keeps selectivity (cross-attn) + high-amplitude injection (FiLM) — stronger than adaln.
    - "dual" (HYBRID memory): TWO channels, each through its OWN cross-attention (separate
      softmax — not concatenated, not FiLM). h_sem (DC) = the moment tokens on the action-head
      KV tail, EXACTLY like 'cross_attn' (the 18.4 baseline, unchanged). h_spatial (AC) = raw
      per-position framesamp tokens, consumed by a NEW per-block spatial cross-attn (residual-add,
      zero-init output projection -> contributes 0 at init, so a fresh dual forward == the
      moment-only baseline). Set mem_framesamp_frames>0 (loader appends the episode-spanning
      frames) so the spatial channel sees whole-episode context."""

    memory_type: Literal["moment_token", "vision_feature"] = "moment_token"
    """What flows through the memory module (action-head VLM conditioning is unchanged).
    "moment_token": learnable moment tokens' post-LLM hidden states.
    "vision_feature": primary view (first modality_key) image tokens, post-LLM, avg-pooled
    to 64/step (no moment tokens added). Supports both mem_cond_type values."""

    mem_film_layers: str = "all"
    """`mem_cond_type='modul'` only — injection-DEPTH control: which DiT blocks get a
    `MemoryFiLM`. Probes showed the every-block `modul` dumps 66-86% of memory->action
    saliency into the SHALLOW first block (the worst injection depth). This restricts the
    FiLM to a chosen depth band; blocks not selected get NO mem_film module (and skip the
    apply), so there is no shallow block-0 dump. Spec forms:
      - "all"            : every block (DEFAULT — exactly the current behavior).
      - "mid"            : a mid-deep band ~[num_layers//4 .. 5*num_layers//8) (e.g. [8..20)
                           of 32) — the empirically good injection depth.
      - "8,10,12,14"     : an explicit comma-separated list of block indices.
      - "8-20" / "8:20"  : a (start, end) half-open range.
    Resolved against the DiT `num_layers` at model build."""

    mem_window_mode: Literal["recent", "linspace"] = "recent"
    """Moment-memory window sampling (mem_source='moment' only). "recent" (DEFAULT, current
    behavior) = the recent-stride K-window (delta_indices = -(K-1-i)*stride). "linspace" =
    CAUSAL episode-spanning: the K memory-window frames are even-`linspace` over the
    history-so-far [0, current_step] (past only — train & inference identical, no FIFO
    mismatch). Tests whether whole-episode coverage (vs a recent window) helps the compressed
    moment memory. Orthogonal to mem_source; keep mem_source='moment'."""

    mem_source: Literal["moment", "framesamp"] = "moment"
    """`mem_cond_type='modul'` only — memory-RICHNESS control: what the per-block FiLM
    cross-attends.
      - "moment"    : the K x n_q compressed memory-transformer output `mq_memory_out`
                      (16-32 learned tokens) — DEFAULT, exactly the current behavior.
      - "framesamp" : MME-VLA-style — many RAW per-frame vision patch tokens spanning the
                      window, even-`linspace` temporal sub-sampling, capped at
                      `mem_framesamp_budget`. Feeds richer/rawer memory than the lossy
                      moment-token summary. NOTE: spanning the WHOLE episode (vs the current
                      K-frame/stride window) needs data-loader changes — see the model code /
                      return notes for the remaining pipeline work."""

    mem_framesamp_budget: int = 512
    """`mem_source='framesamp'` only — max number of raw vision tokens fed to the FiLM
    cross-attention (even `linspace` sub-sample of the available per-frame patch tokens).
    MME-VLA reference uses up to 512."""

    mem_fs_select: Literal["fifo", "diff", "patch_union", "var_pyramid"] = "fifo"
    """`mem_source='framesamp'` frame-selection FAMILY. "fifo" (default) = current
    behavior: acausal whole-episode linspace frames at train, recent-F FIFO at inference.
    "diff" = TokenDrop-style pixel-difference keyframes at BOTH train (loader: causal
    frame 0 + top-diff <= anchor + anchor) and inference (DiffFrameSelector; stamped into
    the checkpoint's model config), so train/eval frame distributions match."""

    mem_fs_diff_stride: int = 8
    """Scoring cadence for mem_fs_select='diff' (TokenDrop stride; env steps at train,
    policy calls at inference)."""

    mem_fs_attn_layer: int = 13
    """mem_fs_select='patch_union': DiT cross-attn layer whose action->patch attention is
    the relevance channel (scored in a no_grad pass over the full candidate set)."""

    mem_fs_diff_share: float = 0.5
    """mem_fs_select='patch_union': novelty-channel share of the patch budget (rest = relevance)."""

    mem_fs_tail_share: float = 0.0
    """mem_fs_select='patch_union': of the NON-novelty budget, the share going to the tail_L15
    channel (patch . frame-summary dot product; note-22). 0.0 = 2-way nov∪act (default,
    backward-compatible); 0.5 = 3-way nov∪act∪tail with equal act/tail."""

    mem_fs_pos_rope: bool = False
    """mem_fs_select='patch_union': PPE-style 3D RoPE (Δt,y,x) on the stored memory keys in the
    DiT cross-attention (note-23). False = no position (default, backward-compatible).
    Also honored by mem_fs_select='var_pyramid' (pyramid-token centers on the latent grid)."""

    mem_varp_ckpt: str = ""
    """mem_fs_select='var_pyramid': path to the official VAR vae checkpoint
    (vae_ch160v4096z32.pth, https://huggingface.co/FoundationVision/var). REQUIRED for real
    training — '' builds a random-weight tokenizer (tests only, warns loudly)."""

    mem_varp_res: int = 128
    """mem_fs_select='var_pyramid': VAR encode resolution. 128 -> latent 8x8, scales
    (1,2,3,4,5,6,8), <=155 tokens/frame; 256 -> latent 16x16, full 10 scales, <=680/frame."""

    mem_varp_budget: int = 0
    """mem_fs_select='var_pyramid': hard cap on emitted memory tokens (top-budget by gate,
    temporal order kept). 0 = no cap (soft gating only). For matched-budget A/B against
    patch_union/tokendrop set this equal to mem_framesamp_budget."""

    mem_varp_gate_hard: bool = False
    """mem_fs_select='var_pyramid': straight-through 0/1 prefix gates (hard forward, soft
    backward) instead of soft gate scaling — train/deploy-consistent hard selection."""

    mem_varp_budget_lambda: float = 0.0
    """mem_fs_select='var_pyramid': weight of the expected-token-fraction penalty on the
    selector gates (pressure to store coarser unless the task needs detail). 0 = off."""

    mem_varp_target_frac: float = 0.0
    """mem_fs_select='var_pyramid': explicit target for the (frac - target)^2 budget penalty.
    0 = AUTO: with a hard mem_varp_budget the target anchors at budget/(F*tokens_per_frame)
    (the cap's operating point — a plain linear penalty would push the capped-out deep gates
    to zero unopposed); without a hard budget, 0 falls back to the linear rate penalty."""

    mem_varp_gist_scales: int = 4
    """mem_fs_select='var_pyramid': number of coarsest pyramid levels pooled (mean + channel-max
    per level) into the selector's per-frame gist. Must reach deep enough to SEE what the
    selector is asked to buy — object identity enters the code around mid-depth."""

    mem_varp_view: int = 0
    """mem_fs_select='var_pyramid': which camera view's pixels feed the VAR tokenizer
    (0 = primary/front). Wrist-view folding into pos-RoPE is future work."""

    mem_framesamp_frames: int = 8
    """`mem_source='framesamp'` only — number of EPISODE-SPANNING video frames the loader
    even-`linspace`-samples across the WHOLE episode (independent of the K-step
    memory_window/stride). These frames are appended AFTER the K-step memory-window frames
    in the SAME video stream, run through the backbone vision encoder, and their raw patch
    tokens (linspace sub-sampled to `mem_framesamp_budget`) feed the per-block MemoryFiLM as
    `mem_seq`. Ignored unless `mem_source='framesamp'` (and `mem_cond_type='modul'`)."""

    mem_image_side: bool = False
    """`mem_cond_type='cross_attn'` only — cross-attn ROUTING control for the memory tokens
    that ride the action-head KV. The DiT splits cross-attn blocks into an IMAGE pathway
    (attends tokens with `image_mask=True`) and a TEXT pathway (`image_mask=False`).
      - False (DEFAULT — exactly the current behavior): memory tokens (moment tail or
              framesamp tokens) are tagged `image_mask=False` -> TEXT pathway.
      - True : memory tokens are tagged `image_mask=True` -> IMAGE pathway instead.
    Only affects the cross_attn paths; modul/adaln slice the moment tail off the KV so they
    are unaffected."""

    load_moment_tokens_from: str | None = None
    """Stage-2 entry. Path to a Stage-1 (TCL) checkpoint or `model.safetensors`
    from which the moment-token parameter is loaded."""

    freeze_moment_tokens: bool = False
    """Stage 2 freezes moment tokens by default (matches GR00T frozen-VLM recipe)."""

    tcl_tau: float = 0.07
    """InfoNCE temperature for the TCL stage."""

    tcl_no_projection_head: bool = False
    """Stage-1 (TCL) only. Drop the `moment_to_repr` projection head (use Identity) and
    compute the InfoNCE directly on the pooled moment-token representation, so the moment
    tokens are the ONLY trainable path — nothing downstream can absorb the contrastive
    signal. Ignored outside `--hamlet-mode tcl`."""

