import os
from typing import Any, Tuple

from gr00t.configs.model.gr00t_n1d6 import Gr00tN1d6Config
from gr00t.model.modules.dit import AlternateVLDiT, DiT
from gr00t.model.modules.eagle_backbone import EagleBackbone
from gr00t.model.modules.embodiment_conditioned_mlp import (
    CategorySpecificMLP,
    MultiEmbodimentActionEncoder,
)
from gr00t.model.modules.memory import MemoryTransformer
import torch
from torch import nn
from torch.distributions import Beta
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, PreTrainedModel
from transformers.feature_extraction_utils import BatchFeature
import tree


class _MemAdaLNPool(nn.Module):
    """Mean-pool memory tokens (B, n_q, d_in) -> (B, d_out) for AdaLN-zero conditioning.

    The output projection is zero-initialized so the AdaLN-memory path starts as an
    exact no-op; it learns to inject memory as training proceeds.
    """

    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.proj = nn.Linear(d_in, d_out)
        self.reset_parameters()

    def reset_parameters(self):
        """Zero-init the projection (no-op at init). Safe to call after
        `from_pretrained` leaves these as missing/uninitialized keys."""
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, mem: torch.Tensor) -> torch.Tensor:  # mem: (B, n_q, d_in)
        pooled = mem.mean(dim=1)  # (B, d_in)
        return self.proj(pooled)  # (B, d_out); zero at init -> no-op


class Gr00tN1d6ActionHead(nn.Module):
    """Action head component for flow matching diffusion policy."""

    supports_gradient_checkpointing = True

    def __init__(self, config: Gr00tN1d6Config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.input_embedding_dim = config.input_embedding_dim

        # memory-as-modulator (mem_cond_type='modul'): the DiT action head cross-attends to the
        # memory SEQUENCE and FiLM-modulates per layer (MME-VLA style). Enabled only with HAMLET memory.
        _use_hamlet_mem = (
            getattr(config, "hamlet_mode", "off") == "finetune"
            and getattr(config, "memory_num_layers", 0) > 0
        )
        _mem_film_dim = (
            config.backbone_embedding_dim
            if (getattr(config, "mem_cond_type", "cross_attn") == "modul" and _use_hamlet_mem)
            else None
        )
        # mem_cond_type='dual': the h_spatial (AC) channel rides a SEPARATE per-block spatial
        # cross-attention over the raw framesamp tokens (backbone hidden dim). Set only for dual
        # (and only with HAMLET memory) -> every other mode passes None and builds NO spatial
        # module, so non-dual behavior is byte-identical.
        _spatial_cross_attn_dim = (
            config.backbone_embedding_dim
            if (getattr(config, "mem_cond_type", "cross_attn") == "dual" and _use_hamlet_mem)
            else None
        )
        # mem_cond_type='modul' depth/richness knobs (defaults reproduce current behavior).
        _mem_film_layers = getattr(config, "mem_film_layers", "all")
        self.mem_source = getattr(config, "mem_source", "moment")
        self.mem_framesamp_budget = int(getattr(config, "mem_framesamp_budget", 512))
        # mem_source='framesamp': number of episode-spanning frames the loader appends after
        # the K-step memory-window frames (0 unless framesamp). Used to split the backbone
        # rows: first memory_window rows = K-window (moment memory), last this-many rows =
        # episode-spanning framesamp frames whose raw patch tokens build the mem_seq.
        self.mem_framesamp_frames = int(getattr(config, "mem_framesamp_frames", 0))
        # Inference-side framesamp frame SELECTION. "fifo" (default) = recent-F rolling
        # window (current behavior); "diff" = TokenDrop-style pixel-difference keyframes
        # (RoboMME mem_buffer.py, frame-level): frame 0 + top-(F-2) diff peaks + current.
        # Motivation: the recent-FIFO loses demo/early events on long episodes while the
        # training frames are episode-spanning (Markdown/vlm_keyframe_labels probe).
        self.mem_fs_select = getattr(config, "mem_fs_select", "fifo")
        self.mem_fs_diff_stride = int(getattr(config, "mem_fs_diff_stride", 8))
        # mem_fs_select='patch_union' (probe: Markdown/patch_memory_labels): patch-level
        # budget = diff top-(share) UNION action->image cross-attn top-(rest) at this layer.
        self.mem_fs_attn_layer = int(getattr(config, "mem_fs_attn_layer", 13))
        self.mem_fs_diff_share = float(getattr(config, "mem_fs_diff_share", 0.5))
        self.mem_fs_tail_share = float(getattr(config, "mem_fs_tail_share", 0.0))
        self.mem_fs_pos_rope = bool(getattr(config, "mem_fs_pos_rope", False))
        self._pu_rope_hd = int(config.diffusion_model_cfg.get("attention_head_dim", 48))
        self._pu_positions = None  # (B, M, 3) Δt,y,x of the current mem_seq, for key RoPE
        # mem_fs_select='var_pyramid' (Markdown/var_pyramid_memory.md): frozen VAR multi-scale
        # VQVAE encodes each candidate frame's RAW pixels into a residual coarseness pyramid;
        # a trainable selector picks the stored prefix depth per frame, conditioned on the
        # task/observation query. mem_seq = gated pyramid tokens; positions ride the same
        # fs_pos_rope channel as patch_union; the budget aux loss rides mem_aux_loss.
        self.var_pyramid = None
        if self.mem_fs_select == "var_pyramid":
            assert (
                self.mem_source == "framesamp"
                and getattr(config, "mem_cond_type", "cross_attn") in ("modul", "cross_attn")
                and int(getattr(config, "mem_framesamp_frames", 0)) > 0
            ), (
                "mem_fs_select='var_pyramid' requires mem_source='framesamp', mem_cond_type in "
                "('modul','cross_attn') and mem_framesamp_frames>0 — otherwise the selector/proj "
                "params are never exercised (DDP unused-parameter crash) and no memory is built."
            )
            from gr00t.model.modules.var_pyramid import VarPyramidMemory

            self.var_pyramid = VarPyramidMemory(
                dim=config.backbone_embedding_dim,
                res=int(getattr(config, "mem_varp_res", 128)),
                max_frames=max(32, int(getattr(config, "mem_framesamp_frames", 0)) + 1),
                budget=int(getattr(config, "mem_varp_budget", 0)),
                gate_hard=bool(getattr(config, "mem_varp_gate_hard", False)),
                budget_lambda=float(getattr(config, "mem_varp_budget_lambda", 0.0)),
                target_frac=float(getattr(config, "mem_varp_target_frac", 0.0)),
                gist_scales=int(getattr(config, "mem_varp_gist_scales", 4)),
                var_ckpt=str(getattr(config, "mem_varp_ckpt", "")),
            )
            self.mem_varp_view = int(getattr(config, "mem_varp_view", 0))
        self._varp_pixels = None  # rolling-inference pixel FIFO (B, F, 3, H, W), detached
        # Per-session selector list at inference (DiffFrameSelector | PatchUnionSelector);
        # Gr00tPolicy round-trips it across calls exactly like the FIFO caches.
        self._fs_state: list | None = None
        # patch_union two-pass scoring: pass A puts ALL candidate patch tokens in the KV and
        # captures the DiT's action->patch cross-attention; pass B selects the union top-budget
        # using those scores. `_pu_rel` holds (B, n_cand) relevance for pass B.
        self._pu_score_pass: bool = False
        self._pu_rel: torch.Tensor | None = None
        # inference post-action write: [(selector, vis_current, kv_image_cols)] per batch item
        self._pu_pending: list | None = None
        # cross_attn ROUTING for the memory tokens that ride the action-head KV. False
        # (default) -> image_mask=False (TEXT cross-attn pathway, current behavior); True ->
        # image_mask=True (IMAGE cross-attn pathway). Only the cross_attn paths read this.
        self.mem_image_side = bool(getattr(config, "mem_image_side", False))

        # Initialize components directly from config
        if config.use_alternate_vl_dit:
            self.model = AlternateVLDiT(
                **config.diffusion_model_cfg,
                cross_attention_dim=config.backbone_embedding_dim,
                mem_cross_attention_dim=_mem_film_dim,
                mem_film_layers=_mem_film_layers,
                spatial_cross_attention_dim=_spatial_cross_attn_dim,
                attend_text_every_n_blocks=config.attend_text_every_n_blocks,
            )
            print("Using AlternateVLDiT for diffusion model")
        else:
            self.model = DiT(
                **config.diffusion_model_cfg,
                cross_attention_dim=config.backbone_embedding_dim,
                mem_cross_attention_dim=_mem_film_dim,
                mem_film_layers=_mem_film_layers,
                spatial_cross_attention_dim=_spatial_cross_attn_dim,
            )
            print("Using DiT for diffusion model")
        self.action_dim = config.max_action_dim
        self.action_horizon = config.action_horizon
        self.num_inference_timesteps = config.num_inference_timesteps

        self.state_encoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=config.max_state_dim,
            hidden_dim=self.hidden_size,
            output_dim=self.input_embedding_dim,
        )
        self.action_encoder = MultiEmbodimentActionEncoder(
            action_dim=self.action_dim,
            hidden_size=self.input_embedding_dim,
            num_embodiments=config.max_num_embodiments,
        )
        self.action_decoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=self.hidden_size,
            hidden_dim=self.hidden_size,
            output_dim=self.action_dim,
        )

        self.vlln = (
            nn.LayerNorm(config.backbone_embedding_dim) if config.use_vlln else nn.Identity()
        )

        if config.add_pos_embed:
            self.position_embedding = nn.Embedding(config.max_seq_len, self.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        # State dropout parameters
        self.state_dropout_prob = config.state_dropout_prob
        self.mask_token = (
            nn.Parameter(0.02 * torch.randn(1, 1, self.input_embedding_dim))
            if self.state_dropout_prob > 0
            else None
        )

        # State noise parameters
        self.state_additive_noise_scale = config.state_additive_noise_scale

        self.beta_dist = Beta(config.noise_beta_alpha, config.noise_beta_beta)
        self.num_timestep_buckets = config.num_timestep_buckets

        # The memory transformer aggregates K timesteps of n_q moment tokens
        # (oldest first to current last) and replaces the current step's tail with the
        # memory-augmented output. memory_num_layers == 0 means identity pass-through.
        self.use_hamlet = getattr(config, "hamlet_mode", "off") == "finetune"
        # memory_type: "moment_token" (n_q tokens/step) or "vision_feature"
        # (primary-view image tokens pooled to 64/step).
        self.memory_type = getattr(config, "memory_type", "moment_token")
        self._mem_tokens_per_step = (
            64 if self.memory_type == "vision_feature" else config.n_moment_tokens
        )
        # MULTI-SCALE memory (V1/V2, Markdown/10_multiscale_temporal_memory.md): one aggregator per
        # window length over suffix windows of the K-step history; "" = off (single-scale path).
        from gr00t.model.modules.multiscale_memory import parse_int_list
        self.memory_scales = parse_int_list(getattr(config, "memory_scales", ""))
        if self.use_hamlet and getattr(config, "memory_num_layers", 0) > 0:
            mem_arch = getattr(config, "memory_arch", "transformer")
            if self.memory_scales:
                assert self.memory_scales[-1] == config.memory_window, (
                    f"max(memory_scales)={self.memory_scales[-1]} must equal "
                    f"memory_window={config.memory_window} (the loader provides exactly K snapshots; "
                    f"shorter scales slice suffixes)"
                )
                assert getattr(config, "mem_source", "moment") == "moment", (
                    "memory_scales requires mem_source='moment' — framesamp bypasses the moment "
                    "memory (and its cross_attn fs-fallback path does not grow masks for M != n_q)"
                )
                from gr00t.model.modules.multiscale_memory import MultiScaleMemory
                self.memory_transformer = MultiScaleMemory(
                    dim=config.backbone_embedding_dim,
                    n_q=self._mem_tokens_per_step,
                    scales=self.memory_scales,
                    num_layers=config.memory_num_layers,
                    arch=mem_arch,
                    hidden=getattr(config, "memory_hidden", 512),
                    n_state=getattr(config, "memory_state_dim", 64),
                    uniform_slots=getattr(config, "memory_scales_uniform", False),
                    dog_tokens=getattr(config, "memory_scales_dog", False),
                    comm=getattr(config, "memory_comm", False),
                    aux_lambda=getattr(config, "memory_aux_lambda", 0.0),
                    aux_horizons=parse_int_list(getattr(config, "memory_aux_horizons", "")),
                    aux_warmup=getattr(config, "memory_aux_warmup_steps", 2000),
                    aux_ema=getattr(config, "memory_aux_ema", 0.996),
                    aux_detach_moment=getattr(config, "memory_aux_detach_moment", False),
                )
            elif mem_arch == "transformer":
                self.memory_transformer = MemoryTransformer(
                    dim=config.backbone_embedding_dim,
                    n_q=self._mem_tokens_per_step,
                    T=config.memory_window,
                    num_layers=config.memory_num_layers,
                )
            else:
                # recurrent / state-space memory (GRU / S4D / Mamba) — drop-in, same I/O contract.
                from gr00t.model.modules.seq_memory import SequenceMemory
                self.memory_transformer = SequenceMemory(
                    dim=config.backbone_embedding_dim,
                    n_q=self._mem_tokens_per_step,
                    T=config.memory_window,
                    num_layers=config.memory_num_layers,
                    kind=mem_arch,
                    hidden=getattr(config, "memory_hidden", 512),
                    n_state=getattr(config, "memory_state_dim", 64),
                )
        else:
            self.memory_transformer = None

        # Inference-time rolling cache. Shape: (B, K*n_q, d), oldest first to current last.
        self._memory_cache: torch.Tensor | None = None
        self._vision_cache: torch.Tensor | None = None
        # mem_cond_type='dual' inference: a SEPARATE rolling raw-vision FIFO for the h_spatial
        # (framesamp) channel, kept distinct from `_vision_cache`/`_memory_cache` so the moment
        # (h_sem) memory state is never clobbered. Shape: (B, F*n_img, d), oldest first.
        self._spatial_cache: torch.Tensor | None = None
        # mem_window_mode: "recent" (recent-stride K FIFO) or "linspace" (CAUSAL episode-spanning:
        # buffer ALL past moment read-outs and subsample linspace(0, now, K) each step, matching
        # the loader's training-time window). _moment_buffer holds the full (B, t*n_q, d) history.
        self.mem_window_mode = getattr(config, "mem_window_mode", "recent")
        self._moment_buffer: torch.Tensor | None = None

        # memory-to-action conditioning. "cross_attn" (default) replaces the moment-token
        # tail of the action-head KV; "adaln" mean-pools the memory output through a
        # zero-init projection added to the DiT timestep embedding.
        self.mem_cond_type = getattr(config, "mem_cond_type", "cross_attn")

        # v2b (Markdown/16_memory_into_image_tokens.md): inject per-frame memory state into the
        # framesamp h_spatial tokens, so episode-spanning frames stop being ALIASED (identical
        # swings get distinct, event-indexed tokens — the diagnosed v1-dual failure).
        #   'none'   (default): byte-identical to v1 dual.
        #   'te'     : sinusoidal frame-index embedding — ordering-only ablation.
        #   'moment' : each F frame's OWN raw moment tail, contextualized IN EPISODE ORDER by a
        #              dedicated block-causal MemoryTransformer (frame f's state = its place in
        #              the event history; block-causal => only past frames visible).
        # Both variants pass through a ZERO-INIT projection: step-0 forward is byte-identical
        # to 'none' (the RT-1 zero-init recipe; NOT multiplicative FiLM — the modul -5.0 lesson).
        self.mem_fs_inject = getattr(config, "mem_fs_inject", "none")
        self.fs_inject_tf = None
        self.fs_inject_proj = None
        self._spatial_tail_cache: torch.Tensor | None = None
        if self.mem_cond_type == "dual" and self.mem_fs_inject != "none":
            assert self.mem_fs_inject in ("te", "moment"), (
                f"mem_fs_inject={self.mem_fs_inject!r} (expected 'none'|'te'|'moment')"
            )
            assert self.mem_framesamp_frames > 0, (
                "mem_fs_inject requires mem_framesamp_frames>0 (the F episode-spanning frames)"
            )
            _d_bb = config.backbone_embedding_dim
            if self.mem_fs_inject == "moment":
                self.fs_inject_tf = MemoryTransformer(
                    dim=_d_bb,
                    n_q=self._mem_tokens_per_step,
                    T=self.mem_framesamp_frames,
                    num_layers=2,
                )
            self.fs_inject_proj = nn.Linear(_d_bb, _d_bb)
            nn.init.zeros_(self.fs_inject_proj.weight)
            nn.init.zeros_(self.fs_inject_proj.bias)
        if (
            self.use_hamlet
            and self.memory_transformer is not None
            and self.mem_cond_type == "adaln"
        ):
            self.mem_adaln_pool = _MemAdaLNPool(
                d_in=config.backbone_embedding_dim,
                d_out=self.model.inner_dim,
            )
        else:
            self.mem_adaln_pool = None

        self.set_trainable_parameters(
            config.tune_projector, config.tune_diffusion_model, config.tune_vlln
        )

    def set_trainable_parameters(
        self, tune_projector: bool, tune_diffusion_model: bool, tune_vlln: bool
    ):
        self.tune_projector = tune_projector
        self.tune_diffusion_model = tune_diffusion_model
        self.tune_vlln = tune_vlln
        for p in self.parameters():
            p.requires_grad = True
        if not tune_projector:
            self.state_encoder.requires_grad_(False)
            self.action_encoder.requires_grad_(False)
            self.action_decoder.requires_grad_(False)
            if self.config.add_pos_embed:
                self.position_embedding.requires_grad_(False)
            if self.state_dropout_prob > 0:
                self.mask_token.requires_grad_(False)
        if not tune_diffusion_model:
            self.model.requires_grad_(False)
        if not tune_vlln:
            self.vlln.requires_grad_(False)
        print(f"Tune action head projector: {self.tune_projector}")
        print(f"Tune action head diffusion model: {self.tune_diffusion_model}")
        print(f"Tune action head vlln: {self.tune_vlln}")
        # Check if any parameters are still trainable. If not, print a warning.
        if not tune_projector and not tune_diffusion_model and not tune_vlln:
            for name, p in self.named_parameters():
                if p.requires_grad:
                    print(f"Action head trainable parameter: {name}")
        if not any(p.requires_grad for p in self.parameters()):
            print("Warning: No action head trainable parameters found.")

    def set_frozen_modules_to_eval_mode(self):
        """
        Huggingface will call model.train() at each training_step. To ensure
        the expected behaviors for modules like dropout, batchnorm, etc., we
        need to call model.eval() for the frozen modules.
        """
        if self.training:
            if not self.tune_projector:
                self.state_encoder.eval()
                self.action_encoder.eval()
                self.action_decoder.eval()
                if self.config.add_pos_embed:
                    self.position_embedding.eval()
            if not self.tune_diffusion_model:
                self.model.eval()

    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        sample = (1 - sample) * self.config.noise_s
        return sample

    def _fs_inject_states(self, tails_BFnd: torch.Tensor) -> torch.Tensor:
        """v2b (note 16): per-frame injection state for the framesamp h_spatial tokens.

        tails_BFnd: (B, F, n_q, d) — each episode-spanning frame's OWN raw moment tail,
        oldest first (episode order).

        'moment': contextualize the tails in order with the dedicated block-causal
        transformer, so frame f's state encodes its position in the EVENT history
        (identical frames -> distinct states: the anti-aliasing ingredient).
        'te': sinusoidal frame-index embedding — the ordering-only ablation (same
        zero-init projection path; if 'te' matches 'moment', the win was mere ordering).

        Returns (B, F, d) AFTER the zero-init projection (=> exactly 0 at step 0).
        """
        B, Fn, n_q, d = tails_BFnd.shape
        if self.mem_fs_inject == "moment":
            ctx = self.fs_inject_tf(tails_BFnd.reshape(B, Fn * n_q, d))
            s = ctx.view(B, Fn, n_q, d).mean(dim=2)
        else:  # 'te'
            half = d // 2
            idx = torch.arange(Fn, device=tails_BFnd.device, dtype=torch.float32)
            freqs = 10000.0 ** (
                -torch.arange(half, device=tails_BFnd.device, dtype=torch.float32)
                / max(half - 1, 1)
            )
            ang = idx[:, None] * freqs[None, :]  # (F, half)
            te = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)
            if te.shape[-1] < d:
                te = F.pad(te, (0, d - te.shape[-1]))
            s = te.to(tails_BFnd.dtype).unsqueeze(0).expand(B, Fn, d)
        return self.fs_inject_proj(s)

    def _framesamp_mem_seq(
        self,
        backbone_features_BKTd: torch.Tensor,  # (B, F, T, d)
        image_mask_BKT: torch.Tensor | None,  # (B, F, T) bool, or None
    ) -> torch.Tensor:
        """mem_source='framesamp' (MME-VLA style): gather the RAW per-frame vision patch
        tokens across the F supplied frames and even-`linspace` sub-sample to
        `mem_framesamp_budget` tokens -> (B, budget_eff, d). Used as `mem_seq` for the
        per-block MemoryFiLM instead of the compressed `mq_memory_out` moment tokens.

        At training the F frames are the EPISODE-SPANNING linspace frames the loader appends
        after the K-step memory window (`mem_framesamp_frames`); the inference path builds the
        same representation from the rolling raw-vision FIFO cache. The model-side budget knob
        is honored across whatever frames are passed in.
        """
        B, K, T, d = backbone_features_BKTd.shape
        flat = backbone_features_BKTd.reshape(B, K * T, d)  # oldest frame first
        if image_mask_BKT is not None:
            # Per-sample image-token counts can differ; gather a uniform count = the min over
            # the batch so the result is a dense (B, n, d) tensor (mask-free, like moment seq).
            mask_flat = image_mask_BKT.reshape(B, K * T).bool()
            counts = mask_flat.sum(dim=1)
            n_img = int(counts.min().item())
            if n_img == 0:
                # No image tokens detected -> nothing raw to sample; caller falls back.
                return None
            rows = []
            for b in range(B):
                idx = mask_flat[b].nonzero(as_tuple=False).squeeze(1)[:n_img]
                rows.append(flat[b, idx, :])
            vis = torch.stack(rows, dim=0)  # (B, n_img, d), temporal order preserved
        else:
            vis = flat  # (B, K*T, d)
        n_tok = vis.shape[1]
        budget = min(self.mem_framesamp_budget, n_tok)
        if budget < n_tok:
            # Even temporal sub-sample over the whole window (MME-VLA linspace).
            sel = torch.linspace(0, n_tok - 1, steps=budget, device=vis.device).round().long()
            sel = torch.unique(sel)
            vis = vis[:, sel, :]
        return vis.contiguous()

    def _patch_union_mem_seq(
        self,
        fs_backbone_BFTd: torch.Tensor,  # (B, F, T, d) candidate frames
        fs_image_mask_BFT: torch.Tensor | None,  # (B, F, T) bool
        moment_q_Bd: torch.Tensor,  # (B, d) relevance query (current moment tokens, mean)
    ) -> torch.Tensor | None:
        """mem_fs_select='patch_union' TRAINING path: pick `mem_framesamp_budget` individual
        PATCH tokens out of the F candidate frames by the union of two channels
        (Markdown/patch_memory_labels: union > either channel alone > z-sum):

          novelty  (diff_share of the budget): token-space |delta| vs the SAME patch index in
                   the previous candidate frame; frame 0 protected (TokenDrop sentinel).
          relevance(rest): dot product with the current step's moment tokens — the model's own
                   action-conditioning summary as the query (prefix-routed relevance;
                   obj2all >> obj2act in the grounding probe). Same quantity is computable at
                   inference from the current frame alone, so train/deploy match.

        Selection is score-only (no_grad); the returned tokens keep their graph so gradients
        flow to the selected patches exactly as in the linspace path.
        """
        B, Fn, T, d = fs_backbone_BFTd.shape
        if fs_image_mask_BFT is not None:
            m = fs_image_mask_BFT.bool()
            n_img = int(m.sum(dim=2).min().item())
            if n_img == 0:
                return None
            rows = []
            for b in range(B):
                fr = [fs_backbone_BFTd[b, f][m[b, f]][:n_img, :] for f in range(Fn)]
                rows.append(torch.stack(fr, dim=0))
            vis = torch.stack(rows, dim=0)  # (B, F, n_img, d)
        else:
            vis = fs_backbone_BFTd
            n_img = T
        flat_all = vis.reshape(B, Fn * n_img, d)
        self._pu_n_cand = Fn * n_img
        self._pu_n_img = n_img
        if self._pu_score_pass:
            if self.mem_fs_pos_rope:  # pass A KV = all candidates, in flat order
                from gr00t.model.modules.fs_pos_rope import positions_from_flat
                allc = torch.arange(Fn * n_img, device=flat_all.device).unsqueeze(0).expand(B, -1)
                self._pu_positions = positions_from_flat(allc, n_img)
            return flat_all.contiguous()  # pass A: score every candidate
        budget = min(self.mem_framesamp_budget, Fn * n_img)
        n_diff = max(1, int(round(budget * self.mem_fs_diff_share)))
        rem = max(0, budget - n_diff)
        want_tail = self.mem_fs_tail_share > 0.0 and fs_image_mask_BFT is not None
        n_tail = max(0, int(round(rem * self.mem_fs_tail_share))) if want_tail else 0
        n_attn = max(0, rem - n_tail)
        with torch.no_grad():
            v32 = vis.detach().float()
            prev = torch.cat([v32[:, :1], v32[:, :-1]], dim=1)
            diff = (v32 - prev).abs().mean(-1)  # (B, F, n_img)
            # frame-0 FRONT sentinel (TokenDrop): matches inference observe() step-0, which
            # protects only the front half; all-view inf would eat 2x the novelty slots.
            _half = n_img // 2 if n_img % 2 == 0 else n_img
            diff[:, 0, :_half] = float("inf")
            if self._pu_rel is not None and self._pu_rel.shape[-1] == Fn * n_img:
                rel = self._pu_rel.to(v32.device).float().view(B, Fn, n_img)
            else:  # fallback (no captured attention): moment-token query
                rel = torch.einsum("bfnd,bd->bfn", v32, moment_q_Bd.detach().float())
            # tail_L15 channel: per-frame summary (mean of POST-IMAGE, PRE-MOMENT tokens on
            # the FULL candidate frame, note-22) . patch, dot product. `vis` is already
            # image-only, so recover the summary from fs_backbone_BFTd's tail region.
            tail = None
            if n_tail > 0:
                n_q = int(getattr(self.config, "n_moment_tokens", 0))
                mm = fs_image_mask_BFT.bool()
                tq = torch.zeros(B, Fn, d, device=v32.device, dtype=torch.float32)
                for b in range(B):
                    for f in range(Fn):
                        idx = mm[b, f].nonzero(as_tuple=True)[0]
                        lo = int(idx[-1]) + 1 if idx.numel() else 0
                        hi = T - n_q if n_q > 0 else T
                        seg = fs_backbone_BFTd[b, f, lo:hi]
                        if seg.shape[0]:
                            tq[b, f] = seg.float().mean(0)
                tail = torch.einsum("bfnd,bfd->bfn", v32, tq)
            flat_d = diff.reshape(B, Fn * n_img)
            flat_r = rel.reshape(B, Fn * n_img)
            flat_t = tail.reshape(B, Fn * n_img) if tail is not None else None
            keep = []
            for b in range(B):
                idx_d = torch.topk(flat_d[b], min(n_diff, flat_d.shape[1])).indices
                idx_r = torch.topk(flat_r[b], min(n_attn, flat_r.shape[1])).indices if n_attn else idx_d[:0]
                cat = [idx_d, idx_r]
                if flat_t is not None and n_tail > 0:
                    cat.append(torch.topk(flat_t[b], min(n_tail, flat_t.shape[1])).indices)
                u = torch.unique(torch.cat(cat))
                u = u.sort().values[:budget]
                if u.numel() < budget:  # pad by repeating the last (kept in temporal order)
                    u = torch.cat([u, u[-1:].expand(budget - u.numel())])
                keep.append(u)
            keep_idx = torch.stack(keep, dim=0)  # (B, budget) into F*n_img, temporal order
        if self.mem_fs_pos_rope:  # pass B KV = the selected patches, in keep_idx order
            from gr00t.model.modules.fs_pos_rope import positions_from_flat
            self._pu_positions = positions_from_flat(keep_idx, n_img)
        return torch.gather(flat_all, 1, keep_idx.unsqueeze(-1).expand(B, budget, d)).contiguous()

    def _var_pyramid_mem_seq(
        self,
        backbone_output,
        B: int,
        K_all: int,
        F: int,
        moment_q_Bd: torch.Tensor,
        reset_memory: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """mem_fs_select='var_pyramid': encode the F candidate frames' RAW pixels
        (`fs_pixels`, stashed by the top-level forward/get_action) with the frozen VAR pyramid
        and emit the selector-gated multi-coarseness tokens as mem_seq. Returns None when
        pixels/config are unavailable — the caller falls back to _framesamp_mem_seq.
        Positions ride self._pu_positions (fs_pos_rope key-RoPE, same channel as patch_union);
        the expected-budget aux loss rides backbone_output['mem_aux_loss'] (same plumbing as
        the multi-scale V2 aux; framesamp excludes the moment-path writers, so no collision)."""
        px = backbone_output.get("fs_pixels", None)
        if px is None or self.var_pyramid is None or F <= 0:
            if self.mem_fs_pos_rope:
                # Never leave positions from a previous successful call: the caller falls back
                # to framesamp tokens of a DIFFERENT length, and stale positions would rotate
                # the wrong keys (or crash) in the DiT key-RoPE.
                self._pu_positions = None
            return None
        if isinstance(px, (list, tuple)):
            # Eagle processor emits pixel_values as a LIST of per-image tensors (tiles, 3, H, W)
            # in row-major image order. RoboMME 256px -> 1 tile of 252x252 each, so the cat
            # reproduces the (B*K_all*V, 3, H, W) layout. Heterogeneous sizes (dynamic tiling
            # on larger images) are resized to the VAR encode res first — VAR resizes anyway.
            res = self.var_pyramid.tokenizer.res
            if len({tuple(t.shape[-2:]) for t in px}) > 1:
                px = [
                    torch.nn.functional.interpolate(
                        t.float(), size=(res, res), mode="bicubic", align_corners=False
                    )
                    for t in px
                ]
            px = torch.cat([t.reshape(-1, *t.shape[-3:]) for t in px], dim=0)
        rows = B * K_all
        if K_all > 1:
            # K-step training: pixel rows cover all K_target+F backbone rows (V views each,
            # row-major). Slice the LAST F rows (the episode-spanning candidates), chosen view.
            if px.shape[0] % rows != 0:
                raise RuntimeError(
                    f"var_pyramid: fs_pixels has {px.shape[0]} rows, not divisible by "
                    f"B*K_all={rows} — unexpected pixel_values layout (views per row must be "
                    f"constant and row-major)."
                )
            V = px.shape[0] // rows
            frames = px.view(B, K_all, V, *px.shape[1:])[
                :, K_all - F :, min(self.mem_varp_view, V - 1)
            ]
        else:
            # Rolling inference: px holds the CURRENT frame's views. Maintain a detached
            # F-frame pixel FIFO (episode state, like _vision_cache; cleared by reset_memory).
            V = max(1, px.shape[0] // B)
            cur = px.view(B, V, *px.shape[1:])[:, min(self.mem_varp_view, V - 1)]
            cur = cur.detach().unsqueeze(1)  # (B, 1, 3, H, W)
            if self._varp_pixels is None or self._varp_pixels.shape[0] != B:
                self._varp_pixels = cur.expand(B, F, *cur.shape[2:]).contiguous()
            elif reset_memory is not None and reset_memory.any():
                # Per-sample episode reset (mirrors _vision_cache): reset rows refill with the
                # current frame, others shift.
                defaults = cur.expand(B, F, *cur.shape[2:])
                shifted = torch.cat([self._varp_pixels[:, 1:], cur], dim=1)
                rb = reset_memory.view(B, 1, 1, 1, 1)
                self._varp_pixels = torch.where(rb, defaults, shifted)
            else:
                self._varp_pixels = torch.cat([self._varp_pixels[:, 1:], cur], dim=1)
            frames = self._varp_pixels
        mem, pos, aux = self.var_pyramid(frames, moment_q_Bd)
        if self.mem_fs_pos_rope:
            self._pu_positions = pos
        if aux is not None:
            prev = backbone_output.get("mem_aux_loss", None)
            backbone_output["mem_aux_loss"] = aux if prev is None else prev + aux
        return mem

    def _pu_rope_on(self):
        """Build cos/sin from self._pu_positions and arm the key-RoPE for the next DiT forward.
        No-op unless mem_fs_pos_rope and positions are available."""
        if not self.mem_fs_pos_rope or self._pu_positions is None:
            return
        # Key-RoPE rides GLOBAL state that is cleared right after the forward. Gradient
        # checkpointing would RECOMPUTE the DiT forward during backward — after clear_rope —
        # so the recomputed keys would be unrotated: silent forward/recompute mismatch and
        # wrong gradients. Our runs keep checkpointing off (training_config default False);
        # refuse loudly rather than corrupt gradients if someone turns it on.
        if getattr(self, "training", False) and getattr(
            getattr(self, "model", None), "gradient_checkpointing", False
        ):
            raise RuntimeError(
                "mem_fs_pos_rope is incompatible with gradient_checkpointing on the DiT: "
                "the rope global state is cleared before backward-time recompute, which "
                "would silently produce wrong gradients. Disable one of the two."
            )
        from gr00t.model.modules.fs_patch_union import set_rope
        from gr00t.model.modules.fs_pos_rope import build_cos_sin
        cos, sin = build_cos_sin(self._pu_positions, self._pu_rope_hd)
        set_rope(cos, sin, self._pu_positions.shape[1])

    def _pu_rope_off(self):
        if not self.mem_fs_pos_rope:
            return
        from gr00t.model.modules.fs_patch_union import clear_rope
        clear_rope()

    @torch.no_grad()
    def _patch_union_score_pass(self, backbone_output, action_input):
        """Pass A: assemble the KV with ALL candidate patch tokens, run ONE DiT forward at a
        fixed timestep, and return the captured action->candidate cross-attention (B, n_cand).

        Cheap: the backbone is NOT re-run (its output is the input here); only the memory
        assembly + one DiT forward over 16 action queries. Returns None on any failure, in
        which case pass B falls back to the moment-token query."""
        from transformers.feature_extraction_utils import BatchFeature as _BF

        from gr00t.model.modules.fs_patch_union import capture_begin, capture_end

        try:
            bo = _BF(data=dict(backbone_output))
            self._pu_score_pass, self._pu_rel = True, None
            B = action_input.state.shape[0]
            bo = self.process_backbone_output(bo, action_inputs_B=B)
            cand = bo.get("mem_seq", None)
            kv = bo["backbone_features"]
            n_cand = cand.shape[1] if cand is not None else int(getattr(self, "_pu_n_cand", 0))
            if n_cand == 0:
                return None
            device, dtype = kv.device, kv.dtype
            state_features = self.state_encoder(action_input.state, action_input.embodiment_id)
            horizon = self.config.action_horizon
            noisy = torch.zeros(B, horizon, self.action_dim, device=device, dtype=dtype)
            t_disc = torch.full((B,), self.num_timestep_buckets // 2, device=device, dtype=torch.long)
            af = self.action_encoder(noisy, t_disc, action_input.embodiment_id)
            if self.config.add_pos_embed:
                pos = torch.arange(af.shape[1], dtype=torch.long, device=device)
                af = af + self.position_embedding(pos).unsqueeze(0)
            sa = torch.cat((state_features, af), dim=1)
            n_cross = int(self.config.diffusion_model_cfg["num_layers"]) // 2
            capture_begin(self.mem_fs_attn_layer, n_cross)
            kwargs = dict(
                hidden_states=sa, encoder_hidden_states=kv,
                encoder_attention_mask=bo.get("backbone_attention_mask", None),
                timestep=t_disc, return_all_hidden_states=True,
                temb_add=bo.get("mem_temb_add", None),
                mem_seq=bo.get("mem_seq", None),
                mem_seq_spatial=bo.get("mem_seq_spatial", None),
            )
            if self.config.use_alternate_vl_dit:
                kwargs["image_mask"] = bo.get("image_mask", None)
                kwargs["backbone_attention_mask"] = bo.get("backbone_attention_mask", None)
            self._pu_rope_on()          # rotate the candidate keys during scoring too
            try:
                self.model(**kwargs)
            finally:
                self._pu_rope_off()
            scores = capture_end()
            if scores is None or scores.shape[0] != B:
                return None
            return scores[:, -n_cand:].detach()  # candidates occupy the KV tail
        except Exception as e:
            try:
                capture_end()
            except Exception:
                pass
            if not getattr(self, "_pu_warned", False):
                print(f"[patch_union] score pass failed ({type(e).__name__}: {e}); "
                      f"falling back to moment-token query", flush=True)
                self._pu_warned = True
            return None
        finally:
            self._pu_score_pass = False

    def process_backbone_output(
        self,
        backbone_output: BatchFeature,
        action_inputs_B: int | None = None,
        reset_memory: torch.Tensor | None = None,
    ) -> BatchFeature:
        backbone_features = backbone_output["backbone_features"]
        backbone_features = self.vlln(backbone_features)

        if (
            self.use_hamlet
            and self.memory_transformer is not None
            and self.memory_type == "vision_feature"
            and "primary_view_feature" in backbone_output
            and action_inputs_B is not None
        ):
            # vision_feature: memory aggregates the primary view's pooled (64) tokens.
            # backbone_features (action-head conditioning) is the vanilla VLM output and is
            # NOT modified except (cross_attn only) by APPENDING the current memory tokens.
            v_nq = self._mem_tokens_per_step  # 64
            primary = backbone_output["primary_view_feature"]  # (B*K, 64, d)
            BK, _, d = primary.shape
            B = action_inputs_B
            K = BK // B
            K_target = self.memory_transformer.T
            Tlen = backbone_features.shape[1]
            if K not in (1, K_target):
                raise RuntimeError(
                    f"HAMLET memory: got K={K} backbone rows per action sample "
                    f"(expected 1 for rolling inference or memory_window={K_target} "
                    f"for K-step training). The video delta_indices / memory_window "
                    f"data config is inconsistent - refusing to silently skip "
                    f"memory augmentation."
                )
            if K == K_target:
                if self.memory_scales:
                    # multi-scale: the appended memory block is the L-scale token bank (B, M, d).
                    mem_aug, _mem_aux_vf = self.memory_transformer.forward_multi(
                        primary.view(B, K, v_nq, d)
                    )
                    if _mem_aux_vf is not None:
                        backbone_output["mem_aux_loss"] = _mem_aux_vf
                else:
                    mem_seq = primary.view(B, K, v_nq, d).view(B, K * v_nq, d)
                    mem_out = self.memory_transformer(mem_seq)
                    mem_aug = mem_out[:, -v_nq:, :]
                current = backbone_features.view(B, K, Tlen, d)[:, -1, :, :]  # unchanged
                am = (
                    backbone_output["backbone_attention_mask"].view(B, K, -1)[:, -1, :]
                    if "backbone_attention_mask" in backbone_output
                    else None
                )
                im = (
                    backbone_output["image_mask"].view(B, K, -1)[:, -1, :]
                    if "image_mask" in backbone_output
                    else None
                )
                if self.mem_cond_type == "adaln":
                    backbone_output["mem_temb_add"] = self.mem_adaln_pool(mem_aug)
                else:
                    _Mv = mem_aug.shape[1]  # v_nq single-scale; L*v_nq(+extras) multi-scale
                    current = torch.cat([current, mem_aug], dim=1)
                    if am is not None:
                        am = torch.cat([am, am.new_ones(B, _Mv)], dim=1)
                    if im is not None:
                        im = torch.cat([im, im.new_zeros(B, _Mv)], dim=1)
                backbone_features = current
                if am is not None:
                    backbone_output["backbone_attention_mask"] = am
                if im is not None:
                    backbone_output["image_mask"] = im
            elif K == 1:
                vis_current = primary  # (B, 64, d)
                if self._vision_cache is None or self._vision_cache.shape[0] != B:
                    self._vision_cache = vis_current.repeat(1, K_target, 1)
                elif reset_memory is not None and reset_memory.any():
                    defaults = vis_current.repeat(1, K_target, 1)
                    shifted = torch.cat([self._vision_cache[:, v_nq:, :], vis_current], dim=1)
                    reset_b = reset_memory.view(B, 1, 1).expand(B, K_target * v_nq, d)
                    self._vision_cache = torch.where(reset_b, defaults, shifted)
                else:
                    self._vision_cache = torch.cat(
                        [self._vision_cache[:, v_nq:, :], vis_current], dim=1
                    )
                if self.memory_scales:
                    mem_aug, _ = self.memory_transformer.forward_multi(
                        self._vision_cache.view(B, K_target, v_nq, d)
                    )
                else:
                    mem_out = self.memory_transformer(self._vision_cache)
                    mem_aug = mem_out[:, -v_nq:, :]
                if self.mem_cond_type == "adaln":
                    backbone_output["mem_temb_add"] = self.mem_adaln_pool(mem_aug)
                else:
                    _Mv = mem_aug.shape[1]
                    backbone_features = torch.cat([backbone_features, mem_aug], dim=1)
                    if "backbone_attention_mask" in backbone_output:
                        am = backbone_output["backbone_attention_mask"]
                        backbone_output["backbone_attention_mask"] = torch.cat(
                            [am, am.new_ones(B, _Mv)], dim=1
                        )
                    if "image_mask" in backbone_output:
                        im = backbone_output["image_mask"]
                        backbone_output["image_mask"] = torch.cat(
                            [im, im.new_zeros(B, _Mv)], dim=1
                        )
            # else: unexpected K — pass through.
        elif (
            self.use_hamlet
            and self.memory_transformer is not None
            and "n_moment_tokens" in backbone_output
            and action_inputs_B is not None
        ):
            n_q = int(backbone_output["n_moment_tokens"])
            BK, T, d = backbone_features.shape
            B = action_inputs_B
            K = BK // B
            assert BK == B * K, f"expected B*K rows, got {BK} for B={B}"
            K_target = self.memory_transformer.T

            # mem_source='framesamp': the loader appends `mem_framesamp_frames` episode-spanning
            # frames AFTER the K-step memory-window frames, so the backbone gives K_target + F
            # rows per sample. Split them: the FIRST K_target rows are the K-window (moment
            # memory), the LAST F rows are the framesamp episode-spanning frames. F==0 (every
            # non-framesamp config) leaves the row layout exactly as before.
            # Both 'modul' (FiLM path) and 'cross_attn' (KV path) consume these episode frames
            # when mem_source=='framesamp' (experiment (d): route raw framesamp tokens through
            # the winning cross_attn KV instead of the moment-token compression).
            F = (
                self.mem_framesamp_frames
                if (
                    (
                        self.mem_cond_type in ("modul", "cross_attn")
                        and self.mem_source == "framesamp"
                    )
                    # mem_cond_type='dual': the h_spatial (AC) channel also consumes the F
                    # episode-spanning framesamp frames (when the loader supplied them).
                    or self.mem_cond_type == "dual"
                )
                else 0
            )
            fs_backbone = None  # (B, F, T, d) raw episode-spanning frames, if framesamp
            fs_image_mask = None  # (B, F, T) bool
            if F > 0 and K == K_target + F:
                feats_all = backbone_features.view(B, K, T, d)
                fs_backbone = feats_all[:, K_target:, :, :].contiguous()  # episode-spanning frames
                if "image_mask" in backbone_output:
                    fs_image_mask = backbone_output["image_mask"].view(B, K, -1)[
                        :, K_target:, :
                    ].contiguous()
                # Drop the framesamp frames from the K-window views so the rest of the path
                # operates exactly as the non-framesamp K-step training path.
                backbone_features = feats_all[:, :K_target, :, :].reshape(B * K_target, T, d)
                if "backbone_attention_mask" in backbone_output:
                    backbone_output["backbone_attention_mask"] = (
                        backbone_output["backbone_attention_mask"]
                        .view(B, K, -1)[:, :K_target, :]
                        .reshape(B * K_target, -1)
                    )
                if "image_mask" in backbone_output:
                    backbone_output["image_mask"] = (
                        backbone_output["image_mask"]
                        .view(B, K, -1)[:, :K_target, :]
                        .reshape(B * K_target, -1)
                    )
                K = K_target
                BK = B * K

            if K not in (1, K_target):
                raise RuntimeError(
                    f"HAMLET memory: got K={K} backbone rows per action sample "
                    f"(expected 1 for rolling inference or memory_window={K_target} "
                    f"for K-step training{', +%d framesamp frames' % F if F else ''}). The "
                    f"video delta_indices / memory_window data config is inconsistent - "
                    f"refusing to silently skip memory augmentation."
                )
            if K == K_target:
                # K-step training path: backbone gave K real timesteps per sample.
                moment_all = backbone_features[:, -n_q:, :].contiguous().view(B, K, n_q, d)
                if self.memory_scales:
                    # MULTI-SCALE: per-scale read-outs over suffix windows, concatenated
                    # (B, M, d) with M = L*n_q (+ uniform/DoG extras). The V2 BYOL aux loss
                    # (training only) rides backbone_output to forward()'s loss.
                    mq_augmented, _mem_aux = self.memory_transformer.forward_multi(moment_all)
                    mq_memory_out = mq_augmented  # mem_seq (modul) = the multi-scale token bank
                    if _mem_aux is not None:
                        backbone_output["mem_aux_loss"] = _mem_aux
                else:
                    mq_mem_seq = moment_all.view(B, K * n_q, d)  # oldest first -> current last
                    mq_memory_out = self.memory_transformer(mq_mem_seq)
                    mq_augmented = mq_memory_out[:, -n_q:, :]
                current = backbone_features.view(B, K, T, d)[:, -1, :, :]
                am = (
                    backbone_output["backbone_attention_mask"].view(B, K, -1)[:, -1, :]
                    if "backbone_attention_mask" in backbone_output
                    else None
                )
                im = (
                    backbone_output["image_mask"].view(B, K, -1)[:, -1, :]
                    if "image_mask" in backbone_output
                    else None
                )
                if self.mem_cond_type == "dual":
                    # HYBRID memory (h_sem + h_spatial), each through its OWN cross-attention:
                    #   h_sem (DC): the moment tokens ride the action-head KV tail EXACTLY like the
                    #               cross_attn moment path (the 18.4 baseline) — UNCHANGED.
                    #   h_spatial (AC): raw per-position framesamp tokens on a SEPARATE key
                    #               `mem_seq_spatial`, consumed by the per-block spatial_cross_attn
                    #               (residual-add, zero-init -> 0 at init). The KV / am / im lengths
                    #               stay exactly the moment path's (do NOT slice the n_q tail; do
                    #               NOT append framesamp to the KV).
                    current = torch.cat([current[:, :-n_q, :], mq_augmented], dim=1)
                    _Mm = mq_augmented.shape[1]
                    if _Mm != n_q:
                        # multi-scale: KV tail grew n_q -> M; masks must grow to match (same
                        # pattern as the framesamp M!=n_q path below).
                        if am is not None:
                            am = torch.cat([am[:, :-n_q], am.new_ones(B, _Mm)], dim=1)
                        if im is not None:
                            _tag = im.new_ones(B, _Mm) if self.mem_image_side else im.new_zeros(B, _Mm)
                            im = torch.cat([im[:, :-n_q], _tag], dim=1)
                    elif self.mem_image_side and im is not None:
                        # Route the n_q moment tail through the IMAGE cross-attn pathway (clone
                        # first: `im` is a (B, T) view sharing storage with the backbone image_mask).
                        im = im.clone()
                        im[:, -n_q:] = True
                    # h_spatial: raw per-position framesamp tokens. With episode-spanning frames
                    # available (F>0) they span the WHOLE episode; otherwise fall back to the
                    # K-window frames. _framesamp_mem_seq may return None if there are no image
                    # tokens — the DiT call site treats a missing key as "no spatial branch".
                    if fs_backbone is not None:
                        if self.mem_fs_inject != "none" and self.fs_inject_proj is not None:
                            # v2b: color each episode frame's tokens with its per-frame state
                            # (broadcast over the row; only image-masked tokens survive the
                            # gather in _framesamp_mem_seq). The F rows carry their OWN raw
                            # n_q moment tails at the row end — previously discarded.
                            fs_tails = fs_backbone[:, :, -n_q:, :]  # (B, F, n_q, d)
                            s_fs = self._fs_inject_states(fs_tails)  # (B, F, d), 0 at step-0
                            fs_backbone = fs_backbone + s_fs[:, :, None, :]
                        fs = self._framesamp_mem_seq(fs_backbone, fs_image_mask)
                    else:
                        im_full = (
                            backbone_output["image_mask"].view(B, K, -1)
                            if "image_mask" in backbone_output
                            else None
                        )
                        fs = self._framesamp_mem_seq(
                            backbone_features.view(B, K, T, d), im_full
                        )
                    if fs is not None:
                        backbone_output["mem_seq_spatial"] = fs
                elif self.mem_cond_type in ("adaln", "modul"):
                    # Memory does NOT ride the KV: slice the moment-token tail off.
                    #   adaln -> pooled memory added to temb;
                    #   modul -> action head cross-attends the full memory seq + per-layer FiLM.
                    if self.mem_cond_type == "adaln":
                        backbone_output["mem_temb_add"] = self.mem_adaln_pool(mq_augmented)
                    elif self.mem_source == "framesamp":
                        # mem_source='framesamp': cross-attend RAW per-frame vision tokens
                        # (linspace sub-sampled) instead of moment tokens. With episode-spanning
                        # frames available (F>0) they span the WHOLE episode; otherwise fall back
                        # to the K-window frames (and finally moment tokens if no image tokens).
                        _mq = backbone_features.view(B, K, T, d)[:, -1, -n_q:, :].mean(1)
                        fs = None
                        if self.mem_fs_select == "var_pyramid":
                            fs = self._var_pyramid_mem_seq(
                                backbone_output,
                                B,
                                K_target + F if fs_backbone is not None else K,
                                F if fs_backbone is not None else self.mem_framesamp_frames,
                                _mq,
                            )
                        if fs is None and fs_backbone is not None:
                            fs = (
                                self._patch_union_mem_seq(fs_backbone, fs_image_mask, _mq)
                                if self.mem_fs_select == "patch_union"
                                else self._framesamp_mem_seq(fs_backbone, fs_image_mask)
                            )
                        elif fs is None:
                            im_full = (
                                backbone_output["image_mask"].view(B, K, -1)
                                if "image_mask" in backbone_output
                                else None
                            )
                            fs = self._framesamp_mem_seq(
                                backbone_features.view(B, K, T, d), im_full
                            )
                        backbone_output["mem_seq"] = fs if fs is not None else mq_memory_out
                    else:
                        backbone_output["mem_seq"] = mq_memory_out
                    current = current[:, :-n_q, :]
                    if am is not None:
                        am = am[:, :-n_q]
                    if im is not None:
                        im = im[:, :-n_q]
                elif self.mem_source == "framesamp":
                    # cross_attn + framesamp (experiment (d)): the RAW per-frame vision tokens
                    # (linspace sub-sampled to <=budget) replace the n_q moment-token tail in the
                    # action-head KV, so the DiT cross-attends [image | text | framesamp(<=budget)]
                    # through the existing cross_attn routing — NO moment-token compression.
                    _mq = backbone_features.view(B, K, T, d)[:, -1, -n_q:, :].mean(1)
                    fs = None
                    if self.mem_fs_select == "var_pyramid":
                        fs = self._var_pyramid_mem_seq(
                            backbone_output,
                            B,
                            K_target + F if fs_backbone is not None else K,
                            F if fs_backbone is not None else self.mem_framesamp_frames,
                            _mq,
                        )
                    if fs is None and fs_backbone is not None:
                        fs = (
                            self._patch_union_mem_seq(fs_backbone, fs_image_mask, _mq)
                            if self.mem_fs_select == "patch_union"
                            else self._framesamp_mem_seq(fs_backbone, fs_image_mask)
                        )
                    elif fs is None:
                        im_full = (
                            backbone_output["image_mask"].view(B, K, -1)
                            if "image_mask" in backbone_output
                            else None
                        )
                        fs = self._framesamp_mem_seq(
                            backbone_features.view(B, K, T, d), im_full
                        )
                    if fs is None:
                        # No raw image tokens available -> fall back to the moment-augmented tail
                        # so the KV length is unchanged (degenerate config).
                        current = torch.cat([current[:, :-n_q, :], mq_augmented], dim=1)
                    else:
                        M = fs.shape[1]
                        # Drop the n_q moment tail, append the M framesamp tokens. The KV length
                        # CHANGES (n_q -> M). am/im must grow to match so the AlternateVLDiT
                        # text/image split stays aligned (it indexes them by the KV length).
                        current = torch.cat([current[:, :-n_q, :], fs], dim=1)
                        if am is not None:
                            am = torch.cat([am[:, :-n_q], am.new_ones(B, M)], dim=1)
                        if im is not None:
                            # Mark framesamp tokens as NON-image (image_mask=False) — same side as
                            # the moment tail they replace — so they ride the alternating TEXT
                            # cross-attn blocks (the mid-deep winning routing).
                            # mem_image_side=True instead tags them image_mask=True so they ride
                            # the IMAGE cross-attn pathway.
                            mem_tag = im.new_ones(B, M) if self.mem_image_side else im.new_zeros(B, M)
                            im = torch.cat([im[:, :-n_q], mem_tag], dim=1)
                else:
                    # cross_attn: memory-augmented tokens replace the moment-token tail.
                    current = torch.cat([current[:, :-n_q, :], mq_augmented], dim=1)
                    _Mm = mq_augmented.shape[1]
                    if _Mm != n_q:
                        # multi-scale: KV tail grew n_q -> M; masks must grow to match.
                        if am is not None:
                            am = torch.cat([am[:, :-n_q], am.new_ones(B, _Mm)], dim=1)
                        if im is not None:
                            _tag = im.new_ones(B, _Mm) if self.mem_image_side else im.new_zeros(B, _Mm)
                            im = torch.cat([im[:, :-n_q], _tag], dim=1)
                    elif self.mem_image_side and im is not None:
                        # Route the n_q moment tail through the IMAGE cross-attn pathway.
                        # Clone first: `im` is a (B, T) view sharing storage with the backbone
                        # image_mask, and in-place writes would corrupt it.
                        im = im.clone()
                        im[:, -n_q:] = True
                backbone_features = current
                if am is not None:
                    backbone_output["backbone_attention_mask"] = am
                if im is not None:
                    backbone_output["image_mask"] = im
            elif K == 1:
                # Inference path with rolling FIFO cache.
                # framesamp at K==1: maintain a rolling FIFO of RAW per-frame vision tokens
                # (mirrors the vision_feature `_vision_cache`, but stores the raw image-token
                # slice instead of the pooled-64 primary view) and build mem_seq from it with
                # the SAME linspace->budget logic as training. The cache lives on
                # `self._vision_cache` so the Gr00tPolicy per-session round-trip (which already
                # checkpoints `_vision_cache`) isolates it per episode with no policy change.
                use_framesamp_infer = (
                    self.mem_cond_type in ("modul", "cross_attn")
                    and self.mem_source == "framesamp"
                    and F > 0
                    and "image_mask" in backbone_output
                )
                # mem_cond_type='dual' h_spatial channel: maintain an INDEPENDENT rolling raw-vision
                # FIFO on `self._spatial_cache` (NOT `_vision_cache`/`_memory_cache`, so the moment
                # h_sem state below is never clobbered) and emit it as `mem_seq_spatial`. The moment
                # h_sem path runs unchanged at the bottom of this K==1 block (dual is not in
                # ('adaln','modul'), so it appends `mq_augmented` to the KV like cross_attn). This
                # block only SETS mem_seq_spatial — it does NOT touch backbone_features/am/im and
                # does NOT early-return.
                if (
                    self.mem_cond_type == "dual"
                    and F > 0
                    and "image_mask" in backbone_output
                ):
                    im_cur_sp = backbone_output["image_mask"].bool()  # (B, T)
                    n_img_sp = int(im_cur_sp.sum(dim=1).min().item())
                    if n_img_sp > 0:
                        rows_sp = [
                            backbone_features[b, im_cur_sp[b]][:n_img_sp, :] for b in range(B)
                        ]
                        vis_cur_sp = torch.stack(rows_sp, dim=0)  # (B, n_img, d), current frame
                        Wcache_sp = F * n_img_sp  # rolling window = F most-recent frames
                        if self._spatial_cache is None or self._spatial_cache.shape[0] != B:
                            self._spatial_cache = vis_cur_sp.repeat(1, F, 1)
                        elif reset_memory is not None and reset_memory.any():
                            defaults_sp = vis_cur_sp.repeat(1, F, 1)
                            shifted_sp = torch.cat(
                                [self._spatial_cache[:, n_img_sp:, :], vis_cur_sp], dim=1
                            )
                            reset_b_sp = reset_memory.view(B, 1, 1).expand(B, Wcache_sp, d)
                            self._spatial_cache = torch.where(reset_b_sp, defaults_sp, shifted_sp)
                        else:
                            self._spatial_cache = torch.cat(
                                [self._spatial_cache[:, n_img_sp:, :], vis_cur_sp], dim=1
                            )
                        # v2b: mirror FIFO of the per-step RAW moment tails (episode order),
                        # kept separate so the moment h_sem cache is never clobbered.
                        if self.mem_fs_inject != "none" and self.fs_inject_proj is not None:
                            tail_cur_sp = backbone_features[:, -n_q:, :]  # (B, n_q, d)
                            _Wt = self.mem_framesamp_frames * n_q
                            if (
                                self._spatial_tail_cache is None
                                or self._spatial_tail_cache.shape[0] != B
                            ):
                                self._spatial_tail_cache = tail_cur_sp.repeat(
                                    1, self.mem_framesamp_frames, 1
                                )
                            elif reset_memory is not None and reset_memory.any():
                                defaults_tl = tail_cur_sp.repeat(1, self.mem_framesamp_frames, 1)
                                shifted_tl = torch.cat(
                                    [self._spatial_tail_cache[:, n_q:, :], tail_cur_sp], dim=1
                                )
                                reset_tl = reset_memory.view(B, 1, 1).expand(B, _Wt, d)
                                self._spatial_tail_cache = torch.where(
                                    reset_tl, defaults_tl, shifted_tl
                                )
                            else:
                                self._spatial_tail_cache = torch.cat(
                                    [self._spatial_tail_cache[:, n_q:, :], tail_cur_sp], dim=1
                                )
                        # Same linspace->budget sub-sample as training (_framesamp_mem_seq).
                        # v2b: inject per-frame states into the cached frame tokens at READ time
                        # (never mutate the cache itself).
                        if self.mem_fs_inject != "none" and self.fs_inject_proj is not None:
                            _Fs = self.mem_framesamp_frames
                            tails_sp = self._spatial_tail_cache.view(B, _Fs, n_q, d)
                            s_sp = self._fs_inject_states(tails_sp)  # (B, F, d)
                            vis_sp = (
                                self._spatial_cache.view(B, _Fs, n_img_sp, d)
                                + s_sp[:, :, None, :]
                            ).reshape(B, _Fs * n_img_sp, d)
                        else:
                            vis_sp = self._spatial_cache
                        n_tok_sp = vis_sp.shape[1]
                        budget_sp = min(self.mem_framesamp_budget, n_tok_sp)
                        if budget_sp < n_tok_sp:
                            sel_sp = (
                                torch.linspace(
                                    0, n_tok_sp - 1, steps=budget_sp, device=vis_sp.device
                                )
                                .round()
                                .long()
                            )
                            sel_sp = torch.unique(sel_sp)
                            vis_sp = vis_sp[:, sel_sp, :]
                        backbone_output["mem_seq_spatial"] = vis_sp.contiguous()
                if use_framesamp_infer:
                    # Current frame's raw image tokens (moment tokens are NOT image tokens, so
                    # the image_mask already excludes them). Per-sample count is uniform (square
                    # vision grid) -> take the per-batch min for a dense (B, n_img, d) tensor.
                    im_cur = backbone_output["image_mask"].bool()  # (B, T)
                    counts = im_cur.sum(dim=1)
                    n_img = int(counts.min().item())
                    if n_img > 0:
                        rows = [backbone_features[b, im_cur[b]][:n_img, :] for b in range(B)]
                        vis_current = torch.stack(rows, dim=0)  # (B, n_img, d), current frame
                        Wcache = F * n_img  # rolling window = F most-recent frames
                        if self._vision_cache is None or self._vision_cache.shape[0] != B:
                            self._vision_cache = vis_current.repeat(1, F, 1)
                        elif reset_memory is not None and reset_memory.any():
                            defaults = vis_current.repeat(1, F, 1)
                            shifted = torch.cat(
                                [self._vision_cache[:, n_img:, :], vis_current], dim=1
                            )
                            reset_b = reset_memory.view(B, 1, 1).expand(B, Wcache, d)
                            self._vision_cache = torch.where(reset_b, defaults, shifted)
                        else:
                            self._vision_cache = torch.cat(
                                [self._vision_cache[:, n_img:, :], vis_current], dim=1
                            )
                        # Same linspace->budget sub-sample as training (_framesamp_mem_seq).
                        # mem_fs_select='diff': override the READ with TokenDrop-style
                        # pixel-diff keyframes (frame 0 + top-(F-2) diff peaks + current).
                        # The FIFO update above still runs unconditionally — it remains the
                        # per-session liveness/round-trip anchor in Gr00tPolicy, so the
                        # default path stays byte-identical and reset semantics are shared.
                        _vp_used = False
                        if self.mem_fs_select == "var_pyramid":
                            # VAR-pyramid memory, INFERENCE: read the rolling pixel FIFO
                            # (updated inside the helper; per-sample reset honored). Falls back
                            # to the raw-vision FIFO if fs_pixels is missing. The generic
                            # linspace->budget trim below is SKIPPED for pyramid tokens — the
                            # selector gates / mem_varp_budget govern their count, and an even
                            # trim would desync self._pu_positions from the kept tokens.
                            fs_vp = self._var_pyramid_mem_seq(
                                backbone_output, B, 1, F,
                                backbone_features[:, -n_q:, :].mean(1),
                                reset_memory=reset_memory,
                            )
                            if fs_vp is not None:
                                vis = fs_vp
                                _vp_used = True
                            else:
                                vis = self._vision_cache
                        elif self.mem_fs_select == "patch_union":
                            # Patch-level union memory, INFERENCE. READ the heap now (memory at
                            # step t = history < t); the current frame's patches are SCORED by
                            # the same quantity training uses — the DiT's action->patch
                            # cross-attention at mem_fs_attn_layer, captured during this very
                            # forward — and committed AFTER the action (post-action write).
                            from gr00t.model.modules.fs_patch_union import (
                                PatchUnionSelector,
                                capture_begin,
                            )

                            if self._fs_state is None or len(self._fs_state) != B:
                                self._fs_state = [None] * B
                            Lm_kv = backbone_features.shape[1] - n_q
                            rows_sel, pending, rows_pos = [], [], []
                            for b in range(B):
                                sel_b = self._fs_state[b]
                                if not isinstance(sel_b, PatchUnionSelector) or (
                                    reset_memory is not None and bool(reset_memory[b])
                                ):
                                    sel_b = PatchUnionSelector(
                                        budget=self.mem_framesamp_budget,
                                        diff_share=self.mem_fs_diff_share,
                                        stride=self.mem_fs_diff_stride,
                                        tail_share=self.mem_fs_tail_share,
                                    )
                                    self._fs_state[b] = sel_b
                                v_b = vis_current[b].detach()
                                idx_b = im_cur[b][:Lm_kv].nonzero(as_tuple=False).squeeze(1)[
                                    : v_b.shape[0]
                                ]
                                # tail_L15 channel: current frame's post-image, pre-moment
                                # summary token -> patch dot product (matches training).
                                tail_b = None
                                if self.mem_fs_tail_share > 0.0:
                                    im_row = im_cur[b][:Lm_kv]
                                    ii = im_row.nonzero(as_tuple=False).squeeze(1)
                                    if ii.numel():
                                        last = int(ii[-1]) + 1
                                        seg = backbone_features[b, last:Lm_kv]
                                        if seg.shape[0]:
                                            summ = seg.float().mean(0)
                                            tail_b = (v_b.float() @ summ).cpu()
                                if self.mem_fs_pos_rope:
                                    r_b, p_b = sel_b.read(
                                        v_b, want_pos=True,
                                        pos_frames=self.mem_framesamp_frames,
                                    )
                                    rows_pos.append(p_b)
                                else:
                                    r_b = sel_b.read(v_b)
                                rows_sel.append(r_b)
                                pending.append((sel_b, v_b, idx_b, tail_b))
                            self._pu_pending = pending
                            self._pu_positions = (torch.stack(rows_pos, dim=0)
                                                  if self.mem_fs_pos_rope and rows_pos else None)
                            n_cross = int(self.config.diffusion_model_cfg["num_layers"]) // 2
                            capture_begin(self.mem_fs_attn_layer, n_cross)
                            vis = torch.stack(rows_sel, dim=0)  # (B, budget, d)
                        elif self.mem_fs_select == "diff":
                            from gr00t.model.modules.fs_diff_select import (
                                DiffFrameSelector,
                                split_fs_pixels,
                            )

                            if self._fs_state is None or len(self._fs_state) != B:
                                self._fs_state = [None] * B
                            pix_rows = split_fs_pixels(backbone_output, B)
                            rows_sel = []
                            for b in range(B):
                                sel_b = self._fs_state[b]
                                if sel_b is None or (
                                    reset_memory is not None and bool(reset_memory[b])
                                ):
                                    sel_b = DiffFrameSelector(
                                        max_frames=F, stride=self.mem_fs_diff_stride
                                    )
                                    self._fs_state[b] = sel_b
                                sel_b.observe(
                                    vis_current[b],
                                    pix_rows[b] if pix_rows is not None else None,
                                )
                                rows_sel.append(sel_b.read(vis_current[b]))
                            vis = torch.stack(rows_sel, dim=0)  # (B, F*n_img, d)
                        else:
                            vis = self._vision_cache
                        n_tok = vis.shape[1]
                        budget = min(self.mem_framesamp_budget, n_tok)
                        if budget < n_tok and not _vp_used:
                            sel = torch.linspace(
                                0, n_tok - 1, steps=budget, device=vis.device
                            ).round().long()
                            sel = torch.unique(sel)
                            vis = vis[:, sel, :]
                        vis = vis.contiguous()
                        if self.mem_cond_type == "cross_attn":
                            # cross_attn + framesamp (experiment (d)): the framesamp tokens
                            # REPLACE the moment-token tail in the KV (mirror of training). The
                            # KV length changes (n_q -> M); grow am/im to match, marking the
                            # framesamp tokens NON-image so they ride the TEXT cross-attn blocks.
                            M = vis.shape[1]
                            backbone_features = torch.cat(
                                [backbone_features[:, :-n_q, :], vis], dim=1
                            )
                            if "backbone_attention_mask" in backbone_output:
                                am0 = backbone_output["backbone_attention_mask"]
                                backbone_output["backbone_attention_mask"] = torch.cat(
                                    [am0[:, :-n_q], am0.new_ones(B, M)], dim=1
                                )
                            im0 = backbone_output["image_mask"]
                            # mem_image_side=True tags the framesamp tokens image_mask=True so
                            # they ride the IMAGE cross-attn pathway (default False = TEXT).
                            mem_tag0 = im0.new_ones(B, M) if self.mem_image_side else im0.new_zeros(B, M)
                            backbone_output["image_mask"] = torch.cat(
                                [im0[:, :-n_q], mem_tag0], dim=1
                            )
                        else:
                            backbone_output["mem_seq"] = vis
                            # Slice the moment-token tail off the KV (modul does not ride the KV).
                            backbone_features = backbone_features[:, :-n_q, :]
                            if "backbone_attention_mask" in backbone_output:
                                backbone_output["backbone_attention_mask"] = backbone_output[
                                    "backbone_attention_mask"
                                ][:, :-n_q]
                            backbone_output["image_mask"] = backbone_output["image_mask"][:, :-n_q]
                        backbone_output["backbone_features"] = backbone_features
                        return backbone_output

                moment_current = backbone_features[:, -n_q:, :]  # (B, n_q, d)

                if self.mem_window_mode == "linspace":
                    # CAUSAL episode-spanning window: keep a growing buffer of ALL past moment
                    # read-outs, then subsample linspace(0, t-1, K_target) -> (B, K_target*n_q, d),
                    # matching the loader's training-time linspace(0, anchor, K). Conservative reset:
                    # any reset flag (or episode-boundary reset_memory()) restarts the buffer.
                    if (
                        self._moment_buffer is None
                        or self._moment_buffer.shape[0] != B
                        or (reset_memory is not None and reset_memory.any())
                    ):
                        self._moment_buffer = moment_current  # (B, n_q, d)
                    else:
                        self._moment_buffer = torch.cat([self._moment_buffer, moment_current], dim=1)
                    t = self._moment_buffer.shape[1] // n_q
                    sel = (
                        torch.linspace(0, t - 1, steps=K_target, device=moment_current.device)
                        .round()
                        .long()
                    )
                    tok_idx = (
                        sel.unsqueeze(1) * n_q
                        + torch.arange(n_q, device=moment_current.device).unsqueeze(0)
                    ).reshape(-1)
                    self._memory_cache = self._moment_buffer[:, tok_idx, :]  # (B, K_target*n_q, d)
                elif self._memory_cache is None or self._memory_cache.shape[0] != B:
                    # First call (or batch-size mismatch): K-replicate current moment-token to fill the window.
                    self._memory_cache = moment_current.repeat(1, K_target, 1)
                else:
                    if reset_memory is not None and reset_memory.any():
                        defaults = moment_current.repeat(1, K_target, 1)
                        shifted = torch.cat([self._memory_cache[:, n_q:, :], moment_current], dim=1)
                        reset_b = reset_memory.view(B, 1, 1).expand(B, K_target * n_q, d)
                        self._memory_cache = torch.where(reset_b, defaults, shifted)
                    else:
                        self._memory_cache = torch.cat(
                            [self._memory_cache[:, n_q:, :], moment_current], dim=1
                        )

                if self.memory_scales:
                    # MULTI-SCALE inference: the K_target FIFO (or linspace buffer) already holds
                    # max-scale snapshots; scales slice suffixes inside forward_multi. No aux at eval.
                    mq_augmented, _ = self.memory_transformer.forward_multi(
                        self._memory_cache.view(B, K_target, n_q, d)
                    )
                    mq_memory_out = mq_augmented
                else:
                    mq_memory_out = self.memory_transformer(self._memory_cache)
                    mq_augmented = mq_memory_out[:, -n_q:, :]
                if os.environ.get("DUAL_DEBUG"):
                    _mc = self._memory_cache
                    _nf = (_mc.shape[1] // n_q) if _mc is not None else 0
                    _std = (_mc.float().view(_mc.shape[0], _nf, n_q, _mc.shape[-1]).std(dim=1).mean().item()
                            if (_mc is not None and _nf > 1) else -1.0)
                    _rm = reset_memory.tolist() if reset_memory is not None else None
                    print(f"[DUAL_DBG] K==1 mc_frames={_nf} interframe_std={_std:.4f} "
                          f"mq_aug_norm={float(mq_augmented.float().norm()):.3f} reset={_rm}", flush=True)
                if self.mem_cond_type in ("adaln", "modul"):
                    if self.mem_cond_type == "adaln":
                        backbone_output["mem_temb_add"] = self.mem_adaln_pool(mq_augmented)
                    else:
                        # framesamp normally builds mem_seq from the rolling raw-vision FIFO
                        # above (and early-returns). We only reach here in the degenerate case
                        # mem_framesamp_frames==0 or no image tokens were present -> fall back to
                        # moment tokens for mem_seq and warn once.
                        if self.mem_source == "framesamp" and not getattr(
                            self, "_warned_framesamp_infer", False
                        ):
                            print(
                                "[HAMLET][WARN] mem_source='framesamp' could not build a raw-vision "
                                "mem_seq at K==1 inference (mem_framesamp_frames==0 or no image "
                                "tokens); using moment tokens. Set mem_framesamp_frames>0 for true "
                                "framesamp inference."
                            )
                            self._warned_framesamp_infer = True
                        backbone_output["mem_seq"] = mq_memory_out
                    backbone_features = backbone_features[:, :-n_q, :]
                    if "backbone_attention_mask" in backbone_output:
                        backbone_output["backbone_attention_mask"] = backbone_output[
                            "backbone_attention_mask"
                        ][:, :-n_q]
                    if "image_mask" in backbone_output:
                        backbone_output["image_mask"] = backbone_output["image_mask"][:, :-n_q]
                else:
                    backbone_features = torch.cat(
                        [backbone_features[:, :-n_q, :], mq_augmented], dim=1
                    )
                    _Mm = mq_augmented.shape[1]
                    if _Mm != n_q:
                        # multi-scale: KV length changed (n_q -> M); grow masks to match
                        # (mirror of the K==1 framesamp pattern above).
                        if "backbone_attention_mask" in backbone_output:
                            am0 = backbone_output["backbone_attention_mask"]
                            backbone_output["backbone_attention_mask"] = torch.cat(
                                [am0[:, :-n_q], am0.new_ones(B, _Mm)], dim=1
                            )
                        if "image_mask" in backbone_output:
                            im0 = backbone_output["image_mask"]
                            _tag = (
                                im0.new_ones(B, _Mm) if self.mem_image_side else im0.new_zeros(B, _Mm)
                            )
                            backbone_output["image_mask"] = torch.cat([im0[:, :-n_q], _tag], dim=1)
                    elif self.mem_image_side and "image_mask" in backbone_output:
                        # Route the n_q moment tail through the IMAGE cross-attn pathway.
                        # image_mask is unchanged length T in this branch; clone before the
                        # in-place write so we never mutate shared backbone storage.
                        im0 = backbone_output["image_mask"].clone()
                        im0[:, -n_q:] = True
                        backbone_output["image_mask"] = im0
            else:
                # Unexpected K — pass through.
                pass

        if os.environ.get("DUAL_NO_SPATIAL"):
            # DEBUG bisection: drop h_spatial so the DiT spatial branch is skipped
            # (mem_seq_spatial=None) -> pure moment h_sem path. Isolates spatial-corrupts vs base.
            backbone_output.pop("mem_seq_spatial", None)
        if os.environ.get("DUAL_DEBUG"):
            _am = backbone_output.get("backbone_attention_mask")
            _im = backbone_output.get("image_mask")
            print(f"[DUAL_KV] bf={tuple(backbone_features.shape)} norm={float(backbone_features.float().norm()):.1f} "
                  f"am.sum={int(_am.sum()) if _am is not None else None} "
                  f"im.sum={int(_im.sum()) if _im is not None else None} "
                  f"has_memspat={'mem_seq_spatial' in backbone_output} "
                  f"has_memseq={'mem_seq' in backbone_output}", flush=True)
        backbone_output["backbone_features"] = backbone_features
        return backbone_output

    def reset_memory(self):
        """Clear the rolling memory cache. Call at episode boundary."""
        self._memory_cache = None
        self._vision_cache = None
        self._spatial_cache = None
        self._moment_buffer = None
        self._varp_pixels = None

    def forward(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:
        """
        Forward pass through the action head.

        Args:
            backbone_output: Output from the backbone model containing:
                - backbone_features: [B, seq_len, backbone_embedding_dim]
                - backbone_attention_mask: [B, seq_len]
            action_input: Input containing:
                - state: [B, state_dim]
                - action: [B, action_horizon, action_dim] (during training)
                - embodiment_id: [B] (embodiment IDs)
                - action_mask: [B, action_horizon, action_dim]

        Returns:
            BatchFeature containing:
                - loss: action prediction loss
        """
        # Set frozen modules to eval
        self.set_frozen_modules_to_eval_mode()

        # K-step HAMLET: pass the action-input batch size so process_backbone_output can
        # collapse the B*K backbone rows back to B current-step rows after memory aggregation.
        B_target = action_input.state.shape[0]
        if self.mem_fs_select == "patch_union" and self.mem_source == "framesamp":
            # Pass A (no_grad): score every candidate patch by the DiT's own action->patch
            # cross-attention, then pass B below selects the union top-budget with it.
            self._pu_rel = self._patch_union_score_pass(backbone_output, action_input)
        backbone_output = self.process_backbone_output(backbone_output, action_inputs_B=B_target)

        # Get vision and language embeddings.
        vl_embeds = backbone_output.backbone_features
        device = vl_embeds.device

        # Get embodiment ID.
        embodiment_id = action_input.embodiment_id

        # Embed state.
        state_features = self.state_encoder(action_input.state, embodiment_id)

        # Dropout state features.
        if self.state_dropout_prob > 0:
            do_dropout = (
                torch.rand(state_features.shape[0], device=state_features.device)
                < self.state_dropout_prob
            )
            do_dropout = do_dropout[:, None, None].to(dtype=state_features.dtype)
            state_features = state_features * (1 - do_dropout) + self.mask_token * do_dropout

        # Add Gaussian noise to state features.
        if self.training and self.state_additive_noise_scale > 0:
            print(
                f"Adding Gaussian noise to state features with scale {self.state_additive_noise_scale}"
            )
            noise = torch.randn_like(state_features) * self.state_additive_noise_scale
            state_features = state_features + noise

        # Embed noised action trajectory.
        actions = action_input.action
        noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
        t = self.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)
        t = t[:, None, None]  # shape (B,1,1) for broadcast

        noisy_trajectory = (1 - t) * noise + t * actions
        velocity = actions - noise

        # Convert (continuous) t -> discrete if needed
        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()
        action_features = self.action_encoder(noisy_trajectory, t_discretized, embodiment_id)

        # Maybe add position embedding.
        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        # Join vision, language, state and action embedding along sequence dimension.
        sa_embs = torch.cat((state_features, action_features), dim=1)
        vl_attn_mask = backbone_output.backbone_attention_mask
        # AdaLN-zero HAMLET: pooled memory vector added to the DiT timestep embedding.
        mem_temb_add = backbone_output.get("mem_temb_add", None)

        self._pu_rope_on()  # PPE key RoPE on the selected memory tail (no-op unless enabled)
        try:
            if self.config.use_alternate_vl_dit:
                image_mask = backbone_output.image_mask
                backbone_attention_mask = backbone_output.backbone_attention_mask
                model_output, _ = self.model(
                    hidden_states=sa_embs,
                    encoder_hidden_states=vl_embeds,
                    encoder_attention_mask=vl_attn_mask,
                    timestep=t_discretized,
                    return_all_hidden_states=True,
                    image_mask=image_mask,
                    backbone_attention_mask=backbone_attention_mask,
                    temb_add=mem_temb_add,
                    mem_seq=backbone_output.get("mem_seq", None),
                    mem_seq_spatial=backbone_output.get("mem_seq_spatial", None),
                )
            else:
                model_output, _ = self.model(
                    hidden_states=sa_embs,
                    encoder_hidden_states=vl_embeds,
                    encoder_attention_mask=vl_attn_mask,
                    timestep=t_discretized,
                    return_all_hidden_states=True,
                    temb_add=mem_temb_add,
                    mem_seq=backbone_output.get("mem_seq", None),
                    mem_seq_spatial=backbone_output.get("mem_seq_spatial", None),
                )
        finally:
            self._pu_rope_off()

        pred = self.action_decoder(model_output, embodiment_id)
        pred_actions = pred[:, -actions.shape[1] :]

        # Slice out only the action portion of pred and target.
        action_mask = action_input.action_mask
        action_loss = F.mse_loss(pred_actions, velocity, reduction="none") * action_mask
        loss = action_loss.sum() / (action_mask.sum() + 1e-6)

        # Multi-scale V2: per-scale BYOL predictive aux loss (lambda + warmup already applied
        # inside MultiScaleMemory; present only in training K-step forwards).
        mem_aux_loss = backbone_output.get("mem_aux_loss", None)
        if mem_aux_loss is not None:
            loss = loss + mem_aux_loss

        out = {
            "loss": loss,
            "action_loss": action_loss,
            "action_mask": action_mask,
            "backbone_features": vl_embeds,
            "state_features": state_features,
        }
        if mem_aux_loss is not None:
            # only include when present — a None value breaks HF Trainer's eval nested_detach
            out["mem_aux_loss"] = mem_aux_loss
        return out

    def _encode_features(
        self,
        backbone_output: BatchFeature,
        action_input: BatchFeature,
        reset_memory: torch.Tensor | None = None,
    ) -> BatchFeature:
        """
        Encode features for the action head.
        """
        B_target = action_input.state.shape[0]
        if self.mem_fs_select == "patch_union" and self.mem_source == "framesamp":
            self._pu_rel = self._patch_union_score_pass(backbone_output, action_input)
        backbone_output = self.process_backbone_output(
            backbone_output, action_inputs_B=B_target, reset_memory=reset_memory
        )

        # Get vision and language embeddings.
        vl_embeds = backbone_output.backbone_features
        embodiment_id = action_input.embodiment_id

        # Embed state.
        state_features = self.state_encoder(action_input.state, embodiment_id)

        return BatchFeature(data={"backbone_features": vl_embeds, "state_features": state_features})

    @torch.no_grad()
    def get_action_with_features(
        self,
        backbone_features: torch.Tensor,
        state_features: torch.Tensor,
        embodiment_id: torch.Tensor,
        backbone_output: BatchFeature,
    ) -> BatchFeature:
        """
        Generate actions using the flow matching diffusion process.

        Args:
            backbone_features: [B, seq_len, backbone_embedding_dim]
            state_features: [B, state_horizon, input_embedding_dim]
            embodiment_id: [B] (embodiment IDs)
            backbone_output: Output from the backbone model
        """
        vl_embeds = backbone_features
        # AdaLN-zero HAMLET: pooled memory vector added to the DiT timestep embedding.
        mem_temb_add = backbone_output.get("mem_temb_add", None)

        # Set initial actions as the sampled noise. When the env var
        # GR00T_INFERENCE_SEED is set, draw from a persistent per-device generator
        # seeded with that value so RoboMME eval is fully deterministic and
        # reproducible across runs (the generator advances per call -> varied but
        # reproducible noise). Default (unset) = nondeterministic global RNG (unchanged).
        batch_size = vl_embeds.shape[0]
        device = vl_embeds.device
        _noise_size = (batch_size, self.config.action_horizon, self.action_dim)
        import os as _os
        _inf_seed = _os.environ.get("GR00T_INFERENCE_SEED")
        if _inf_seed is not None:
            _gen = getattr(self, "_inference_gen", None)
            if _gen is None:
                _gen = torch.Generator(device=device)
                _gen.manual_seed(int(_inf_seed))
                self._inference_gen = _gen
            actions = torch.randn(size=_noise_size, dtype=vl_embeds.dtype, device=device, generator=_gen)
        else:
            actions = torch.randn(size=_noise_size, dtype=vl_embeds.dtype, device=device)

        dt = 1.0 / self.num_inference_timesteps

        # Run denoising steps.
        for t in range(self.num_inference_timesteps):
            t_cont = t / float(self.num_inference_timesteps)  # e.g. goes 0, 1/N, 2/N, ...
            t_discretized = int(t_cont * self.num_timestep_buckets)

            # Embed noised action trajectory.
            timesteps_tensor = torch.full(
                size=(batch_size,), fill_value=t_discretized, device=device
            )
            action_features = self.action_encoder(actions, timesteps_tensor, embodiment_id)
            # Add position embedding.
            if self.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs

            # Join vision, language, state and action embedding along sequence dimension.
            sa_embs = torch.cat((state_features, action_features), dim=1)

            # Run model forward.
            self._pu_rope_on()  # PPE key RoPE on the memory tail (no-op unless enabled)
            try:
                if self.config.use_alternate_vl_dit:
                    model_output = self.model(
                        hidden_states=sa_embs,
                        encoder_hidden_states=vl_embeds,
                        timestep=timesteps_tensor,
                        image_mask=backbone_output.image_mask,
                        backbone_attention_mask=backbone_output.backbone_attention_mask,
                        temb_add=mem_temb_add,
                        mem_seq=backbone_output.get("mem_seq", None),
                        mem_seq_spatial=backbone_output.get("mem_seq_spatial", None),
                    )
                else:
                    model_output = self.model(
                        hidden_states=sa_embs,
                        encoder_hidden_states=vl_embeds,
                        timestep=timesteps_tensor,
                        temb_add=mem_temb_add,
                        mem_seq=backbone_output.get("mem_seq", None),
                        mem_seq_spatial=backbone_output.get("mem_seq_spatial", None),
                    )
            finally:
                self._pu_rope_off()
            pred = self.action_decoder(model_output, embodiment_id)

            pred_velocity = pred[:, -self.action_horizon :]

            # Update actions using euler integration.
            actions = actions + dt * pred_velocity
        return BatchFeature(
            data={
                "action_pred": actions,
                "backbone_features": vl_embeds,
                "state_features": state_features,
            }
        )

    @torch.no_grad()
    def get_action(
        self,
        backbone_output: BatchFeature,
        action_input: BatchFeature,
        options: dict[str, Any] | None = None,
    ) -> BatchFeature:
        """
        Generate actions using the flow matching diffusion process.
        """
        reset_memory = options.get("reset_memory") if options else None
        features = self._encode_features(backbone_output, action_input, reset_memory=reset_memory)
        if options and options.get("prime_only"):
            # Memory-priming call: the cache update in _encode_features is all that is
            # needed. Skip flow-matching denoising so the (seeded) action-noise RNG is
            # NOT advanced — keeps action noise call-aligned with non-primed policies
            # for strict paired comparison (and makes priming much cheaper).
            if self.mem_fs_select == "patch_union":
                from gr00t.model.modules.fs_patch_union import capture_end

                capture_end()
                self._pu_commit(None)  # no denoise ran -> novelty channel only
            ref = features.backbone_features
            zeros = torch.zeros(
                ref.shape[0], self.config.action_horizon, self.action_dim,
                device=ref.device, dtype=ref.dtype,
            )
            return BatchFeature(data={"action_pred": zeros})
        out = self.get_action_with_features(
            backbone_features=features.backbone_features,
            state_features=features.state_features,
            embodiment_id=action_input.embodiment_id,
            backbone_output=backbone_output,
        )
        if self.mem_fs_select == "patch_union":
            from gr00t.model.modules.fs_patch_union import capture_end

            self._pu_commit(capture_end())
        return out

    def _pu_commit(self, attn_scores):
        """Post-action write (inference): score the current frame's patches with the captured
        action->patch attention (same quantity as training pass A) and push them into the
        per-session heaps. Novelty is token-space, as in training."""
        pending, self._pu_pending = self._pu_pending, None
        if not pending:
            return
        for b, (sel_b, v_b, idx_b, tail_b) in enumerate(pending):
            a = None
            if attn_scores is not None and b < attn_scores.shape[0]:
                row = attn_scores[b]
                if idx_b is not None and idx_b.numel() > 0 and int(idx_b.max()) < row.shape[0]:
                    a = row[idx_b.to(row.device)].float().cpu()
            sel_b.observe(v_b, None, a, tail_b)

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype

    def prepare_input(self, batch: dict) -> BatchFeature:
        """Prepare input batch for the action head."""
        return BatchFeature(data=batch)


def get_backbone_cls(config: Gr00tN1d6Config):
    if "NVEagle" in config.model_name or "nvidia/Eagle" in config.model_name:
        return EagleBackbone
    else:
        raise ValueError(f"Unsupported model name: {config.model_name}")


class Gr00tN1d6(PreTrainedModel):
    """Gr00tN1d6: Vision-Language-Action model with backbone."""

    config_class = Gr00tN1d6Config
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: Gr00tN1d6Config,
        transformers_loading_kwargs: dict = {"trust_remote_code": True},
    ):
        """
        Initialize Gr00tN1d6 model.

        Args:
            config: Model configuration
            transformers_loading_kwargs: Dict with transformers loading parameters:
                - transformers_trust_remote_code: Whether to trust remote code when loading from HF Hub
                - transformers_local_files_only: Whether to only use local files
                - model_revision: Specific model revision to use
                - transformers_cache_dir: Directory to cache downloaded models
                - transformers_access_token: HuggingFace access token for gated models

        Note: During training, transformers parameters are passed from training config.
              During inference (e.g., from_pretrained), defaults are used.
        """
        super().__init__(config)
        self.config = config

        backbone_cls = get_backbone_cls(config)
        backbone_kwargs = dict(
            model_name=config.model_name,
            tune_llm=config.tune_llm,
            tune_visual=config.tune_visual,
            select_layer=config.select_layer,
            reproject_vision=config.reproject_vision,
            use_flash_attention=config.use_flash_attention,
            load_bf16=config.load_bf16,
            tune_top_llm_layers=config.tune_top_llm_layers,
            trainable_params_fp32=config.backbone_trainable_params_fp32,
            transformers_loading_kwargs=transformers_loading_kwargs,
        )
        if getattr(config, "hamlet_mode", "off") != "off":
            mem_type = getattr(config, "memory_type", "moment_token")
            backbone_kwargs["memory_type"] = mem_type
            if mem_type == "vision_feature":
                backbone_kwargs["n_moment_tokens"] = 0
                backbone_kwargs["freeze_moment_tokens"] = False
            else:
                backbone_kwargs["n_moment_tokens"] = config.n_moment_tokens
                backbone_kwargs["freeze_moment_tokens"] = config.freeze_moment_tokens
        self.backbone = backbone_cls(**backbone_kwargs)

        # Initialize action head (TCL mode swaps in a contrastive head).
        if getattr(config, "hamlet_mode", "off") == "tcl":
            from .tcl_head import Gr00tN1d6TCLHead
            self.action_head = Gr00tN1d6TCLHead(
                backbone_embedding_dim=config.backbone_embedding_dim,
                tcl_tau=config.tcl_tau,
                no_projection_head=getattr(config, "tcl_no_projection_head", False),
            )
            # TCL stage: freeze everything except moment_tokens and moment_to_repr.
            for p in self.backbone.parameters():
                p.requires_grad = False
            if getattr(self.backbone, "moment_tokens", None) is not None:
                self.backbone.moment_tokens.requires_grad = True
            for p in self.action_head.parameters():
                p.requires_grad = True
        else:
            self.action_head = Gr00tN1d6ActionHead(config)
        from .processing_gr00t_n1d6 import Gr00tN1d6DataCollator

        self.collator = Gr00tN1d6DataCollator(
            model_name=config.model_name,
            model_type=config.backbone_model_type,
            transformers_loading_kwargs=transformers_loading_kwargs,
        )

    def prepare_input(self, inputs: dict) -> Tuple[BatchFeature, BatchFeature]:
        """Prepare inputs for backbone and action head."""

        # NOTE -- currently the eval code doesn't use collator, so we need to add it here
        # this should ideally be fixed upstream
        if "vlm_content" in inputs:
            # Fix for n_envs > 1: Process all environments' VLM content, not just the first
            vlm_content_list = inputs["vlm_content"]
            # Ensure vlm_content_list is always a list for consistent processing
            if not isinstance(vlm_content_list, list):
                vlm_content_list = [vlm_content_list]

            # Process all VLM contents through the collator
            prep = self.collator([{"vlm_content": vlm} for vlm in vlm_content_list])["inputs"]
            inputs.pop("vlm_content")
            inputs.update(prep)

        backbone_inputs = self.backbone.prepare_input(inputs)
        action_inputs = self.action_head.prepare_input(inputs)

        # Move to device and dtype
        def to_device_with_dtype(x):
            if torch.is_floating_point(x):
                return x.to(self.device, dtype=self.dtype)
            else:
                return x.to(self.device)

        backbone_inputs = tree.map_structure(to_device_with_dtype, backbone_inputs)
        action_inputs = tree.map_structure(to_device_with_dtype, action_inputs)

        return backbone_inputs, action_inputs

    def forward(self, inputs: dict) -> BatchFeature:
        """Forward pass.
        - hamlet_mode == "tcl": run backbone 3× (anchor / aug / neg) and pass to TCL head.
        - else: standard single-pass.
        """
        if getattr(self.config, "hamlet_mode", "off") == "tcl":
            return self._forward_tcl(inputs)

        # Prepare inputs for backbone and action head
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        backbone_outputs = self.backbone(backbone_inputs)
        if (
            getattr(self.action_head, "mem_fs_select", "fifo") == "var_pyramid"
            and "pixel_values" in backbone_inputs
        ):
            # Raw pixels ([-1,1], Eagle-normalized) for the frozen VAR pyramid tokenizer.
            backbone_outputs["fs_pixels"] = backbone_inputs["pixel_values"]
        action_outputs = self.action_head(backbone_outputs, action_inputs)

        return action_outputs

    def _forward_tcl(self, inputs: dict) -> BatchFeature:
        """TCL Stage-1: split inputs into anchor / aug / neg streams, run backbone three
        times, pass the three backbone_features tensors to the TCL head for InfoNCE.
        """
        backbone_inputs, action_inputs = self.prepare_input(inputs)

        def _slice(prefix: str) -> BatchFeature:
            data = {
                "input_ids": backbone_inputs[f"{prefix}input_ids"],
                "attention_mask": backbone_inputs[f"{prefix}attention_mask"],
                "pixel_values": backbone_inputs[f"{prefix}pixel_values"],
            }
            # Forward optional fields when present (Eagle processor may include extras).
            for opt in ("image_grid_thw",):
                if f"{prefix}{opt}" in backbone_inputs:
                    data[opt] = backbone_inputs[f"{prefix}{opt}"]
            return BatchFeature(data=data)

        anc_out = self.backbone(_slice(""))
        aug_out = self.backbone(_slice("aug_"))
        neg_out = self.backbone(_slice("neg_"))
        return self.action_head(anc_out, aug_out, neg_out, action_inputs)

    def get_action(
        self, inputs: dict, options: dict[str, Any] | None = None
    ) -> BatchFeature:
        """
        Generate actions using the complete model.
        """
        # Prepare inputs for backbone and action head
        backbone_inputs, action_inputs = self.prepare_input(inputs)

        # Forward through backbone
        backbone_outputs = self.backbone(backbone_inputs)
        if (
            getattr(self.action_head, "mem_fs_select", "fifo")
            in ("diff", "patch_union", "var_pyramid")
            and "pixel_values" in backbone_inputs
        ):
            # Raw pixels for the diff-keyframe write score (fs_diff_select). Score-only:
            # never enters the computation graph or the memory tokens themselves.
            backbone_outputs["fs_pixels"] = backbone_inputs["pixel_values"]
        action_outputs = self.action_head.get_action(backbone_outputs, action_inputs, options)

        return action_outputs

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype


# Register the model with HuggingFace
AutoConfig.register("Gr00tN1d6", Gr00tN1d6Config)
AutoModel.register(Gr00tN1d6Config, Gr00tN1d6)
