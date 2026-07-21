# SPDX-License-Identifier: Apache-2.0
"""Multi-scale temporal memory (V1) + HKSL-style specialization (V2) for HAMLET read-out tokens.

Motivation (Markdown/10_multiscale_temporal_memory.md): a softmax read-out is provably low-pass, and
its attenuation is bounded by the window it pools over — short window => provably flat(ter) response.
So run the SAME read-out aggregator at several window lengths in parallel and feed all scales to the
action expert:

  V1 (this module, always on): per-scale aggregators over suffix windows of the K-step moment-token
      history + optional parameter-free UNIFORM-pooling slots (the only scale with a guaranteed
      sinc/Dirichlet response) + optional DoG tokens (m_short - m_long: the only genuinely band-pass
      channel a simplex read-out bank can emit).
  V2 (optional, training-only): per-scale BYOL predictive loss (short scale predicts near-term
      change, long scale predicts slow state) against a momentum-EMA target — the explicit signal
      that FORCES scales to specialize (window length alone does not pin the realized softmax
      spread; cf. the n_q=4 vs 128 null) — plus zero-init coarse->fine cross-attention (HKSL's
      communication manager).

I/O contract: forward_multi(moment_all (B, K, n_q, d)) -> (tokens (B, M, d), aux_loss | None) where
M = L*n_q [+ L*n_q uniform] [+ (L-1)*n_q DoG]. The caller tail-replaces the KV exactly like the
framesamp M!=n_q path (masks grown by M - n_q).

Numerical safety (bf16 lessons from seq_memory.py's NaN): bias-free Linears, zero-init comm output
(exact no-op at init), EMA held in fp32 OUTSIDE the module state (plain dict; bf16 EMA with
1-m=0.004 underflows below bf16 eps), target forward in fp32 via functional_call with autocast off.
The BYOL predictor is NOT zero-init (cosine at 0 is singular); lambda-warmup provides the safe start.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.func import functional_call
from torch.nn import functional as F

from gr00t.model.modules.memory import MemoryTransformer


def parse_int_list(spec: str) -> list[int]:
    """'2,8,16' -> [2, 8, 16]; ''/None -> []."""
    if not spec:
        return []
    return [int(tok) for tok in str(spec).replace(" ", "").split(",") if tok]


class _RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dt = x.dtype
        x32 = x.float()
        rms = x32.pow(2).mean(-1, keepdim=True).clamp_min(self.eps).rsqrt()
        return ((x32 * rms) * self.weight).to(dt)


class _ZeroInitCrossAttn(nn.Module):
    """Pre-norm multi-head cross-attention with a ZERO-INIT output projection (exact no-op at
    init, matching the repo's zero-init warm-start convention). Bias-free. Residual added by caller."""

    def __init__(self, dim: int, num_heads: int = 8, init_range: float = 0.02):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q_norm = _RMSNorm(dim)
        self.kv_norm = _RMSNorm(dim)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        for m in (self.q_proj, self.k_proj, self.v_proj):
            nn.init.normal_(m.weight, std=init_range)
        nn.init.zeros_(self.o_proj.weight)  # no-op at init

    def forward(self, q_tokens: torch.Tensor, kv_tokens: torch.Tensor) -> torch.Tensor:
        B, Lq, D = q_tokens.shape
        Lk = kv_tokens.shape[1]
        q = self.q_proj(self.q_norm(q_tokens)).view(B, Lq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(self.kv_norm(kv_tokens)).view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(self.kv_norm(kv_tokens)).view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v)  # (B, H, Lq, hd)
        out = out.transpose(1, 2).reshape(B, Lq, D)
        return self.o_proj(out)


class _Predictor(nn.Module):
    """BYOL online predictor (standard init — zero-init would make the cosine loss singular)."""

    def __init__(self, dim: int, hidden: int = 512, init_range: float = 0.02):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden, bias=False),
            _RMSNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, dim, bias=False),
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=init_range)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _make_aggregator(arch: str, dim: int, n_q: int, T: int, num_layers: int, hidden: int, n_state: int):
    if arch == "transformer":
        return MemoryTransformer(dim=dim, n_q=n_q, T=T, num_layers=num_layers)
    from gr00t.model.modules.seq_memory import SequenceMemory
    return SequenceMemory(dim=dim, n_q=n_q, T=T, num_layers=num_layers, kind=arch,
                          hidden=hidden, n_state=n_state)


class MultiScaleMemory(nn.Module):
    """Bank of read-out aggregators over suffix windows of the moment-token history.

    Args:
        dim, n_q:      backbone embedding dim / moment tokens per step.
        scales:        window lengths, ascending (e.g. [2, 8, 16]); max(scales) MUST equal the
                       loader's memory_window K (the K-step batch provides exactly K snapshots).
        arch:          aggregator kind per scale ("transformer" | "gru" | "ssm" | "mamba").
        uniform_slots: append per-scale parameter-free uniform-mean tokens (guaranteed sinc response).
        dog_tokens:    append adjacent-scale differences (short - long; genuinely band-pass).
        comm:          V2 coarse->fine communication (zero-init cross-attn, coarsest -> finest cascade).
        aux_lambda:    V2 BYOL aux-loss weight; 0 disables V2's predictive loss entirely.
        aux_horizons:  per-scale prediction horizon h_l (steps ahead). Empty -> auto:
                       h_l = min(max(1, K_l // 2), K - K_l), and skip scales with no room (K_l == K).
        aux_warmup:    linear lambda warmup steps (counted per training forward; NOT checkpointed —
                       resume re-ramps, harmless).
        aux_ema:       momentum for the fp32 EMA target encoder.
        aux_detach_moment: detach moment_all for the ONLINE aux branch too (aux shapes only the
                       memory modules, not the backbone moment tokens).
    """

    def __init__(
        self,
        dim: int,
        n_q: int,
        scales: list[int],
        num_layers: int = 2,
        arch: str = "transformer",
        hidden: int = 512,
        n_state: int = 64,
        uniform_slots: bool = False,
        dog_tokens: bool = False,
        comm: bool = False,
        aux_lambda: float = 0.0,
        aux_horizons: list[int] | None = None,
        aux_warmup: int = 2000,
        aux_ema: float = 0.996,
        aux_detach_moment: bool = False,
    ):
        super().__init__()
        assert len(scales) >= 1 and scales == sorted(scales), f"scales must be ascending, got {scales}"
        assert len(set(scales)) == len(scales), f"duplicate scales: {scales}"
        self.dim = dim
        self.n_q = n_q
        self.scales = list(scales)
        self.K = scales[-1]           # loader window; validated by the caller against memory_window
        self.T = self.K               # compat with MemoryTransformer attr surface
        self.uniform_slots = uniform_slots
        self.dog_tokens = dog_tokens
        self.aux_lambda = float(aux_lambda)
        self.aux_warmup = int(aux_warmup)
        self.aux_ema = float(aux_ema)
        self.aux_detach_moment = aux_detach_moment

        self.aggs = nn.ModuleList(
            [_make_aggregator(arch, dim, n_q, K_l, num_layers, hidden, n_state) for K_l in scales]
        )
        L = len(scales)
        self.comm = (
            nn.ModuleList([_ZeroInitCrossAttn(dim) for _ in range(L - 1)]) if (comm and L > 1) else None
        )

        # ---- V2: per-scale horizons + predictors -------------------------------------------------
        if aux_horizons:
            assert len(aux_horizons) == L, f"aux_horizons needs {L} entries, got {aux_horizons}"
            self.horizons = [int(h) for h in aux_horizons]
        else:
            self.horizons = [min(max(1, K_l // 2), self.K - K_l) for K_l in scales]
        # A scale participates in the aux loss iff h > 0 and its shifted online window fits.
        self._aux_scales = [
            l for l, (K_l, h) in enumerate(zip(scales, self.horizons)) if h > 0 and K_l + h <= self.K
        ]
        if self.aux_lambda > 0 and self._aux_scales:
            self.predictors = nn.ModuleDict(
                {str(l): _Predictor(dim, hidden) for l in self._aux_scales}
            )
        else:
            self.predictors = None
        # fp32 EMA of the per-scale aggregator params, kept OUTSIDE module state (plain dict of
        # fp32 tensors keyed [scale][param_name]) so .to(bf16) can never degrade it and the
        # checkpoint stays drop-in. Lazily initialized on first training forward.
        # NOTE: reads aggs[l].state_dict() -> correct under DDP / DeepSpeed ZeRO-1/2 (full params
        # on every rank; this repo's default is stage 2). ZeRO-3 shards params and would silently
        # EMA shards — do not use memory_aux_lambda>0 with deepspeed_stage=3.
        self._ema_state: dict[int, dict[str, torch.Tensor]] | None = None
        self._aux_step = 0  # python int; not checkpointed (warmup re-ramps on resume)

    # ------------------------------------------------------------------ V2 internals
    def _ema_init(self):
        self._ema_state = {
            l: {k: v.detach().float().clone() for k, v in self.aggs[l].state_dict().items()}
            for l in self._aux_scales
        }

    @torch.no_grad()
    def _ema_update(self):
        m = self.aux_ema
        for l in self._aux_scales:
            online = self.aggs[l].state_dict()
            for k, v in self._ema_state[l].items():
                v.mul_(m).add_(online[k].detach().float(), alpha=1.0 - m)

    def _aux_loss(self, moment_all: torch.Tensor) -> torch.Tensor | None:
        """BYOL: predict the EMA read-out at t from the online read-out at t-h (per scale)."""
        if self.predictors is None or not self.training:
            return None
        if self._ema_state is None:
            self._ema_init()
        B, K, n_q, d = moment_all.shape
        src = moment_all.detach() if self.aux_detach_moment else moment_all
        dev_type = moment_all.device.type
        losses = []
        for l in self._aux_scales:
            K_l, h = self.scales[l], self.horizons[l]
            # ONLINE branch (grad ON): window ending h steps in the past.
            sub_on = src[:, K - h - K_l : K - h].reshape(B, K_l * n_q, d)
            z_on = self.aggs[l](sub_on)[:, -n_q:, :].mean(dim=1)          # (B, d)
            pred = self.predictors[str(l)](z_on)
            # TARGET branch (grad OFF, fp32 EMA weights, autocast off): current window.
            with torch.no_grad(), torch.autocast(device_type=dev_type, enabled=False):
                sub_tg = moment_all.detach()[:, K - K_l :].reshape(B, K_l * n_q, d).float()
                y = functional_call(self.aggs[l], self._ema_state[l], (sub_tg,))[:, -n_q:, :].mean(dim=1)
            pred_n = F.normalize(pred.float(), dim=-1)
            y_n = F.normalize(y, dim=-1)
            losses.append((2.0 - 2.0 * (pred_n * y_n).sum(dim=-1)).mean())
        if not losses:
            return None
        self._aux_step += 1
        self._ema_update()
        warm = min(1.0, self._aux_step / max(1, self.aux_warmup))
        return self.aux_lambda * warm * torch.stack(losses).mean()

    # ------------------------------------------------------------------ main entry
    def forward_multi(self, moment_all: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """moment_all: (B, K, n_q, d), oldest first. Returns (tokens (B, M, d), aux_loss | None)."""
        B, K, n_q, d = moment_all.shape
        assert K == self.K and n_q == self.n_q and d == self.dim, (
            f"expected (B, {self.K}, {self.n_q}, {self.dim}), got {tuple(moment_all.shape)}"
        )
        # Per-scale read-out over each suffix window; keep the current slice (last n_q tokens).
        slices = [
            self.aggs[l](moment_all[:, -K_l:].reshape(B, K_l * n_q, d))[:, -n_q:, :]
            for l, K_l in enumerate(self.scales)
        ]
        # V2 comm: coarse -> fine cascade (finest gets context enriched through every coarser scale).
        if self.comm is not None:
            for l in range(len(slices) - 2, -1, -1):
                slices[l] = slices[l] + self.comm[l](slices[l], slices[l + 1])
        parts = list(slices)
        if self.uniform_slots:
            # Parameter-free uniform mean over each window (per token position) — the one read-out
            # whose frequency response is EXACTLY the Dirichlet kernel with cutoff ~ 1/K_l.
            parts += [moment_all[:, -K_l:].mean(dim=1) for K_l in self.scales]
        if self.dog_tokens and len(slices) > 1:
            # Difference-of-scales (short - long): zero DC gain -> genuinely band-pass tokens.
            parts += [slices[l] - slices[l + 1] for l in range(len(slices) - 1)]
        tokens = torch.cat(parts, dim=1)  # (B, M, d)
        return tokens, self._aux_loss(moment_all)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Flat-sequence compat shim: (B, K*n_q, d) -> multi-scale tokens (B, M, d)."""
        B, L, d = x.shape
        tokens, _ = self.forward_multi(x.view(B, self.K, self.n_q, d))
        return tokens

    def current_slice(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, -self.n_q :, :]
