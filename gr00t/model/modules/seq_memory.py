# SPDX-License-Identifier: Apache-2.0
"""Recurrent / state-space HAMLET memory — a drop-in replacement for MemoryTransformer.

Aggregates the K-step moment-token history with an EXPLICIT recurrent state (GRU) or state-space
model (S4D / Mamba-style selective SSM) instead of block-causal attention — making "what events
have happened" an accumulating state rather than implicit attention slots.

Same I/O as MemoryTransformer (drop-in): forward (B, T*n_q, d) oldest-first -> (B, T*n_q, d);
.T, .n_q, current_slice().

Stability (learned from a real bf16 NaN, root-caused by an in-model probe):
  (1) BIAS-FREE — every Linear is bias=False, matching the original MemoryTransformer (LLaMA-style
      RMSNorm + no biases). The probe caught `up.bias` going NaN: a bias is the term that turns a
      finite-but-huge activation into a NaN (huge·W + NaN_bias = NaN); a bias-free Linear of a finite
      input stays finite and is then bounded by the next RMSNorm.
  (2) the recurrent SCAN runs in fp32 (autocast off) — the sequential accumulation is precision-sensitive.
  (3) each SSM layer is a PRE-NORM RESIDUAL block (like the transformer's per-block norm).
  (4) RMSNorm is OVERFLOW-SAFE (factors out the per-row max before squaring) — the SSM residual stream
      was observed to transiently spike to ~1e31, which would overflow fp32 in a plain mean(x**2).
The up-projection is ZERO-INIT -> the module starts as ~identity (final_norm(x)) for a safe warm-start.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class _RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dt = x.dtype
        x32 = x.float()
        # overflow-safe: factor out the per-row max so x**2 can't overflow fp32 for huge x.
        # Mathematically identical to plain RMSNorm (the scale cancels) but numerically robust to
        # the transient ~1e31 spikes the SSM residual stream can produce.
        s = x32.abs().amax(-1, keepdim=True).clamp_min(1.0)
        xn = x32 / s
        rms = xn.pow(2).mean(-1, keepdim=True).clamp_min(self.eps).rsqrt()
        return ((xn * rms) * self.weight).to(dt)


class _S4DLayer(nn.Module):
    """Diagonal SSM (S4D / LRU core): per-channel leaky integrator, FIXED learned decay.
    Bias-free, pure transform (residual handled by the caller); fp32 scan."""

    def __init__(self, d: int, n_state: int = 64):
        super().__init__()
        self.inp = nn.Linear(d, n_state, bias=False)
        self.log_a = nn.Parameter(torch.linspace(0.5, 4.0, n_state))
        self.C = nn.Linear(n_state, d, bias=False)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dt_in = x.dtype
        with torch.autocast(device_type=x.device.type, enabled=False):
            xf = x.float()
            u = F.linear(xf, self.inp.weight.float())
            a = torch.sigmoid(self.log_a.float())          # (n_state,) in (0,1)
            B, L, N = u.shape
            h = u.new_zeros(B, N)
            ys = []
            for t in range(L):
                h = a * h + (1 - a) * u[:, t]
                ys.append(h)
            # clamp the C-projection (the unbounded GELU(C(scan)) path) before GELU: a hard guard
            # against fp32 overflow of C(scan) that would otherwise reach +-inf before any norm.
            out = self.act(F.linear(torch.stack(ys, 1), self.C.weight.float()).clamp(-1e4, 1e4))
        return out.to(dt_in)


class _MambaLayer(nn.Module):
    """Selective SSM (Mamba core): INPUT-DEPENDENT decay (content-based gating). Bias-free; fp32 scan."""

    def __init__(self, d: int, n_state: int = 64):
        super().__init__()
        self.inp = nn.Linear(d, n_state, bias=False)
        self.dt = nn.Linear(d, n_state, bias=False)
        self.C = nn.Linear(n_state, d, bias=False)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dt_in = x.dtype
        with torch.autocast(device_type=x.device.type, enabled=False):
            xf = x.float()
            u = F.linear(xf, self.inp.weight.float())
            a = torch.sigmoid(F.linear(xf, self.dt.weight.float()))   # (B,L,n_state) selective decay
            B, L, N = u.shape
            h = u.new_zeros(B, N)
            ys = []
            for t in range(L):
                h = a[:, t] * h + (1 - a[:, t]) * u[:, t]
                ys.append(h)
            out = self.act(F.linear(torch.stack(ys, 1), self.C.weight.float()).clamp(-1e4, 1e4))
        return out.to(dt_in)


class SequenceMemory(nn.Module):
    """GRU / SSM recurrent memory over the (B, T*n_q, d) moment-token sequence.

    kind: "gru" | "ssm" (S4D) | "mamba" (selective SSM); hidden: bottleneck; n_state: SSM state.
    Bias-free (matches MemoryTransformer); up is zero-init -> identity at start.
    """

    def __init__(self, dim, n_q, T, num_layers=2, kind="gru", hidden=512, n_state=64, init_range=0.02):
        super().__init__()
        self.dim = dim
        self.n_q = n_q
        self.T = T
        self.kind = kind
        self.in_norm = _RMSNorm(dim)
        self.down = nn.Linear(dim, hidden, bias=False)
        self.up = nn.Linear(hidden, dim, bias=False)
        self.final_norm = _RMSNorm(dim)
        if kind == "gru":
            self.core = nn.GRU(hidden, hidden, num_layers, batch_first=True, bias=False)
            self.norms = None
        elif kind in ("ssm", "mamba"):
            Layer = _S4DLayer if kind == "ssm" else _MambaLayer
            self.core = nn.ModuleList([Layer(hidden, n_state) for _ in range(num_layers)])
            self.norms = nn.ModuleList([_RMSNorm(hidden) for _ in range(num_layers)])  # pre-norm/layer
        else:
            raise ValueError(f"unknown SequenceMemory kind={kind!r}")
        # init-parity with the stable MemoryTransformer: all inner Linears ~ normal(std=init_range).
        # PyTorch's default kaiming init gives the SSM C-projection ~3.6x larger std, which feeds the
        # unbounded GELU(C(scan)) path; matching std=0.02 keeps early-step magnitudes small.
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=init_range)
        nn.init.zeros_(self.up.weight)            # RE-zero AFTER apply -> identity at start (safe warm-start)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T*n_q, d) oldest-first -> (B, T*n_q, d)."""
        assert x.shape[-1] == self.dim, f"expected dim {self.dim}, got {x.shape[-1]}"
        h = self.down(self.in_norm(x))
        if self.kind == "gru":
            h, _ = self.core(h)
        else:
            for layer, norm in zip(self.core, self.norms):
                h = h + layer(norm(h))            # pre-norm residual block (magnitude control)
        h = h.clamp(-1e4, 1e4)                     # hard guard: caps the residual stream before `up`
        return self.final_norm(x + self.up(h))    # up is zero-init -> starts as final_norm(x)

    def current_slice(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, -self.n_q:, :]
