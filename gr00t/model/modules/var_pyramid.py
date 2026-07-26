# SPDX-License-Identifier: Apache-2.0
"""VAR-pyramid memory: multi-coarseness patch memory with task-conditioned scale selection.

`mem_fs_select='var_pyramid'` (Markdown/var_pyramid_memory.md): instead of selecting which
BACKBONE patch tokens to store (patch_union), encode each candidate frame's RAW PIXELS with a
frozen VAR multi-scale VQVAE (FoundationVision/VAR, NeurIPS'24) into a RESIDUAL pyramid — scale s
quantizes what scales 1..s-1 missed — and let a small trainable selector pick, per frame and
conditioned on (task/observation query, frame gist), HOW DEEP into the pyramid to store.

Why VAR and not more backbone patches: the pyramid is a learned conditional code — scale s tokens
carry ONLY information absent from coarser scales, so "coarseness" is a well-posed prefix depth
(rate), not an ad-hoc subsample; and the frozen quantized codebook gives a stable memory substrate
that does not drift as the policy trains.

Structure (VAR vae, ch=160, vocab 4096, Cvae=32, /16 latent):
    scales (1,2,3,4,5,6,8[,10,13,16])   tokens/scale p_s^2   cumulative 1,5,14,30,55,91,155[,...]
    input res 128 -> latent 8x8  -> scales up to 8   (155 tokens full-depth; DEFAULT)
    input res 256 -> latent 16x16 -> scales up to 16 (680 tokens full-depth)

Selector: nested gates z_s = prod_{i<=s} sigmoid(l_i) (monotone by construction => always a
prefix), trained end-to-end through the action loss (gates scale the emitted tokens) plus an
expected-token-budget penalty. `gate_hard` switches to straight-through 0/1 gates.

I/O: forward(frames (B,F,3,H,W) in [-1,1], query (B,d)) -> (mem_seq (B,M,d), positions (B,M,3),
budget_loss|None). positions are (dt, y, x) floats on the latent grid, ready for fs_pos_rope
key-RoPE (scale-s token centers are mapped into the full-grid coordinate frame).

Vendored VAR encode-side code (Encoder/Phi/quantizer nearest-neighbour loop) is adapted from
https://github.com/FoundationVision/VAR (MIT License, Copyright (c) 2024 FoundationVision), with
attribute names kept IDENTICAL so the official `vae_ch160v4096z32.pth` checkpoint loads directly.
Decoder / VAE-training / dist code paths are dropped.
"""
from __future__ import annotations

import logging
import os

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)

VAR_FULL_PATCH_NUMS = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)


def var_patch_nums_for_res(res: int) -> tuple[int, ...]:
    """Official scale schedule truncated to the latent side of `res` (res/16)."""
    side = res // 16
    assert side in VAR_FULL_PATCH_NUMS, f"res {res} -> latent {side} not in {VAR_FULL_PATCH_NUMS}"
    assert side >= 2, f"res {res} gives a single-scale pyramid (latent {side}x{side}); use res>=32"
    return tuple(p for p in VAR_FULL_PATCH_NUMS if p <= side)


# ---------------------------------------------------------------------------- vendored VAR (MIT)
def _normalize(in_channels, num_groups=32):
    return nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


class _Downsample2x(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        return self.conv(F.pad(x, pad=(0, 1, 0, 1), mode="constant", value=0))


class _ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, dropout=0.0):
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels
        self.in_channels, self.out_channels = in_channels, out_channels
        self.norm1 = _normalize(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = _normalize(out_channels)
        self.dropout = nn.Identity()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if in_channels != out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.nin_shortcut = nn.Identity()

    def forward(self, x):
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return self.nin_shortcut(x) + h


class _AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.C = in_channels
        self.norm = _normalize(in_channels)
        self.qkv = nn.Conv2d(in_channels, 3 * in_channels, kernel_size=1, stride=1, padding=0)
        self.w_ratio = int(in_channels) ** (-0.5)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        qkv = self.qkv(self.norm(x))
        B, _, H, W = qkv.shape
        C = self.C
        q, k, v = qkv.reshape(B, 3, C, H, W).unbind(1)
        q = q.view(B, C, H * W).permute(0, 2, 1).contiguous()
        k = k.view(B, C, H * W).contiguous()
        w = torch.bmm(q, k).mul_(self.w_ratio)
        w = F.softmax(w, dim=2)
        v = v.view(B, C, H * W).contiguous()
        h = torch.bmm(v, w.permute(0, 2, 1).contiguous()).view(B, C, H, W)
        return x + self.proj_out(h)


class _Encoder(nn.Module):
    """VAR vae encoder (basic_vae.Encoder), attribute-compatible with the official ckpt."""

    def __init__(self, *, ch=160, ch_mult=(1, 1, 2, 2, 4), num_res_blocks=2, in_channels=3,
                 z_channels=32, using_sa=True, using_mid_sa=True):
        super().__init__()
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.downsample_ratio = 2 ** (self.num_resolutions - 1)
        self.conv_in = nn.Conv2d(in_channels, ch, kernel_size=3, stride=1, padding=1)
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block, attn = nn.ModuleList(), nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                block.append(_ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
                if i_level == self.num_resolutions - 1 and using_sa:
                    attn.append(_AttnBlock(block_in))
            down = nn.Module()
            down.block, down.attn = block, attn
            if i_level != self.num_resolutions - 1:
                down.downsample = _Downsample2x(block_in)
            self.down.append(down)
        self.mid = nn.Module()
        self.mid.block_1 = _ResnetBlock(in_channels=block_in)
        self.mid.attn_1 = _AttnBlock(block_in) if using_mid_sa else nn.Identity()
        self.mid.block_2 = _ResnetBlock(in_channels=block_in)
        self.norm_out = _normalize(block_in)
        self.conv_out = nn.Conv2d(block_in, z_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        h = self.conv_in(x)
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](h)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
            if i_level != self.num_resolutions - 1:
                h = self.down[i_level].downsample(h)
        h = self.mid.block_2(self.mid.attn_1(self.mid.block_1(h)))
        return self.conv_out(F.silu(self.norm_out(h)))


class _Phi(nn.Conv2d):
    def __init__(self, embed_dim, quant_resi):
        super().__init__(in_channels=embed_dim, out_channels=embed_dim, kernel_size=3, stride=1, padding=1)
        self.resi_ratio = abs(quant_resi)

    def forward(self, h_BChw):
        return h_BChw.mul(1 - self.resi_ratio) + super().forward(h_BChw).mul_(self.resi_ratio)


class _PhiPartiallyShared(nn.Module):
    def __init__(self, qresi_ls: nn.ModuleList):
        super().__init__()
        self.qresi_ls = qresi_ls
        K = len(qresi_ls)
        self.ticks = np.linspace(1 / 3 / K, 1 - 1 / 3 / K, K) if K == 4 else np.linspace(1 / 2 / K, 1 - 1 / 2 / K, K)

    def __getitem__(self, at_from_0_to_1: float) -> _Phi:
        return self.qresi_ls[np.argmin(np.abs(self.ticks - at_from_0_to_1)).item()]


class _VarQuantizerEncodeOnly(nn.Module):
    """Encode-side of VAR's VectorQuantizer2: the residual nearest-neighbour loop, emitting the
    per-scale codebook embeddings instead of reconstruction f_hats. Attribute names match the
    official ckpt (`quantize.embedding`, `quantize.quant_resi.qresi_ls.*`)."""

    def __init__(self, vocab_size=4096, Cvae=32, quant_resi=0.5, share_quant_resi=4):
        super().__init__()
        self.Cvae = Cvae
        self.embedding = nn.Embedding(vocab_size, Cvae)
        self.quant_resi = _PhiPartiallyShared(
            nn.ModuleList([_Phi(Cvae, quant_resi) for _ in range(share_quant_resi)])
        )

    @torch.no_grad()
    def encode(self, f_BChw: torch.Tensor, patch_nums, want_idx: bool = False):
        """f: (N, Cvae, G, G) fp32 -> [ (N, p_s^2, Cvae) codebook embeddings ] per scale.
        want_idx=True additionally returns the per-scale code INDICES concatenated to a single
        (N, sum p^2) int32 tensor — the lossless minimal storage form (embeddings are a pure
        codebook lookup away)."""
        N, C, H, W = f_BChw.shape
        assert patch_nums[-1] == H == W, f"patch_nums[-1]={patch_nums[-1]} must equal latent side {H}"
        f_rest = f_BChw.clone()
        SN = len(patch_nums)
        out: list[torch.Tensor] = []
        idx_parts: list[torch.Tensor] = []
        for si, pn in enumerate(patch_nums):
            z_NC = (
                F.interpolate(f_rest, size=(pn, pn), mode="area") if si != SN - 1 else f_rest
            ).permute(0, 2, 3, 1).reshape(-1, C)
            d = torch.sum(z_NC.square(), dim=1, keepdim=True) + torch.sum(
                self.embedding.weight.square(), dim=1, keepdim=False
            )
            d.addmm_(z_NC, self.embedding.weight.T, alpha=-2, beta=1)
            idx = torch.argmin(d, dim=1).view(N, pn, pn)
            if want_idx:
                idx_parts.append(idx.reshape(N, pn * pn).to(torch.int32))
            e = self.embedding(idx)                                   # (N, pn, pn, Cvae)
            out.append(e.reshape(N, pn * pn, C))
            h = e.permute(0, 3, 1, 2)
            if si != SN - 1:
                h = F.interpolate(h, size=(H, W), mode="bicubic")
            h = self.quant_resi[si / (SN - 1)](h.contiguous())
            f_rest.sub_(h)
        if want_idx:
            return out, torch.cat(idx_parts, dim=1)
        return out


class VarPyramidTokenizer(nn.Module):
    """Frozen VAR vae encode path: [-1,1] images -> per-scale residual codebook embeddings."""

    def __init__(self, res: int = 128, ch: int = 160, vocab_size: int = 4096, Cvae: int = 32,
                 var_ckpt: str = ""):
        super().__init__()
        self.res = res
        self.patch_nums = var_patch_nums_for_res(res)
        self.Cvae = Cvae
        self.encoder = _Encoder(ch=ch, z_channels=Cvae)
        self.quant_conv = nn.Conv2d(Cvae, Cvae, 3, stride=1, padding=1)
        self.quantize = _VarQuantizerEncodeOnly(vocab_size=vocab_size, Cvae=Cvae)
        # _weights_pending: var_ckpt was configured but absent on THIS machine (typical eval
        # box: the stamped training path doesn't exist there). Don't crash at construction —
        # the REAL weights arrive via the model checkpoint's load_state_dict (they're saved in
        # it), which clears the flag. If they never arrive, forward() raises loudly instead of
        # silently encoding with random weights.
        self._weights_pending = False
        if var_ckpt and os.path.isfile(var_ckpt):
            self._load_var_ckpt(var_ckpt)
        elif var_ckpt:
            logger.warning(
                f"VarPyramidTokenizer: mem_varp_ckpt not found at {var_ckpt!r} — expecting the "
                "tokenizer weights from the MODEL checkpoint load; forward() raises if none "
                "arrives."
            )
            self._weights_pending = True
        else:
            logger.warning(
                "VarPyramidTokenizer: no var_ckpt given — RANDOM VAR weights (tests only; "
                "set mem_varp_ckpt to the official vae_ch160v4096z32.pth for training)."
            )
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)
        # EAGER fp32 shadow of the frozen weights, kept OUTSIDE module state (plain dict, same
        # convention as multiscale_memory's EMA). DeepSpeed bf16 casts frozen params too, and
        # the residual-quantization loop is precision-sensitive (deep scales quantize ever-
        # smaller residuals that bf16 rounding would drown). Built NOW — while the loaded
        # weights are still full-precision — and only MOVED (never rebuilt) later, so a cast
        # can never degrade it. ~2x the 44M encode-side params in RAM; negligible on A100.
        self._fp32_sd: dict[str, torch.Tensor] = {
            k: v.detach().float().clone() for k, v in self.state_dict().items()
        }
        # The shadow must FOLLOW checkpoint loads: at eval, __init__ may build from random
        # weights (mem_varp_ckpt path absent on the eval machine) and the real weights arrive
        # via the MODEL checkpoint's load_state_dict — without a refresh, a bf16-cast forward
        # would route through the stale random shadow (silent garbage). The post-hook covers
        # torch load_state_dict recursion; a fingerprint check in forward() backstops HF/
        # DeepSpeed loaders that bypass module post-hooks.
        self.register_load_state_dict_post_hook(self._refresh_fp32_shadow)
        # Weight fingerprint as of end-of-init; a post-load change beyond bf16 rounding means
        # real weights arrived (clears _weights_pending in the post-hook).
        self._init_fingerprint = self.quant_conv.weight.detach().float().flatten()[:64].clone()

    @staticmethod
    def _refresh_fp32_shadow(module, incompatible_keys=None):
        module._fp32_sd = {
            k: v.detach().float().clone() for k, v in module.state_dict().items()
        }
        if getattr(module, "_weights_pending", False) and hasattr(module, "_init_fingerprint"):
            w = module.quant_conv.weight.detach().float().flatten()[:64]
            fp = module._init_fingerprint.to(w.device)
            if not torch.allclose(w, fp, atol=2e-2, rtol=5e-2):
                module._weights_pending = False

    def _load_var_ckpt(self, path: str):
        sd = torch.load(path, map_location="cpu")
        sd = sd.get("state_dict", sd)
        drop = ("decoder.", "post_quant_conv.", "quantize.ema_vocab_hit_SV")
        sd = {k: v for k, v in sd.items() if not k.startswith(drop)}
        missing, unexpected = self.load_state_dict(sd, strict=False)
        assert not missing, f"VAR ckpt missing keys: {missing[:8]}"
        assert not unexpected, f"VAR ckpt unexpected keys: {unexpected[:8]}"
        self._weights_pending = False  # real VAR weights are in place
        logger.info(f"VarPyramidTokenizer: loaded VAR vae from {path} ({len(sd)} tensors)")

    @torch.no_grad()
    def forward(self, imgs: torch.Tensor, want_idx: bool = False):
        """imgs: (N, 3, H, W) in [-1, 1] (any H,W; resized to self.res). Returns per-scale
        (N, p_s^2, Cvae) fp32 embeddings; with want_idx=True, (embs, (N, sum p^2) int32 codes).
        fp32 + autocast off: frozen VAR is precision-sensitive and cheap vs the backbone."""
        if self._weights_pending:
            raise RuntimeError(
                "VarPyramidTokenizer: encoding with RANDOM weights — mem_varp_ckpt was set but "
                "missing at init, and no checkpoint load supplied action_head.var_pyramid.* "
                "weights. Fix the path, or load a model checkpoint that contains them."
            )
        w = self.quant_conv.weight
        if w.dtype != torch.float32:
            # Weights were cast (e.g. DeepSpeed bf16 casts the whole module, frozen included).
            # Forward through the eager fp32 shadow via functional_call: the re-entered forward
            # sees fp32 weights and takes the normal branch below (single bounded re-entry).
            # The shadow predates the cast, so full precision is preserved; move-only on device
            # change (rebuilding from the live weights would bake in the bf16 rounding).
            ref = self._fp32_sd["quant_conv.weight"]
            if ref.device != w.device:
                self._fp32_sd = {k: v.to(w.device) for k, v in self._fp32_sd.items()}
                ref = self._fp32_sd["quant_conv.weight"]
            # Staleness backstop (loaders that bypass module post-hooks): live weights should
            # be the bf16 rounding of the shadow; a larger gap means a checkpoint replaced the
            # weights after the shadow was built -> rebuild from live (rounded, best available).
            if not torch.allclose(
                ref.flatten()[:64], w.detach().float().flatten()[:64], atol=2e-2, rtol=5e-2
            ):
                logger.warning(
                    "VarPyramidTokenizer: fp32 shadow stale vs live weights (checkpoint loaded "
                    "after cast?) — rebuilding from the live cast weights."
                )
                self._refresh_fp32_shadow(self)
            from torch.func import functional_call

            return functional_call(self, self._fp32_sd, (imgs,), {"want_idx": want_idx})
        with torch.autocast(device_type=imgs.device.type, enabled=False):
            x = imgs.float()
            if x.shape[-2:] != (self.res, self.res):
                x = F.interpolate(x, size=(self.res, self.res), mode="bicubic", align_corners=False)
            f = self.quant_conv(self.encoder(x))
            return self.quantize.encode(f, self.patch_nums, want_idx=want_idx)

    def _codebook_fp32(self) -> torch.Tensor:
        """The full-precision codebook, regardless of module casts (bf16 lookups would perturb
        the stored-code round-trip; the fp32 shadow predates any cast)."""
        w = self.quantize.embedding.weight
        if w.dtype != torch.float32:
            return self._fp32_sd["quantize.embedding.weight"].to(w.device)
        return w

    @torch.no_grad()
    def indices_to_embs(self, idx_NP: torch.Tensor) -> list[torch.Tensor]:
        """Inverse of the storage form: (N, sum p^2) int codes -> per-scale (N, p^2, Cvae)
        fp32 embeddings, bit-equal to what forward() would emit for the same frames."""
        book = self._codebook_fp32()
        out, off = [], 0
        for p in self.patch_nums:
            n = p * p
            out.append(F.embedding(idx_NP[:, off : off + n].long(), book))
            off += n
        return out


# ---------------------------------------------------------------------------- trainable side
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


class CoarsenessSelector(nn.Module):
    """Per-frame nested prefix gates over the S pyramid scales.

    Inputs: query (B, d) — the current step's moment-token mean (the model's own task+observation
    summary, same relevance query as patch_union) — and per-frame gist (B, F, gist_dim) pooled
    from the coarsest `gist_scales` pyramid levels. Output z (B, F, S): z_s = prod_{i<=s}
    sigmoid(l_i / temp), monotonically non-increasing in s, so the kept set is ALWAYS a pyramid
    prefix (VAR tokens are residuals — a deep scale without its base is meaningless). The
    +init_bias logit bias starts every gate near 1 (keep-everything).

    Cross-frame mixing: a purely additive per-frame h can express "this frame matters for this
    task" but never "this frame is redundant given its neighbour" — arguably THE allocation
    judgement for a memory. One self-attention layer over the F frame embeddings (zero-init
    output projection => exact no-op at init) lets the budget express "spend here, not there"."""

    def __init__(self, d_query: int, n_scales: int, gist_dim: int = 64, hidden: int = 256,
                 temp: float = 1.0, init_bias: float = 2.0):
        super().__init__()
        self.temp = float(temp)
        self.init_bias = float(init_bias)
        self.q_proj = nn.Linear(d_query, hidden, bias=False)
        self.g_proj = nn.Linear(gist_dim, hidden, bias=False)
        self.fa_norm = _RMSNorm(hidden)
        self.fa_q = nn.Linear(hidden, hidden, bias=False)
        self.fa_k = nn.Linear(hidden, hidden, bias=False)
        self.fa_v = nn.Linear(hidden, hidden, bias=False)
        self.fa_o = nn.Linear(hidden, hidden, bias=False)
        self.mlp = nn.Sequential(_RMSNorm(hidden), nn.GELU(), nn.Linear(hidden, n_scales, bias=True))
        for m in (self.q_proj, self.g_proj, self.mlp[-1]):
            nn.init.normal_(m.weight, std=0.02)
        # The no-op guarantee lives ENTIRELY in fa_o = 0, so q/k/v cost nothing to init at full
        # strength (1/sqrt(hidden)). At std 0.02 the liftoff would be second-order twice over:
        # fa_q/k/v get exactly zero gradient until fa_o moves, AND near-zero qk logits make the
        # softmax uniform, so attn@v ≈ the frame-MEAN of v — a frame-independent vector that
        # cannot differentiate even with a healthy |fa_o|.
        for m in (self.fa_q, self.fa_k, self.fa_v):
            nn.init.normal_(m.weight, std=hidden ** -0.5)
        nn.init.zeros_(self.fa_o.weight)  # cross-frame layer is an exact no-op at init
        # Diagnostics only (detached, overwritten per forward): near-uniform rows here mean the
        # branch is "connected but not reasoning" — attention entropy is the statistic that
        # distinguishes the two, not |fa_o|.
        self.last_attn: torch.Tensor | None = None
        # positive bias -> sigmoid ~ 0.88/gate at init -> starts near "keep everything".
        nn.init.constant_(self.mlp[-1].bias, init_bias)

    def forward(self, query_Bd: torch.Tensor, gist_BFg: torch.Tensor) -> torch.Tensor:
        # Inputs follow the module's (possibly bf16-cast) weight dtype; attention softmax and
        # the gate nonlinearity run in fp32 (saturation + cumprod are precision-sensitive).
        wd = self.q_proj.weight.dtype
        h = self.q_proj(query_Bd.to(wd)).unsqueeze(1) + self.g_proj(gist_BFg.to(wd))  # (B,F,H)
        hn = self.fa_norm(h)
        q, k, v = self.fa_q(hn), self.fa_k(hn), self.fa_v(hn)
        attn = torch.softmax(
            q.float() @ k.float().transpose(1, 2) / (q.shape[-1] ** 0.5), dim=-1
        )
        self.last_attn = attn.detach()
        h = h + self.fa_o(attn.to(wd) @ v)
        logits = self.mlp(h).float() / self.temp             # (B, F, S)
        return torch.sigmoid(logits).cumprod(dim=-1)         # nested prefix gates, fp32


class VarpCodeStore:
    """Full-episode per-session store of VAR code INDICES (~2 bytes x 155/frame at res128).

    Storing codes instead of pixels makes retention trivial — the whole episode fits in ~10s of
    KB — so frame selection becomes a READ-time decision that replays the TRAINING keyframe
    indexer (`_fs_diff_indices`: frame-0 + top pixel-diff peaks <= anchor + anchor) EXACTLY,
    instead of the recency-FIFO's 112-step bound and greedy approximations. Each frame is
    VAR-encoded exactly once (at observe); reads are index gathers + codebook lookups.

    Plain CPU object: round-trips per session through the policy's `_fs_state` channel like
    DiffFrameSelector/PatchUnionSelector; a reset slot arrives as None -> fresh store.
    """

    def __init__(self, patch_nums, sig_side: int = 32):
        self.patch_nums = tuple(patch_nums)
        self.sig_side = int(sig_side)
        self.codes: list[torch.Tensor] = []   # per call: (sum p^2,) int16 CPU
        self.scores: list[float] = []         # pixel-diff vs previous call (frame-0 = inf)
        self.last_sig: torch.Tensor | None = None
        self.t = -1

    def observe(self, idx_P: torch.Tensor, frame_3HW: torch.Tensor) -> None:
        """Advance one policy call: store this frame's codes + its novelty score. The score is
        computed on a small pixel signature (area-pooled), mirroring the training scorer's
        pixel-diff up to resolution; only the PREVIOUS signature is retained, never frames."""
        self.t += 1
        sig = F.interpolate(
            frame_3HW.detach().float().unsqueeze(0), size=(self.sig_side, self.sig_side),
            mode="area",
        )[0].cpu()
        self.scores.append(
            float("inf") if self.last_sig is None else float((sig - self.last_sig).abs().mean())
        )
        self.last_sig = sig
        self.codes.append(idx_P.detach().to(torch.int16).cpu())

    def select(self, n_frames: int) -> list[int]:
        """Replay the training keyframe indexer over the full stored history."""
        import numpy as np

        from gr00t.data.dataset.sharded_single_step_dataset import _fs_diff_indices

        steps = np.arange(self.t + 1)
        scores = np.array(self.scores, dtype=np.float64)
        return _fs_diff_indices(steps, scores, anchor=self.t, fs_frames=n_frames)

    def gather(self, sel: list[int]) -> torch.Tensor:
        return torch.stack([self.codes[i] for i in sel])  # (F, sum p^2) int16 CPU


class VarPyramidMemory(nn.Module):
    """frames + query -> gated multi-coarseness mem_seq for the DiT memory cross-attention.

    Args:
        dim:            backbone embedding dim (mem_seq token dim).
        res:            VAR encode resolution (128 -> scales to 8x8, 155 tok/frame max).
        max_frames:     age-embedding table size (>= F candidate frames).
        budget:         hard cap on emitted tokens (top-`budget` by gate, temporal order kept);
                        0 = emit all F*sum(p^2) gated tokens.
        gate_hard:      straight-through 0/1 gates (train/deploy-consistent hard selection).
        budget_lambda:  weight of the expected-token-fraction penalty; 0 disables.
        target_frac:    if >0, penalize (frac - target)^2 instead of plain frac.
        var_ckpt:       official VAR vae checkpoint path ('' = random, tests only).
    """

    def __init__(self, dim: int, res: int = 128, max_frames: int = 32, budget: int = 0,
                 gate_hard: bool = False, budget_lambda: float = 0.0, target_frac: float = 0.0,
                 gist_scales: int = 4, selector_hidden: int = 256, gate_temp: float = 1.0,
                 var_ckpt: str = ""):
        super().__init__()
        self.tokenizer = VarPyramidTokenizer(res=res, var_ckpt=var_ckpt)
        pn = self.tokenizer.patch_nums
        self.patch_nums = pn
        S = len(pn)
        self.budget = int(budget)
        self.gate_hard = gate_hard
        self.budget_lambda = float(budget_lambda)
        self.target_frac = float(target_frac)
        Cvae = self.tokenizer.Cvae
        self.dim = dim
        self.proj = nn.Linear(Cvae, dim, bias=False)
        nn.init.normal_(self.proj.weight, std=0.02)
        self.scale_emb = nn.Parameter(0.02 * torch.randn(S, dim))
        self.age_emb = nn.Parameter(0.02 * torch.randn(max_frames, dim))
        self.norm = _RMSNorm(dim)
        # Gist = mean ⊕ channel-wise max over each of the `gist_scales` coarsest levels. The
        # conditioning input must CONTAIN what the selector is asked to buy: object identity
        # only enters the code around mid-depth (ladder figure — the red cube appears at 6x6),
        # so a scale-1/2 gist provably cannot drive "this frame holds the cube" decisions. The
        # channel-max matters at the deeper gist levels: a spatial mean dilutes a one-cell
        # object 1/p^2, while max keeps "some cell fired this code direction".
        self.gist_scales = max(1, min(int(gist_scales), S))
        # mean and amax coincide elementwise on a single-token level (p^2 == 1) — skip the
        # duplicate block rather than carry 32 identical dims.
        gist_dim = sum(Cvae * (1 if p * p == 1 else 2) for p in pn[: self.gist_scales])
        self.selector = CoarsenessSelector(dim, S, gist_dim=gist_dim, hidden=selector_hidden,
                                           temp=gate_temp)

        # static per-token (scale_idx, y, x) tables for one frame, latent-grid coords: a scale-s
        # token covers a (G/p_s)^2 cell; its center in full-grid coords feeds fs_pos_rope.
        G = pn[-1]
        scale_ids, ys, xs = [], [], []
        for si, p in enumerate(pn):
            iy, ix = torch.meshgrid(torch.arange(p), torch.arange(p), indexing="ij")
            ys.append((iy.reshape(-1).float() + 0.5) * (G / p) - 0.5)
            xs.append((ix.reshape(-1).float() + 0.5) * (G / p) - 0.5)
            scale_ids.append(torch.full((p * p,), si, dtype=torch.long))
        self.register_buffer("tok_scale", torch.cat(scale_ids), persistent=False)   # (P,)
        self.register_buffer("tok_y", torch.cat(ys), persistent=False)              # (P,)
        self.register_buffer("tok_x", torch.cat(xs), persistent=False)              # (P,)
        self.tokens_per_frame = int(self.tok_scale.numel())                         # P = sum p^2
        costs = torch.tensor([p * p for p in pn], dtype=torch.float32)
        self.register_buffer("scale_cost", costs / costs.sum(), persistent=False)   # (S,)

    def reinit_missing_from_ckpt(self, var_ckpt: str = ""):
        """setup.py hook for when this module's keys were MISSING from the model checkpoint —
        HF fast-init can leave missing params as torch.empty garbage (see the hamlet missing-key
        re-init block). Re-initializes the trainable side exactly like __init__ and re-loads the
        frozen VAR vae (idempotent when the in-memory weights are already correct)."""
        with torch.no_grad():
            nn.init.normal_(self.proj.weight, std=0.02)
            self.scale_emb.data.normal_(mean=0.0, std=0.02)
            self.age_emb.data.normal_(mean=0.0, std=0.02)
            self.norm.weight.data.fill_(1.0)
            for m in self.selector.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=0.02)
                    if m.bias is not None:
                        m.bias.data.zero_()
                elif isinstance(m, _RMSNorm):
                    m.weight.data.fill_(1.0)
            sel = self.selector
            for m in (sel.fa_q, sel.fa_k, sel.fa_v):  # full-strength attn init (see __init__)
                nn.init.normal_(m.weight, std=m.weight.shape[1] ** -0.5)
            nn.init.zeros_(sel.fa_o.weight)  # cross-frame layer: no-op at (re)init
            nn.init.constant_(sel.mlp[-1].bias, sel.init_bias)
        tk = self.tokenizer
        if var_ckpt and os.path.isfile(var_ckpt):
            tk._load_var_ckpt(var_ckpt)
        elif var_ckpt:
            logger.warning(
                f"var_pyramid reinit: mem_varp_ckpt not found at {var_ckpt!r} — tokenizer "
                "weights may be garbage; forward() will raise until real weights are loaded."
            )
            tk._weights_pending = True
        tk._refresh_fp32_shadow(tk)

    def _gist(self, embs: list[torch.Tensor], B: int, Fn: int) -> torch.Tensor:
        """Per-frame conditioning gist: [mean ⊕ channel-max] over each of the coarsest
        `gist_scales` levels (single block for p^2==1 levels, where the two coincide), fp32."""
        parts = []
        for s in range(self.gist_scales):
            e = embs[s].reshape(B, Fn, -1, embs[s].shape[-1])
            parts.append(e.mean(dim=2))
            if e.shape[2] > 1:
                parts.append(e.amax(dim=2))
        return torch.cat(parts, dim=-1)

    def forward(
        self, frames_BF3HW: torch.Tensor, query_Bd: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """frames: (B, F, 3, H, W) in [-1,1], oldest first. query: (B, dim).
        Returns (mem_seq (B, M, dim), positions (B, M, 3) float (dt,y,x), budget_loss|None)."""
        B, Fn, _, H, W = frames_BF3HW.shape
        assert Fn <= self.age_emb.shape[0], f"F={Fn} exceeds max_frames={self.age_emb.shape[0]}"
        embs = self.tokenizer(frames_BF3HW.reshape(B * Fn, 3, H, W))  # [(B*F, p^2, Cvae)] fp32
        return self._forward_embs(embs, query_Bd, B, Fn)

    def forward_from_indices(
        self, idx_BFP: torch.Tensor, query_Bd: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Code-native read path: (B, F, sum p^2) int code indices -> the SAME outputs as
        forward() would produce for the frames those codes came from (codebook lookup is
        bit-exact) — the action head reads VAR codes, so storing indices is lossless."""
        B, Fn, P = idx_BFP.shape
        assert P == self.tokens_per_frame, f"expected {self.tokens_per_frame} codes/frame, got {P}"
        assert Fn <= self.age_emb.shape[0], f"F={Fn} exceeds max_frames={self.age_emb.shape[0]}"
        embs = self.tokenizer.indices_to_embs(idx_BFP.reshape(B * Fn, P))
        return self._forward_embs(embs, query_Bd, B, Fn)

    def _forward_embs(
        self, embs: list[torch.Tensor], query_Bd: torch.Tensor, B: int, Fn: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        gist = self._gist(embs, B, Fn)
        z = self.selector(query_Bd, gist)                                  # (B, F, S) fp32
        if self.gate_hard:
            z_h = (z > 0.5).float()
            z_g = z_h + z - z.detach()          # straight-through: hard forward, soft backward
        else:
            z_g = z

        dt = self.proj.weight.dtype
        toks, gates = [], []
        for si, e in enumerate(embs):
            t = self.proj(e.to(dt)).view(B, Fn, -1, self.dim)
            t = t + self.scale_emb[si].view(1, 1, 1, -1) + self.age_emb[:Fn].view(1, Fn, 1, -1)
            toks.append(self.norm(t))                                     # (B, F, p^2, dim)
            gates.append(z_g[:, :, si : si + 1].expand(-1, -1, t.shape[2]))
        tok = torch.cat(toks, dim=2)                                      # (B, F, P, dim)
        gate = torch.cat(gates, dim=2)                                    # (B, F, P) fp32
        mem = tok * gate.unsqueeze(-1).to(dt)

        # positions (dt, y, x): dt = recency rank (0 = most recent, matches fs_pos_rope).
        dev = embs[0].device
        age = torch.arange(Fn - 1, -1, -1, device=dev, dtype=torch.float32)          # (F,)
        pos = torch.stack(
            [
                age.view(Fn, 1).expand(Fn, self.tokens_per_frame),
                # .float(): non-persistent buffers follow module casts (bf16 under DeepSpeed);
                # positions must stay fp32 for stack-dtype consistency and RoPE angles.
                self.tok_y.float().view(1, -1).expand(Fn, -1),
                self.tok_x.float().view(1, -1).expand(Fn, -1),
            ],
            dim=-1,
        ).view(1, Fn * self.tokens_per_frame, 3).expand(B, -1, -1)                   # (B, F*P, 3)

        mem = mem.view(B, Fn * self.tokens_per_frame, -1)
        gate_flat = gate.view(B, Fn * self.tokens_per_frame)
        if self.budget and self.budget < mem.shape[1]:
            # Hard cap: top-`budget` by gate value. Ties (exact 1.0s under gate_hard) break
            # COARSE-scale-first then MOST-RECENT-first — every frame's pyramid base survives
            # before anyone's deep detail, and detail drops from the oldest frames first.
            # (A flat static-order penalty would keep the oldest frames whole and drop the
            # most recent frames entirely — the worst choice for a memory.)
            scale_pen = self.tok_scale.float().repeat(Fn) * 1e-4              # (F*P,)
            age_pen = age.repeat_interleave(self.tokens_per_frame) * 1e-6     # (F*P,) dt=0 recent
            keep = (
                torch.topk(gate_flat - scale_pen - age_pen, self.budget, dim=1)
                .indices.sort(dim=1).values
            )
            mem = torch.gather(mem, 1, keep.unsqueeze(-1).expand(-1, -1, mem.shape[-1]))
            pos = torch.gather(pos.contiguous(), 1, keep.unsqueeze(-1).expand(-1, -1, 3))

        aux = None
        if self.budget_lambda > 0 and self.training:
            # The tokenizer is no-grad, so the gist carries no gradient — query_Bd is the ONLY
            # route by which the budget penalty could reach the backbone, and it would teach the
            # moment tokens to look like "tasks that need less memory". Recompute the gates from
            # a DETACHED query for the penalty: forward-identical values, identical gradient to
            # the selector's own parameters, exactly zero to the backbone. The action-loss path
            # through z (into the query) stays fully intact.
            z_aux = self.selector(query_Bd.detach(), gist)
            frac = (z_aux * self.scale_cost.float().view(1, 1, -1)).sum(-1).mean()  # E[frac]
            tgt = self.target_frac
            if tgt <= 0 and self.budget:
                # Under a hard cap, top-k already drops the deep tokens, so no action-loss
                # gradient reaches their gates — a plain linear penalty would push them to zero
                # UNOPPOSED (irreversible collapse of the deep scales). Anchor the penalty at
                # the cap's own operating point instead: stationary there, and it pushes gates
                # back UP when the soft allocation falls below what the cap can afford.
                tgt = min(1.0, self.budget / float(Fn * self.tokens_per_frame))
            if tgt > 0:
                aux = self.budget_lambda * (frac - tgt).pow(2)
            else:
                # No hard cap (soft mode): every token keeps an action-loss gradient to oppose
                # the pressure, so the plain linear rate penalty is meaningful here.
                aux = self.budget_lambda * frac
        return mem, pos, aux
