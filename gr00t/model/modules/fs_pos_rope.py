"""PPE-style 3D position RoPE for patch memory (`mem_fs_pos_rope=True`).

Motivation (Markdown/23_patch_memory_grounding.md): a dual linear probe showed the stored
patch tokens ALREADY encode (view,y,x) almost perfectly (1.00/0.99/0.95) and coarse time,
BUT the DiT memory cross-attention has NO RoPE, so its QK dot product cannot USE that
position — "decodable by a probe" != "usable by attention". PPE (arXiv:2510.22936) fixes the
analogous gap for merged tokens by restoring their original RoPE indices.

Two adaptations vs PPE:
  * PPE lives in Qwen SELF-attention where every token (incl. text) carries an mRoPE position.
    Our action queries are separate diffusion tokens with NO spatial position, so we rotate
    the KEYS only (memory patches) and leave the queries at identity. <q, R(pos) k> then
    exposes each key's absolute (Δt,y,x) to the dot product — exactly the missing signal.
  * PPE assigns K>1 position ids per MERGED token (top-K cluster members). We SELECT (never
    merge), so every stored patch has ONE true (Δt,y,x,view) => plain 3D RoPE, K=1.

Coordinate scheme per stored patch:
    Δt   recency rank of its candidate frame (0 = most recent), in [0, F-1]
    y    row in its view's SxS grid, with the WRIST view folded to y+S so the two cameras
         separate in the y band; range [0, 2S-1]
    x    col in [0, S-1]
The head_dim/2 rotary frequencies are split into three contiguous bands (Δt, y, x), mRoPE
style. view is folded into y (front: y, wrist: y+S) rather than given its own band.
"""

from __future__ import annotations

import torch


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def build_cos_sin(pos_BM3: torch.Tensor, head_dim: int, bands=(0.34, 0.33, 0.33),
                  base: float = 10000.0) -> tuple[torch.Tensor, torch.Tensor]:
    """pos_BM3: (B, M, 3) int/float positions (Δt, y, x). Returns cos,sin (B, M, head_dim)
    ready to rotate a (B, heads, M, head_dim) key via k*cos + rotate_half(k)*sin."""
    device = pos_BM3.device
    half = head_dim // 2
    # split `half` frequency slots into 3 contiguous bands
    n = [max(1, int(round(b * half))) for b in bands]
    n[-1] = half - n[0] - n[1]                       # make them sum to `half`
    freqs = []
    for d, nd in enumerate(n):
        inv = base ** (-torch.arange(0, nd, device=device, dtype=torch.float32) / max(1, nd))
        freqs.append(pos_BM3[..., d].float().unsqueeze(-1) * inv)   # (B, M, nd)
    ang = torch.cat(freqs, dim=-1)                   # (B, M, half)
    ang = torch.cat([ang, ang], dim=-1)              # (B, M, head_dim) — mirror for rotate_half
    return ang.cos(), ang.sin()


def rotate_keys(k_BHMd: torch.Tensor, cos_BMd: torch.Tensor, sin_BMd: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to keys only. k: (B, heads, M, head_dim); cos/sin: (B, M, head_dim)."""
    cos = cos_BMd.unsqueeze(1).to(k_BHMd.dtype)      # (B, 1, M, head_dim)
    sin = sin_BMd.unsqueeze(1).to(k_BHMd.dtype)
    return k_BHMd * cos + _rotate_half(k_BHMd) * sin


def positions_from_flat(keep_idx_BM: torch.Tensor, n_img: int, n_views: int = 2) -> torch.Tensor:
    """Recover (Δt, y, x) for each selected flat index into F*n_img.
    keep_idx: (B, M) into F*n_img (frame f = idx // n_img, patch i = idx % n_img).
    Returns (B, M, 3): Δt = recency rank (max_f - f), y (wrist folded +side), x."""
    f = keep_idx_BM // n_img
    i = keep_idx_BM % n_img
    per_view = n_img // n_views
    side = int(round(per_view ** 0.5))
    view = (i // per_view).clamp(max=n_views - 1)
    within = i % per_view
    y = within // side + view * side                 # wrist folded below front
    x = within % side
    fmax = f.max(dim=1, keepdim=True).values
    dt = fmax - f                                    # 0 = most recent candidate frame
    return torch.stack([dt, y, x], dim=-1).to(torch.float32)
