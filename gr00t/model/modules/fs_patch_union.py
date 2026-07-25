"""Patch-level union memory (mem_fs_select='patch_union') — probe-validated design.

Scores (identical at train and inference, cf. Gr00tN1d6._patch_union_mem_seq):
  novelty   = token-space |delta| vs the previous scored frame (frame-0 protected)
  relevance = the DiT's own action->patch cross-attention at `mem_fs_attn_layer`
              (TRUE action query). Train: captured in a no_grad scoring pass over all
              candidates (_patch_union_score_pass). Inference: captured during the real
              forward and committed AFTER the action (_pu_commit), so the memory read at
              step t only contains history < t.

Instead of selecting FRAMES (fifo/diff) and linspace-subsampling their tokens to the
budget, maintain a running budget of individual PATCH tokens selected by the union of
two channels (Markdown/patch_memory_labels probe: union beats both single channels and
the z-sum combination on redundancy/coverage/demo-share).

Stored entries are the raw backbone vision tokens of the selected cells — the same
token type the framesamp mem_seq carries, so a framesamp-trained policy consumes them
without architecture changes. (t, y, x) tagging: off by default (content identity only,
the v1 behavior); with `mem_fs_pos_rope=True` each stored key is rotated by a PPE-style
3D RoPE over (Δt=recency rank, y with wrist folded, x) — see fs_pos_rope.py / note-21 §8.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- attention capture
# Minimal, flag-gated sdpa patch: during the patch_union SCORING pass we need the DiT's
# action->candidate cross-attention at one layer. Identified by Lq != Lk (cross-attn) and
# call-count mod n_cross == layer. Inactive (zero overhead) outside the scoring pass.
_CAP = {"scores": None, "count": 0, "layer": 13, "n_cross": 16, "active": False}
_PATCHED = False
# PPE-style key RoPE (note-23): when active, rotate the LAST `rope_M` keys of every cross-attn
# (Lq != Lk) by cos/sin, so the memory patches' (Δt,y,x) enter the QK dot product. Applied
# BEFORE the act-capture scoring so both the capture and the real sdpa see rotated keys.
_ROPE = {"cos": None, "sin": None, "M": 0, "active": False}


def set_rope(cos, sin, M: int) -> None:
    _ROPE["cos"], _ROPE["sin"], _ROPE["M"], _ROPE["active"] = cos, sin, int(M), True


def clear_rope() -> None:
    _ROPE["cos"] = _ROPE["sin"] = None
    _ROPE["M"], _ROPE["active"] = 0, False


def install_capture(layer: int, n_cross: int) -> None:
    global _PATCHED
    _CAP["layer"], _CAP["n_cross"] = int(layer), int(n_cross)
    if _PATCHED:
        return
    orig = F.scaled_dot_product_attention

    def patched(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **kw):
        # PPE key RoPE: rotate the last M keys of a cross-attn (queries left at identity).
        if _ROPE["active"] and _ROPE["cos"] is not None and q.shape[-2] != k.shape[-2]:
            try:
                from gr00t.model.modules.fs_pos_rope import rotate_keys
                M = _ROPE["M"]
                if 0 < M <= k.shape[-2] and _ROPE["cos"].shape[-2] == M:
                    kk = k.clone()
                    kk[..., -M:, :] = rotate_keys(k[..., -M:, :], _ROPE["cos"], _ROPE["sin"])
                    k = kk
            except Exception:
                pass
        out = orig(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p,
                   is_causal=is_causal, scale=scale, **kw)
        if _CAP["active"]:
            try:
                if q.shape[-2] != k.shape[-2]:
                    li = _CAP["count"] % _CAP["n_cross"]
                    _CAP["count"] += 1
                    if li == _CAP["layer"]:
                        sc = scale if scale is not None else q.shape[-1] ** -0.5
                        sco = (q.float() @ k.float().transpose(-2, -1)) * sc
                        if attn_mask is not None and attn_mask.dtype != torch.bool:
                            sco = sco + attn_mask.float()
                        a = sco.softmax(-1).mean(dim=(1, 2)).detach()  # (B, Lk)
                        prev = _CAP["scores"]
                        _CAP["scores"] = a if prev is None or prev.shape != a.shape else prev + a
            except Exception:
                pass
        return out

    F.scaled_dot_product_attention = patched
    torch.nn.functional.scaled_dot_product_attention = patched
    _PATCHED = True


def capture_begin(layer: int, n_cross: int):
    install_capture(layer, n_cross)
    _CAP["scores"] = None
    _CAP["count"] = 0
    _CAP["active"] = True


def capture_end():
    _CAP["active"] = False
    return _CAP["scores"]


def pix_cells(pix_flat: torch.Tensor, n_img: int) -> torch.Tensor | None:
    """Flattened per-sample pixels -> per-token mean intensity cells (n_img,).

    Token order assumption (same as the probe / img_heat_from_cross): view-major,
    row-major side x side grid per view; n_img = n_views * side^2. Pixels arrive as
    (V*C*H*W,) with V views sample-major."""
    if pix_flat is None:
        return None
    ppv_candidates = [2, 1]
    for n_views in ppv_candidates:
        ppv = n_img // n_views
        side = int(round(ppv ** 0.5))
        if side * side == ppv and n_img % n_views == 0:
            break
    else:
        return None
    numel = pix_flat.numel()
    if numel % n_views != 0:
        return None
    per_view = pix_flat.float().view(n_views, -1)
    chw = per_view.shape[1]
    # assume (C, H, W) with H == W
    hw = chw // 3
    h = int(round(hw ** 0.5))
    if 3 * h * h != chw:
        return None
    img = per_view.view(n_views, 3, h, h).mean(1)  # (V, H, W) grayscale
    cell = F.adaptive_avg_pool2d(img.unsqueeze(1), side).squeeze(1)  # (V, side, side)
    return cell.reshape(-1)  # (n_img,) view-major row-major


class PatchUnionSelector:
    """Per-session running patch memory: novelty top-(budget*diff_share) UNION act-relevance
    UNION (optional) tail_L15, the non-novelty budget split by `tail_share`."""

    def __init__(self, budget: int = 512, diff_share: float = 0.5, stride: int = 1,
                 tail_share: float = 0.0):
        self.budget = int(budget)
        self.n_diff = max(1, int(round(budget * diff_share)))
        rem = max(0, self.budget - self.n_diff)
        self.n_tail = max(0, int(round(rem * tail_share)))
        self.n_attn = max(0, rem - self.n_tail)
        self.stride = max(1, int(stride))
        self.reset()

    def reset(self):
        self.diff_heap: list = []   # (score, step, idx, token)
        self.attn_heap: list = []
        self.tail_heap: list = []
        self.last_cells: torch.Tensor | None = None
        self.last_tok: torch.Tensor | None = None   # previous scored frame's (n_img, d) tokens
        self.step = -1

    @staticmethod
    def _push(heap, cap, item):
        import heapq
        heapq.heappush(heap, item)
        if len(heap) > cap:
            heapq.heappop(heap)

    def observe(self, vis: torch.Tensor, cells: torch.Tensor | None,
                attn: torch.Tensor | None, tail: torch.Tensor | None = None) -> None:
        """vis: (n_img, d) current-frame raw image tokens; cells: (n_img,) pixel cell
        intensities (diff signal; falls back to token-space); attn: (n_img,) captured
        cross-attn scores; tail: (n_img,) patch . frame-summary dot product (tail_L15
        channel). attn/tail are None on the first call."""
        self.step += 1
        n_img = vis.shape[0]
        if self.step == 0:
            ppv = n_img // 2 if n_img % 2 == 0 else n_img
            for i in range(min(ppv, n_img)):  # frame-0 FRONT cells: TokenDrop sentinel
                self._push(self.diff_heap, self.n_diff, (1e9, 0, i, vis[i].detach()))
            self.last_cells = cells
            self.last_tok = vis.detach().float()
            return
        if self.step % self.stride == 0:
            if cells is not None and self.last_cells is not None:
                d = (cells - self.last_cells).abs()          # pixel-cell variant (unused today)
                self.last_cells = cells
            else:
                # TRAIN-MATCHING novelty: per-dim mean|Δ| of the full token, exactly
                # `_patch_union_mem_seq`'s (v_t - v_{t-1}).abs().mean(-1). The old scalar
                # shortcut |mean(v_t) - mean(v_{t-1})| under-detects semantic change
                # (|mean Δ| <= mean|Δ|) and was a train/infer formula mismatch.
                d = (vis.detach().float() - self.last_tok).abs().mean(-1)
            self.last_tok = vis.detach().float()
            for i in range(n_img):
                s = float(d[i])
                if s > 1e-4:
                    self._push(self.diff_heap, self.n_diff, (s, self.step, i, vis[i].detach()))
        if attn is not None and self.n_attn > 0:
            for i in range(n_img):
                self._push(self.attn_heap, self.n_attn, (float(attn[i]), self.step, i, vis[i].detach()))
        if tail is not None and self.n_tail > 0:
            for i in range(n_img):
                self._push(self.tail_heap, self.n_tail, (float(tail[i]), self.step, i, vis[i].detach()))

    def read(self, vis_current: torch.Tensor, want_pos: bool = False, n_views: int = 2,
             pos_frames: int = 8):
        """(budget, d): union of all heaps in (step, idx) order, deduped, padded with
        current-frame tokens to exactly budget rows. If want_pos, also return (budget, 3)
        (Δt, y, x) for PPE key RoPE.

        Δt is the RECENCY RANK over the distinct steps held in memory, normalized onto
        [0, pos_frames-1] (0 = newest) — matching the TRAINING side, where Δt is the rank
        among the F causally-selected candidate frames (fs_pos_rope.positions_from_flat).
        Raw step-distance (self.step - t) would be OOD: training only ever produces Δt in
        [0, F-1], and RoPE angles outside that support were never trained."""
        seen, entries = set(), []
        for heap in (self.diff_heap, self.attn_heap, self.tail_heap):
            for (s, t, i, tok) in heap:
                if (t, i) not in seen:
                    seen.add((t, i))
                    entries.append((t, i, tok))
        entries.sort(key=lambda e: (e[0], e[1]))
        dev = vis_current.device
        kept = entries[: self.budget]
        toks = [tok.to(dev) for (_t, _i, tok) in kept]
        n_img = vis_current.shape[0]
        pos = None
        if want_pos:
            per_view = max(1, n_img // n_views)
            side = int(round(per_view ** 0.5))
            steps_desc = sorted({t for (t, _i, _tok) in kept}, reverse=True)  # newest first
            rank_of = {t: r for r, t in enumerate(steps_desc)}
            denom = max(1, len(steps_desc) - 1)
            span = max(0, pos_frames - 1)
            pos = []
            for (t, i, _tok) in kept:
                view = min(i // per_view, n_views - 1)
                within = i % per_view
                dt = int(round(rank_of[t] / denom * span)) if denom else 0
                pos.append([dt, within // side + view * side, within % side])
        k = 0
        while len(toks) < self.budget:
            idx = k % n_img
            toks.append(vis_current[idx])
            if want_pos:
                # padding = real current-frame patches: true (y, x), Δt=0 (newest)
                view = min(idx // per_view, n_views - 1)
                within = idx % per_view
                pos.append([0, within // side + view * side, within % side])
            k += 1
        out = torch.stack(toks, dim=0)
        if want_pos:
            return out, torch.tensor(pos, dtype=torch.float32, device=dev)
        return out
