"""Learned patch-memory selection (`mem_fs_learned_select=True`) — note-24.

The patch_union selector has never been trained: pass A is `no_grad` and top-k is not
differentiable, so "score -> selection -> future loss" has no gradient path. Measured
consequence (note-24 §1): training V2 from 30k to 50k tripled success (4.6% -> 14.4%) while
the memory's geometry was bit-for-bit unchanged — the hand-written novelty/attention rule is
effectively a fixed heuristic that training cannot move.

This module supplies the two pieces needed to put a gradient on selection:

  PatchScoreHead   one logit per candidate patch, from local + (masked) global features and
                   optionally its (dt, y, x) position. DynamicViT's prediction module.

  soft_topk        logits -> alpha in (0,1) with sum(alpha) == budget EXACTLY, via a
                   bisected threshold. LKV's budgeted Soft-TopK. Fully differentiable in the
                   logits, so no straight-through estimator is needed (DiffPrune shows
                   surrogate gradients cost ~8 points: the forward runs hard top-k while the
                   backward differentiates a different, smooth function).

alpha is consumed as an ADDITIVE ATTENTION BIAS log(alpha) on the memory keys, which makes
exp(P_ij + log alpha_j) = exp(P_ij) * alpha_j — exactly DynamicViT's masked attention
    A~_ij = exp(P_ij) G_ij / sum_k exp(P_ik) G_ik,   G_ij = alpha_j
so alpha=0 is *identical* to deleting the token (not merely zeroing its value, which would
still let it occupy the softmax denominator). That equivalence is what lets training (all
candidates, soft alpha) converge to inference (hard top-budget, alpha in {0,1}) as the
temperature anneals — the failure mode that killed patch_union v1 was exactly this kind of
train/deploy divergence, so it is bought deliberately.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

NEG = -30.0   # log(alpha) floor: exp(-30) ~ 1e-13, dead in bf16/fp32 but never NaN


class PatchScoreHead(nn.Module):
    """(B, N, d) candidate patch tokens -> (B, N) selection logits.

    Local branch sees the patch; global branch sees the whole candidate pool (mean-pooled,
    optionally alpha-weighted like DynamicViT's Agg) so the score is *competitive* rather than
    absolute — the same patch should win in a boring frame and lose in an eventful one.

    `use_pos` feeds the normalised (dt, y, x) that fs_pos_rope already computes, which is what
    lets the head express recency/spatial priors instead of us hand-coding a novelty channel.
    """

    def __init__(self, d_model: int, hidden: int | None = None, use_pos: bool = True):
        super().__init__()
        h = int(hidden or max(32, d_model // 8))
        self.use_pos = bool(use_pos)
        self.norm = nn.LayerNorm(d_model)
        self.local = nn.Linear(d_model, h)
        self.glob = nn.Linear(d_model, h)
        self.pos = nn.Linear(3, h) if use_pos else None
        # (dt, y, x) upper bounds: F-1=7 recency ranks, 2*9 folded rows, 9 columns.
        self.register_buffer("pos_scale", torch.tensor([8.0, 18.0, 9.0]), persistent=False)
        n_in = h * (3 if use_pos else 2)
        self.out = nn.Sequential(nn.GELU(), nn.Linear(n_in, h), nn.GELU(), nn.Linear(h, 1))
        # start near-neutral so the first steps behave like the (already working) heuristic
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    @property
    def pdtype(self) -> torch.dtype:
        """The head's parameter dtype. Under DeepSpeed bf16 the whole module is cast, so inputs
        must follow the WEIGHTS (a fp32 input into a bf16 LayerNorm raises "expected scalar type
        Float but found BFloat16"). Precision where it matters — the budget bisection and the
        alpha/bias math — is fp32 regardless: `score` lifts the logits back to fp32 and
        soft_topk/alpha_to_bias work there."""
        return self.norm.weight.dtype

    def embed_local(self, v_Nd: torch.Tensor) -> torch.Tensor:
        return self.local(self.norm(v_Nd.to(self.pdtype)))

    def embed_pool(self, v_Nd: torch.Tensor) -> torch.Tensor:
        """Frame-level context vector (h,). At inference this is computed ONCE per frame at
        push time over the frame's full patch set and stored frozen — which is exactly what
        training sees (per-frame pooling over the complete candidate frame)."""
        return self.glob(self.norm(v_Nd.to(self.pdtype))).mean(dim=-2)

    def score(self, zl_Nh: torch.Tensor, zg_h: torch.Tensor,
              pos_N3: torch.Tensor | None) -> torch.Tensor:
        # zg may arrive per-frame (..., h) to broadcast, or already per-entry (..., N, h) when
        # the caller caches one context vector per stored patch (LearnedPatchSelector).
        zg = zg_h if zg_h.shape == zl_Nh.shape else zg_h.unsqueeze(-2).expand_as(zl_Nh)
        parts = [zl_Nh, zg]
        if self.use_pos:
            wd = self.pos.weight.dtype
            p = torch.zeros(*zl_Nh.shape[:-1], 3, device=zl_Nh.device, dtype=wd) \
                if pos_N3 is None else pos_N3.to(wd)
            # Normalise RAW (dt, y, x) here, not at the call sites: training reads them from
            # positions_from_flat and inference from the selector, and the head must see the
            # same scale from both or the learned position prior does not transfer.
            parts.append(self.pos(p / self.pos_scale.to(device=p.device, dtype=wd)))
        # fp32 out: the budget bisection, gumbel noise and log(alpha) bias all want full
        # precision, and the caller's residual z-scores are fp32 too.
        return self.out(torch.cat([q.to(zl_Nh.dtype) for q in parts], dim=-1)).squeeze(-1).float()

    def forward(self, v_BNd: torch.Tensor, pos_BN3: torch.Tensor | None = None) -> torch.Tensor:
        """(B, N, d) [+ (B, N, 3) normalised (dt,y,x)] -> (B, N) logits, pooling over N.
        Callers that need per-frame pooling reshape to (B*F, n_img, d) first."""
        zl = self.embed_local(v_BNd)
        zg = self.embed_pool(v_BNd)
        return self.score(zl, zg, pos_BN3)


def soft_topk(logits_BN: torch.Tensor, budget: int, tau: float = 1.0,
              iters: int = 40) -> torch.Tensor:
    """alpha = sigmoid((logits - lambda) / tau) with lambda bisected so sum(alpha) == budget.

    lambda is solved under no_grad (it is a function of the logits, but treating it as a
    constant is the standard budgeted-gate relaxation and keeps the backward cheap); the
    gradient flows through `logits`. The forward satisfies the budget exactly, so the KV the
    student attends to always carries the same total attention mass as the deployed top-k.
    """
    B, N = logits_BN.shape
    if budget >= N:
        return torch.ones_like(logits_BN)
    x = logits_BN.detach().float()
    lo = x.min(dim=1, keepdim=True).values - 20.0 * tau
    hi = x.max(dim=1, keepdim=True).values + 20.0 * tau
    with torch.no_grad():
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            s = torch.sigmoid((x - mid) / tau).sum(dim=1, keepdim=True)
            too_many = s > budget          # raise the threshold to keep fewer
            lo = torch.where(too_many, mid, lo)
            hi = torch.where(too_many, hi, mid)
        lam = 0.5 * (lo + hi)
    return torch.sigmoid((logits_BN.float() - lam) / tau)


def hard_topk(logits_BN: torch.Tensor, budget: int) -> torch.Tensor:
    """Deployment gate: 1 on the top-`budget` logits, 0 elsewhere (tau -> 0 of soft_topk)."""
    B, N = logits_BN.shape
    if budget >= N:
        return torch.ones_like(logits_BN)
    a = torch.zeros_like(logits_BN)
    idx = torch.topk(logits_BN, budget, dim=1).indices
    return a.scatter(1, idx, 1.0)


def gumbel_perturb(logits_BN: torch.Tensor, scale: float) -> torch.Tensor:
    """Gumbel-top-k exploration. Without it only the CURRENTLY selected patches receive
    gradient, so the head can never learn to promote a patch it has not already picked."""
    if scale <= 0.0:
        return logits_BN
    u = torch.rand_like(logits_BN).clamp_(1e-9, 1.0 - 1e-9)
    return logits_BN + scale * (-torch.log(-torch.log(u)))


def alpha_to_bias(alpha_BN: torch.Tensor, dtype=None) -> torch.Tensor:
    """log(alpha), floored — the additive attention bias that realises G_ij = alpha_j."""
    b = torch.log(alpha_BN.float().clamp_min(1e-13)).clamp_min(NEG)
    return b if dtype is None else b.to(dtype)


class LearnedPatchSelector:
    """Inference-side running memory ranked by the LEARNED score (mem_fs_learned_select).

    Differs from PatchUnionSelector in two ways that matter:

    * one heap, not two. The learned logit is a single comparable quantity, so the budget is
      allocated by global competition across all frames -- which is the mechanism LKV's
      ablation found dominant ("even weak selectors excel with proper allocation"). The
      hand-written novelty/attn channels could never do this: separate heaps, different units.
    * entries are RE-SCORED every call. dt shifts as the episode advances, so a score computed
      once at push time goes stale -- the limitation LKV explicitly concedes. Re-scoring costs
      one tiny MLP over ~674 cached h-dim embeddings, so it is essentially free here.

    Cached per entry: the local embedding (h,) and its frame's pooled context (h,), NOT the
    raw token -- so re-scoring never re-runs the LayerNorm/Linear over d-dim tokens.
    """

    def __init__(self, head: PatchScoreHead, budget: int = 512, n_views: int = 2,
                 pos_frames: int = 8):
        self.head, self.budget = head, int(budget)
        self.n_views, self.pos_frames = int(n_views), int(pos_frames)
        self.reset()

    def reset(self):
        self.zl = self.zg = self.tok = None      # (M, h), (M, h), (M, d)
        self.step_of = self.idx_of = None        # (M,) long
        self.step = -1

    def _yx(self, idx, n_img: int):
        per_view = n_img // self.n_views
        side = int(round(per_view ** 0.5))
        view = (idx // per_view).clamp(max=self.n_views - 1)
        within = idx % per_view
        return (within // side + view * side).float(), (within % side).float()

    def _dt(self):
        """Recency RANK over distinct steps, normalised to [0, pos_frames-1] -- the support the
        head was trained on. Raw step distance would be far outside it on long episodes."""
        uniq = torch.unique(self.step_of)
        rank = torch.searchsorted(uniq, self.step_of)           # 0 = oldest
        r = (uniq.numel() - 1 - rank).float()                   # 0 = most recent
        if uniq.numel() > 1:
            r = r * (self.pos_frames - 1) / float(uniq.numel() - 1)
        return r

    def _positions(self, n_img: int):
        y, x = self._yx(self.idx_of, n_img)
        return torch.stack([self._dt(), y, x], dim=-1)

    def _scores(self, n_img: int):
        with torch.no_grad():
            return self.head.score(self.zl, self.zg, self._positions(n_img))

    @torch.no_grad()
    def observe(self, vis_Nd: torch.Tensor, *_ignored, **_kw) -> None:
        """Push the current frame's patches, then prune back to budget by re-scored rank."""
        self.step += 1
        n_img = vis_Nd.shape[0]
        v = vis_Nd.detach()
        p = next(self.head.parameters())
        zl = self.head.embed_local(v.to(p.device)).to(p.dtype)
        zg = self.head.embed_pool(v.to(p.device)).to(p.dtype).unsqueeze(0).expand_as(zl)
        idx = torch.arange(n_img, device=zl.device)
        stp = torch.full((n_img,), self.step, device=zl.device, dtype=torch.long)
        tok = v.to(zl.device)
        if self.zl is None:
            self.zl, self.zg, self.tok, self.step_of, self.idx_of = zl, zg, tok, stp, idx
        else:
            self.zl = torch.cat([self.zl, zl]); self.zg = torch.cat([self.zg, zg])
            self.tok = torch.cat([self.tok, tok])
            self.step_of = torch.cat([self.step_of, stp])
            self.idx_of = torch.cat([self.idx_of, idx])
        if self.zl.shape[0] > self.budget:
            keep = torch.topk(self._scores(n_img), self.budget).indices
            keep = keep[torch.argsort(self.step_of[keep] * n_img + self.idx_of[keep])]
            self.zl, self.zg, self.tok = self.zl[keep], self.zg[keep], self.tok[keep]
            self.step_of, self.idx_of = self.step_of[keep], self.idx_of[keep]

    @torch.no_grad()
    def read(self, vis_current: torch.Tensor, want_pos: bool = False, n_views: int = 2,
             pos_frames: int = 8):
        """(budget, d) memory tokens in temporal order, padded with current-frame tokens.
        Mirrors PatchUnionSelector.read's signature so the call sites are interchangeable."""
        self.n_views, self.pos_frames = int(n_views), int(pos_frames)
        n_img = vis_current.shape[0]
        dev, dt_ = vis_current.device, vis_current.dtype
        if self.tok is None:
            rows, pos = vis_current[:0], torch.zeros(0, 3, device=dev)
        else:
            rows, pos = self.tok.to(dev, dt_), self._positions(n_img).to(dev)
        need = self.budget - rows.shape[0]
        if need > 0:                                  # pad with the current frame (dt = 0)
            k = min(need, n_img)
            pad_i = torch.arange(k, device=dev)
            y, x = self._yx(pad_i, n_img)
            rows = torch.cat([rows, vis_current[:k]])
            pos = torch.cat([pos, torch.stack([torch.zeros_like(y), y, x], -1).to(dev)])
            if rows.shape[0] < self.budget:
                r = self.budget - rows.shape[0]
                rows = torch.cat([rows, vis_current[-1:].expand(r, -1)])
                pos = torch.cat([pos, pos[-1:].expand(r, -1)])
        return (rows[: self.budget], pos[: self.budget]) if want_pos else rows[: self.budget]


def anneal(step: int, total: int, hi: float, lo: float) -> float:
    """Linear schedule, clamped. Used for both the Soft-TopK temperature and the Gumbel scale
    (LKV / DiffPrune both anneal soft -> hard so the train forward converges to deployment)."""
    if total <= 0:
        return lo
    f = min(1.0, max(0.0, step / float(total)))
    return hi + (lo - hi) * f
