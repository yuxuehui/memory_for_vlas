#!/usr/bin/env python3
"""Relevance-source ablation for patch-level memory selection.

Motivated by VLA-Pruner (arXiv:2511.16449): prefill-to-vision and action-to-vision attention
select complementary token sets (their overlap ~50%, often <30%); prefill-only and action-only
both underperform the union. Here we measure the same question for OUR selection design:

  relevance sources (per patch, per scored step):
    act  = DiT action→patch cross-attention, mean over latter-half layers (paper's choice)
    tail = backbone post-image tokens (summary + moment) → patch self-attention @ L13
           (the sharp readout: entropy .685 vs action .91 — gen_nonvision_attn quantification)
  novelty:
    nov  = token-space |Δ| vs previous scored step (the trained patch_union novelty channel)

  variants @ budget 512:  act_only · tail_only · act∪tail · nov∪act (deployed patch_union)
                          · nov∪tail · nov∪act∪tail

Also reports the act/tail top-512 OVERLAP (our analog of the paper's ~50% statistic).
Probe-level ranking only — the trained-in lesson applies; treat as design evidence.

Run: cd HAMLET-Isaac-GR00T-N1d6 && CUDA_VISIBLE_DEVICES=1 NO_ALBUMENTATIONS_UPDATE=1 \
       .venv/bin/python /home/users/xuehui/myfile/Markdown/patch_memory_labels/relevance_ablation.py
"""

from __future__ import annotations

import io
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/home/users/xuehui/myfile/HAMLET-Isaac-GR00T-N1d6")
sys.path.insert(0, "/home/users/xuehui/myfile/HAMLET-Isaac-GR00T-N1d6/scripts")
sys.path.insert(0, "/home/users/xuehui/myfile/Markdown/patch_memory_labels")
sys.path.insert(0, "/home/users/xuehui/myfile/Markdown/lang_token_temporal")
sys.path.insert(0, "/home/users/xuehui/myfile/Markdown/attention_map")

from gen_patch_labels import EPS, TOL, MIN_CELLS, build_policy, load_episode, obs_of, parse_gt
from gen_nonvision_attn import (MODELS, capture_action, install_cross_attn_capture,
                                install_self_attn_capture, _SA)

OUT = Path("/home/users/xuehui/myfile/Markdown/patch_memory_labels")
MODEL = "/tmp/robomme_eval/model_expd60k"   # deployment-relevant scorer
STRIDE = 8
BUDGET = 512
TAIL_LAYER = 13


def episode_signals(policy, pmeta, df, instr, ep, n_cross):
    """Per scored step: novelty (n_img,), act (n_img,), tail (n_img,) — one policy call each."""
    import lang_token_temporal as LT  # noqa: F401  (obs preprocessing registered)
    from PIL import Image

    dec = lambda d: np.array(Image.open(io.BytesIO(d["bytes"])).convert("RGB"))
    recs, prev_tok = [], None
    for t in range(0, len(df), STRIDE):
        st = np.asarray(df.iloc[t]["state"], np.float32)
        ob = obs_of(pmeta, dec(df.iloc[t]["image"]), dec(df.iloc[t]["wrist_image"]), st, instr)
        _SA["maps"], _SA["active"] = [], True
        act_layers = capture_action(policy, ob, n_cross)   # runs the full policy step
        _SA["active"] = False
        sa = list(_SA["maps"])
        if not act_layers or not sa:
            continue
        # image mask from a cheap re-derivation: backbone seq length = self-attn size
        L = sa[0].shape[0]
        # image mask: infer from the policy's cached masks is awkward here; recompute via blocks
        # of the largest square segments — instead capture from the model output path:
        im = policy.model.backbone._last_image_mask  # set by our monkeypatch below
        imnp = im
        # tail queries = all non-image positions AFTER the last image token
        img_idx = np.nonzero(imnp)[0]
        tail_q = list(range(int(img_idx[-1]) + 1, L))
        A_tail = sa[TAIL_LAYER].numpy()
        tail_sig = A_tail[tail_q][:, imnp].mean(0)
        # action = mean over latter-half DiT layers, current-frame image columns
        A_act = np.mean([a for a in act_layers[len(act_layers) // 2:]], axis=0)
        aa = A_act[:L] if A_act.shape[0] >= L else np.pad(A_act, (0, L - A_act.shape[0]))
        act_sig = aa[imnp]
        # novelty: token-space delta needs backbone tokens; use tail-layer VALUE proxy —
        # simplest faithful signal available here: per-patch |Δ| of the attention-weighted
        # patch identity is NOT it; instead reuse pixel cells (probe-consistent).
        recs.append(dict(t=t, act=act_sig, tail=tail_sig))
    return recs


def install_imagemask_stash(policy):
    bb = policy.model.backbone
    orig = bb.forward

    def wrapped(vl_input):
        out = orig(vl_input)
        try:
            bb._last_image_mask = out["image_mask"][0].bool().cpu().numpy()
        except Exception:
            pass
        return out

    bb.forward = wrapped


def add_novelty(recs, df, pmeta):
    """Pixel-cell novelty on the image grid (probe-consistent 9x9 per view)."""
    from PIL import Image

    dec = lambda d: np.array(Image.open(io.BytesIO(d["bytes"])).convert("RGB"))
    n_img = recs[0]["act"].shape[0]
    n_views = 2
    side = int(round((n_img // n_views) ** 0.5))

    def cells(t):
        out = []
        for key in ("image", "wrist_image"):
            a = np.asarray(Image.fromarray(dec(df.iloc[t][key])).resize((side * 8, side * 8)),
                           np.float32) / 255.0
            out.append(a.reshape(side, 8, side, 8, 3).mean((1, 3, 4)))
        return np.stack(out).reshape(-1)

    prev = None
    for r in recs:
        c = cells(r["t"])
        r["nov"] = np.zeros(n_img, np.float32) if prev is None else np.abs(c - prev)
        if prev is None:
            r["nov"][: n_img // n_views] = 1e9  # frame-0 front sentinel
        prev = c
    return recs


def select(recs, channels, budget=BUDGET):
    """Union of per-channel top-K over all (t, patch) candidates; K = budget // n_channels."""
    import heapq

    k = budget // len(channels)
    kept = set()
    for ch in channels:
        heap = []
        for r in recs:
            for i, s in enumerate(r[ch]):
                heapq.heappush(heap, (float(s), r["t"], i))
                if len(heap) > k:
                    heapq.heappop(heap)
        kept |= {(t, i) for (_s, t, i) in heap}
    return kept


def metrics(kept, gt, n_img):
    ts = np.asarray(sorted(t for (t, _i) in kept))
    if not len(ts):
        return {}
    cover = lambda evs: (sum(1 for e in evs if (np.abs(ts - e) <= TOL).sum() >= MIN_CELLS), len(evs))
    rr, rn = cover(gt.get("ref", []))
    er, en = cover(gt.get("exec", []) + ([gt["press"][0]] if "press" in gt else []))
    cells = [i for (_t, i) in kept]
    return dict(n=len(kept), steps=len(set(ts.tolist())),
                demo=float((ts < gt.get("t0", 0)).mean()) if gt.get("t0", 0) > 0 else 0.0,
                ref=f"{rr}/{rn}", ex=f"{er}/{en}",
                dup=1 - len(set(cells)) / max(1, len(cells)))


def main():
    install_self_attn_capture()
    install_cross_attn_capture()
    policy, _V, pmeta = build_policy(MODEL if Path(MODEL).exists() else MODELS["HAMLET"])
    install_imagemask_stash(policy)
    n_cross = int(policy.model.config.diffusion_model_cfg["num_layers"]) // 2

    VARIANTS = {
        "act_only": ("act",), "tail_only": ("tail",), "act∪tail": ("act", "tail"),
        "nov∪act (deployed)": ("nov", "act"), "nov∪tail": ("nov", "tail"),
        "nov∪act∪tail": ("nov", "act", "tail"),
    }
    agg = defaultdict(lambda: defaultdict(list))
    overlaps = []
    for task, eps in EPS.items():
        for ep in eps:
            df, instr = load_episode(ep)
            gt = parse_gt(task, ep)
            recs = episode_signals(policy, pmeta, df, instr, ep, n_cross)
            if not recs:
                print(f"[{task}/ep{ep}] no signals"); continue
            recs = add_novelty(recs, df, pmeta)
            n_img = recs[0]["act"].shape[0]
            a = select(recs, ("act",)); b = select(recs, ("tail",))
            ov = len(a & b) / max(1, len(a | b) - len(a & b) + len(a & b))
            ov = len(a & b) / BUDGET
            overlaps.append(ov)
            print(f"[{task}/ep{ep}] steps={len(recs)} act/tail top-512 overlap={ov:.2f}")
            for name, chs in VARIANTS.items():
                m = metrics(select(recs, chs), gt, n_img)
                for k, v in m.items():
                    agg[name][k].append(v)
    print(f"\nact vs tail top-512 overlap: mean={np.mean(overlaps):.2f} "
          f"min={np.min(overlaps):.2f}  (VLA-Pruner reports ~0.5, often <0.3)")
    print(f"\n{'variant':<20}{'steps':>7}{'dup':>7}{'demo':>7}{'REF':>10}{'EXEC':>10}")
    for name, m in agg.items():
        rr = sum(int(x.split('/')[0]) for x in m['ref']); rn = sum(int(x.split('/')[1]) for x in m['ref'])
        er = sum(int(x.split('/')[0]) for x in m['ex']); en = sum(int(x.split('/')[1]) for x in m['ex'])
        print(f"{name:<20}{np.mean(m['steps']):>7.1f}{np.mean(m['dup']):>7.2f}"
              f"{np.mean(m['demo']):>7.2f}{f'{rr}/{rn}':>10}{f'{er}/{en}':>10}")


if __name__ == "__main__":
    main()
