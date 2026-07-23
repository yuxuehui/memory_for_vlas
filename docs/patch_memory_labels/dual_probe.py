#!/usr/bin/env python3
"""Dual probe on stored patch-memory tokens — decides H1 vs H2 in one pass.

H1 (position loss): can the stored per-patch token h recover (t, y, x, view)?
    Linear probe h -> label vs a shuffled-label baseline. If t is unrecoverable while
    (y,x) is, the gap is a TIME tag; if (y,x) also fails, full 3D position is needed.

H2 (fragments lose content): do the SELECTED patches retain the scene content the model
    itself uses? Target = the model's own summary/tail token (its scene code). Compare a
    ridge map pooled(subset)->summary for three pools of the SAME size:
        selected (patch_union)   vs   random   vs   whole frame (upper bound)
    selected >> random  => selection keeps content; selected << whole => fragments do lose
    content (H2). Random is the fair "same budget, no smarts" control.

Tokens are the real post-ViT backbone features that become mem_seq (captured via a forward
hook), so this is exactly what the DiT cross-attn reads — no proxy for the input side.

Run: cd HAMLET-Isaac-GR00T-N1d6 && CUDA_VISIBLE_DEVICES=<gpu> NO_ALBUMENTATIONS_UPDATE=1 \
       .venv/bin/python .../dual_probe.py --model /tmp/robomme_eval/model_pu40k
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/home/users/xuehui/myfile/Markdown/patch_memory_labels")
sys.path.insert(0, "/home/users/xuehui/myfile/HAMLET-Isaac-GR00T-N1d6/scripts")

from gen_act_vs_tail_patches import (EPS, build_policy, dec, img_blocks, obs_of, ACT_LAYER)
from gen_nonvision_attn import _XA, install_cross_attn_capture
from gen_patch_labels import load_episode
import visualize_memory_attention as V

STRIDE = 8
BUDGET = 512
DIFF_SHARE = 0.5
OUT = Path("/home/users/xuehui/myfile/Markdown/patch_memory_labels")

_BB = {"feat": None, "mask": None}


def install_feature_hook(model):
    def hook(_m, _i, out):
        try:
            _BB["feat"] = out["backbone_features"].detach().float().cpu()
            _BB["mask"] = out["image_mask"].detach().bool().cpu()
        except Exception:
            pass
    return model.backbone.register_forward_hook(hook)


def collect(policy, pmeta, df, instr, ep):
    """Per scored step: real backbone patch tokens h with labels + selection + summary token."""
    n_cross, n_q = pmeta["n_cross"], pmeta["n_q"]
    recs, prev_tok = [], None
    for t in range(0, len(df), STRIDE):
        front, wrist = dec(df.iloc[t]["image"]), dec(df.iloc[t]["wrist_image"])
        state = np.asarray(df.iloc[t]["state"], np.float32)
        _BB["feat"] = _BB["mask"] = None
        _XA["maps"], _XA["active"] = [], True
        with torch.inference_mode():
            policy.get_action(obs_of(pmeta, front, wrist, state, instr),
                              options={"session_ids": [f"dp{ep}"], "reset_memory": [t == 0]})
        _XA["active"] = False
        if _BB["feat"] is None or _BB["mask"] is None:
            continue
        feat, mask = _BB["feat"][0], _BB["mask"][0]           # (L,d), (L,)
        blocks = img_blocks(mask.numpy())
        if len(blocks) < 2:
            continue
        side = int(round((blocks[0][1] - blocks[0][0] + 1) ** 0.5))
        n_per = side * side
        # per-patch token vectors + labels for the two image views
        toks, labs = [], []                                   # h ; (view,y,x)
        for view, (b0, b1) in enumerate(blocks[:2]):
            block = feat[b0:b1 + 1]                           # (n_per, d)
            if block.shape[0] != n_per:
                continue
            for p in range(n_per):
                toks.append(block[p]); labs.append((view, p // side, p % side))
        if len(toks) != 2 * n_per:
            continue
        H = torch.stack(toks).numpy()                         # (2*n_per, d)
        lab = np.asarray(labs, np.int64)
        # summary token = mean of post-image summary slots (exclude the n_q moment tail)
        L = feat.shape[0]
        tail = feat[blocks[-1][1] + 1: L - n_q] if (L - n_q) > blocks[-1][1] + 1 else feat[blocks[-1][1] + 1:]
        summ = tail.mean(0).numpy() if tail.shape[0] else feat[-1].numpy()
        # relevance = DiT act->patch at ACT_LAYER (same signal deployment uses)
        rel = np.zeros(2 * n_per, np.float32)
        xa = list(_XA["maps"])
        if xa:
            n_den = max(1, len(xa) // n_cross)
            As = [xa[d * n_cross + ACT_LAYER] for d in range(n_den) if d * n_cross + ACT_LAYER < len(xa)]
            if As:
                av = (sum(A.float().mean(0) for A in As) / len(As)).numpy()
                for view, (b0, b1) in enumerate(blocks[:2]):
                    if b1 < len(av):
                        rel[view * n_per:(view + 1) * n_per] = av[b0:b1 + 1][:n_per]
        recs.append(dict(t=t, H=H, lab=lab, summ=summ, rel=rel, n_per=n_per, prev=prev_tok))
        prev_tok = H
    return recs


def patch_union_subset(recs, budget=BUDGET, diff_share=DIFF_SHARE):
    """Deployment-faithful selection over the whole episode: token-space novelty ∪ act-relevance.
    Returns, per rec, the boolean mask of selected patches (competitive global top-k)."""
    n_diff = int(round(budget * diff_share))
    n_attn = budget - n_diff
    # global candidate pool across all steps
    cand = []
    for k, r in enumerate(recs):
        nov = (np.abs(r["H"] - r["prev"]).mean(1) if r["prev"] is not None
               else np.full(r["H"].shape[0], np.inf))       # frame-0 sentinel
        if r["prev"] is None:
            nov[r["n_per"]:] = -np.inf                       # only front frame-0 protected
        for i in range(r["H"].shape[0]):
            cand.append((k, i, float(nov[i]), float(r["rel"][i])))
    order_d = sorted(range(len(cand)), key=lambda j: -cand[j][2])[:n_diff]
    order_r = sorted(range(len(cand)), key=lambda j: -cand[j][3])[:n_attn]
    keep = set(order_d) | set(order_r)
    sel = [np.zeros(r["H"].shape[0], bool) for r in recs]
    for j in keep:
        k, i = cand[j][0], cand[j][1]
        sel[k][i] = True
    return sel


def ridge_r2(X, Y, seed=0):
    """5-fold-ish single split ridge R² (multi-output), standardized."""
    n = len(X); rng = np.random.RandomState(seed); idx = rng.permutation(n)
    ntr = int(0.8 * n); tr, te = idx[:ntr], idx[ntr:]
    Xtr, Xte, Ytr, Yte = X[tr], X[te], Y[tr], Y[te]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    lam = 10.0
    W = np.linalg.solve(Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1]), Xtr.T @ Ytr)
    pred = Xte @ W
    ss_res = ((Yte - pred) ** 2).sum(); ss_tot = ((Yte - Yte.mean(0)) ** 2).sum()
    return 1 - ss_res / (ss_tot + 1e-9)


def logistic_acc(X, y, seed=0):
    """Multinomial logistic accuracy vs majority baseline, single 80/20 split."""
    from sklearn.linear_model import LogisticRegression
    n = len(X); rng = np.random.RandomState(seed); idx = rng.permutation(n)
    ntr = int(0.8 * n); tr, te = idx[:ntr], idx[ntr:]
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    clf = LogisticRegression(max_iter=200, C=1.0)
    clf.fit((X[tr] - mu) / sd, y[tr])
    acc = clf.score((X[te] - mu) / sd, y[te])
    maj = np.bincount(y[te]).max() / len(te)
    return acc, maj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/tmp/robomme_eval/model_pu40k")
    ap.add_argument("--tasks", default="SwingXtimes,ButtonUnmask,VideoPlaceButton,PatternLock,RouteStick")
    args = ap.parse_args()

    install_cross_attn_capture()
    policy, _V, pmeta = build_policy(args.model)
    install_feature_hook(policy.model)

    all_recs = []
    for task in args.tasks.split(","):
        for ep in EPS.get(task, []):
            df, instr = load_episode(ep)
            recs = collect(policy, pmeta, df, instr, ep)
            if recs:
                sel = patch_union_subset(recs)
                for r, s in zip(recs, sel):
                    r["sel"] = s
                all_recs.append((task, ep, recs))
                print(f"[{task}/ep{ep}] {len(recs)} steps, {sum(s.sum() for s in sel)} selected", flush=True)
    del policy; torch.cuda.empty_cache()

    # ---- assemble global token matrix + labels
    H = np.concatenate([r["H"] for _, _, recs in all_recs for r in recs])
    lab = np.concatenate([r["lab"] for _, _, recs in all_recs for r in recs])
    tvec = np.concatenate([np.full(r["H"].shape[0], r["t"]) for _, _, recs in all_recs for r in recs])
    print(f"\n{H.shape[0]} tokens, d={H.shape[1]}")

    print("\n=== H1: position/time recoverable from the stored token? ===")
    view, y, x = lab[:, 0], lab[:, 1], lab[:, 2]
    for name, yv in [("view (2-way)", view), ("y-row", y), ("x-col", x)]:
        acc, maj = logistic_acc(H, yv)
        print(f"  {name:16} acc {acc:.3f}  (majority {maj:.3f}, lift {acc-maj:+.3f})")
    # time: 4-way quartile-within-episode classification (robust, interpretable) — is the
    # token's position in the episode readable at all? Chance = 0.25.
    tbin = []
    for _, _, recs in all_recs:
        ts = np.array([r["t"] for r in recs], float); lo, hi = ts.min(), ts.max()
        for r in recs:
            q = min(3, int(4 * (r["t"] - lo) / (hi - lo + 1e-9)))
            tbin.append(np.full(r["H"].shape[0], q))
    tbin = np.concatenate(tbin).astype(np.int64)
    acc, maj = logistic_acc(H, tbin)
    print(f"  time (quartile)  acc {acc:.3f}  (chance 0.25, majority {maj:.3f}, lift {acc-maj:+.3f})")

    print("\n=== H2: do SELECTED patches keep the scene content? ===")
    # Same-space frame retrieval: can pooled(subset) identify ITS OWN frame among all frames,
    # using the WHOLE-frame pool as the gallery? Query & gallery are both mean vision tokens
    # (same space, unlike the summary token), so the metric can actually discriminate. If
    # selected retrieves its own frame ~ as well as whole-vs-whole, fragments keep the scene;
    # selected << whole and ~ random => H2 (fragments lose content). Random = fair control.
    rng = np.random.RandomState(0)
    gallery, pools = [], {"sel": [], "rand": [], "whole": []}
    for _, _, recs in all_recs:
        for r in recs:
            s = r["sel"]; k = int(s.sum())
            if k < 4:
                continue
            Hs = r["H"]
            rand = np.zeros(len(s), bool); rand[rng.choice(len(s), k, replace=False)] = True
            gallery.append(Hs.mean(0))
            pools["sel"].append(Hs[s].mean(0))
            pools["rand"].append(Hs[rand].mean(0))
            pools["whole"].append(Hs.mean(0))
    G = np.asarray(gallery)
    norm = lambda A: A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
    Gn = norm(G)
    print(f"  same-space own-frame retrieval (query=pool, gallery=whole-frame, N={len(G)}):")
    for which, tag in [("sel", "selected (patch_union)"), ("rand", "random (same size)"),
                       ("whole", "whole frame (identity=1.0)")]:
        P = norm(np.asarray(pools[which]))
        sim = P @ Gn.T
        diag = sim[np.arange(len(P)), np.arange(len(P))]
        rank = (sim >= diag[:, None]).sum(1)
        # exclude the trivial whole==whole identity by reporting rank among OTHERS for 'whole'
        top1 = (rank == 1).mean(); mrr = (1.0 / rank).mean()
        print(f"    {tag:26} top1 {top1:.3f}  MRR {mrr:.3f}")
    np.savez("/tmp/robomme_eval/dualprobe_cache.npz",
             H=H, lab=lab, tbin=tbin, G=G,
             sel=np.asarray(pools["sel"]), rand=np.asarray(pools["rand"]))
    print("\nPROBE_DONE")


if __name__ == "__main__":
    main()
