#!/usr/bin/env python3
"""Held-out P3b: does the directional count effect survive when ŵ is fit OUT-OF-SAMPLE?

In-sample P3b fit the count axis ŵ on all baseline steps, then measured within-step slope(proj vs c′).
Here we GroupKFold by episode: fit ŵ on TRAIN episodes' baselines, project TEST episodes' swap Δactions,
and compute the within-step slope on the held-out steps. Compare to the in-sample number (+0.41, 79%).

  python scripts/held_out_p3b.py --npz /tmp/robomme_eval/probe/swap_dir3.npz
"""
import argparse, collections
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument("--npz", required=True)
ap.add_argument("--nfold", type=int, default=5); ap.add_argument("--seeds", type=int, default=5)
a = ap.parse_args()

d = np.load(a.npz, allow_pickle=True)
bx = d["bx"]; by = d["by"].astype(np.float64); count = bx[:, 0]; phase = bx[:, 1]
sw_ep, sw_t, sw_c, sw_cp = d["sw_ep"], d["sw_t"], d["sw_c"], d["sw_cp"]
delta = d["sw_delta"].astype(np.float64)

# episode per baseline row (first-appearance order of (ep,t) == baseX append order)
seen, s = [], set()
for e, t in zip(sw_ep, sw_t):
    k = (int(e), int(t))
    if k not in s:
        s.add(k); seen.append(k)
assert len(seen) == len(by), (len(seen), len(by))
ep_base = np.array([k[0] for k in seen])


def fit_w(mask):
    X = np.column_stack([count[mask], phase[mask], np.ones(mask.sum())])
    coef, *_ = np.linalg.lstsq(X, by[mask], rcond=None)
    w = coef[0]
    return w / (np.linalg.norm(w) + 1e-9)


def within_step_slopes(w, test_eps):
    """slope(proj vs c′) per test recipient step, using axis w."""
    per = collections.defaultdict(list)
    sel = np.isin(sw_ep, list(test_eps))
    for i in np.where(sel)[0]:
        per[(int(sw_ep[i]), int(sw_t[i]))].append((sw_cp[i], float(delta[i] @ w)))
    out = []
    for v in per.values():
        cp = np.array([x[0] for x in v]); pr = np.array([x[1] for x in v])
        if len(v) >= 3 and cp.std() > 0:
            out.append(np.polyfit(cp, pr, 1)[0])
    return out

ug = np.unique(ep_base)
held = []
for seed in range(a.seeds):
    folds = np.array_split(np.random.RandomState(seed).permutation(ug), min(a.nfold, len(ug)))
    for f in folds:
        test_eps = set(f.tolist())
        tr = ~np.isin(ep_base, list(test_eps))
        if tr.sum() < 8 or len(test_eps) < 1:
            continue
        w = fit_w(tr)
        held += within_step_slopes(w, test_eps)

held = np.array(held)
# in-sample reference (ŵ on all)
w_all = fit_w(np.ones(len(by), bool))
insample = np.array(within_step_slopes(w_all, set(ug.tolist())))

print(f"steps: {len(by)} | episodes: {len(ug)} | held-out slopes collected: {len(held)}\n")
print("within-step slope(proj vs c′):")
print(f"  IN-SAMPLE ŵ : mean {insample.mean():+.3f} ± {insample.std():.3f} | {(insample>0).mean()*100:.0f}% positive (n={len(insample)})")
if len(held):
    se = held.std() / np.sqrt(len(held))
    print(f"  HELD-OUT  ŵ : mean {held.mean():+.3f} ± {held.std():.3f} | {(held>0).mean()*100:.0f}% positive (n={len(held)}, SE={se:.3f}, t={held.mean()/ (se+1e-9):.1f})")
    print(f"\n  → directional effect {'SURVIVES held-out' if held.mean() > 2*se else 'does NOT clearly survive held-out (≈ noise)'}")
