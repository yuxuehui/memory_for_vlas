#!/usr/bin/env python3
"""Held-out test: does the BASELINE ACTION linearly predict count? (replaces the in-sample LDA panel.)
Reconstructs per-baseline episode id from the swap arrays, then runs the hardened CV (logreg,
GroupKFold by episode, mean±std, label-shuffle null). Near chance ⇒ panel B's LDA was in-sample optimistic.

  python scripts/probe_count_from_action.py --npz /tmp/robomme_eval/probe/swap_dir2.npz
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import probe_moment_acts_hardened as H

ap = argparse.ArgumentParser(); ap.add_argument("--npz", required=True); ap.add_argument("--early-frac", type=float, default=0.4)
a = ap.parse_args()
d = np.load(a.npz, allow_pickle=True)
by = d["by"].astype(np.float32); count = d["bx"][:, 0].astype(int); phase = d["bx"][:, 1]
sw_ep, sw_t = d["sw_ep"], d["sw_t"]

# reconstruct episode per baseline row (first-appearance order of (ep,t) == baseX append order)
seen, seenset = [], set()
for e, t in zip(sw_ep, sw_t):
    k = (int(e), int(t))
    if k not in seenset:
        seenset.add(k); seen.append(k)
assert len(seen) == len(by), (len(seen), len(by))
ep = np.array([k[0] for k in seen])
print(f"{len(by)} baseline steps · {len(np.unique(ep))} episodes · counts {sorted(set(count.tolist()))}\n")

print("count-from-ACTION  (held-out, GroupKFold-by-episode, logreg, 3 seeds):")
for tag, sel in [("ALL  ", np.ones(len(by), bool)),
                 ("EARLY", phase < a.early_frac), ("LATE ", phase >= a.early_frac)]:
    if sel.sum() < 20 or len(np.unique(ep[sel])) < 4:
        print(f"  {tag}: too few"); continue
    acc, std, chance, _ = H.cv_classify(by[sel], count[sel], ep[sel], "logreg")
    # label-shuffle null
    rng = np.random.RandomState(0); ys = count[sel].copy(); rng.shuffle(ys)
    nul, *_ = H.cv_classify(by[sel], ys, ep[sel], "logreg", seeds=(0,))
    print(f"  {tag}: acc = {acc:.3f} ± {std:.3f}   (chance {chance:.3f}, shuffle-null {nul:.3f})")

print("\nrecall — count from the MEMORY rep (hardened.log): mem_agg LATE .46 / m′ₜ LATE .40")
