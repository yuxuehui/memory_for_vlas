#!/usr/bin/env python3
"""Is each relevance channel EVENT-like or PLATEAU-like over time?

VLA-Pruner (arXiv:2511.16449) reports top-K attended-patch overlap of 0.89/0.93/0.95 between
CONSECUTIVE control steps, and uses that continuity to predict the current step's action
attention from the past w steps via a decaying-window EMA. We never measured the analogous
number for our models, and it decides two things:

  1. whether "dup" (same (view,r,c) selected at many timesteps) is redundancy or a TEMPORAL
     TRACE of a few key locations — the latter is what Counting-style tasks need;
  2. whether an EMA-RESIDUAL score (S_t - EMA of past) would preserve or destroy that
     information: it fires at each onset of a spiky signal, but only once for a plateau.

For each channel (act_L13 / tail_L15 / tail_L13 / nov) we report top-K overlap at lags
1,2,4,8 (lag-1 alone cannot separate a plateau from a slow drift) plus a persistence-run
statistic: how many consecutive scored steps a cell stays in the top-K once it enters.

Run: cd HAMLET-Isaac-GR00T-N1d6 && CUDA_VISIBLE_DEVICES=<gpu> NO_ALBUMENTATIONS_UPDATE=1 \
       .venv/bin/python .../temporal_persistence.py [--models expd_framesamp]
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/users/xuehui/myfile/Markdown/patch_memory_labels")

from gen_act_vs_tail_patches import (EPS, MODELS, build_policy, episode_signals,
                                     install_cross_attn_capture, install_self_attn_capture)
from gen_patch_labels import load_episode

OUT = Path("/home/users/xuehui/myfile/Markdown/patch_memory_labels")
CHANNELS = ["act", "tail15", "tail13", "nov"]
FRACS = [0.125, 0.25, 0.50]     # VLA-Pruner's top-12.5% / 25% / 50%
LAGS = [1, 2, 4, 8]


def topk_sets(recs, ch, k):
    """Per scored step: the set of flat cell indices in that step's top-k for this channel."""
    out = []
    for r in recs:
        g = r.get(ch)
        if g is None:
            out.append(None); continue
        flat = g.reshape(-1)
        out.append(set(np.argpartition(-flat, k - 1)[:k].tolist()))
    return out


def overlap_at_lag(sets, lag):
    vals = [len(a & b) / len(a) for a, b in zip(sets, sets[lag:]) if a and b]
    return float(np.mean(vals)) if vals else float("nan")


def run_lengths(sets):
    """Mean number of CONSECUTIVE scored steps a cell stays inside the top-k once it enters."""
    active, runs = {}, []
    for i, s in enumerate(sets):
        if s is None:
            continue
        for c in list(active):
            if c not in s:
                runs.append(i - active.pop(c))
        for c in s:
            active.setdefault(c, i)
    runs += [len(sets) - st for st in active.values()]
    return float(np.mean(runs)) if runs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="expd_framesamp,HAMLET")
    args = ap.parse_args()
    install_self_attn_capture(); install_cross_attn_capture()
    agg = defaultdict(lambda: defaultdict(list))

    for mname in args.models.split(","):
        policy, V, pmeta = build_policy(MODELS[mname])
        for task, eplist in EPS.items():
            for ep in eplist:
                df, instr = load_episode(ep)
                recs, side = episode_signals(policy, V, pmeta, df, instr, ep)
                if len(recs) < max(LAGS) + 2:
                    print(f"[{mname}/{task}/ep{ep}] too short"); continue
                n_cells = 2 * side * side
                for ch in CHANNELS:
                    if ch not in recs[0]:
                        continue
                    for fr in FRACS:
                        k = max(1, int(round(fr * n_cells)))
                        sets = topk_sets(recs, ch, k)
                        for lag in LAGS:
                            agg[(mname, ch, fr)][f"lag{lag}"].append(overlap_at_lag(sets, lag))
                        agg[(mname, ch, fr)]["run"].append(run_lengths(sets))
                print(f"[{mname}/{task}/ep{ep}] steps={len(recs)} done", flush=True)
        del policy
        import torch; torch.cuda.empty_cache()

    hdr = f"{'model':16}{'channel':9}{'top-K':>7}" + "".join(f"{f'lag{l}':>8}" for l in LAGS) + f"{'run(steps)':>12}"
    print("\n=== top-K attended-cell overlap across scored steps (stride 8) ===")
    print("(VLA-Pruner, consecutive CONTROL steps: 0.89 / 0.93 / 0.95 at top-12.5/25/50%)")
    print(hdr)
    lines = [hdr]
    for (m, ch, fr), d in agg.items():
        row = (f"{m:16}{ch:9}{int(fr*100):>6}%"
               + "".join(f"{np.nanmean(d[f'lag{l}']):>8.3f}" for l in LAGS)
               + f"{np.nanmean(d['run']):>12.2f}")
        print(row); lines.append(row)
    (OUT / "temporal_persistence.txt").write_text("\n".join(lines) + "\n")
    print("\nPERSIST_DONE")


if __name__ == "__main__":
    main()
