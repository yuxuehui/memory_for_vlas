#!/usr/bin/env python3
"""Visualize the P3b count-axis directional readout from causal_memory_swap.py's enriched npz.
  python scripts/plot_count_axis.py --npz /tmp/robomme_eval/probe/swap_dir2.npz
"""
import argparse, collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--npz", required=True)
ap.add_argument("--out", default="/tmp/robomme_eval/probe/count_axis.png")
a = ap.parse_args()

d = np.load(a.npz, allow_pickle=True)
bx = d["bx"]; by = d["by"].astype(np.float64); w = d["w"].astype(np.float64)
count, phase = bx[:, 0], bx[:, 1]
sw_c, sw_cp, sw_ep, sw_t, proj = d["sw_c"], d["sw_cp"], d["sw_ep"], d["sw_t"], d["sw_proj"]

fig, ax = plt.subplots(2, 2, figsize=(13, 10))

# (1) PCA of baseline actions colored by count + the count axis ŵ projected in
X = by - by.mean(0)
U, S, Vt = np.linalg.svd(X, full_matrices=False)
PC = X @ Vt[:2].T
sc = ax[0, 0].scatter(PC[:, 0], PC[:, 1], c=count, cmap="viridis", s=24, edgecolor="k", linewidth=0.2)
wp = Vt[:2] @ (w / (np.linalg.norm(w) + 1e-9))
scale = 0.7 * np.abs(PC).max()
ax[0, 0].annotate("", xy=(wp[0] * scale, wp[1] * scale), xytext=(0, 0),
                  arrowprops=dict(arrowstyle="-|>", lw=3, color="red"))
ax[0, 0].text(wp[0] * scale, wp[1] * scale, "  ŵ (count axis)", color="red", fontweight="bold")
plt.colorbar(sc, ax=ax[0, 0], label="count_so_far")
ax[0, 0].set_title("(1) Baseline actions (PCA-2D), colored by count\naction drifts along ŵ as count↑")
ax[0, 0].set_xlabel("PC1"); ax[0, 0].set_ylabel("PC2")

# group swaps by recipient step
per = collections.defaultdict(list)
for c, cp, ep, t, p in zip(sw_c, sw_cp, sw_ep, sw_t, proj):
    per[(ep, t)].append((cp, p))

# (2) within-step proj vs c' (example steps)
ex = [k for k in per if len(per[k]) >= 4][:12]
for k in ex:
    v = sorted(per[k]); cs = [x[0] for x in v]; ps = [x[1] for x in v]
    ax[0, 1].plot(cs, ps, marker="o", alpha=0.55)
ax[0, 1].set_title("(2) Within-step: proj vs injected count c'\n(each line = one recipient step; mostly rising)")
ax[0, 1].set_xlabel("injected count c'"); ax[0, 1].set_ylabel("proj = Δaction · ŵ")

# (3) histogram of within-step slopes
slopes = []
for k, v in per.items():
    if len(v) >= 3:
        cs = np.array([x[0] for x in v]); ps = np.array([x[1] for x in v])
        if cs.std() > 0:
            slopes.append(np.polyfit(cs, ps, 1)[0])
slopes = np.array(slopes)
ax[1, 0].hist(slopes, bins=25, color="steelblue", edgecolor="k")
ax[1, 0].axvline(0, color="k", ls="--")
ax[1, 0].axvline(slopes.mean(), color="red", lw=2, label=f"mean = {slopes.mean():+.3f}")
ax[1, 0].set_title(f"(3) Within-step slope(proj vs c'):  {(slopes>0).mean()*100:.0f}% positive (n={len(slopes)})")
ax[1, 0].set_xlabel("slope (proj per +1 count)"); ax[1, 0].legend()

# (4) pooled mean proj vs signed offset (shows the Simpson confound: flat/− pooled)
dc = sw_cp - sw_c
xs = sorted(set(dc.tolist()))
m = [proj[dc == k].mean() for k in xs]
e = [proj[dc == k].std() / max(1, np.sqrt((dc == k).sum())) for k in xs]
ax[1, 1].errorbar(xs, m, yerr=e, marker="o", capsize=3, color="darkorange")
ax[1, 1].axhline(0, color="k", ls=":")
ax[1, 1].set_title("(4) Pooled mean proj vs (c'−c)\n(per-step offset confounds pooled slope → use panel 3)")
ax[1, 1].set_xlabel("c' − c (signed count offset)"); ax[1, 1].set_ylabel("mean proj")

plt.tight_layout()
plt.savefig(a.out, dpi=110)
print("saved", a.out)
