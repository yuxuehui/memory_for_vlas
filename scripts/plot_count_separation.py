#!/usr/bin/env python3
"""Better view of 'does the action separate by count' than unsupervised PCA/t-SNE.
Supervised projections onto the count direction:
  (A) action · ŵ  (the fitted count axis)  as a violin per count
  (B) LDA-2D on count  (best *linear* count separation; PCA-pre-reduced)
  (C) t-SNE colored by count  IF sklearn is available (else skipped, with note)

  python scripts/plot_count_separation.py --npz /tmp/robomme_eval/probe/swap_dir2.npz
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--npz", required=True)
ap.add_argument("--out", default="/tmp/robomme_eval/probe/count_separation.png")
a = ap.parse_args()

d = np.load(a.npz, allow_pickle=True)
by = d["by"].astype(np.float64)          # (n, 320) baseline actions
count = d["bx"][:, 0].astype(int)        # count_so_far
w = d["w"].astype(np.float64); w = w / (np.linalg.norm(w) + 1e-9)
classes = sorted(set(count.tolist()))
Xc = by - by.mean(0)


def lda_2d(X, y, pca_k=40):
    Xz = X - X.mean(0)
    U, S, Vt = np.linalg.svd(Xz, full_matrices=False)
    k = min(pca_k, Vt.shape[0], Xz.shape[0] - len(set(y)) - 1)
    Z = Xz @ Vt[:k].T
    cls = np.unique(y); mu = Z.mean(0); Sw = np.zeros((k, k)); Sb = np.zeros((k, k))
    for c in cls:
        Zc = Z[y == c]; mc = Zc.mean(0)
        Sw += (Zc - mc).T @ (Zc - mc)
        Sb += len(Zc) * np.outer(mc - mu, mc - mu)
    ev, evec = np.linalg.eig(np.linalg.solve(Sw + 1e-2 * np.eye(k), Sb))
    W = evec[:, np.argsort(-ev.real)[:2]].real
    return Z @ W


fig, ax = plt.subplots(1, 3, figsize=(17, 5))

# (A) violin of action·ŵ per count
proj = Xc @ w
data = [proj[count == c] for c in classes]
ax[0].violinplot(data, positions=classes, showmeans=True)
means = [v.mean() for v in data]
ax[0].plot(classes, means, "r-o", lw=2, label="per-count mean")
r = np.corrcoef(count, proj)[0, 1]
ax[0].set_title(f"(A) action · ŵ  by count  (supervised axis)\nin-sample corr={r:+.2f}  |  CV count-acc=.46 is the rigorous #")
ax[0].set_xlabel("count_so_far"); ax[0].set_ylabel("action · ŵ"); ax[0].legend()

# (B) LDA-2D
L = lda_2d(by, count)
sc = ax[1].scatter(L[:, 0], L[:, 1], c=count, cmap="viridis", s=26, edgecolor="k", linewidth=0.2)
plt.colorbar(sc, ax=ax[1], label="count")
ax[1].set_title("(B) LDA-2D on count (best *linear* separation)")
ax[1].set_xlabel("LD1"); ax[1].set_ylabel("LD2")

# (C) t-SNE if available
try:
    from sklearn.manifold import TSNE
    T = TSNE(n_components=2, perplexity=min(30, len(by) // 4), init="pca", random_state=0).fit_transform(by)
    sc = ax[2].scatter(T[:, 0], T[:, 1], c=count, cmap="viridis", s=26, edgecolor="k", linewidth=0.2)
    plt.colorbar(sc, ax=ax[2], label="count")
    ax[2].set_title("(C) t-SNE colored by count (unsupervised)")
except Exception as e:
    # fallback: PCA-2D, to make the unsupervised-can't-see-count point
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False); P = Xc @ Vt[:2].T
    sc = ax[2].scatter(P[:, 0], P[:, 1], c=count, cmap="viridis", s=26, edgecolor="k", linewidth=0.2)
    plt.colorbar(sc, ax=ax[2], label="count")
    ax[2].set_title("(C) PCA-2D, unsupervised (t-SNE n/a: no sklearn)\nunsupervised → dominated by task/phase, count smeared")

plt.tight_layout()
plt.savefig(a.out, dpi=110)
print("saved", a.out)
