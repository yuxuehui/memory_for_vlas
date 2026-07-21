#!/usr/bin/env python3
"""Phase-1 probe (moment-token test plan, P1). Numpy-only (no sklearn).

Linear-decode targets from m'_t (current-frame) vs mem_agg (history-aggregated):
  - phase        : call_step / n_calls            (regression, R^2)         -> does it encode WHEN
  - subtask      : simple_subgoal string          (classification, acc)     -> current sub-task
  - task_id      : task_index                      (classification, acc)     -> task identity
  - count_so_far : ordinal in "for the Nth time"  (classification, acc)     -> HISTORY-integrated var

Decisive contrast: a history variable (count_so_far) should decode from **mem_agg >> m'_t**,
and the gap should be larger on EARLY steps (where the current frame is least informative).
If mem_agg ~= m'_t everywhere, the "memory" is not adding history -> epiphenomenal.

  python3 scripts/probe_moment_acts.py --acts /tmp/robomme_eval/probe/acts_hamlet.npz
"""
import argparse, re
import numpy as np


# ---------- numpy linear probes (closed-form) ----------
def _zfit(X):
    mu = X.mean(0); sd = X.std(0) + 1e-6
    return mu, sd


def ridge_r2_cv(X, y, groups, lam=50.0, nfold=5, seed=0):
    ug = np.unique(groups)
    if len(ug) < nfold:
        nfold = max(2, len(ug))
    folds = np.array_split(np.random.RandomState(seed).permutation(ug), nfold)
    r2 = []
    for f in folds:
        te = np.isin(groups, f); tr = ~te
        if te.sum() < 2 or tr.sum() < 5:
            continue
        mu, sd = _zfit(X[tr]); Xtr = (X[tr] - mu) / sd; Xte = (X[te] - mu) / sd
        muy = y[tr].mean(); ytr = y[tr] - muy
        d = Xtr.shape[1]
        w = np.linalg.solve(Xtr.T @ Xtr + lam * np.eye(d), Xtr.T @ ytr)
        pred = Xte @ w + muy
        ss = ((y[te] - pred) ** 2).sum(); tot = ((y[te] - y[te].mean()) ** 2).sum() + 1e-9
        r2.append(1 - ss / tot)
    return float(np.mean(r2)) if r2 else float("nan")


def lsq_acc_cv(X, y, groups, lam=50.0, nfold=5, seed=0):
    """Least-squares (one-hot) linear classifier, group K-fold. Returns (acc, chance)."""
    classes = np.unique(y); K = len(classes); cmap = {c: i for i, c in enumerate(classes)}
    yi = np.array([cmap[v] for v in y])
    ug = np.unique(groups)
    if len(ug) < nfold:
        nfold = max(2, len(ug))
    folds = np.array_split(np.random.RandomState(seed).permutation(ug), nfold)
    accs = []
    for f in folds:
        te = np.isin(groups, f); tr = ~te
        if te.sum() < 2 or tr.sum() < 5 or len(np.unique(yi[tr])) < 2:
            continue
        mu, sd = _zfit(X[tr]); Xtr = (X[tr] - mu) / sd; Xte = (X[te] - mu) / sd
        Xtr = np.hstack([Xtr, np.ones((Xtr.shape[0], 1))]); Xte = np.hstack([Xte, np.ones((Xte.shape[0], 1))])
        Y = np.eye(K)[yi[tr]]
        d = Xtr.shape[1]
        W = np.linalg.solve(Xtr.T @ Xtr + lam * np.eye(d), Xtr.T @ Y)
        pred = (Xte @ W).argmax(1)
        accs.append((pred == yi[te]).mean())
    chance = np.bincount(yi).max() / len(yi)
    return (float(np.mean(accs)) if accs else float("nan")), float(chance)


# ---------- history-variable extractor: count-so-far from the subgoal ----------
ORD = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6,
       "1st": 1, "2nd": 2, "3rd": 3, "4th": 4, "5th": 5, "6th": 6,
       "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6}


def count_so_far(subgoal):
    s = str(subgoal).lower()
    best = None
    for m in re.finditer(r"for the (\w+) time", s):
        v = ORD.get(m.group(1))
        if v: best = v if best is None else max(best, v)
    if best is None:
        m = re.search(r"(\d+)\s*(?:st|nd|rd|th)?\s*time", s)
        if m: best = int(m.group(1))
    return best  # None if not a counting subgoal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", required=True)
    ap.add_argument("--early-frac", type=float, default=0.4, help="phase < this = EARLY")
    args = ap.parse_args()

    d = np.load(args.acts, allow_pickle=True)
    MP = d["m_prime"].astype(np.float32); MA = d["mem_agg"].astype(np.float32)
    ep = d["episode"]; ti = d["task_index"]; cs = d["call_step"]; nc = d["n_calls"]
    sg = d["subgoal"]
    phase = cs / np.maximum(nc - 1, 1)
    has_mem = bool(np.abs(MA).sum() > 0)
    print(f"loaded {MP.shape[0]} steps | {len(np.unique(ep))} eps | {len(np.unique(ti))} task_idx | "
          f"mem_agg {'present' if has_mem else 'EMPTY (vanilla)'}\n")

    reps = [("m'_t (current frame)", MP)]
    if has_mem:
        reps.append(("mem_agg (history) ", MA))

    def line(name, fn):
        cells = []
        for rn, X in reps:
            cells.append(fn(X))
        return name, cells

    print(f"{'target':<26}" + "".join(f"{rn:>26}" for rn, _ in reps))
    print("-" * (26 + 26 * len(reps)))

    # phase (R^2)
    row = [ridge_r2_cv(X, phase.astype(np.float32), ep) for _, X in reps]
    print(f"{'phase  (R^2)':<26}" + "".join(f"{v:>26.3f}" for v in row))

    # subtask (acc); restrict to a few common task_idx-agnostic; encode globally
    acc = [lsq_acc_cv(X, sg, ep) for _, X in reps]
    ch = acc[0][1]
    print(f"{f'subtask acc (chance {ch:.2f})':<26}" + "".join(f"{a:>26.3f}" for a, _ in acc))

    # task_id (acc)
    acc = [lsq_acc_cv(X, ti, ep) for _, X in reps]
    ch = acc[0][1]
    print(f"{f'task_id acc (chance {ch:.2f})':<26}" + "".join(f"{a:>26.3f}" for a, _ in acc))

    # count_so_far (acc) -- the HISTORY variable, on counting subgoals only
    csf = np.array([count_so_far(s) if count_so_far(s) is not None else -1 for s in sg])
    mask = csf >= 1
    print(f"\n[count_so_far] {mask.sum()} counting-subgoal steps over {len(np.unique(ep[mask]))} eps "
          f"(values {sorted(set(csf[mask].tolist()))})")
    if mask.sum() > 30 and len(np.unique(ep[mask])) >= 4:
        for tag, sel in [("ALL  ", mask),
                         ("EARLY", mask & (phase < args.early_frac)),
                         ("LATE ", mask & (phase >= args.early_frac))]:
            if sel.sum() < 20 or len(np.unique(ep[sel])) < 4:
                print(f"  count_so_far {tag}: too few ({sel.sum()})"); continue
            accs = [lsq_acc_cv(X[sel], csf[sel], ep[sel]) for _, X in reps]
            ch = accs[0][1]
            print(f"  count_so_far {tag} acc (chance {ch:.2f}): " +
                  "  ".join(f"{rn.strip()}={a:.3f}" for (rn, _), (a, _) in zip(reps, accs)))
        print("\n  >>> DECISIVE: if mem_agg >> m'_t on count_so_far (esp. EARLY) -> real history/memory;"
              "\n      if mem_agg ~= m'_t -> memory not adding history (epiphenomenal).")
    else:
        print("  (not enough counting-subgoal steps; widen the dump to PickXtimes/SwingXtimes/ButtonUnmask)")


if __name__ == "__main__":
    main()
