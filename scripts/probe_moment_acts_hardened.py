#!/usr/bin/env python3
"""Phase-1 probe -- HARDENED (torch). Addresses "is the linear model well-trained?".

vs the quick numpy probe (least-squares-on-one-hot, fixed lambda, 1 seed), this uses:
  - multinomial LOGISTIC regression, L-BFGS (strong-wolfe) to convergence  [proper linear probe]
  - an MLP ceiling (1x256 ReLU, Adam + early stop)                          [non-linear: "absent" means absent]
  - L2 (weight_decay) chosen by CV from a small sweep                       [tuned regularization]
  - GroupKFold by episode x multiple seeds -> mean +/- std                   [error bars]
  - a label-shuffle NULL per target                                         [leakage sanity -> must be ~chance]

  cd $REPO && CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/probe_moment_acts_hardened.py \
      --acts /tmp/robomme_eval/probe/acts_hamlet.npz
"""
import argparse, re
import numpy as np
import torch, torch.nn as nn

DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ---------- splits / standardize ----------
def group_folds(groups, nfold, seed):
    ug = np.unique(groups)
    perm = np.random.RandomState(seed).permutation(ug)
    for f in np.array_split(perm, min(nfold, len(ug))):
        te = np.isin(groups, f)
        if te.any() and (~te).any():
            yield ~te, te


def zfit(Xtr, Xte):
    mu = Xtr.mean(0, keepdims=True); sd = Xtr.std(0, keepdims=True) + 1e-6
    return (Xtr - mu) / sd, (Xte - mu) / sd


# ---------- linear logistic (L-BFGS, convex -> global opt) ----------
def logreg_fit_pred(Xtr, ytr, Xte, K, wd):
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=DEV)
    ytr_t = torch.tensor(ytr, dtype=torch.long, device=DEV)
    Xte_t = torch.tensor(Xte, dtype=torch.float32, device=DEV)
    W = torch.zeros(Xtr.shape[1], K, device=DEV, requires_grad=True)
    b = torch.zeros(K, device=DEV, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], max_iter=120, line_search_fn="strong_wolfe")
    cel = nn.CrossEntropyLoss()

    def closure():
        opt.zero_grad()
        loss = cel(Xtr_t @ W + b, ytr_t) + wd * (W * W).sum()
        loss.backward()
        return loss
    opt.step(closure)
    with torch.no_grad():
        return (Xte_t @ W + b).argmax(1).cpu().numpy()


# ---------- MLP ceiling ----------
def mlp_fit_pred(Xtr, ytr, Xte, K, task="clf", seed=0, hidden=256, epochs=400, wd=1e-4, lr=1e-3):
    torch.manual_seed(seed)
    n = Xtr.shape[0]; vi = np.random.RandomState(seed).permutation(n)
    nval = max(16, n // 6); val, tr = vi[:nval], vi[nval:]
    Xt = torch.tensor(Xtr, dtype=torch.float32, device=DEV)
    out_dim = K if task == "clf" else 1
    yt = torch.tensor(ytr if task == "clf" else ytr.reshape(-1, 1),
                      dtype=torch.long if task == "clf" else torch.float32, device=DEV)
    net = nn.Sequential(nn.Linear(Xtr.shape[1], hidden), nn.ReLU(), nn.Dropout(0.1),
                        nn.Linear(hidden, out_dim)).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    lossf = nn.CrossEntropyLoss() if task == "clf" else nn.MSELoss()
    best = (1e18, None); bad = 0
    for _ in range(epochs):
        net.train(); opt.zero_grad()
        loss = lossf(net(Xt[tr]), yt[tr]); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            vl = lossf(net(Xt[val]), yt[val]).item()
        if vl < best[0] - 1e-5:
            best = (vl, {k: v.detach().clone() for k, v in net.state_dict().items()}); bad = 0
        else:
            bad += 1
            if bad > 50:
                break
    if best[1]:
        net.load_state_dict(best[1])
    net.eval()
    with torch.no_grad():
        o = net(torch.tensor(Xte, dtype=torch.float32, device=DEV))
        return (o.argmax(1).cpu().numpy() if task == "clf" else o.squeeze(1).cpu().numpy())


def ridge_pred(Xtr, ytr, Xte, lam):
    Xt = torch.tensor(Xtr, dtype=torch.float32, device=DEV)
    yt = torch.tensor(ytr, dtype=torch.float32, device=DEV)
    muy = yt.mean()
    d = Xt.shape[1]
    W = torch.linalg.solve(Xt.T @ Xt + lam * torch.eye(d, device=DEV), Xt.T @ (yt - muy))
    return (torch.tensor(Xte, dtype=torch.float32, device=DEV) @ W + muy).cpu().numpy()


# ---------- CV drivers ----------
def cv_classify(X, y, groups, probe, seeds=(0, 1, 2), nfold=5, wd_grid=(1e-2, 1e-1)):
    classes = np.unique(y); K = len(classes); cmap = {c: i for i, c in enumerate(classes)}
    yi = np.array([cmap[v] for v in y])
    chance = np.bincount(yi).max() / len(yi)
    grids = wd_grid if probe == "logreg" else (1e-4,)
    best = (-1, None)
    for wd in grids:
        accs = []
        for s in seeds:
            for tr, te in group_folds(groups, nfold, s):
                Xtr, Xte = zfit(X[tr], X[te])
                if len(np.unique(yi[tr])) < 2:
                    continue
                if probe == "logreg":
                    p = logreg_fit_pred(Xtr, yi[tr], Xte, K, wd)
                else:
                    p = mlp_fit_pred(Xtr, yi[tr], Xte, K, "clf", seed=s)
                accs.append((p == yi[te]).mean())
        m = np.mean(accs) if accs else -1
        if m > best[0]:
            best = (m, (np.std(accs) if accs else 0, wd))
    return best[0], best[1][0], chance, best[1][1]


def cv_regress(X, y, groups, probe, seeds=(0, 1, 2), nfold=5, lam_grid=(10.0, 50.0, 200.0)):
    grids = lam_grid if probe == "ridge" else (None,)
    best = (-1e9, None)
    for lam in grids:
        r2s = []
        for s in seeds:
            for tr, te in group_folds(groups, nfold, s):
                Xtr, Xte = zfit(X[tr], X[te])
                pred = ridge_pred(Xtr, y[tr], Xte, lam) if probe == "ridge" else \
                    mlp_fit_pred(Xtr, y[tr].astype(np.float32), Xte, 1, "reg", seed=s)
                ss = ((y[te] - pred) ** 2).sum(); tot = ((y[te] - y[te].mean()) ** 2).sum() + 1e-9
                r2s.append(1 - ss / tot)
        m = np.mean(r2s) if r2s else -1e9
        if m > best[0]:
            best = (m, (np.std(r2s) if r2s else 0, lam))
    return best[0], best[1][0], best[1][1]


# ---------- count_so_far ----------
ORD = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6,
       "1st": 1, "2nd": 2, "3rd": 3, "4th": 4, "5th": 5, "6th": 6,
       "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6}


def count_so_far(sg):
    s = str(sg).lower(); best = None
    for m in re.finditer(r"for the (\w+) time", s):
        v = ORD.get(m.group(1))
        if v:
            best = v if best is None else max(best, v)
    if best is None:
        m = re.search(r"(\d+)\s*(?:st|nd|rd|th)?\s*time", s)
        if m:
            best = int(m.group(1))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", required=True)
    ap.add_argument("--early-frac", type=float, default=0.4)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    seeds = tuple(range(args.seeds))

    d = np.load(args.acts, allow_pickle=True)
    MP = d["m_prime"].astype(np.float32); MA = d["mem_agg"].astype(np.float32)
    ep = d["episode"]; ti = d["task_index"]; cs = d["call_step"]; nc = d["n_calls"]; sg = d["subgoal"]
    phase = (cs / np.maximum(nc - 1, 1)).astype(np.float32)
    has_mem = bool(np.abs(MA).sum() > 0)
    reps = [("m'_t", MP)] + ([("mem_agg", MA)] if has_mem else [])
    print(f"device={DEV} | {MP.shape[0]} steps | {len(np.unique(ep))} eps | seeds={seeds}\n")

    def row(name, fn_per_rep, fmt="{:.3f}"):
        cells = [fn_per_rep(X) for _, X in reps]
        print(f"{name:<26}" + "".join((fmt.format(m) + f"±{sd:.3f}").rjust(20) for (m, sd) in cells))

    print(f"{'TARGET  [probe]':<26}" + "".join(rn.rjust(20) for rn, _ in reps) + "    (chance/null)")
    print("-" * (26 + 20 * len(reps) + 18))

    # phase (R^2)
    for pr in ("ridge", "mlp"):
        row(f"phase R2 [{pr}]", lambda X: cv_regress(X, phase, ep, pr, seeds)[:2])

    # categorical targets
    for tname, y, sub in [("subtask", sg, None), ("task_id", ti, None)]:
        for pr in ("logreg", "mlp"):
            res = [cv_classify(X, y, ep, pr, seeds) for _, X in reps]
            ch = res[0][2]
            print(f"{tname+' acc ['+pr+']':<26}" +
                  "".join((f"{a:.3f}±{s:.3f}").rjust(20) for a, s, _, _ in res) +
                  f"    chance {ch:.3f}")
        # label-shuffle null (logreg)
        rng = np.random.RandomState(0); yp = y.copy(); rng.shuffle(yp)
        nullres = [cv_classify(X, yp, ep, "logreg", (0,)) for _, X in reps]
        print(f"{'  '+tname+' NULL [shuf]':<26}" +
              "".join((f"{a:.3f}").rjust(20) for a, s, _, _ in nullres) + "    (want ~chance)")

    # count_so_far ALL / EARLY / LATE
    csf = np.array([count_so_far(s) or -1 for s in sg]); m0 = csf >= 1
    print(f"\n[count_so_far] {m0.sum()} steps, {len(np.unique(ep[m0]))} eps, values {sorted(set(csf[m0].tolist()))}")
    for tag, sel in [("ALL", m0), ("EARLY", m0 & (phase < args.early_frac)), ("LATE", m0 & (phase >= args.early_frac))]:
        if sel.sum() < 30 or len(np.unique(ep[sel])) < 5:
            print(f"  count_so_far {tag}: too few ({sel.sum()})"); continue
        for pr in ("logreg", "mlp"):
            res = [cv_classify(X[sel], csf[sel], ep[sel], pr, seeds) for _, X in reps]
            ch = res[0][2]
            print(f"  count {tag} acc [{pr}]".ljust(26) +
                  "".join((f"{a:.3f}±{s:.3f}").rjust(20) for a, s, _, _ in res) + f"    chance {ch:.3f}")


if __name__ == "__main__":
    main()
