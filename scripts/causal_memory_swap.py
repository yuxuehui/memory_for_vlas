#!/usr/bin/env python3
"""P3 causal memory-SWAP. Tests whether the count *in* the aggregated memory is *used*.

At a recipient decision step we OVERRIDE the memory-transformer output (mem_agg) with one
recorded from a DIFFERENT episode at a chosen count c', then measure the relative action change:
  dact(c') = || a(swap=c') - a0 || / ||a0||      (a0 = action with the real memory)
If the count is causally used, dact should grow with |c' - c_true|, and exceed a same-count
swap and a random-vector swap of matched norm.

Reuses visualize_memory_attention (build_policy/load_episode/dec) + the causal_ablation idiom
(seed before each call; save/restore policy._memory_cache[sid]). Counting episodes are read from
the P1 dump so we don't re-scan the dataset.

  cd $REPO && CUDA_VISIBLE_DEVICES=2 NO_ALBUMENTATIONS_UPDATE=1 .venv/bin/python \
      scripts/causal_memory_swap.py --model /tmp/robomme_eval/model \
      --acts /tmp/robomme_eval/probe/acts_hamlet.npz --n-donor 16 --n-recip 12 \
      --out /tmp/robomme_eval/probe/swap.npz
"""
import argparse, os, sys, re, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
import visualize_memory_attention as V

CAP = {"rec": None}    # last memory-transformer output (1, K*nq, d)
SWAP = {"out": None}   # if set, this tensor REPLACES the memory-transformer output


def hook_mt(mod, inp, out):
    CAP["rec"] = out.detach().clone()
    if SWAP["out"] is not None:
        return SWAP["out"]


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


def seed_all(s=0):
    torch.manual_seed(s); torch.cuda.manual_seed_all(s); np.random.seed(s); random.seed(s)


def flat(a):
    return np.concatenate([np.asarray(v, dtype=np.float32).reshape(-1) for v in a.values()])


def reldiff(a, b):
    return float(np.linalg.norm(flat(a) - flat(b)) / (np.linalg.norm(flat(b)) + 1e-8))


def get(policy, obs, sid, reset):
    seed_all()
    return policy.get_action(obs, options={"session_ids": [sid], "reset_memory": [reset]})[0]


def obs_from(row, pm, instr):
    vk, sk, lk = pm["vkeys"], pm["skeys"], pm["lkey"]
    front = V.dec(row["image"]); wrist = V.dec(row["wrist_image"]); st = np.asarray(row["state"], np.float32)
    return {"video": {vk[0]: front[None, None], vk[1]: wrist[None, None]},
            "state": {sk[0]: st[None, None, :7], sk[1]: st[None, None, 7:8]},
            "language": {lk: [[instr]]}}


def cget(policy, sid):
    c = policy._memory_cache.get(sid)
    return c.clone() if c is not None else None


def cset(policy, sid, c):
    if c is not None:
        policy._memory_cache[sid] = c.clone()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--repo", default="/home/users/xuehui/myfile/HAMLET-Isaac-GR00T-N1d6")
    ap.add_argument("--acts", required=True)
    ap.add_argument("--n-donor", type=int, default=16)
    ap.add_argument("--n-recip", type=int, default=12)
    ap.add_argument("--warmup", type=int, default=2, help="min ci before measuring (real history)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="/tmp/robomme_eval/probe/swap.npz")
    args = ap.parse_args()

    policy, pm = V.build_policy(args.model, args.device)
    stride = pm["stride"]
    policy.model.action_head.memory_transformer.register_forward_hook(hook_mt)

    d = np.load(args.acts, allow_pickle=True)
    cnt_eps = sorted({int(e) for e, s in zip(d["episode"], d["subgoal"]) if count_so_far(s)})
    donor_eps = cnt_eps[:args.n_donor]
    recip_eps = cnt_eps[args.n_donor:args.n_donor + args.n_recip]
    print(f"counting eps {len(cnt_eps)} | donors {len(donor_eps)} | recipients {len(recip_eps)}")

    # ---- 1) DONOR BANK: count -> [(full mem-transformer output, ep)] ----
    donors = {}
    for ep in donor_eps:
        df, instr = V.load_episode(args.repo, ep); sid = f"don{ep}"
        for ci, t in enumerate(range(0, len(df), stride)):
            SWAP["out"] = None; CAP["rec"] = None
            get(policy, obs_from(df.iloc[t], pm, instr), sid, ci == 0)
            c = count_so_far(df.iloc[t].get("simple_subgoal", ""))
            if ci >= args.warmup and c and CAP["rec"] is not None:
                donors.setdefault(int(c), []).append((CAP["rec"].clone(), ep))
    print("donor bank:", {k: len(v) for k, v in sorted(donors.items())})
    donor_counts = sorted(donors)
    if len(donor_counts) < 2:
        print("ERROR: <2 distinct donor counts"); return
    # one fixed donor per count (first available), for consistency
    fixed = {c: donors[c][0] for c in donor_counts}

    # ---- 2) RECIPIENTS ----
    rows = []
    baseX, baseY = [], []       # (count, phase) -> flat(a0), for the count-axis fit
    dirr = []                   # per swap: dict(ep,t,c,cprime, delta=flat(a)-flat(a0))
    for ep in recip_eps:
        df, instr = V.load_episode(args.repo, ep); sid = f"rec{ep}"
        ncalls = len(range(0, len(df), stride))
        for ci, t in enumerate(range(0, len(df), stride)):
            obs = obs_from(df.iloc[t], pm, instr)
            c = count_so_far(df.iloc[t].get("simple_subgoal", ""))
            pre = cget(policy, sid)
            SWAP["out"] = None; CAP["rec"] = None
            a0 = get(policy, obs, sid, ci == 0)          # baseline (advances cache)
            post = cget(policy, sid); ref = CAP["rec"]
            if ci < args.warmup or c is None or ref is None:
                continue
            fa0 = flat(a0)
            baseX.append([float(c), ci / max(ncalls - 1, 1)]); baseY.append(fa0)
            for cprime in donor_counts:
                donor, de = fixed[cprime]
                if de == ep:                              # avoid self-donor
                    alt = next(((o, e) for (o, e) in donors[cprime] if e != ep), None)
                    if alt is None:
                        continue
                    donor = alt[0]
                cset(policy, sid, pre); SWAP["out"] = donor
                a = get(policy, obs, sid, False)
                rows.append(dict(ep=ep, t=int(t), c=int(c), cprime=int(cprime),
                                 dact=reldiff(a, a0), kind="swap"))
                dirr.append(dict(ep=ep, t=int(t), c=int(c), cprime=int(cprime), delta=flat(a) - fa0))
            # random-memory control (matched norm)
            cset(policy, sid, pre)
            rnd = torch.randn_like(ref); rnd = rnd / rnd.norm() * ref.norm()
            SWAP["out"] = rnd
            a = get(policy, obs, sid, False)
            rows.append(dict(ep=ep, t=int(t), c=int(c), cprime=-1, dact=reldiff(a, a0), kind="random"))
            SWAP["out"] = None
            cset(policy, sid, post)                       # restore for correct continuation

    # ---- 3) magnitude analysis ----
    sw = [r for r in rows if r["kind"] == "swap"]
    rnd = [r for r in rows if r["kind"] == "random"]
    print(f"\n[MAGNITUDE] {len(sw)} swaps over {len(set(r['ep'] for r in sw))} recipient eps")
    if sw:
        dd = np.array([abs(r["cprime"] - r["c"]) for r in sw]); da = np.array([r["dact"] for r in sw])
        for k in sorted(set(dd.tolist())):
            m = dd == k
            print(f"  |dc|={k}: dact={da[m].mean():.4f} ± {da[m].std():.4f}  (n={int(m.sum())})")
        if dd.std() > 0:
            print(f"  corr(|c'-c|, dact) = {np.corrcoef(dd, da)[0,1]:+.3f}")
        same, diff = da[dd == 0], da[dd >= 1]
        if len(same) and len(diff):
            print(f"  same={same.mean():.4f} diff={diff.mean():.4f} ratio={diff.mean()/(same.mean()+1e-9):.2f}×")
    if rnd:
        print(f"  random={np.mean([r['dact'] for r in rnd]):.4f} [scale ref]")

    # ---- 4) DIRECTIONAL analysis (P3b) ----
    print(f"\n[DIRECTIONAL] count-axis from {len(baseX)} baseline steps")
    if len(baseX) >= 10 and dirr:
        BX = np.asarray(baseX, np.float64); BY = np.asarray(baseY, np.float64)
        X = np.column_stack([BX[:, 0], BX[:, 1], np.ones(len(BX))])   # count, phase, intercept
        coef, *_ = np.linalg.lstsq(X, BY, rcond=None)                 # (3, D)
        w = coef[0]; w = w / (np.linalg.norm(w) + 1e-9)               # count direction (phase partialled out)
        import collections
        proj = np.array([float(r["delta"] @ w) for r in dirr])
        dc = np.array([r["cprime"] - r["c"] for r in dirr], float)    # SIGNED count offset
        if dc.std() > 0:
            print(f"  corr(signed c'-c, proj on count-axis) = {np.corrcoef(dc, proj)[0,1]:+.3f}  (>0 ⇒ higher injected count → action toward 'more-done')")
        hi, lo = proj[dc > 0], proj[dc < 0]
        if len(hi) and len(lo):
            print(f"  proj when c'>c : {hi.mean():+.4f} (n={len(hi)})   c'<c : {lo.mean():+.4f} (n={len(lo)})   gap={hi.mean()-lo.mean():+.4f}")
        # within-step slope of proj vs signed dc (controls for step)
        per = collections.defaultdict(list)
        for r, p in zip(dirr, proj):
            per[(r["ep"], r["t"])].append((r["cprime"] - r["c"], p))
        slopes = [np.polyfit([x[0] for x in v], [x[1] for x in v], 1)[0]
                  for v in per.values() if len(v) >= 3 and np.std([x[0] for x in v]) > 0]
        if slopes:
            s = np.array(slopes)
            print(f"  within-step slope(proj vs c'-c): mean={s.mean():+.5f} ± {s.std():.5f}  ({(s>0).mean()*100:.0f}% positive, n={len(s)} steps)")

    save = dict(dact=np.array([r["dact"] for r in rows], np.float32),
                c=np.array([r["c"] for r in rows], np.int32),
                cprime=np.array([r["cprime"] for r in rows], np.int32),
                kind=np.array([r["kind"] for r in rows], object))
    if len(baseX) >= 10 and dirr:   # directional data for plotting
        save.update(bx=np.asarray(baseX, np.float32), by=np.asarray(baseY, np.float32),
                    w=w.astype(np.float32),
                    sw_c=np.array([r["c"] for r in dirr], np.int32),
                    sw_cp=np.array([r["cprime"] for r in dirr], np.int32),
                    sw_ep=np.array([r["ep"] for r in dirr], np.int32),
                    sw_t=np.array([r["t"] for r in dirr], np.int32),
                    sw_proj=proj.astype(np.float32),
                    sw_delta=np.stack([r["delta"] for r in dirr]).astype(np.float32))
    np.savez_compressed(args.out, **save)
    print(f"\nsaved {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
