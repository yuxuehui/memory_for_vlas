#!/usr/bin/env python3
"""Causal test: does HAMLET's action actually USE the memory tokens at inference, despite the
action head putting only ~1% of its cross-attention mass on them?

At chosen replay steps, we run get_action three ways on the SAME observation + SAME memory cache
+ SAME flow-matching noise (fixed seed), differing only in an attention mask on the DiT cross-attn:
  - baseline : no mask
  - memory   : -inf on the last n_q KV columns (the memory-conditioned tokens) -> action can't see memory
  - random   : -inf on n_q RANDOM non-memory columns (control) -> remove an equal-size random set
Metric: Δaction = ||a_variant - a_base|| / ||a_base||  (relative L2 over the predicted chunk).

Read: Δmem >> Δrand  => memory is causally important (high info per attention, low-mass but used).
      Δmem ≈ Δrand ≈ 0 => memory is nearly inert at inference (the +10pp came from training-time effects).
"""
from __future__ import annotations
import argparse, collections
import numpy as np
import torch
import torch.nn.functional as F

# import data/model helpers from the viz script (same repo)
import scripts.visualize_memory_attention as V

ABLATE = {"mode": None, "nq": 4, "rand_cols": None}
LAST_LK = [None]
_orig_sdpa = F.scaled_dot_product_attention


def patched(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **kw):
    Lq, Lk = q.shape[-2], k.shape[-2]
    if Lq != Lk:  # DiT cross-attn (action queries x backbone KV)
        LAST_LK[0] = Lk
        if ABLATE["mode"] in ("memory", "random"):
            nq = ABLATE["nq"]
            cols = list(range(Lk - nq, Lk)) if ABLATE["mode"] == "memory" else ABLATE["rand_cols"]
            m = torch.zeros(Lk, device=q.device, dtype=torch.float32)
            m[cols] = float("-inf")
            am = m.view(1, 1, 1, Lk)
            if attn_mask is not None and attn_mask.dtype != torch.bool:
                am = am + attn_mask.float()
            return _orig_sdpa(q, k, v, attn_mask=am.to(q.dtype), dropout_p=dropout_p, is_causal=is_causal, scale=scale, **kw)
    return _orig_sdpa(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale, **kw)


F.scaled_dot_product_attention = patched
torch.nn.functional.scaled_dot_product_attention = patched


def seed_all(s=0):
    torch.manual_seed(s); torch.cuda.manual_seed_all(s); np.random.seed(s)


def flat(act):
    return np.concatenate([np.asarray(v, dtype=np.float32).reshape(-1) for v in act.values()])


def reldiff(a, b):
    fa, fb = flat(a), flat(b)
    return float(np.linalg.norm(fa - fb) / (np.linalg.norm(fb) + 1e-8))


def get(policy, obs, sid, reset):
    return policy.get_action(obs, options={"session_ids": [sid], "reset_memory": [reset]})[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/tmp/hamlet_runs/hamlet_e2e_model")
    ap.add_argument("--repo", default="/home/users/xuehui/myfile/HAMLET-Isaac-GR00T-N1d6")
    ap.add_argument("--episodes", default="400,995,1561")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n-rand", type=int, default=4, help="random-control draws per step")
    args = ap.parse_args()
    episodes = [int(x) for x in args.episodes.split(",") if x.strip()]

    policy, pmeta = V.build_policy(args.model, args.device)
    K, nq, stride = pmeta["K"], pmeta["nq"], pmeta["stride"]
    ABLATE["nq"] = nq
    vk, sk, lk = pmeta["vkeys"], pmeta["skeys"], pmeta["lkey"]
    print(f"K={K} nq={nq} stride={stride} | metric = ||a_variant - a_base|| / ||a_base||")

    all_mem, all_rand = [], []
    for ep in episodes:
        df, instr = V.load_episode(args.repo, ep)
        call_steps = list(range(0, len(df), stride))
        sid = f"ab{ep}"
        prev_post = None
        print(f"\n== ep{ep} (len {len(df)}) :: {instr[:60]}")
        for ci, t in enumerate(call_steps):
            row = df.iloc[t]
            front = V.dec(row["image"]); wrist = V.dec(row["wrist_image"])
            st = np.asarray(row["state"], np.float32)
            obs = {"video": {vk[0]: front[None, None], vk[1]: wrist[None, None]},
                   "state": {sk[0]: st[None, None, :7], sk[1]: st[None, None, 7:8]},
                   "language": {lk: [[instr]]}}
            reset = (ci == 0)
            used = None if reset else (prev_post.clone() if prev_post is not None else None)
            # baseline
            ABLATE["mode"] = None
            if used is not None:
                policy._memory_cache[sid] = used.clone()
            seed_all()
            a_base = get(policy, obs, sid, reset)
            post = policy._memory_cache[sid].clone()
            # ablations once the cache holds K real snapshots
            if ci >= K and used is not None and LAST_LK[0] is not None:
                Lk = LAST_LK[0]
                ABLATE["mode"] = "memory"
                policy._memory_cache[sid] = used.clone(); seed_all()
                d_mem = reldiff(get(policy, obs, sid, False), a_base)
                drs = []
                for r in range(args.n_rand):
                    ABLATE["rand_cols"] = sorted(np.random.RandomState(1000 + r).choice(Lk - nq, nq, replace=False).tolist())
                    ABLATE["mode"] = "random"
                    policy._memory_cache[sid] = used.clone(); seed_all()
                    drs.append(reldiff(get(policy, obs, sid, False), a_base))
                d_rand = float(np.mean(drs))
                all_mem.append(d_mem); all_rand.append(d_rand)
                print(f"  step {t:4d}: Δmem={d_mem:.4f}  Δrand={d_rand:.4f}  ratio={d_mem/(d_rand+1e-9):.2f}")
            ABLATE["mode"] = None
            policy._memory_cache[sid] = post.clone()
            prev_post = post

    mem = np.array(all_mem); rand = np.array(all_rand)
    print("\n=== SUMMARY (n=%d ablation steps) ===" % len(mem))
    print(f"Δaction(memory) : mean {mem.mean():.4f}  median {np.median(mem):.4f}")
    print(f"Δaction(random) : mean {rand.mean():.4f}  median {np.median(rand):.4f}")
    print(f"ratio mem/random: mean {mem.mean()/(rand.mean()+1e-9):.2f}")
    print("READ: ratio >> 1 => memory causally used (low-mass but high-leverage); ratio ~1 => memory ~inert at inference.")


if __name__ == "__main__":
    main()
