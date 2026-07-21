#!/usr/bin/env python3
"""Phase-1 dump (moment-token test plan, P1).

Capture, per step, the two tensors that decide whether HAMLET has *memory*:
  - m'_t      = the current-step moment-token summary (post-LLM, PRE-aggregation) -> sees only o_t
  - mem_agg   = the memory-transformer's history-aggregated output that conditions the action head
We REPLAY RoboMME dataset episodes through the policy (no sim needed) and dump mean-pooled
(over n_q) reps + per-step metadata (subgoal/instruction/step) to one .npz.

Both are pure-read forward hooks (behavior-preserving):
  m'_t    : hook on policy.model.backbone           -> out["backbone_features"][:, -nq:, :].mean(1)
  mem_agg : hook on action_head.memory_transformer  -> out[:, -nq:, :].mean(1)

Run from the repo root, server venv, single GPU:
  cd $REPO && CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/dump_moment_acts.py \
      --model /tmp/robomme_eval/model --per-task 4 --max-episodes 160 \
      --out /tmp/robomme_eval/probe/acts_hamlet.npz
"""
from __future__ import annotations
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import visualize_memory_attention as V  # reuse build_policy / load_episode / dec

CAP = {}


def register_hooks(model):
    """Two read-only forward hooks. Returns (handles, has_memory)."""
    handles = []

    def hk_mprime(mod, inp, out):
        try:
            nq = int(out["n_moment_tokens"])
            bf = out["backbone_features"]                       # (B, N+nq, d)
            CAP["m_prime"] = bf[:, -nq:, :].detach().float().mean(1)[0].cpu().numpy()
        except Exception as e:
            CAP["m_prime_err"] = repr(e)

    handles.append(model.backbone.register_forward_hook(hk_mprime))

    mt = getattr(getattr(model, "action_head", None), "memory_transformer", None)
    if mt is not None:
        def hk_memagg(mod, inp, out):
            try:
                nq = int(getattr(mod, "n_q", 4))
                CAP["mem_agg"] = out[:, -nq:, :].detach().float().mean(1)[0].cpu().numpy()
            except Exception as e:
                CAP["mem_agg_err"] = repr(e)
        handles.append(mt.register_forward_hook(hk_memagg))
    return handles, (mt is not None)


def build_episode_index(repo):
    """[(episode_index, task_index, length)] via tasks.jsonl(instruction->ti) + episodes.jsonl."""
    ds = f"{repo}/data/robomme"
    instr2ti = {}
    with open(f"{ds}/meta/tasks.jsonl") as f:
        for l in f:
            o = json.loads(l); instr2ti[o["task"]] = o["task_index"]
    eps = []
    with open(f"{ds}/meta/episodes.jsonl") as f:
        for l in f:
            o = json.loads(l)
            ti = instr2ti.get(o["tasks"][0], -1)
            eps.append((int(o["episode_index"]), int(ti), int(o.get("length", 0))))
    return eps


def parse_range(s):
    out = set()
    for part in s.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-"); out.update(range(int(a), int(b) + 1))
        elif part:
            out.add(int(part))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--repo", default="/home/users/xuehui/myfile/HAMLET-Isaac-GR00T-N1d6")
    ap.add_argument("--per-task", type=int, default=4, help="episodes per task_index (balanced)")
    ap.add_argument("--max-episodes", type=int, default=160)
    ap.add_argument("--episodes", default="", help="explicit list e.g. 0-9,20,33 (overrides --per-task)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--reset-every", action="store_true",
                    help="P2: reset_memory=True EVERY step -> FIFO = K copies of current -> NO real history (effective K=1)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    policy, pmeta = V.build_policy(args.model, args.device)
    print("policy meta:", {k: pmeta[k] for k in ("K", "nq", "stride", "is_hamlet")})
    handles, has_mem = register_hooks(policy.model)
    if not has_mem:
        print("WARNING: no memory_transformer (vanilla?) -> mem_agg stays zero")

    idx = build_episode_index(args.repo)
    if args.episodes:
        want = parse_range(args.episodes)
        chosen = [(e, ti, ln) for (e, ti, ln) in idx if e in want]
    else:
        by_ti = {}
        for (e, ti, ln) in idx:
            by_ti.setdefault(ti, []).append((e, ti, ln))
        chosen = []
        for ti in sorted(by_ti):
            chosen += by_ti[ti][:args.per_task]
        chosen = chosen[:args.max_episodes]
    print(f"dumping {len(chosen)} episodes across {len(set(c[1] for c in chosen))} task_indices")

    K, nq, stride, is_hamlet = pmeta["K"], pmeta["nq"], pmeta["stride"], pmeta["is_hamlet"]
    vkeys, skeys, lkey = pmeta["vkeys"], pmeta["skeys"], pmeta["lkey"]
    R = {k: [] for k in ("m_prime", "mem_agg", "episode", "task_index", "call_step",
                          "n_calls", "step_idx", "subgoal", "grounded", "instruction")}
    t0 = time.time()
    for n, (ep, ti, ln) in enumerate(chosen):
        df, instr = V.load_episode(args.repo, ep)
        call_steps = list(range(0, len(df), stride))
        sid = f"d{ep}"
        for ci, t in enumerate(call_steps):
            row = df.iloc[t]
            front = V.dec(row["image"]); wrist = V.dec(row["wrist_image"])
            state = np.asarray(row["state"], np.float32)
            obs = {"video": {vkeys[0]: front[None, None], vkeys[1]: wrist[None, None]},
                   "state": {skeys[0]: state[None, None, :7], skeys[1]: state[None, None, 7:8]},
                   "language": {lkey: [[instr]]}}
            CAP.clear()
            if is_hamlet:
                reset = (ci == 0) or args.reset_every
                policy.get_action(obs, options={"session_ids": [sid], "reset_memory": [reset]})
            else:
                policy.get_action(obs)
            if "m_prime" not in CAP:
                print("  !! no m_prime capture:", CAP.get("m_prime_err"))
                continue
            mp = CAP["m_prime"]
            R["m_prime"].append(mp.astype(np.float16))
            R["mem_agg"].append(CAP.get("mem_agg", np.zeros_like(mp)).astype(np.float16))
            R["episode"].append(ep); R["task_index"].append(ti)
            R["call_step"].append(ci); R["n_calls"].append(len(call_steps)); R["step_idx"].append(int(t))
            R["subgoal"].append(str(row.get("simple_subgoal", "")))
            R["grounded"].append(str(row.get("grounded_subgoal", "")))
            R["instruction"].append(instr)
        if (n + 1) % 10 == 0:
            print(f"  {n+1}/{len(chosen)} eps | {len(R['m_prime'])} steps | {time.time()-t0:.0f}s")

    out = dict(
        m_prime=np.stack(R["m_prime"]).astype(np.float16),
        mem_agg=np.stack(R["mem_agg"]).astype(np.float16),
        episode=np.array(R["episode"], np.int32),
        task_index=np.array(R["task_index"], np.int32),
        call_step=np.array(R["call_step"], np.int32),
        n_calls=np.array(R["n_calls"], np.int32),
        step_idx=np.array(R["step_idx"], np.int32),
        subgoal=np.array(R["subgoal"], object),
        grounded=np.array(R["grounded"], object),
        instruction=np.array(R["instruction"], object),
    )
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(args.out, **out)
    for h in handles:
        h.remove()
    print(f"saved m_prime{out['m_prime'].shape} mem_agg{out['mem_agg'].shape} -> {args.out} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
