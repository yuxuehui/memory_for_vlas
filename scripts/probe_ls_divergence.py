"""Does the LEARNED patch selection diverge from the patch_union heuristic?

The decisive read for note-24's A/B: if the trained head keeps (nearly) the same 512 patches
the hand-written union would keep, the A/B is null — the head merely re-learned the prior it was
warm-started from.

WHERE THE SELECTION ACTUALLY HAPPENS (it took three attempts to observe it — both wrong turns
are worth recording): at inference the policy feeds K=1 rows and the model's `K == 1` branch
drives the streaming selectors (LearnedPatchSelector / PatchUnionSelector) directly; the
training-side builders (`_learned_select_mem_seq` etc.) are never called there, so hooking them
records nothing — that is correct behaviour, not a bug. And the selector object does not stay on
the model: the policy round-trips it through `action_head._fs_state` and NULLS the attribute
after every call (gr00t_policy.py:503), keeping the live object in
`policy._fs_session_state[session_id]`. Read that, after the call.

  python scripts/probe_ls_divergence.py \
      --ls-ckpt /home/storage/xuehui/robomme_eval/model_ls_20k \
      --pu-ckpt /home/storage/xuehui/robomme_eval/model_puv2_50k \
      --repo . --per-task 1 --device cuda:0

Reading it:
  iou_patch ~1.0                    -> null A/B (head == heuristic)
  iou_patch low, iou_frame high     -> same frames, different patches (spatial re-selection)
  iou_frame low                     -> the head keeps different MOMENTS (temporal re-selection)
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(__file__))
import visualize_memory_attention as V  # build_policy / load_episode / dec

SUITE = {
    "BinFill": "Counting", "PickXtimes": "Counting", "StopCube": "Counting",
    "SwingXtimes": "Counting",
    "ButtonUnmask": "Permanence", "ButtonUnmaskSwap": "Permanence",
    "VideoUnmask": "Permanence", "VideoUnmaskSwap": "Permanence",
    "PickHighlight": "Reference", "VideoPlaceButton": "Reference",
    "VideoPlaceOrder": "Reference", "VideoRepick": "Reference",
    "InsertPeg": "Imitation", "MoveCube": "Imitation",
    "PatternLock": "Imitation", "RouteStick": "Imitation",
}
def kept_set(sel):
    """(step, patch_idx) pairs held by either selector class.

    WHERE THE STATE LIVES (the two failed probe versions both misread this): at inference the
    policy round-trips selector state through `action_head._fs_state` per call and then NULLS
    the model attribute (gr00t_policy.py:503), keeping the live object in
    `policy._fs_session_state[session_id]`. Read THAT, after the call."""
    if sel is None:
        return set()
    if hasattr(sel, "step_of") and sel.step_of is not None:      # LearnedPatchSelector
        return set(zip(sel.step_of.tolist(), sel.idx_of.tolist()))
    out = set()
    for heap in ("diff_heap", "attn_heap", "tail_heap"):         # PatchUnionSelector
        for item in getattr(sel, heap, []) or []:
            out.add((int(item[1]), int(item[2])))
    return out


def replay_keeps(policy, meta, repo, episode):
    vkeys, skeys, lkey = meta["vkeys"], meta["skeys"], meta["lkey"]
    stride = meta["stride"]
    df, instr = V.load_episode(repo, episode)
    sid = f"probe{episode}"
    per_call = []
    for ci, t in enumerate(range(0, len(df), stride)):
        row = df.iloc[t]
        state = np.asarray(row["state"], np.float32)
        obs = {"video": {vkeys[0]: V.dec(row["image"])[None, None],
                         vkeys[1]: V.dec(row["wrist_image"])[None, None]},
               "state": {skeys[0]: state[None, None, :7], skeys[1]: state[None, None, 7:8]},
               "language": {lkey: [[instr]]}}
        policy.get_action(obs, options={"session_ids": [sid], "reset_memory": [ci == 0]})
        sel = getattr(policy, "_fs_session_state", {}).get(sid)
        per_call.append(kept_set(sel))
    n_img = 162  # 2 views x 81 (9x9); only used for the frame decomposition below
    return per_call, n_img


def task_of(repo, episode):
    for line in open(f"{repo}/data/robomme/meta/episodes.jsonl"):
        o = json.loads(line)
        if o["episode_index"] == episode:
            t = (o.get("tasks") or ["?"])[0]
            for k in SUITE:
                if k.lower() in t.lower().replace(" ", ""):
                    return k
            return t[:20]
    return "?"


def task_from_instr(t):
    """episodes.jsonl carries INSTRUCTIONS, not task names — map by distinctive phrases.
    (Dataset quirk: PickHighlight's instruction spells it 'highlighteted'.)"""
    t = t.lower()
    for phrase, name in (
        ("retrace the same pattern", "PatternLock"),
        ("navigate around", "RouteStick"),
        ("same end of the same peg", "InsertPeg"),
        ("same manner", "MoveCube"),
        ("highlight", "PickHighlight"),
        ("into the bin", "BinFill"),
        ("stop the cube just as it reaches", "StopCube"),
        ("right-side target", "SwingXtimes"),
        ("place it on the target", "PickXtimes"),
        ("previously placed", "VideoPlaceOrder"),
        ("right after the button", "VideoPlaceButton"),
        ("right before the button", "VideoPlaceButton"),
        ("same block", "VideoRepick"),
    ):
        if phrase in t:
            return name
    if "hiding" in t:
        watch, fin = "watch" in t, "finally" in t
        if watch:
            return "VideoUnmaskSwap" if fin else "VideoUnmask"
        return "ButtonUnmaskSwap" if fin else "ButtonUnmask"
    return None


def pick_episodes(repo, per_task):
    by = collections.defaultdict(list)
    for line in open(f"{repo}/data/robomme/meta/episodes.jsonl"):
        o = json.loads(line)
        key = task_from_instr((o.get("tasks") or [""])[0])
        if key and len(by[key]) < per_task:
            by[key].append(o["episode_index"])
    return [(k, e) for k, eps in sorted(by.items()) for e in eps]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ls-ckpt", required=True)
    ap.add_argument("--pu-ckpt", required=True)
    ap.add_argument("--repo", default=".")
    ap.add_argument("--per-task", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="Markdown/ls_divergence")
    args = ap.parse_args()

    eps = pick_episodes(args.repo, args.per_task)
    print(f"{len(eps)} episodes across {len({t for t,_ in eps})} tasks")
    keeps = {}
    for tag, ckpt in (("learned", args.ls_ckpt), ("heuristic", args.pu_ckpt)):
        policy, meta = V.build_policy(ckpt, args.device)
        for task, ep in eps:
            with torch.no_grad():
                per_call, n_img = replay_keeps(policy, meta, args.repo, ep)
            keeps.setdefault((task, ep), {})[tag] = (per_call, n_img)
            sizes = [len(s) for s in per_call]
            print(f"  [{tag}] {task} ep{ep}: {len(per_call)} calls, kept "
                  f"{min(sizes) if sizes else 0}-{max(sizes) if sizes else 0} tokens")
        del policy
        torch.cuda.empty_cache()
    os.makedirs(args.out, exist_ok=True)
    recs = []
    for (task, ep), arms in sorted(keeps.items()):
        if len(arms) != 2:
            continue
        (A, n_img), (Bk, _) = arms["learned"], arms["heuristic"]
        n = min(len(A), len(Bk))
        if not n or not n_img:
            continue
        for ci in (n // 2, n - 1):
            a, b = A[ci], Bk[ci]
            iou = len(a & b) / max(1, len(a | b))
            fa = {st for st, _ in a}
            fb = {st for st, _ in b}
            fiou = len(fa & fb) / max(1, len(fa | fb))
            recs.append({"task": task, "suite": SUITE.get(task, "?"), "episode": ep,
                         "call": "final" if ci == n - 1 else "mid",
                         "iou_patch": round(iou, 4), "iou_frame": round(fiou, 4),
                         "n_learned": len(a), "n_heuristic": len(b),
                         "frames_learned": len(fa), "frames_heuristic": len(fb)})
    df = pd.DataFrame(recs)
    if df.empty:
        print("no paired records — check the hook"); return
    df.to_csv(f"{args.out}/divergence.csv", index=False)
    fin = df[df["call"] == "final"]
    print("\n=== per-suite mean (final call) ===")
    print(fin.groupby("suite")[["iou_patch", "iou_frame"]].mean().round(3))
    print("\n=== per-task (final call) ===")
    print(fin[["task", "iou_patch", "iou_frame"]].to_string(index=False))
    print("\nOVERALL  iou_patch %.3f   iou_frame %.3f" %
          (fin["iou_patch"].mean(), fin["iou_frame"].mean()))
    print(f"wrote {args.out}/divergence.csv")


if __name__ == "__main__":
    main()
