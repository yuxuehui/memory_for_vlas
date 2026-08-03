"""Visualize WHICH image patches the learned selector keeps vs the patch_union heuristic.

Companion to probe_ls_divergence.py (which measured iou_patch ~0.14 / iou_frame ~1.0): render
the actual frames with the 9x9 patch grid overlaid, colored by who keeps each patch —
  green  = learned only          blue = heuristic only
  yellow = both                  dim  = neither
One HTML per task (base64-embedded JPEGs, same style as Markdown/patch_memory_labels/*.html).

  python scripts/vis_ls_patches.py \
      --ls-ckpt /home/storage/xuehui/robomme_eval/model_ls_20k \
      --pu-ckpt /home/storage/xuehui/robomme_eval/model_puv2_50k \
      --tasks VideoPlaceButton VideoUnmask PatternLock PickXtimes \
      --repo . --device cuda:0 --out Markdown/ls_divergence
"""
from __future__ import annotations

import argparse
import base64
import io
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
import visualize_memory_attention as V
from probe_ls_divergence import kept_set, pick_episodes, replay_keeps  # noqa: F401

SIDE, PER_VIEW = 9, 81


def idx_to_cell(idx):
    """flat patch idx -> (view, row, col); view 0 = front, 1 = wrist."""
    view = 0 if idx < PER_VIEW else 1
    w = idx % PER_VIEW
    return view, w // SIDE, w % SIDE


def overlay(img, cells, alpha=0.45):
    """cells: {(row,col): color}; img: HxWx3 uint8."""
    im = img.astype(np.float32).copy()
    H, W = im.shape[:2]
    ch, cw = H / SIDE, W / SIDE
    for (r, c), col in cells.items():
        y0, y1 = int(r * ch), int((r + 1) * ch)
        x0, x1 = int(c * cw), int((c + 1) * cw)
        im[y0:y1, x0:x1] = (1 - alpha) * im[y0:y1, x0:x1] + alpha * np.array(col, np.float32)
        im[y0:y0 + 2, x0:x1] = col; im[y1 - 2:y1, x0:x1] = col
        im[y0:y1, x0:x0 + 2] = col; im[y0:y1, x1 - 2:x1] = col
    return im.astype(np.uint8)


def b64(img):
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, "JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode()


GREEN, BLUE, YELLOW = (60, 200, 80), (70, 130, 240), (240, 210, 60)


def replay_with_frames(policy, meta, repo, episode):
    """Same call cadence as the probe, additionally caching the raw frames per call step."""
    vkeys, skeys, lkey = meta["vkeys"], meta["skeys"], meta["lkey"]
    stride = meta["stride"]
    df, instr = V.load_episode(repo, episode)
    sid = f"vis{episode}"
    per_call, frames = [], {}
    for ci, t in enumerate(range(0, len(df), stride)):
        row = df.iloc[t]
        front, wrist = V.dec(row["image"]), V.dec(row["wrist_image"])
        frames[ci] = (front, wrist)
        state = np.asarray(row["state"], np.float32)
        obs = {"video": {vkeys[0]: front[None, None], vkeys[1]: wrist[None, None]},
               "state": {skeys[0]: state[None, None, :7], skeys[1]: state[None, None, 7:8]},
               "language": {lkey: [[instr]]}}
        policy.get_action(obs, options={"session_ids": [sid], "reset_memory": [ci == 0]})
        per_call.append(kept_set(getattr(policy, "_fs_session_state", {}).get(sid)))
    return per_call, frames, instr


def render_task(task, ep, learned, heuristic, frames, instr, out_dir):
    """At the FINAL call: group kept patches by source step, show ~8 evenly spaced steps."""
    A, B = learned[-1], heuristic[-1]
    steps = sorted({s for s, _ in A} | {s for s, _ in B})
    show = steps if len(steps) <= 8 else [steps[i] for i in
                                          np.linspace(0, len(steps) - 1, 8).round().astype(int)]
    cards = []
    for st in show:
        cells = [dict(), dict()]  # per view
        for (s, i), col in [((s, i), YELLOW) for (s, i) in (A & B)] + \
                           [((s, i), GREEN) for (s, i) in (A - B)] + \
                           [((s, i), BLUE) for (s, i) in (B - A)]:
            if s != st:
                continue
            v, r, c = idx_to_cell(i)
            cells[v][(r, c)] = col
        if st not in frames:
            continue
        fr, wr = frames[st]
        na, nb = sum(1 for s, _ in A if s == st), sum(1 for s, _ in B if s == st)
        cards.append(
            f"<div class='card'><h4>memory step {st} — learned {na} / heuristic {nb} patches</h4>"
            f"<img src='data:image/jpeg;base64,{b64(overlay(fr, cells[0]))}'>"
            f"<img src='data:image/jpeg;base64,{b64(overlay(wr, cells[1]))}'></div>")
    iou = len(A & B) / max(1, len(A | B))
    html = f"""<html><head><style>
body{{font-family:sans-serif;background:#111;color:#eee;margin:20px}}
.card{{display:inline-block;margin:8px;background:#1c1c1c;padding:8px;border-radius:6px}}
.card img{{width:300px;margin:2px}} h4{{margin:2px 0;font-size:13px;color:#bbb}}
.legend span{{padding:2px 10px;margin-right:8px;border-radius:4px;font-size:13px}}
</style></head><body>
<h2>{task} ep{ep} — final memory bank, learned(20k) vs heuristic(puv2-50k)</h2>
<p style='color:#999'>{instr}</p>
<p class='legend'><span style='background:rgb{GREEN};color:#000'>learned only</span>
<span style='background:rgb{BLUE};color:#000'>heuristic only</span>
<span style='background:rgb{YELLOW};color:#000'>both</span>
&nbsp; iou_patch = {iou:.3f} &nbsp; bank: learned {len(A)} / heuristic {len(B)}</p>
{''.join(cards)}</body></html>"""
    path = os.path.join(out_dir, f"patches_{task}_ls20k_vs_pu50k.html")
    open(path, "w").write(html)
    print(f"  wrote {path}  (iou {iou:.3f}, {len(show)} steps shown)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ls-ckpt", required=True)
    ap.add_argument("--pu-ckpt", required=True)
    ap.add_argument("--tasks", nargs="+", default=["VideoPlaceButton", "VideoUnmask",
                                                   "PatternLock", "PickXtimes"])
    ap.add_argument("--repo", default=".")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="Markdown/ls_divergence")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    eps = dict(pick_episodes(args.repo, 1))          # task -> episode
    chosen = [(t, eps[t]) for t in args.tasks if t in eps]
    print("visualizing:", chosen)

    results = {}
    for tag, ckpt in (("learned", args.ls_ckpt), ("heuristic", args.pu_ckpt)):
        policy, meta = V.build_policy(ckpt, args.device)
        for task, ep in chosen:
            with torch.no_grad():
                per_call, frames, instr = replay_with_frames(policy, meta, args.repo, ep)
            results.setdefault((task, ep), {})[tag] = (per_call, frames, instr)
            print(f"  [{tag}] {task} ep{ep}: {len(per_call)} calls")
        del policy
        torch.cuda.empty_cache()

    for (task, ep), arms in results.items():
        A, framesA, instr = arms["learned"]
        Bk, _, _ = arms["heuristic"]
        render_task(task, ep, A, Bk, framesA, instr, args.out)


if __name__ == "__main__":
    main()
