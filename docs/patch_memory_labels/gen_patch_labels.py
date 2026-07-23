#!/usr/bin/env python3
"""Patch-level memory-selection probe — the vlm_keyframe_labels analog for PATCH selection.

For each probe episode, teacher-forced stride-8 policy calls capture the per-DiT-layer
action->image cross-attention over current-frame patches (both views), plus per-cell pixel
diffs (frame-level TokenDrop channel on the same grid). Then we SIMULATE the running
512-patch memory each method would keep:

  td_diff   : per-cell |pixel diff| vs last scored step (TokenDrop, task-blind)
  attn_L{k} : action->image cross-attn at DiT layer k (task-relevance, candidate layers)
  split     : budget split 256 diff + 256 attn_L13 (union of channels)

Outputs per-model per-task HTML (filmstrips with kept cells outlined + temporal-allocation
strips + GT rows parsed from vlm_keyframe_labels) + a summary table with: temporal span,
demo-phase budget share, ref/exec event coverage, same-cell redundancy — the evidence needed
for the attention-scored TokenDrop design (relevance-at-write vs read, layer choice, demo
blindness).

Run:  cd HAMLET-Isaac-GR00T-N1d6 && CUDA_VISIBLE_DEVICES=2 NO_ALBUMENTATIONS_UPDATE=1 \
        .venv/bin/python /home/users/xuehui/myfile/Markdown/patch_memory_labels/gen_patch_labels.py [--smoke]
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HAMLET = Path("/home/users/xuehui/myfile/HAMLET-Isaac-GR00T-N1d6")
sys.path.insert(0, str(HAMLET))
sys.path.insert(0, str(HAMLET / "scripts"))

OUT = Path("/home/users/xuehui/myfile/Markdown/patch_memory_labels")
DATA = HAMLET / "data" / "robomme"
PROBE_HTML = Path("/home/users/xuehui/myfile/Markdown/vlm_keyframe_labels")

MODELS = {
    "vanilla": "/tmp/robomme_eval/model_vanilla",
    "expd_framesamp": "/tmp/robomme_eval/model_expd60k",
}
EPS = {
    "SwingXtimes": [757, 760, 767],
    "PatternLock": [26, 30],
    "ButtonUnmask": [200, 201],
    "VideoPlaceButton": [300],
    "RouteStick": [1112, 1149],
}
STRIDE = 8
BUDGET = 512
ATTN_LAYERS = [5, 10, 13]
TOL = 14  # probe's ref-event tolerance
MIN_CELLS = 3  # event covered iff >= this many kept cells within +-TOL


# ---------------------------------------------------------------- GT from probe HTML
def parse_gt(task: str, ep: int):
    """Parse reference/exec GT + t0 for one episode from the existing probe HTML."""
    f = PROBE_HTML / f"{task}.html"
    if task == "SwingXtimes":
        f = PROBE_HTML / "swingXtimes.html"
    if not f.exists():
        return {}
    s = f.read_text(errors="ignore")
    s = re.sub(r"data:image/[^\"')]+", "", s)  # strip embedded base64 (blows regex windows)
    m = re.search(rf"ep{ep} \(T=(\d+)(?:, t0=(\d+))?\)", s)
    gt = {"t0": int(m.group(2)) if (m and m.group(2)) else 0}
    seg = s[m.end(): m.end() + 3000] if m else ""
    if task == "SwingXtimes":
        m2 = re.search(rf"ep{ep} \(T=\d+\).{{0,80}}?GT apexes \[([0-9, ]+)\], press dwell \[(\d+),\s*(\d+)\]", s, re.S)
        if m2:
            gt["ref"] = []
            gt["exec"] = [int(x) for x in m2.group(1).split(",")]
            gt["press"] = [int(m2.group(2)), int(m2.group(3))]
    else:
        m2 = re.search(r"Visual reference events \[([0-9, ]*)\]", seg)
        m3 = re.search(r"EXEC GT \[([0-9, ]*)\]", seg)
        gt["ref"] = [int(x) for x in m2.group(1).split(",") if x.strip()] if m2 and m2.group(1).strip() else []
        gt["exec"] = [int(x) for x in m3.group(1).split(",") if x.strip()] if m3 and m3.group(1).strip() else []
    return gt


# ---------------------------------------------------------------- episode + obs
def dec(d):
    from PIL import Image
    return np.array(Image.open(io.BytesIO(d["bytes"])).convert("RGB"))


def load_episode(ep: int):
    import pandas as pd
    chunk = "chunk-000" if ep < 1000 else "chunk-001"
    df = pd.read_parquet(DATA / "data" / chunk / f"episode_{ep:06d}.parquet")
    tasks = {json.loads(l)["task_index"]: json.loads(l)["task"]
             for l in open(DATA / "meta" / "tasks.jsonl")}
    return df, tasks[int(df.iloc[0]["task_index"])]


def build_policy(model_path: str):
    import visualize_memory_attention as V
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy
    policy = Gr00tPolicy(model_path=model_path, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT, device="cuda")
    V.install_attn_hooks()
    V.install_imagemask_hook(policy.model)
    vkeys = policy.modality_configs["video"].modality_keys
    skeys = policy.modality_configs["state"].modality_keys
    lkey = policy.modality_configs["language"].modality_keys[0]
    n_cross = int(policy.model.config.diffusion_model_cfg["num_layers"]) // 2
    return policy, V, dict(vkeys=vkeys, skeys=skeys, lkey=lkey, n_cross=n_cross)


def obs_of(pmeta, front, wrist, state, instr):
    v, sk, lk = pmeta["vkeys"], pmeta["skeys"], pmeta["lkey"]
    return {
        "video": {v[0]: front[None, None], v[1]: wrist[None, None]},
        "state": {sk[0]: state[None, None, :7], sk[1]: state[None, None, 7:8]},
        "language": {lk: [[instr]]},
    }


# ---------------------------------------------------------------- scoring pass
def episode_scores(policy, V, pmeta, df, instr, ep):
    """Teacher-forced stride-8 calls. Returns per-scored-step dict with per-view attention
    grids per candidate layer, per-view pixel-diff grids on the same grid, frames."""
    import torch
    n_cross = pmeta["n_cross"]
    T = len(df)
    steps = list(range(0, T, STRIDE))
    out = []
    prev_small = None
    side_holder = {}
    for t in steps:
        front = dec(df.iloc[t]["image"])
        wrist = dec(df.iloc[t]["wrist_image"])
        state = np.asarray(df.iloc[t]["state"], np.float32)
        V._CAP["cross"].clear(); V._CAP["image_mask"] = None
        with torch.inference_mode():
            policy.get_action(obs_of(pmeta, front, wrist, state, instr),
                              options={"session_ids": [f"probe{ep}"], "reset_memory": [t == 0]})
        im = V._CAP["image_mask"]; raw = list(V._CAP["cross"])
        rec = {"t": t, "front": front, "wrist": wrist, "attn": {}, "img_mass": None}
        nq = int(getattr(policy.model.config, "n_moment_tokens", 4))
        if im is not None and raw:
            mask = im[0].bool()
            # framesamp/cross_attn models replace the nq-token moment tail with M memory
            # tokens -> DiT KV is longer than the backbone mask. The CURRENT-frame columns
            # are the first (L - nq) positions; align both av and mask to that prefix.
            Lm = mask.shape[0] - nq
            side_holder.setdefault("side", int(round((int(mask[:Lm].sum()) // 2) ** 0.5)))
            n_den = max(1, len(raw) // n_cross)
            img_mass = []
            for li in range(n_cross):
                As = [raw[d * n_cross + li] for d in range(n_den) if d * n_cross + li < len(raw)]
                av = sum(A[0].float().mean(0) for A in As) / len(As)
                if av.shape[0] == mask.shape[0]:
                    av2, m2 = av, mask
                elif av.shape[0] >= Lm > 0:
                    av2, m2 = av[:Lm], mask[:Lm]
                else:
                    continue
                img_mass.append(float(av2[m2].sum()))
                if li in ATTN_LAYERS:
                    g0 = V.img_heat_from_cross(av2[m2], n_views=2, view=0)
                    g1 = V.img_heat_from_cross(av2[m2], n_views=2, view=1)
                    if g0 is not None:
                        rec["attn"][li] = np.stack([g0, g1])  # (2, side, side)
                        side_holder["side"] = g0.shape[0]
            rec["img_mass"] = img_mass
        side = side_holder.get("side", 16)
        small = np.stack([_cellize(front, side), _cellize(wrist, side)])  # (2, side, side, 3-mean)
        rec["diff"] = np.abs(small - prev_small).mean(-1) if prev_small is not None else np.zeros((2, side, side), np.float32)
        prev_small = small
        out.append(rec)
    return out


def _cellize(frame, side):
    from PIL import Image
    im = Image.fromarray(frame).resize((side * 8, side * 8))
    a = np.asarray(im, np.float32) / 255.0
    return a.reshape(side, 8, side, 8, 3).mean((1, 3))  # (side, side, 3)


# ---------------------------------------------------------------- selection sim
def select(recs, mode, layer=None):
    """Return kept set: list of (t, view, r, c, score). Running top-BUDGET heap;
    frame-0 FRONT cells protected for the diff channel (TokenDrop's sentinel)."""
    import heapq
    heap = []
    for k, rec in enumerate(recs):
        if mode == "diff":
            g = rec["diff"]
        else:
            g = rec["attn"].get(layer)
            if g is None:
                continue
        side = g.shape[-1]
        for view in range(2):
            for r in range(side):
                for c in range(side):
                    s = float(g[view, r, c])
                    if mode == "diff" and k == 0 and view == 0:
                        s = 1e9  # frame-0 protection (front view), TokenDrop sentinel
                    if mode == "diff" and s < 1e-4:
                        continue
                    heapq.heappush(heap, (s, rec["t"], view, r, c))
                    if len(heap) > BUDGET:
                        heapq.heappop(heap)
    return [(t, v, r, c, s) for (s, t, v, r, c) in heap]


def select_split(recs, layer):
    import heapq
    def top(mode, lay, budget):
        h = []
        for k, rec in enumerate(recs):
            g = rec["diff"] if mode == "diff" else rec["attn"].get(lay)
            if g is None:
                continue
            side = g.shape[-1]
            for view in range(2):
                for r in range(side):
                    for c in range(side):
                        s = float(g[view, r, c])
                        if mode == "diff" and k == 0 and view == 0:
                            s = 1e9
                        if mode == "diff" and s < 1e-4:
                            continue
                        heapq.heappush(h, (s, rec["t"], view, r, c))
                        if len(h) > budget:
                            heapq.heappop(h)
        return {(t, v, r, c): s for (s, t, v, r, c) in h}
    a = top("diff", None, BUDGET // 2)
    b = top("attn", layer, BUDGET // 2)
    merged = dict(a); merged.update(b)
    return [(t, v, r, c, s) for (t, v, r, c), s in merged.items()]


def select_sum(recs, layer):
    """Additive combination: per-episode z-score each channel, per-cell SUM -> top-BUDGET.
    (The soft alternative to the union split; needs the normalization because attention is
    a softmax distribution ~0.006/cell while diff is mean |pixel delta| — incommensurable raw.)"""
    diffs, attns, keys = [], [], []
    for rec in recs:
        g_a = rec["attn"].get(layer)
        if g_a is None:
            continue
        g_d = rec["diff"]
        side = g_d.shape[-1]
        for view in range(2):
            for r in range(side):
                for c in range(side):
                    keys.append((rec["t"], view, r, c))
                    diffs.append(float(g_d[view, r, c]))
                    attns.append(float(g_a[view, r, c]))
    if not keys:
        return []
    d = np.asarray(diffs); a = np.asarray(attns)
    z = (d - d.mean()) / (d.std() + 1e-9) + (a - a.mean()) / (a.std() + 1e-9)
    order = np.argsort(z)[::-1][:BUDGET]
    return [(*keys[i], float(z[i])) for i in order]


# ---------------------------------------------------------------- metrics + render
def metrics(kept, gt, T):
    ts = sorted(t for (t, *_rest) in kept)
    if not ts:
        return {}
    arr = np.asarray(ts)
    uniq_steps = sorted(set(ts))
    t0 = gt.get("t0", 0)
    cover = lambda evs: (sum(1 for e in evs if (np.abs(arr - e) <= TOL).sum() >= MIN_CELLS), len(evs)) if evs else (0, 0)
    rr, rn = cover(gt.get("ref", []))
    er, en = cover(gt.get("exec", []) + ([gt["press"][0]] if "press" in gt else []))
    cells = [(v, r, c) for (_t, v, r, c, _s) in kept]
    return dict(
        n=len(kept), steps=len(uniq_steps),
        span=f"{int(np.percentile(arr, 5))}-{int(np.percentile(arr, 95))}",
        demo_share=float((arr < t0).mean()) if t0 > 0 else 0.0,
        ref=f"{rr}/{rn}", exec=f"{er}/{en}",
        dup=1.0 - len(set(cells)) / max(1, len(cells)),
    )


def draw_kept(frame, kept_cells, side):
    from PIL import Image, ImageDraw
    im = Image.fromarray(frame).convert("RGB")
    W, H = im.size
    d = ImageDraw.Draw(im)
    for (r, c) in kept_cells:
        x0, y0 = c * W / side, r * H / side
        d.rectangle([x0, y0, x0 + W / side, y0 + H / side], outline=(255, 60, 60), width=2)
    return im


def b64(im, q=70):
    buf = io.BytesIO(); im.save(buf, "JPEG", quality=q)
    return base64.b64encode(buf.getvalue()).decode()


def alloc_strip(kept, T, t0, width=560, height=42):
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (width, height), (255, 255, 255))
    d = ImageDraw.Draw(im)
    if t0 > 0:
        d.rectangle([0, 0, int(t0 / T * width), height], fill=(235, 235, 250))
    hist = defaultdict(int)
    for (t, *_r) in kept:
        hist[t] += 1
    mx = max(hist.values()) if hist else 1
    for t, n in hist.items():
        x = int(t / T * width)
        d.line([x, height, x, height - int(n / mx * (height - 4))], fill=(200, 40, 40), width=2)
    return im


def render_episode_html(task, ep, T, gt, instr, results, recs, side):
    rows = [f"<h3>ep{ep} (T={T}, t0={gt.get('t0', 0)}) — {instr[:140]}</h3>"]
    if gt.get("ref"):
        rows.append(f"<p>ref events: {gt['ref']} · exec GT: {gt.get('exec', [])}</p>")
    elif gt.get("exec"):
        rows.append(f"<p>apex GT: {gt['exec']} · press: {gt.get('press')}</p>")
    frames_by_t = {rec["t"]: rec for rec in recs}
    for name, kept in results.items():
        m = metrics(kept, gt, T)
        rows.append(f"<h4>{name} — n={m.get('n')} steps={m.get('steps')} span={m.get('span')} "
                    f"demo_share={m.get('demo_share', 0):.2f} REF {m.get('ref')} EXEC {m.get('exec')} "
                    f"dup={m.get('dup', 0):.2f}</h4>")
        strip = alloc_strip(kept, T, gt.get("t0", 0))
        rows.append(f'<img src="data:image/jpeg;base64,{b64(strip)}" style="border:1px solid #999">')
        per_t = defaultdict(lambda: defaultdict(list))
        for (t, v, r, c, _s) in kept:
            per_t[t][v].append((r, c))
        top_ts = sorted(per_t, key=lambda t: -sum(len(x) for x in per_t[t].values()))[:8]
        front_html, wrist_html = [], []
        for t in sorted(top_ts):
            rec = frames_by_t.get(t)
            if rec is None:
                continue
            n_f, n_w = len(per_t[t].get(0, [])), len(per_t[t].get(1, []))
            im_f = draw_kept(rec["front"], per_t[t].get(0, []), side)
            im_w = draw_kept(rec["wrist"], per_t[t].get(1, []), side)
            front_html.append(
                f'<td><img src="data:image/jpeg;base64,{b64(im_f)}" width="130"><br>'
                f"<small>t={t}·{n_f}f+{n_w}w</small></td>")
            wrist_html.append(
                f'<td><img src="data:image/jpeg;base64,{b64(im_w)}" width="130"><br>'
                f"<small>wrist·{n_w}</small></td>")
        rows.append("<table><tr><td><small><b>front</b></small></td>" + "".join(front_html)
                    + "</tr><tr><td><small><b>wrist</b></small></td>" + "".join(wrist_html)
                    + "</tr></table>")
    return "\n".join(rows)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--models", default="vanilla,expd_framesamp")
    args = ap.parse_args()
    eps = {"SwingXtimes": [760]} if args.smoke else EPS
    OUT.mkdir(exist_ok=True)
    summary = []
    for mname in args.models.split(","):
        policy, V, pmeta = build_policy(MODELS[mname])
        print(f"[{mname}] n_cross={pmeta['n_cross']} vkeys={pmeta['vkeys']}")
        for task, ep_list in eps.items():
            sections = []
            for ep in ep_list:
                df, instr = load_episode(ep)
                gt = parse_gt(task, ep)
                print(f"[{mname}/{task}/ep{ep}] T={len(df)} t0={gt.get('t0', 0)} scoring...")
                recs = episode_scores(policy, V, pmeta, df, instr, ep)
                got_layers = sorted(recs[len(recs) // 2]["attn"].keys())
                side = recs[0]["diff"].shape[-1]
                results = {"td_diff": select(recs, "diff")}
                for li in got_layers:
                    results[f"attn_L{li}"] = select(recs, "attn", li)
                if got_layers:
                    best = 13 if 13 in got_layers else got_layers[-1]
                    results["split_diff+L13"] = select_split(recs, best)
                    results["sum_diff+L13"] = select_sum(recs, best)
                sections.append(render_episode_html(task, ep, len(df), gt, instr, results, recs, side))
                for name, kept in results.items():
                    m = metrics(kept, gt, len(df))
                    summary.append(dict(model=mname, task=task, ep=ep, method=name, **m))
                # demo-phase attention informativeness (per candidate layer)
                t0 = gt.get("t0", 0)
                if t0 > 0:
                    for li in got_layers:
                        obs_pk = [float(r["attn"][li][0].max() / (r["attn"][li][0].mean() + 1e-9))
                                  for r in recs if r["t"] < t0 and li in r["attn"]]
                        ex_pk = [float(r["attn"][li][0].max() / (r["attn"][li][0].mean() + 1e-9))
                                 for r in recs if r["t"] >= t0 and li in r["attn"]]
                        if obs_pk and ex_pk:
                            sections.append(f"<p><small>L{li} front peakiness OBS {np.mean(obs_pk):.1f} "
                                            f"vs EXEC {np.mean(ex_pk):.1f}</small></p>")
            page = (f"<meta charset='utf-8'><title>patch selection — {task} — {mname}</title>"
                    f"<h1>{task} — {mname}: simulated 512-patch memory per selection method</h1>"
                    f"<p>grid {side}x{side}x2views, stride {STRIDE}, budget {BUDGET}. "
                    f"Methods: TokenDrop pixel-diff | action-queried attention (candidate DiT layers) | split budget.</p>"
                    + "\n<hr>\n".join(sections))
            (OUT / f"{task}_{mname}.html").write_text(page)
            print(f"wrote {OUT}/{task}_{mname}.html")
        del policy
        import torch
        torch.cuda.empty_cache()
    import pandas as pd
    pd.DataFrame(summary).to_csv(OUT / "summary.csv", index=False)
    print(pd.DataFrame(summary).to_string())


if __name__ == "__main__":
    main()
