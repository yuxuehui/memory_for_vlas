#!/usr/bin/env python3
"""act (action-to-vision) vs tail (prefill-to-vision): WHICH patches does each relevance
channel actually keep?

Companion to gen_patch_labels.py (which compared pixel-diff vs action attention). Here both
relevance sources are captured in the SAME policy call and used to drive the same 512-patch
running memory, so the kept sets are directly comparable:

  act_L13    DiT cross-attention, 16 action queries -> current-frame patches, layer 13
             (the deployed patch_union relevance channel; needs a full DiT forward)
  tail_L13   backbone SELF-attention, post-image summary tokens -> patches, layer 13
             (the "consumed-bottleneck readout" — sharp in HAMLET, absent in framesamp models)
  tail_L15   same, layer 15 (pretrained summarization layer; note-22's correction says THIS
             is the layer to use on framesamp-family models)
  nov        per-cell |pixel diff| vs previous scored step (TokenDrop channel, for reference)
  nov∪act    the deployed patch_union split budget (256+256)
  nov∪tailL15  the queued trained-in variant

Renders per model×task HTML: per-method filmstrips with kept cells outlined (front + wrist
rows, annotated with which timestep holds how many patches) PLUS a direct same-frame overlay
where act-only cells are BLUE, tail-only cells are ORANGE and shared cells are GREEN.

Run:  cd HAMLET-Isaac-GR00T-N1d6 && CUDA_VISIBLE_DEVICES=1 NO_ALBUMENTATIONS_UPDATE=1 \
        .venv/bin/python /home/users/xuehui/myfile/Markdown/patch_memory_labels/gen_act_vs_tail_patches.py
"""

from __future__ import annotations

import argparse
import heapq
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HAMLET = Path("/home/users/xuehui/myfile/HAMLET-Isaac-GR00T-N1d6")
sys.path.insert(0, str(HAMLET))
sys.path.insert(0, str(HAMLET / "scripts"))
sys.path.insert(0, "/home/users/xuehui/myfile/Markdown/patch_memory_labels")
sys.path.insert(0, "/home/users/xuehui/myfile/Markdown/lang_token_temporal")
sys.path.insert(0, "/home/users/xuehui/myfile/Markdown/attention_map")

from gen_patch_labels import (BUDGET, MIN_CELLS, STRIDE, TOL, _cellize, b64, dec,
                              draw_kept, load_episode, obs_of, parse_gt)
from gen_nonvision_attn import _SA, _XA, install_cross_attn_capture, install_self_attn_capture

OUT = Path("/home/users/xuehui/myfile/Markdown/patch_memory_labels")
MODELS = {
    "vanilla": "/tmp/robomme_eval/model_vanilla",          # hamlet_mode=off: no moment tokens at all
    "expd_framesamp": "/tmp/robomme_eval/model_expd60k",   # framesamp family: NO L13 readout circuit
    "HAMLET": "/tmp/robomme_eval/model_B_60k",             # moment consumed: L13 circuit exists
}
EPS = {
    "SwingXtimes": [760],
    "PatternLock": [26],
    "ButtonUnmask": [200],
    "VideoPlaceButton": [300],
    "RouteStick": [1112],
}
ACT_LAYER = 13     # DiT cross-attn layer (deployed --mem-fs-attn-layer)
TAIL_LAYERS = [13, 15]


# ---------------------------------------------------------------- capture
def build_policy(model_path: str):
    import visualize_memory_attention as V
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy
    policy = Gr00tPolicy(model_path=model_path, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
                         device="cuda")
    V.install_imagemask_hook(policy.model)
    return policy, V, dict(
        vkeys=policy.modality_configs["video"].modality_keys,
        skeys=policy.modality_configs["state"].modality_keys,
        lkey=policy.modality_configs["language"].modality_keys[0],
        n_cross=int(policy.model.config.diffusion_model_cfg["num_layers"]) // 2,
        # `n_moment_tokens` sits in every config as a leftover; `hamlet_mode` is the real gate —
        # with it off no moment slots are appended, and chopping 4 would eat real tail tokens.
        n_q=(int(getattr(policy.model.config, "n_moment_tokens", 0))
             if getattr(policy.model.config, "hamlet_mode", "off") != "off" else 0),
    )


def img_blocks(mask_np):
    """Contiguous image-token blocks -> [(start, end)] per view, in sequence coords."""
    idx = np.nonzero(mask_np)[0]
    br = np.nonzero(np.diff(idx) > 1)[0]
    blocks, start = [], idx[0]
    for b in br:
        blocks.append((int(start), int(idx[b])))
        start = idx[b + 1]
    blocks.append((int(start), int(idx[-1])))
    return blocks


def episode_signals(policy, V, pmeta, df, instr, ep):
    """One policy call per scored step; both attention families captured in that call.

    Returns per-step dicts with (2, side, side) grids for: act, tail13, tail15, nov."""
    import torch
    n_cross, n_q = pmeta["n_cross"], pmeta["n_q"]
    recs, prev_small, side = [], None, None
    for t in range(0, len(df), STRIDE):
        front, wrist = dec(df.iloc[t]["image"]), dec(df.iloc[t]["wrist_image"])
        state = np.asarray(df.iloc[t]["state"], np.float32)
        V._CAP["image_mask"] = None
        _SA["maps"], _SA["active"] = [], True
        _XA["maps"], _XA["active"] = [], True
        with torch.inference_mode():
            policy.get_action(obs_of(pmeta, front, wrist, state, instr),
                              options={"session_ids": [f"avt{ep}"], "reset_memory": [t == 0]})
        _SA["active"] = _XA["active"] = False
        sa, xa, im = list(_SA["maps"]), list(_XA["maps"]), V._CAP["image_mask"]
        if im is None or not sa or not xa:
            continue
        mask = im[0].bool().cpu().numpy()
        blocks = img_blocks(mask)
        if len(blocks) < 2:
            continue
        side = int(round((blocks[0][1] - blocks[0][0] + 1) ** 0.5))
        rec = {"t": t, "front": front, "wrist": wrist}

        # ---- tail (prefill): backbone self-attn, post-image summary queries -> patches.
        # The trailing n_q slots are the moment tokens (HAMLET) / the slots framesamp models
        # overwrite downstream; the SUMMARY tokens are what note 22 calls "tail".
        L = sa[0].shape[0]
        tail_q = list(range(blocks[-1][1] + 1, L - n_q))
        if not tail_q:
            tail_q = list(range(blocks[-1][1] + 1, L))
        for li in TAIL_LAYERS:
            if li >= len(sa):
                continue
            A = sa[li].numpy()[tail_q]                       # (n_tail, L)
            g = [A[:, b0:b1 + 1].mean(0).reshape(side, side) for (b0, b1) in blocks[:2]]
            rec[f"tail{li}"] = np.stack(g)                   # (2, side, side)

        # ---- act: DiT action -> current-frame patch columns at ACT_LAYER
        n_den = max(1, len(xa) // n_cross)
        As = [xa[d * n_cross + ACT_LAYER] for d in range(n_den)
              if d * n_cross + ACT_LAYER < len(xa)]
        if As:
            av = (sum(A.float().mean(0) for A in As) / len(As)).numpy()   # (Lk,)
            Lm = len(mask) - n_q          # framesamp models: memory tokens replace the n_q tail
            if av.shape[0] == len(mask):
                av2, m2, off = av, mask, 0
            elif av.shape[0] >= Lm > 0:
                av2, m2, off = av[:Lm], mask[:Lm], 0
            else:
                av2 = None
            if av2 is not None:
                g = []
                for (b0, b1) in blocks[:2]:
                    if b1 < len(av2):
                        g.append(av2[b0:b1 + 1].reshape(side, side))
                if len(g) == 2:
                    rec["act"] = np.stack(g)

        # ---- nov: per-cell pixel diff on the same grid
        small = np.stack([_cellize(front, side), _cellize(wrist, side)])
        rec["nov"] = (np.abs(small - prev_small).mean(-1) if prev_small is not None
                      else np.zeros((2, side, side), np.float32))
        prev_small = small

        # ---- f0: hard frame-0 reservation (no scoring — an anchor, not a channel).
        # Front view of the first scored frame, matching TokenDrop's sentinel.
        rec["f0f"] = np.zeros((2, side, side), np.float32)
        if not recs:                       # this is the first scored step
            rec["f0f"][0] = 1.0
        if "act" in rec and f"tail{TAIL_LAYERS[0]}" in rec:
            recs.append(rec)
    return recs, side


# ---------------------------------------------------------------- selection sim
def top_k(recs, ch, k, sentinel_frame0=False):
    """Running top-k heap over all (t, view, r, c) candidates for one channel."""
    h = []
    if ch.startswith("f0"):
        k = 10 ** 9      # reservation: take every frame-0 cell, never compete for slots
    for j, rec in enumerate(recs):
        g = rec.get(ch)
        if g is None:
            continue
        side = g.shape[-1]
        for v in range(2):
            for r in range(side):
                for c in range(side):
                    s = float(g[v, r, c])
                    if ch.startswith("f0") and s <= 0:
                        continue              # f0 is a fixed reservation, not a ranked channel
                    if sentinel_frame0 and j == 0 and v == 0:
                        s = 1e9                       # TokenDrop frame-0 front sentinel
                    elif sentinel_frame0 and s < 1e-4:
                        continue
                    heapq.heappush(h, (s, rec["t"], v, r, c))
                    if len(h) > k:
                        heapq.heappop(h)
    return {(t, v, r, c): s for (s, t, v, r, c) in h}


def select(recs, channels, budget=BUDGET):
    """Union of per-channel top-(budget/n) sets (the deployed split-budget rule).

    The deployed rule UNDERFILLS when channels overlap (|union| < budget); we round-robin
    down each channel's ranked list instead, so every variant spends exactly `budget` slots
    and 2-way vs 3-way unions are compared at equal cost."""
    ranked = []
    for ch in channels:
        pool = top_k(recs, ch, budget, sentinel_frame0=(ch == "nov"))
        ranked.append([k for k, _s in sorted(pool.items(), key=lambda kv: -kv[1])])
    merged, ptr = {}, [0] * len(ranked)
    while len(merged) < budget and any(ptr[i] < len(ranked[i]) for i in range(len(ranked))):
        for i, lst in enumerate(ranked):
            while ptr[i] < len(lst) and lst[ptr[i]] in merged:
                ptr[i] += 1
            if ptr[i] < len(lst) and len(merged) < budget:
                merged[lst[ptr[i]]] = i          # value = which channel paid for the slot
                ptr[i] += 1
    return merged


def metrics(kept, gt):
    ts = np.asarray(sorted(t for (t, *_r) in kept))
    if not len(ts):
        return {}
    cover = lambda evs: ((sum(1 for e in evs if (np.abs(ts - e) <= TOL).sum() >= MIN_CELLS),
                          len(evs)) if evs else (0, 0))
    rr, rn = cover(gt.get("ref", []))
    er, en = cover(gt.get("exec", []) + ([gt["press"][0]] if "press" in gt else []))
    cells = [(v, r, c) for (_t, v, r, c) in kept]
    t0 = gt.get("t0", 0)
    return dict(n=len(kept), steps=len(set(ts.tolist())),
                span=f"{int(np.percentile(ts, 5))}-{int(np.percentile(ts, 95))}",
                demo=float((ts < t0).mean()) if t0 > 0 else 0.0,
                wrist=float(np.mean([v for (_t, v, _r, _c) in kept])),
                ref=f"{rr}/{rn}", exec=f"{er}/{en}",
                dup=1.0 - len(set(cells)) / max(1, len(cells)))


# ---------------------------------------------------------------- render
def draw_two(frame, act_cells, tail_cells, side):
    """act-only BLUE · tail-only ORANGE · shared GREEN."""
    from PIL import Image, ImageDraw
    im = Image.fromarray(frame).convert("RGB")
    W, H = im.size
    d = ImageDraw.Draw(im)
    a, b = set(act_cells), set(tail_cells)
    for cells, col in ((a - b, (70, 130, 255)), (b - a, (255, 150, 40)), (a & b, (40, 210, 90))):
        for (r, c) in cells:
            x0, y0 = c * W / side, r * H / side
            d.rectangle([x0, y0, x0 + W / side, y0 + H / side], outline=col, width=2)
    return im


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


CHCOL = {"nov": (220, 40, 40), "f0f": (220, 40, 40), "f0b": (220, 40, 40),
         "act": (70, 130, 255), "tail15": (255, 150, 40), "tail13": (255, 150, 40)}


def draw_by_channel(frame, cells, side):
    """cells = [(r, c, channel_name)] — outline colored by which channel paid for the slot."""
    from PIL import Image, ImageDraw
    im = Image.fromarray(frame).convert("RGB")
    W, H = im.size
    d = ImageDraw.Draw(im)
    for (r, c, ch) in cells:
        x0, y0 = c * W / side, r * H / side
        d.rectangle([x0, y0, x0 + W / side, y0 + H / side],
                    outline=CHCOL.get(ch, (150, 150, 150)), width=2)
    return im


def filmstrip_channels(sel, chs, recs_by_t, side, n_cols=12):
    """Filmstrip where each kept cell is colored by the channel that selected it."""
    per_t = defaultdict(lambda: defaultdict(list))
    for (t, v, r, c), ci in sel.items():
        per_t[t][v].append((r, c, chs[ci]))
    top_ts = span_ts(per_t, n_cols)
    f_html, w_html = [], []
    for t in top_ts:
        rec = recs_by_t.get(t)
        if rec is None:
            continue
        cnt = defaultdict(int)
        for v in (0, 1):
            for (_r, _c, ch) in per_t[t].get(v, []):
                cnt[ch] += 1
        tag = " ".join(f"{ch}:{n}" for ch, n in cnt.items())
        f_html.append(f'<td><img src="data:image/jpeg;base64,'
                      f'{b64(draw_by_channel(rec["front"], per_t[t].get(0, []), side))}"'
                      f' width="130"><br><small>t={t} · {tag}</small></td>')
        w_html.append(f'<td><img src="data:image/jpeg;base64,'
                      f'{b64(draw_by_channel(rec["wrist"], per_t[t].get(1, []), side))}"'
                      f' width="130"><br><small>wrist</small></td>')
    return ("<table><tr><td><small><b>front</b></small></td>" + "".join(f_html)
            + "</tr><tr><td><small><b>wrist</b></small></td>" + "".join(w_html) + "</tr></table>")


def span_ts(per_t, n_cols):
    """Timesteps that SPAN THE WHOLE EPISODE: evenly spaced over the steps this variant
    actually holds cells for (first and last always included). Picking the top-N by cell
    count instead would cluster the strip in one phase and hide the rest of the episode."""
    have = sorted(per_t)
    if len(have) <= n_cols:
        return have
    idx = np.linspace(0, len(have) - 1, n_cols).round().astype(int)
    return [have[i] for i in sorted(set(idx.tolist()))]


def union_compare_section(sels, names, recs, side, n_cols=12):
    """The union variants grouped together, each with its OWN kept set, each spanning the
    whole episode."""
    by_t = {r["t"]: r for r in recs}
    out = []
    for name in names:
        if name not in sels:
            continue
        sel, chs = sels[name]
        share = defaultdict(int)
        for ci in sel.values():
            share[chs[ci]] += 1
        out.append(f"<h4 style='margin-bottom:2px'>{name}</h4><small>slot share "
                   + " · ".join(
                       f"<b style='color:{'#dc2828' if c.startswith(('nov', 'f0')) else '#4682ff' if c == 'act' else '#ff9628'}'>{c}</b> {share[c]}"
                       for c in chs) + "</small>")
        out.append(filmstrip_channels(sel, chs, by_t, side, n_cols=n_cols))
    return "\n".join(out)


def filmstrip(kept, recs_by_t, side, n_cols=12):
    per_t = defaultdict(lambda: defaultdict(list))
    for (t, v, r, c) in kept:
        per_t[t][v].append((r, c))
    top_ts = span_ts(per_t, n_cols)
    f_html, w_html = [], []
    for t in top_ts:
        rec = recs_by_t.get(t)
        if rec is None:
            continue
        nf, nw = len(per_t[t].get(0, [])), len(per_t[t].get(1, []))
        f_html.append(f'<td><img src="data:image/jpeg;base64,'
                      f'{b64(draw_kept(rec["front"], per_t[t].get(0, []), side))}" width="130">'
                      f"<br><small>t={t} · {nf}f+{nw}w</small></td>")
        w_html.append(f'<td><img src="data:image/jpeg;base64,'
                      f'{b64(draw_kept(rec["wrist"], per_t[t].get(1, []), side))}" width="130">'
                      f"<br><small>wrist · {nw}</small></td>")
    return ("<table><tr><td><small><b>front</b></small></td>" + "".join(f_html)
            + "</tr><tr><td><small><b>wrist</b></small></td>" + "".join(w_html) + "</tr></table>")


def overlay_section(recs, side, act_ch, tail_ch, n_cols=8):
    """Same-frame per-step top-16 comparison: where does each channel look RIGHT NOW."""
    idx = np.linspace(0, len(recs) - 1, min(n_cols, len(recs))).astype(int)
    f_html, w_html = [], []
    for j in idx:
        rec = recs[j]
        cells = {}
        for ch in (act_ch, tail_ch):
            g = rec.get(ch)
            if g is None:
                cells[ch] = {0: [], 1: []}
                continue
            flat = [(float(g[v, r, c]), v, r, c) for v in range(2)
                    for r in range(side) for c in range(side)]
            top = sorted(flat, reverse=True)[:16]
            cells[ch] = {v: [(r, c) for (_s, vv, r, c) in top if vv == v] for v in (0, 1)}
        ov = len(set(map(tuple, [(v, r, c) for v in (0, 1) for (r, c) in cells[act_ch][v]]))
                 & set(map(tuple, [(v, r, c) for v in (0, 1) for (r, c) in cells[tail_ch][v]])))
        f_html.append(f'<td><img src="data:image/jpeg;base64,'
                      f'{b64(draw_two(rec["front"], cells[act_ch][0], cells[tail_ch][0], side))}"'
                      f' width="130"><br><small>t={rec["t"]} · ∩{ov}/16</small></td>')
        w_html.append(f'<td><img src="data:image/jpeg;base64,'
                      f'{b64(draw_two(rec["wrist"], cells[act_ch][1], cells[tail_ch][1], side))}"'
                      f' width="130"><br><small>wrist</small></td>')
    return ("<table><tr><td><small><b>front</b></small></td>" + "".join(f_html)
            + "</tr><tr><td><small><b>wrist</b></small></td>" + "".join(w_html) + "</tr></table>")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--models", default="vanilla,expd_framesamp,HAMLET")
    args = ap.parse_args()
    eps = {"SwingXtimes": [760]} if args.smoke else EPS

    install_self_attn_capture()
    install_cross_attn_capture()
    summary, overlap_rows = [], []

    for mname in args.models.split(","):
        policy, V, pmeta = build_policy(MODELS[mname])
        print(f"[{mname}] n_cross={pmeta['n_cross']} n_q={pmeta['n_q']}", flush=True)
        for task, ep_list in eps.items():
            sections = []
            for ep in ep_list:
                df, instr = load_episode(ep)
                gt = parse_gt(task, ep)
                print(f"[{mname}/{task}/ep{ep}] T={len(df)} scoring...", flush=True)
                recs, side = episode_signals(policy, V, pmeta, df, instr, ep)
                if not recs:
                    print("  no signals"); continue
                by_t = {r["t"]: r for r in recs}
                tl = [f"tail{li}" for li in TAIL_LAYERS if f"tail{TAIL_LAYERS[0]}" in recs[0]]
                variants = {"nov": ("nov",), "act_L13": ("act",)}
                for ch in tl:
                    variants[ch.replace("tail", "tail_L")] = (ch,)
                variants["act∪tail_L15"] = ("act", "tail15")
                variants["nov∪act (deployed)"] = ("nov", "act")
                variants["nov∪tail_L15"] = ("nov", "tail15")
                variants["nov∪act∪tail_L15"] = ("nov", "act", "tail15")
                variants["nov∪act∪tail_L13"] = ("nov", "act", "tail13")
                # no-novelty proposal: hard frame-0 anchor + the two attention channels.
                # Front-only (81 cells ≈ 16% budget) — reserving BOTH views was dropped: 162
                # cells is the entire grid, so it pins the cell-count metric at its ceiling and
                # measures nothing (it also eats 31% of the budget on a single frame).
                variants["f0front∪act∪tail_L15"] = ("f0f", "act", "tail15")

                kept_sets, paid_by, sels = {}, {}, {}
                for name, chs in variants.items():
                    chs = tuple(c for c in chs if c in recs[0])
                    if not chs:
                        continue
                    sel = select(recs, chs)
                    kept_sets[name] = set(sel.keys())
                    sels[name] = (sel, chs)
                    if len(chs) > 1:   # slot share actually won by each channel
                        cnt = np.bincount(list(sel.values()), minlength=len(chs))
                        paid_by[name] = {chs[i]: int(cnt[i]) for i in range(len(chs))}

                # channel overlap at equal budget (VLA-Pruner's statistic)
                full = {ch: set(top_k(recs, ch, BUDGET, sentinel_frame0=(ch == "nov")).keys())
                        for ch in ("act", "nov") + tuple(tl) if ch in recs[0]}
                for a, b in (("act", "tail13"), ("act", "tail15"), ("tail13", "tail15"),
                             ("act", "nov"), ("tail15", "nov")):
                    if a in full and b in full:
                        overlap_rows.append(dict(model=mname, task=task, ep=ep, pair=f"{a}∩{b}",
                                                 overlap=len(full[a] & full[b]) / BUDGET))

                head = [f"<h3>ep{ep} (T={len(df)}, t0={gt.get('t0', 0)}) — {instr[:150]}</h3>"]
                if gt.get("ref") or gt.get("exec"):
                    head.append(f"<p>ref {gt.get('ref', [])} · exec {gt.get('exec', [])}"
                                f"{' · press ' + str(gt['press']) if 'press' in gt else ''}</p>")
                head.append("<h4 style='color:#444'>act (blue) vs tail_L15 (orange) — per-step "
                            "top-16 on the SAME frame; green = both</h4>")
                head.append(overlay_section(recs, side, "act",
                                            "tail15" if "tail15" in recs[0] else tl[0]))
                if "tail13" in recs[0]:
                    head.append("<h4 style='color:#444'>act (blue) vs tail_L13 (orange)</h4>")
                    head.append(overlay_section(recs, side, "act", "tail13"))
                head.append("<hr><h3 style='color:#444'>Union comparison — each variant over "
                            "the WHOLE episode, box color = channel that paid for the slot "
                            "(<span style='color:#dc2828'>novelty</span> · "
                            "<span style='color:#4682ff'>act_L13</span> · "
                            "<span style='color:#ff9628'>tail_L15</span>)</h3>")
                head.append(union_compare_section(
                    sels, ["nov∪act (deployed)", "nov∪tail_L15", "nov∪act∪tail_L15"],
                    recs, side))
                head.append("<hr><h3 style='color:#444'>Per-variant filmstrips "
                            "(each variant's own kept set, sampled across the whole episode)"
                            "</h3>")
                for name, kept in kept_sets.items():
                    m = metrics(kept, gt)
                    pb = paid_by.get(name)
                    summary.append(dict(model=mname, task=task, ep=ep, method=name, **m,
                                        paid=("/".join(f"{v}" for v in pb.values()) if pb else "")))
                    head.append(f"<h4>{name} — n={m.get('n')} steps={m.get('steps')} "
                                f"span={m.get('span')} demo={m.get('demo', 0):.2f} "
                                f"wrist={m.get('wrist', 0):.2f} REF {m.get('ref')} "
                                f"EXEC {m.get('exec')} dup={m.get('dup', 0):.2f}</h4>")
                    head.append(f'<img src="data:image/jpeg;base64,'
                                f'{b64(alloc_strip(kept, len(df), gt.get("t0", 0)))}"'
                                f' style="border:1px solid #999">')
                    sel, chs = sels[name]
                    if len(chs) > 1:
                        head.append("<small>slot share " + " · ".join(
                            f"<b>{c}</b> {paid_by[name][c]}" for c in chs)
                            + " — box color: <span style='color:#dc2828'>anchor/novelty</span> · "
                              "<span style='color:#4682ff'>act</span> · "
                              "<span style='color:#ff9628'>tail</span></small>")
                        head.append(filmstrip_channels(sel, chs, by_t, side))
                    else:
                        head.append(filmstrip(kept, by_t, side))
                sections.append("\n".join(head))
            if sections:
                page = (f"<meta charset='utf-8'><title>act vs tail patches — {task} — {mname}</title>"
                        f"<h1>{task} — {mname}: act (action→vision) vs tail (prefill→vision) "
                        f"patch selection</h1><p>grid {side}×{side}×2 views, stride {STRIDE}, "
                        f"budget {BUDGET}. act = DiT cross-attn L{ACT_LAYER}; tail = backbone "
                        f"self-attn post-image summary tokens → patches.</p>"
                        + "\n<hr>\n".join(sections))
                (OUT / f"acttail_{task}_{mname}.html").write_text(page)
                print(f"wrote {OUT}/acttail_{task}_{mname}.html", flush=True)
        del policy
        import torch
        torch.cuda.empty_cache()

    import pandas as pd
    pd.DataFrame(summary).to_csv(OUT / "acttail_summary.csv", index=False)
    ov = pd.DataFrame(overlap_rows)
    ov.to_csv(OUT / "acttail_overlap.csv", index=False)
    print("\n=== kept-set metrics ===")
    print(pd.DataFrame(summary).to_string())
    if len(ov):
        print("\n=== channel overlap @ budget 512 ===")
        print(ov.groupby(["model", "pair"])["overlap"].agg(["mean", "min", "max"]).to_string())
    print("ACTTAIL_DONE")


if __name__ == "__main__":
    main()
