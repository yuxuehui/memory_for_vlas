#!/usr/bin/env python3
"""Visualize SpecPrune-VLA's action-aware token selection, adapted to GR00T N1.6.

SpecPrune-VLA (ICML 2026, arXiv:2509.05614, alexwhz-sjtu/SpecPrune-VLA) keeps visual tokens
by two signals (see its spec_prune_vla.py, read 2026-07-23):
  1. TEXT->VISION attention at deep "goal" layers (vlm_layer_attn): text-prompt query rows ->
     image key cols, averaged over heads/text-tokens, top-k per layer, unioned across layers.
     This is its "action-aware token". Its insight: shallow layers attend to background,
     deep layers concentrate on action-centric regions.
  2. FRAME-COSINE dynamic region (get_similarity_indices): 14x14 patchify + per-patch cosine
     vs the previous frame; HIGH-sim = static (reused), the complement = dynamic, protected.

We don't have OpenVLA-OFT, so we replicate the RULE on GR00T. ⚠️ ARCHITECTURE MISMATCH: in
OpenVLA-OFT the text prompt comes AFTER the vision tokens, so text can causally attend to
vision. GR00T N1.6 puts the instruction in the causal PREFIX (before images) → prefix→image
mass is EXACTLY 0.0 (measured, all layers). The only nonimage tokens that can causally attend
to vision are the POST-IMAGE summary tokens ("tail", note-22). So the faithful GR00T port of
SpecPrune's action-aware text→vision is TAIL→vision @ L15 (the pretrained summarization layer;
L13's readout circuit exists only in moment-consuming models). This is the same signal as our
tail_L15 memory channel. Rendered per frame:
  GREEN = top-k tail(nonimage)→vision @ L15 (SpecPrune "action-aware", GR00T port)
  RED   = dynamic region (frame cosine < threshold)
Plus a shallow-vs-deep layer-profile panel to show the hierarchical-attention insight.

Run: cd HAMLET-Isaac-GR00T-N1d6 && CUDA_VISIBLE_DEVICES=<gpu> NO_ALBUMENTATIONS_UPDATE=1 \
       .venv/bin/python .../spec_prune_viz.py --model /tmp/robomme_eval/model_pu50k
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, "/home/users/xuehui/myfile/Markdown/patch_memory_labels")
from gen_act_vs_tail_patches import (EPS, build_policy, dec, img_blocks, obs_of, b64)
from gen_patch_labels import _cellize, load_episode
from gen_nonvision_attn import _SA, install_self_attn_capture

OUT = Path("/home/users/xuehui/myfile/Markdown/patch_memory_labels")
STRIDE = 8
SHALLOW, DEEP = 5, 15          # SpecPrune goal_layers were deep [14,30] on 32-layer LLaMA.
                               # Our Eagle keeps 16 layers; for framesamp-family models (pu50k)
                               # the L13 readout circuit does NOT exist (note-22), so the valid
                               # deep summarization layer is L15.
SIM_THRESHOLD = 0.986          # SpecPrune PRIMARY_SIM_THRESHOLD
KEEP_FRAC = 0.20               # top fraction as "action-aware" per view (viz clarity)
NCOL = 10


def cos_dynamic(front, wrist, prev, side):
    """SpecPrune get_similarity_indices, on the cell grid: cells with cosine < threshold vs
    the previous frame are the DYNAMIC region. Returns (2, side, side) bool + new cell state."""
    cur = np.stack([_cellize(front, side), _cellize(wrist, side)])   # (2, side, side, 3)
    if prev is None:
        return np.zeros((2, side, side), bool), cur
    a = cur.reshape(2, side * side, 3).astype(np.float32)
    b = prev.reshape(2, side * side, 3).astype(np.float32)
    cos = (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-8)
    dyn = (cos < SIM_THRESHOLD).reshape(2, side, side)
    return dyn, cur


def text_to_vision(sa_maps, blocks, side, layer, n_q=0):
    """SpecPrune vlm_layer_attn analog for GR00T. ⚠️ SpecPrune's TEXT rows come AFTER the
    vision tokens (BOS,imgs,text) so they causally attend to vision. GR00T N1.6 puts the
    instruction in the causal PREFIX (before images) → prefix→image mass is EXACTLY 0
    (measured). The only tokens that can causally attend to vision are the POST-IMAGE summary
    tokens (our note-22 'tail'). So the faithful GR00T port of SpecPrune's action-aware
    text→vision is TAIL→vision — which is exactly the tail_L15 channel."""
    if layer >= len(sa_maps):
        return None
    A = sa_maps[layer].numpy()
    L = A.shape[0]
    lo, hi = blocks[-1][1] + 1, (L - n_q if n_q > 0 else L)
    q = list(range(lo, hi))
    if not q:
        return None
    g = [A[q][:, b0:b1 + 1].mean(0).reshape(side, side) for (b0, b1) in blocks[:2]]
    return np.stack(g)                                              # (2, side, side)


def topk_mask(grid, frac):
    """Per-view top-`frac` cells -> bool mask (2, side, side)."""
    out = np.zeros_like(grid, bool)
    for v in range(grid.shape[0]):
        flat = grid[v].reshape(-1)
        k = max(1, int(round(frac * flat.size)))
        idx = np.argpartition(-flat, k - 1)[:k]
        m = np.zeros_like(flat, bool); m[idx] = True
        out[v] = m.reshape(grid[v].shape)
    return out


def draw(frame, mask, side, col, width=2):
    im = Image.fromarray(frame).convert("RGB"); W, H = im.size; d = ImageDraw.Draw(im)
    for r in range(side):
        for c in range(side):
            if mask[r, c]:
                x0, y0 = c * W / side, r * H / side
                d.rectangle([x0, y0, x0 + W / side, y0 + H / side], outline=col, width=width)
    return im


def collect(policy, pmeta, df, instr, ep):
    n_q = pmeta["n_q"]
    recs, prev = [], None
    for t in range(0, len(df), STRIDE):
        front, wrist = dec(df.iloc[t]["image"]), dec(df.iloc[t]["wrist_image"])
        state = np.asarray(df.iloc[t]["state"], np.float32)
        import visualize_memory_attention as V
        V._CAP["image_mask"] = None
        _SA["maps"], _SA["active"] = [], True
        with torch.inference_mode():
            policy.get_action(obs_of(pmeta, front, wrist, state, instr),
                              options={"session_ids": [f"sp{ep}"], "reset_memory": [t == 0]})
        _SA["active"] = False
        im, sa = V._CAP["image_mask"], list(_SA["maps"])
        if im is None or not sa:
            continue
        blocks = img_blocks(im[0].bool().cpu().numpy())
        if len(blocks) < 2:
            continue
        side = int(round((blocks[0][1] - blocks[0][0] + 1) ** 0.5))
        sem_deep = text_to_vision(sa, blocks, side, DEEP, pmeta["n_q"])
        sem_shallow = text_to_vision(sa, blocks, side, SHALLOW, pmeta["n_q"])
        if sem_deep is None:
            continue
        dyn, prev = cos_dynamic(front, wrist, prev, side)
        recs.append(dict(t=t, front=front, wrist=wrist, side=side,
                         sem_deep=sem_deep, sem_shallow=sem_shallow, dyn=dyn))
    return recs


def render(task, ep, T, instr, recs):
    if not recs:
        return ""
    side = recs[0]["side"]
    idx = sorted(set(np.linspace(0, len(recs) - 1, min(NCOL, len(recs))).astype(int)))
    picks = [recs[i] for i in idx]
    rows = [f"<hr><h3>ep{ep} (T={T}) — {instr[:90]}</h3>"]
    # action-aware (deep top-k green) + dynamic (red) overlay
    for view, vn in [(0, "front"), (1, "wrist")]:
        cells = []
        for r in picks:
            frame = r[vn]
            m_sem = topk_mask(r["sem_deep"], KEEP_FRAC)[view]
            im = draw(frame, r["dyn"][view], side, (220, 60, 60), 1)     # dynamic region
            im = draw(np.array(im), m_sem, side, (40, 200, 80), 2)       # action-aware top-k
            cells.append(f'<td><img src="data:image/jpeg;base64,{b64(im)}" width="140"><br>'
                         f'<small>t={r["t"]}</small></td>')
        rows.append(f"<b>{vn}</b><table><tr>" + "".join(cells) + "</tr></table>")
    # hierarchical insight: shallow vs deep text->vision heat, front view
    def heat(grid, v):
        g = grid[v]; g = (g - g.min()) / (g.ptp() + 1e-9)
        a = (g * 255).astype(np.uint8)
        return Image.fromarray(a).resize((140, 140), Image.NEAREST)
    for label, key in [("shallow L%d text→vision" % SHALLOW, "sem_shallow"),
                       ("deep L%d text→vision (action-aware)" % DEEP, "sem_deep")]:
        if picks[0].get(key) is None:
            continue
        cells = [f'<td><img src="data:image/jpeg;base64,{b64(heat(r[key], 0))}" width="140">'
                 f'<br><small>t={r["t"]}</small></td>' for r in picks]
        rows.append(f"<b>{label}</b><table><tr>" + "".join(cells) + "</tr></table>")
    return "\n".join(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/tmp/robomme_eval/model_pu50k")
    ap.add_argument("--tasks", default="SwingXtimes,ButtonUnmask,VideoPlaceButton,PatternLock,RouteStick")
    args = ap.parse_args()
    install_self_attn_capture()
    policy, V, pmeta = build_policy(args.model)
    for task in args.tasks.split(","):
        secs = []
        for ep in EPS.get(task, []):
            df, instr = load_episode(ep)
            recs = collect(policy, pmeta, df, instr, ep)
            secs.append(render(task, ep, len(df), instr, recs))
            print(f"[{task}/ep{ep}] {len(recs)} steps", flush=True)
        page = (f"<meta charset='utf-8'><title>SpecPrune action-aware — {task}</title>"
                f"<h1>{task}: SpecPrune-VLA action-aware token (adapted to GR00T)</h1>"
                f"<p>GREEN = top-{int(KEEP_FRAC*100)}% tail→vision @L{DEEP} (action-aware, GR00T port; prefix→vision is causally 0) · "
                f"RED = dynamic region (frame cosine &lt; {SIM_THRESHOLD}). "
                f"Heat rows: shallow L{SHALLOW} vs deep L{DEEP} text→vision — SpecPrune's "
                f"hierarchical-attention insight (shallow=scattered, deep=action-centric).</p>"
                + "\n".join(secs))
        (OUT / f"specprune_{task}.html").write_text(page)
        print(f"wrote specprune_{task}.html")
    print("SPEC_DONE")


if __name__ == "__main__":
    main()
