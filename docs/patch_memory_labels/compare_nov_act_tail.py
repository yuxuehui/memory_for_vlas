#!/usr/bin/env python3
"""Focused history-selection comparison: deployed nov∪act  vs  nov∪act∪tail_L15.

The deployed patch_union keeps novelty∪relevance (relevance = act = DiT action→patch).
This renders, over the WHOLE episode, what each rule keeps and — the point — the DELTA:
which history patches adding the tail_L15 channel INSERTS, and which it DISPLACES. Uses the
actual deployed model (model_pu50k) so the act/tail attention is the real one.

Run: cd HAMLET-Isaac-GR00T-N1d6 && CUDA_VISIBLE_DEVICES=<gpu> NO_ALBUMENTATIONS_UPDATE=1 \
       .venv/bin/python .../compare_nov_act_tail.py --model /tmp/robomme_eval/model_pu50k
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, "/home/users/xuehui/myfile/Markdown/patch_memory_labels")
from gen_act_vs_tail_patches import (EPS, build_policy, episode_signals, select, span_ts, b64)
from gen_patch_labels import load_episode

OUT = Path("/home/users/xuehui/myfile/Markdown/patch_memory_labels")
NCOL = 12
A_CH = ("nov", "act")               # deployed
B_CH = ("nov", "act", "tail15")     # candidate


def draw(frame, cells, side, col):
    """cells = list of (r,c); col = RGB."""
    im = Image.fromarray(frame).convert("RGB")
    W, H = im.size
    d = ImageDraw.Draw(im)
    for (r, c) in cells:
        x0, y0 = c * W / side, r * H / side
        d.rectangle([x0, y0, x0 + W / side, y0 + H / side], outline=col, width=2)
    return im


def by_step(sel):
    per = defaultdict(lambda: defaultdict(list))          # t -> view -> [(r,c)]
    for (t, v, r, c) in sel:
        per[t][v].append((r, c))
    return per


def strip(title, per_by_view, recs_by_t, side, ts, colors):
    """One filmstrip row-pair (front,wrist). per_by_view maps a color name -> {t:{v:[(r,c)]}}."""
    fh, wh = [], []
    for t in ts:
        rec = recs_by_t.get(t)
        if rec is None:
            continue
        imf = rec["front"]; imw = rec["wrist"]
        # overlay each color group
        f = Image.fromarray(imf).convert("RGB"); w = Image.fromarray(imw).convert("RGB")
        for name, col in colors.items():
            per = per_by_view.get(name, {})
            f = draw(np.array(f), per.get(t, {}).get(0, []), side, col)
            w = draw(np.array(w), per.get(t, {}).get(1, []), side, col)
        fh.append(f'<td><img src="data:image/jpeg;base64,{b64(f)}" width="130"><br>'
                  f'<small>t={t}</small></td>')
        wh.append(f'<td><img src="data:image/jpeg;base64,{b64(w)}" width="130"><br><small>wrist</small></td>')
    return (f"<h4>{title}</h4><table><tr><td><small><b>front</b></small></td>"
            + "".join(fh) + "</tr><tr><td><small><b>wrist</b></small></td>" + "".join(wh)
            + "</tr></table>")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/tmp/robomme_eval/model_pu50k")
    ap.add_argument("--tasks", default="SwingXtimes,ButtonUnmask,VideoPlaceButton,PatternLock,RouteStick")
    args = ap.parse_args()
    from gen_nonvision_attn import install_cross_attn_capture, install_self_attn_capture
    install_self_attn_capture(); install_cross_attn_capture()
    policy, V, pmeta = build_policy(args.model)

    for task in args.tasks.split(","):
        secs = []
        for ep in EPS.get(task, []):
            df, instr = load_episode(ep)
            recs, side = episode_signals(policy, V, pmeta, df, instr, ep)
            if not recs:
                continue
            selA = set(select(recs, A_CH).keys())
            selB = set(select(recs, B_CH).keys())
            added = selB - selA                      # tail_L15 inserts
            dropped = selA - selB                    # displaced from deployed
            kept = selA & selB
            by_t = {r["t"]: r for r in recs}
            per_t_all = defaultdict(lambda: defaultdict(list))
            for (t, v, r, c) in selA | selB:
                per_t_all[t][v].append((r, c))
            ts = span_ts(per_t_all, NCOL)

            # row 1: deployed nov∪act (red)   row 2: +tail_L15 (red)   row 3: DELTA
            colsA = {"A": (220, 60, 60)}
            colsB = {"B": (220, 60, 60)}
            colsD = {"add": (40, 200, 80), "drop": (150, 150, 150)}
            pa = {"A": by_step(selA)}
            pb = {"B": by_step(selB)}
            pd = {"add": by_step(added), "drop": by_step(dropped)}
            secs.append(f"<hr><h3>ep{ep} (T={len(df)}) — {instr[:90]}</h3>"
                        f"<p>deployed nov∪act keeps {len(selA)} · +tail_L15 keeps {len(selB)} · "
                        f"tail INSERTS <b style='color:#28c850'>{len(added)}</b>, "
                        f"DISPLACES <b style='color:#999'>{len(dropped)}</b>, "
                        f"shared {len(kept)}</p>")
            secs.append(strip("deployed: nov ∪ act (relevance)", pa, by_t, side, ts, colsA))
            secs.append(strip("candidate: nov ∪ act ∪ tail_L15", pb, by_t, side, ts, colsB))
            secs.append(strip("DELTA — green = tail_L15 inserts · gray = displaced from deployed",
                              pd, by_t, side, ts, colsD))
            print(f"[{task}/ep{ep}] A {len(selA)} B {len(selB)} +{len(added)} -{len(dropped)}", flush=True)
        page = (f"<meta charset='utf-8'><title>nov∪act vs nov∪act∪tail_L15 — {task}</title>"
                f"<h1>{task}: history selection — deployed nov∪act vs nov∪act∪tail_L15</h1>"
                f"<p>model {Path(args.model).name}, budget {512}, whole episode ({NCOL} cols).</p>"
                + "\n".join(secs))
        (OUT / f"cmp_navt_{task}.html").write_text(page)
        print(f"wrote cmp_navt_{task}.html")
    print("CMP_DONE")


if __name__ == "__main__":
    main()
