#!/usr/bin/env python3
"""Does the DEPLOYED tail_L15 proxy match the REAL backbone L15 tail->vision attention?

The probe validated tail_L15 using REAL backbone self-attention (post-image summary rows ->
image cols @ L15). But the deployed `mem_fs_tail_share` channel scores patches by a PROXY:
`<patch_token, frame_summary_token>` dot product (cheap, no capture). If they rank patches the
same way, the proxy is faithful and the deployed 3-way is training the right signal; if not,
we must capture the real attention.

Per frame we compute BOTH from ONE forward:
  real[i]  = mean over tail rows of A_L15[tail, image_i]   (probe signal, via _SA capture)
  proxy[i] = <v_i, mean(tail tokens)>                       (deployed signal, via feature hook)
and report Spearman rank-corr + top-k selection overlap (selection is top-k, so overlap is
what actually decides which patches are kept).

Run: cd HAMLET-Isaac-GR00T-N1d6 && CUDA_VISIBLE_DEVICES=<gpu> NO_ALBUMENTATIONS_UPDATE=1 \
       .venv/bin/python .../proxy_vs_real.py --model /tmp/robomme_eval/model_pu50k
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/home/users/xuehui/myfile/Markdown/patch_memory_labels")
from gen_act_vs_tail_patches import EPS, build_policy, dec, img_blocks, obs_of
from gen_patch_labels import load_episode
from gen_nonvision_attn import _SA, install_self_attn_capture

STRIDE = 8
LAYER = 15
KFRAC = 0.25          # top-25% selection overlap (tail's budget share ~128/512 ≈ 25%)

_BB = {"feat": None, "mask": None}


def install_feature_hook(model):
    def hook(_m, _i, out):
        try:
            _BB["feat"] = out["backbone_features"].detach().float().cpu()
            _BB["mask"] = out["image_mask"].detach().bool().cpu()
        except Exception:
            pass
    return model.backbone.register_forward_hook(hook)


def spearman(a, b):
    ra = a.argsort().argsort().astype(np.float64)
    rb = b.argsort().argsort().astype(np.float64)
    ra -= ra.mean(); rb -= rb.mean()
    den = np.sqrt((ra**2).sum() * (rb**2).sum()) + 1e-12
    return float((ra * rb).sum() / den)


def topk_overlap(a, b, frac):
    k = max(1, int(round(frac * len(a))))
    sa = set(np.argpartition(-a, k - 1)[:k].tolist())
    sb = set(np.argpartition(-b, k - 1)[:k].tolist())
    return len(sa & sb) / k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/tmp/robomme_eval/model_pu50k")
    ap.add_argument("--tasks", default="SwingXtimes,ButtonUnmask,VideoPlaceButton,PatternLock,RouteStick")
    args = ap.parse_args()
    install_self_attn_capture()
    policy, V, pmeta = build_policy(args.model)
    install_feature_hook(policy.model)
    n_q = pmeta["n_q"]

    per_task = {}
    for task in args.tasks.split(","):
        rows = []                      # (spearman_all, overlap_all, sp_front, sp_wrist)
        for ep in EPS.get(task, []):
            df, instr = load_episode(ep)
            for t in range(0, len(df), STRIDE):
                _BB["feat"] = None
                _SA["maps"], _SA["active"] = [], True
                ob = obs_of(pmeta, dec(df.iloc[t]["image"]), dec(df.iloc[t]["wrist_image"]),
                            np.asarray(df.iloc[t]["state"], np.float32), instr)
                with torch.inference_mode():
                    policy.get_action(ob, options={"session_ids": [f"pr{ep}"], "reset_memory": [t == 0]})
                _SA["active"] = False
                sa, feat, mask = list(_SA["maps"]), _BB["feat"], _BB["mask"]
                if not sa or feat is None or LAYER >= len(sa):
                    continue
                m = mask[0]; blocks = img_blocks(m.numpy())
                if len(blocks) < 2:
                    continue
                L = sa[LAYER].shape[0]
                lo, hi = blocks[-1][1] + 1, (L - n_q if n_q > 0 else L)
                if hi <= lo:
                    continue
                A = sa[LAYER].numpy()
                f = feat[0]
                summ = f[lo:hi].mean(0).numpy()                       # frame summary token
                imcols = []
                for (b0, b1) in blocks[:2]:
                    imcols.extend(range(b0, b1 + 1))
                imcols = np.array(imcols)
                real = A[lo:hi][:, imcols].mean(0)                    # (n_img,) real L15 tail->img
                patches = f[imcols].numpy()                           # (n_img, d)
                proxy = patches @ summ                                # (n_img,) deployed proxy
                if real.shape != proxy.shape or len(real) < 4:
                    continue
                nv = len(imcols) // 2
                rows.append((spearman(proxy, real), topk_overlap(proxy, real, KFRAC),
                             spearman(proxy[:nv], real[:nv]), spearman(proxy[nv:], real[nv:])))
            print(f"[{task}/ep{ep}] {len(rows)} frames so far", flush=True)
        if rows:
            arr = np.array(rows)
            per_task[task] = arr.mean(0)
    del policy; torch.cuda.empty_cache()

    print("\n=== proxy <patch,summary>  vs  REAL backbone L15 tail->vision ===")
    print(f"{'task':18}{'spearman':>10}{'top25%overlap':>15}{'sp_front':>10}{'sp_wrist':>10}")
    allrows = []
    for task, v in per_task.items():
        print(f"{task:18}{v[0]:>10.3f}{v[1]:>15.3f}{v[2]:>10.3f}{v[3]:>10.3f}")
        allrows.append(v)
    if allrows:
        m = np.array(allrows).mean(0)
        print(f"{'MEAN':18}{m[0]:>10.3f}{m[1]:>15.3f}{m[2]:>10.3f}{m[3]:>10.3f}")
    print("\nPROXY_DONE")


if __name__ == "__main__":
    main()
