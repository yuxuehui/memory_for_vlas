#!/usr/bin/env python3
"""Visualize v2b-fix framesamp keyframes + moment→framesamp inject (mem_fs_inject=moment).

IMPORTANT: injection is FEATURE-SPACE only (broadcast s_f ∈ R^d onto every token of frame f).
Pixels are never rewritten. This script shows:
  FIG A — which RGB frames are selected (TRAIN episode-linspace vs INFER recent-F FIFO)
  FIG B — what the fuse does in feature space: ||s_f|| over time, pairwise cos(s_i,s_j)
           (anti-aliasing: identical swings should get DISTINCT states), and a
           before/after patch-feature PCA of one frame (uniform shift = constant Δ).

Usage:
  cd HAMLET-Isaac-GR00T-N1d6
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/visualize_v2b_fs_inject.py \
    --model /nas/pollux/xuehui/v2b_ckpts/ckpt60k_fix \
    --episodes 700,308 --device cuda:0 \
    --outdir /home/users/xuehui/myfile/Markdown/v2b_fs_inject_viz
"""
from __future__ import annotations

import argparse
import collections
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

_CAP: dict = {}


def dec(d):
    return np.array(Image.open(io.BytesIO(d["bytes"])).convert("RGB"))


def load_episode(repo, ep):
    chunk = "chunk-000" if ep < 1000 else "chunk-001"
    df = pd.read_parquet(f"{repo}/data/robomme/data/{chunk}/episode_{ep:06d}.parquet")
    tasks = {
        json.loads(l)["task_index"]: json.loads(l)["task"]
        for l in open(f"{repo}/data/robomme/meta/tasks.jsonl")
    }
    return df, tasks[int(df.iloc[0]["task_index"])]


def train_linspace_idxs(L: int, F: int) -> np.ndarray:
    return np.linspace(0, L - 1, num=F).round().astype(int)


def build_policy(model_path, device):
    from gr00t.policy.gr00t_policy import Gr00tPolicy
    from gr00t.data.embodiment_tags import EmbodimentTag

    policy = Gr00tPolicy(
        model_path=model_path, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT, device=device
    )
    ah = policy.model.action_head
    cfg = policy.model.config
    meta = dict(
        K=int(getattr(cfg, "memory_window", 1)),
        nq=int(getattr(cfg, "n_moment_tokens", 0)),
        stride=int(getattr(cfg, "memory_stride", 16)),
        F=int(getattr(cfg, "mem_framesamp_frames", 0) or 0),
        budget=int(getattr(cfg, "mem_framesamp_budget", 0) or 0),
        inject=str(getattr(cfg, "mem_fs_inject", "none")),
        cond=str(getattr(cfg, "mem_cond_type", "")),
        vkeys=policy.modality_configs["video"].modality_keys,
        skeys=policy.modality_configs["state"].modality_keys,
        lkey=policy.modality_configs["language"].modality_keys[0],
        is_hamlet=bool(getattr(policy, "use_hamlet_inference", False)),
    )
    print(
        f"  config: mem_cond_type={meta['cond']} mem_fs_inject={meta['inject']} "
        f"F={meta['F']} budget={meta['budget']} K={meta['K']} n_q={meta['nq']} stride={meta['stride']}"
    )
    assert meta["cond"] == "dual" and meta["inject"] == "moment", (
        f"expected dual+moment, got cond={meta['cond']} inject={meta['inject']}"
    )
    assert meta["F"] > 0, "mem_framesamp_frames must be >0"
    return policy, meta, ah


def install_inject_hooks(ah):
    """Capture (B,F,d) inject states and optional pre-add backbone rows."""
    _CAP.clear()
    _CAP["s"] = []  # list of (F,d) cpu float
    _CAP["tails"] = []
    orig = ah._fs_inject_states

    def wrapped(tails_BFnd):
        s = orig(tails_BFnd)
        _CAP["s"].append(s.detach().float().cpu()[0])  # (F,d)
        _CAP["tails"].append(tails_BFnd.detach().float().cpu()[0])  # (F,nq,d)
        return s

    ah._fs_inject_states = wrapped
    return orig


def replay_with_inject(policy, pmeta, ah, repo, episode):
    K, nq, stride, F = pmeta["K"], pmeta["nq"], pmeta["stride"], pmeta["F"]
    vkeys, skeys, lkey = pmeta["vkeys"], pmeta["skeys"], pmeta["lkey"]
    df, instr = load_episode(repo, episode)
    ep_len = len(df)
    call_steps = list(range(0, ep_len, stride))
    # Infer FIFO of front RGB that feed the spatial path (last F observed fronts).
    fifo_front = collections.deque(maxlen=F)
    sid = f"v2b_viz_{episode}"
    out = []
    for ci, t in enumerate(call_steps):
        row = df.iloc[t]
        front = dec(row["image"])
        wrist = dec(row["wrist_image"])
        state = np.asarray(row["state"], np.float32)
        obs = {
            "video": {
                vkeys[0]: front[None, None],
                vkeys[1]: wrist[None, None],
            },
            "state": {
                skeys[0]: state[None, None, :7],
                skeys[1]: state[None, None, 7:8],
            },
            "language": {lkey: [[instr]]},
        }
        fifo_front.append(front.copy())
        _CAP["s"].clear()
        _CAP["tails"].clear()
        policy.get_action(obs, options={"session_ids": [sid], "reset_memory": [ci == 0]})
        rec = {
            "step": t,
            "front": front,
            "wrist": wrist,
            "fifo": [f.copy() for f in fifo_front],  # oldest→newest, len≤F
        }
        if _CAP["s"]:
            s = _CAP["s"][-1].numpy()  # (F,d) — may be shorter early if cache not full
            rec["s"] = s
            rec["s_norm"] = np.linalg.norm(s, axis=-1)
        out.append(rec)
        if ci == 0:
            print(f"  first call: fifo={len(fifo_front)} s_shape={None if 's' not in rec else rec['s'].shape}")
    return out, {**pmeta, "instr": instr, "ep_len": ep_len}


def fig_keyframes(recs, meta, ep, df_fronts, out_path):
    """FIG A: train linspace strip vs infer FIFO at late step."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    F, L, instr = meta["F"], meta["ep_len"], meta["instr"]
    train_idx = train_linspace_idxs(L, F)

    # pick a late call where FIFO is full
    late = next((r for r in reversed(recs) if len(r["fifo"]) >= F and "s" in r), recs[-1])
    fig = plt.figure(figsize=(2.1 * F + 1.5, 6.2))
    gs = fig.add_gridspec(3, F, height_ratios=[1.0, 1.0, 0.55], hspace=0.35, wspace=0.15)

    # row 0: TRAIN linspace keyframes
    for i, ti in enumerate(train_idx):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(df_fronts[ti])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"t={ti}", fontsize=9)
        if i == 0:
            ax.set_ylabel("TRAIN\nlinspace F", fontsize=9)
    # row 1: INFER FIFO at late step
    fifo = late["fifo"]
    # pad left if early
    pad = F - len(fifo)
    for i in range(F):
        ax = fig.add_subplot(gs[1, i])
        if i < pad:
            ax.imshow(np.zeros_like(df_fronts[0]))
            ax.set_title("—", fontsize=9)
        else:
            ax.imshow(fifo[i - pad])
            ax.set_title(f"FIFO[{i - pad}]", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        if i == 0:
            ax.set_ylabel(f"INFER FIFO\n@step {late['step']}", fontsize=9)

    # row 2: timeline
    ax = fig.add_subplot(gs[2, :])
    ax.set_xlim(-1, L)
    ax.set_ylim(0, 3)
    ax.axhline(1.5, color="0.85", lw=4)
    ax.scatter(train_idx, np.full(F, 2.0), c="#c44e52", s=60, zorder=3, label="train linspace")
    # map FIFO frames to nearest episode steps of the late call window
    # approximate: late step and previous (F-1)*stride policy calls
    stride = meta["stride"]
    infer_steps = [max(0, late["step"] - (F - 1 - i) * stride) for i in range(F)]
    ax.scatter(infer_steps, np.full(F, 1.0), c="#4c72b0", s=60, zorder=3, label="infer FIFO≈")
    ax.axvline(late["step"], color="0.3", ls="--", lw=1, label=f"current step {late['step']}")
    ax.set_yticks([1, 2])
    ax.set_yticklabels(["infer", "train"])
    ax.set_xlabel("episode frame index")
    ax.legend(loc="upper right", fontsize=8, ncol=3)
    ax.set_title(
        "Train uses FULL-episode linspace (incl. future frames); "
        "infer uses recent-F FIFO — documented mismatch",
        fontsize=9,
    )

    fig.suptitle(
        f"v2b-fix keyframes — ep{ep} (L={L}, F={F})\n\"{instr[:110]}\"",
        fontsize=11,
    )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def fig_inject(recs, meta, ep, out_path):
    """FIG B: ||s_f|| trajectory + FIFO strip + cos-sim (anti-alias probe)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    F, instr = meta["F"], meta["instr"]
    usable = [r for r in recs if "s" in r and r["s"].shape[0] == F]
    if len(usable) < 2:
        print(f"fig_inject: not enough full-F captures ({len(usable)})")
        return

    steps = np.array([r["step"] for r in usable])
    norms = np.stack([r["s_norm"] for r in usable], 0)  # (Tcall, F)
    late = usable[-1]
    s = late["s"]  # (F,d)
    sn = s / (np.linalg.norm(s, axis=-1, keepdims=True) + 1e-8)
    cos = sn @ sn.T
    off = float((cos.sum() - np.trace(cos)) / (F * (F - 1)))

    fig = plt.figure(figsize=(2.15 * F, 8.0))
    gs = fig.add_gridspec(3, F, height_ratios=[1.15, 1.0, 1.05], hspace=0.45, wspace=0.12)

    ax = fig.add_subplot(gs[0, :])
    for fi in range(F):
        ax.plot(steps, norms[:, fi], lw=1.6, label=f"slot {fi}")
    ax.set_xlabel("episode step")
    ax.set_ylabel(r"$\|s_f\|_2$")
    ax.set_title("Moment→framesamp inject magnitude (feature-space; RGB pixels unchanged)")
    ax.legend(fontsize=7, ncol=4, loc="best")
    ax.grid(True, alpha=0.3)

    fifo = late["fifo"]
    pad = F - len(fifo)
    for i in range(F):
        ax = fig.add_subplot(gs[1, i])
        if i < pad:
            ax.imshow(np.zeros_like(late["front"]))
            ax.set_title("—", fontsize=8)
        else:
            ax.imshow(fifo[i - pad])
            nrm = float(late["s_norm"][i]) if i < len(late["s_norm"]) else 0.0
            ax.set_title(f"‖s‖={nrm:.3f}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        if i == 0:
            ax.set_ylabel(f"INFER FIFO\n@ {late['step']}", fontsize=9)

    ax = fig.add_subplot(gs[2, : F // 2 + 1])
    im = ax.imshow(cos, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(F))
    ax.set_yticks(range(F))
    ax.set_title(f"cos(s_i, s_j) @ step {late['step']}", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.05)

    ax = fig.add_subplot(gs[2, F // 2 + 1 :])
    ax.axis("off")
    note = (
        "What 'fuse moment into the image' does\n"
        "──────────────────────────────────\n"
        "fs_backbone[f] += s_f   (broadcast over\n"
        "                        all tokens of frame f)\n"
        "s_f = Proj(MemoryTF(tails)_f)\n"
        "  • lives in d=2048 feature space\n"
        "  • SAME vector on every patch → RGB\n"
        "    spatial layout is untouched\n"
        "  • zero-init Proj ⇒ ‖s‖ grows with train\n"
        "  • identical RGB swings → DISTINCT s_f\n"
        "    if block-causal TF binds event order\n"
        f"\nmean ‖s‖ @late = {float(late['s_norm'].mean()):.4f}\n"
        f"mean off-diag cos = {off:.3f}"
    )
    ax.text(0.0, 0.95, note, family="monospace", fontsize=8.5, va="top", transform=ax.transAxes)

    fig.suptitle(
        f"v2b-fix inject — ep{ep}  (mem_fs_inject=moment, F={F})\n\"{instr[:100]}\"",
        fontsize=11,
    )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def fig_train_strip(ep, df_fronts, meta, out_path):
    """Standalone high-res train-linspace collage."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    F, L = meta["F"], len(df_fronts)
    idxs = train_linspace_idxs(L, F)
    fig, axes = plt.subplots(1, F, figsize=(2.2 * F, 2.6))
    for i, (ax, ti) in enumerate(zip(axes, idxs)):
        ax.imshow(df_fronts[ti])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"#{i}  t={ti}/{L-1}", fontsize=9)
    fig.suptitle(
        f"TRAIN framesamp keyframes = round(linspace(0,{L-1},{F})) — ep{ep}",
        fontsize=11,
    )
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/nas/pollux/xuehui/v2b_ckpts/ckpt60k_fix")
    ap.add_argument("--repo", default="/home/users/xuehui/myfile/HAMLET-Isaac-GR00T-N1d6")
    ap.add_argument(
        "--episodes",
        default="700,308",
        help="700=SwingXtimes (aliasing), 308=VideoPlaceButton (content suite where v2b won)",
    )
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument(
        "--outdir",
        default="/home/users/xuehui/myfile/Markdown/v2b_fs_inject_viz",
    )
    args = ap.parse_args()
    episodes = [int(x) for x in str(args.episodes).split(",") if x.strip()]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("=== load v2b-fix ===")
    policy, pmeta, ah = build_policy(args.model, args.device)
    orig_inject = install_inject_hooks(ah)

    for ep in episodes:
        print(f"\n=== ep{ep} ===")
        df, instr = load_episode(args.repo, ep)
        fronts = [dec(df.iloc[t]["image"]) for t in range(len(df))]
        meta_ep = {**pmeta, "instr": instr, "ep_len": len(df)}
        fig_train_strip(ep, fronts, meta_ep, outdir / f"train_linspace_ep{ep}.png")

        recs, meta_ep = replay_with_inject(policy, pmeta, ah, args.repo, ep)
        n_s = sum(1 for r in recs if "s" in r)
        print(f"  calls={len(recs)} with_inject_capture={n_s}")
        fig_keyframes(recs, meta_ep, ep, fronts, outdir / f"keyframes_train_vs_infer_ep{ep}.png")
        fig_inject(recs, meta_ep, ep, outdir / f"inject_feature_ep{ep}.png")

    ah._fs_inject_states = orig_inject
    print(f"\nDone. Figures in {outdir}")


if __name__ == "__main__":
    main()
