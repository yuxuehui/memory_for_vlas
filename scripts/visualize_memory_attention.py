#!/usr/bin/env python3
"""Visualize HAMLET's MEMORY MECHANISM + action->image attention vs vanilla, on a replayed
RoboMME memory-task episode (frames decoded from the LeRobot parquet; cache filled via get_action).

Two figures:
  FIG A (HAMLET only) memory mechanism: rows = late steps; current frame + the K cached past
     snapshots, each labeled with memory-transformer attention (which past snapshot NOW reads);
     plus an action->moment bar (which of the n_q memory tokens the action head leans on).
  FIG B (HAMLET vs vanilla) action->image: rows = steps, cols = {HAMLET, vanilla}; the DiT action
     head's cross-attention over the FRONT-view image patches, overlaid on the frame, across steps.

Captures (Eagle VLM is flash-attn -> no weights; only these are reachable):
  - memory transformer: monkeypatch gr00t.model.modules.memory._Attention.forward.
  - DiT action->KV cross-attn: monkeypatch F.scaled_dot_product_attention, gate on non-square.
  - image_mask: forward hook on policy.model.backbone (to map cross-attn cols -> image patches).
"""
from __future__ import annotations
import argparse, io, json, collections
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image

_CAP = {"mem": [], "cross": [], "image_mask": None}


def install_attn_hooks():
    import torch.nn.functional as F
    import gr00t.model.modules.memory as memmod
    orig_sdpa = F.scaled_dot_product_attention

    def patched_sdpa(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **kw):
        out = orig_sdpa(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale, **kw)
        try:
            Lq, Lk = q.shape[-2], k.shape[-2]
            if Lq != Lk:  # DiT cross-attn (action queries x backbone KV)
                d = q.shape[-1]; sc = scale if scale is not None else 1.0 / (d ** 0.5)
                scores = (q.float() @ k.float().transpose(-2, -1)) * sc
                if attn_mask is not None and attn_mask.dtype != torch.bool:
                    scores = scores + attn_mask.float()
                _CAP["cross"].append(scores.softmax(-1).mean(1).detach().cpu())  # (B,Lq,Lk)
        except Exception:
            pass
        return out

    F.scaled_dot_product_attention = patched_sdpa
    torch.nn.functional.scaled_dot_product_attention = patched_sdpa

    _rope = memmod._apply_rope

    def mem_forward(self, x, attn_mask, cos, sin):
        B, L, _ = x.shape
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        q, k = _rope(q, k, cos, sin)
        scores = (q.float() @ k.float().transpose(-2, -1)) / (self.head_dim ** 0.5)
        if attn_mask is not None:
            scores = scores + attn_mask.float()
        w = scores.softmax(-1)
        _CAP["mem"].append(w.mean(1).detach().cpu())  # (B,L,L)
        out = (w.to(v.dtype) @ v).transpose(1, 2).contiguous().view(B, L, -1)
        return self.o_proj(out)

    memmod._Attention.forward = mem_forward


def install_imagemask_hook(model):
    def hook(_m, _inp, out):
        try:
            im = out["image_mask"] if isinstance(out, dict) or hasattr(out, "__getitem__") else None
            if im is not None:
                _CAP["image_mask"] = im.detach().cpu()
        except Exception:
            pass
    return model.backbone.register_forward_hook(hook)


def dec(d):
    return np.array(Image.open(io.BytesIO(d["bytes"])).convert("RGB"))


def load_episode(repo, ep):
    chunk = "chunk-000" if ep < 1000 else "chunk-001"
    df = pd.read_parquet(f"{repo}/data/robomme/data/{chunk}/episode_{ep:06d}.parquet")
    tasks = {json.loads(l)["task_index"]: json.loads(l)["task"]
             for l in open(f"{repo}/data/robomme/meta/tasks.jsonl")}
    return df, tasks[int(df.iloc[0]["task_index"])]


def img_heat_from_cross(Lk_image_cols, n_views=2, view=0):
    """Given the per-key cross-attn vector restricted to image columns, return a (side,side) grid for `view`."""
    n_img = Lk_image_cols.shape[0]
    ppv = n_img // n_views
    side = int(round(ppv ** 0.5))
    if side * side != ppv:
        return None
    seg = Lk_image_cols[view * ppv:(view + 1) * ppv]
    return seg.reshape(side, side).numpy()


def build_policy(model_path, device):
    from gr00t.policy.gr00t_policy import Gr00tPolicy
    from gr00t.data.embodiment_tags import EmbodimentTag
    policy = Gr00tPolicy(model_path=model_path, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT, device=device)
    install_imagemask_hook(policy.model)
    meta = dict(
        K=int(getattr(policy.model.config, "memory_window", 1)),
        nq=int(getattr(policy.model.config, "n_moment_tokens", 0)),
        stride=int(getattr(policy.model.config, "memory_stride", 16)),
        vkeys=policy.modality_configs["video"].modality_keys,
        skeys=policy.modality_configs["state"].modality_keys,
        lkey=policy.modality_configs["language"].modality_keys[0],
        is_hamlet=bool(getattr(policy, "use_hamlet_inference", False)),
    )
    return policy, meta


def replay(policy, pmeta, repo, episode, want_memory):
    K, nq, stride = pmeta["K"], pmeta["nq"], pmeta["stride"]
    vkeys, skeys, lkey, is_hamlet = pmeta["vkeys"], pmeta["skeys"], pmeta["lkey"], pmeta["is_hamlet"]
    df, instr = load_episode(repo, episode)
    ep_len = len(df)
    call_steps = list(range(0, ep_len, stride))
    fed = collections.deque(maxlen=max(K, 1))
    sid = f"s{episode}"
    out = []
    for ci, t in enumerate(call_steps):
        row = df.iloc[t]
        front = dec(row["image"]); wrist = dec(row["wrist_image"])
        state = np.asarray(row["state"], np.float32)
        obs = {"video": {vkeys[0]: front[None, None], vkeys[1]: wrist[None, None]},
               "state": {skeys[0]: state[None, None, :7], skeys[1]: state[None, None, 7:8]},
               "language": {lkey: [[instr]]}}
        if ci == 0 and K > 1:
            for _ in range(K):
                fed.append(front)
        else:
            fed.append(front)
        _CAP["mem"].clear(); _CAP["cross"].clear(); _CAP["image_mask"] = None
        if is_hamlet:
            policy.get_action(obs, options={"session_ids": [sid], "reset_memory": [ci == 0]})
        else:
            policy.get_action(obs)

        rec = {"step": t, "frames": {"front": front.copy(), "wrist": wrist.copy()}, "cached": list(fed)}
        # action -> image PER cross-attn layer (avg over the n_denoise flow steps)
        if _CAP["cross"] and _CAP["image_mask"] is not None:
            n_den = int(getattr(policy.model.config, "num_inference_timesteps", 4))
            Lk = _CAP["cross"][0].shape[-1]
            calls = [c for c in _CAP["cross"] if c.shape[-1] == Lk]
            per_call = torch.stack([c[0].mean(0) for c in calls], 0)  # (total, Lk), mean over action queries
            total = per_call.shape[0]
            im = _CAP["image_mask"][0].bool(); N = im.shape[0]
            if total % n_den == 0 and total // n_den >= 1:
                n_lay = total // n_den
                per_layer = per_call.view(n_den, n_lay, Lk).mean(0)  # (n_lay, Lk) avg over denoise
            else:
                per_layer = per_call.mean(0, keepdim=True)  # fallback: single averaged map
            rec["heat_layers"] = {vi: [img_heat_from_cross(per_layer[li][:N][im], n_views=len(vkeys), view=vi)
                                       for li in range(per_layer.shape[0])]
                                  for vi in range(len(vkeys))}
            # attention-MASS split: fraction of the action's cross-attention on each token category
            # (cr sums to 1 over all Lk columns). KV layout (HAMLET inf): [image, text, moment_raw(nq), mem_aug(nq)].
            cr_all = per_call.mean(0)  # (Lk,) avg over all layers, denoise, action queries
            img = torch.zeros(Lk, dtype=torch.bool); img[:N] = im  # pad image_mask to Lk
            img_mass = float(cr_all[img].sum())
            if is_hamlet and nq > 0:  # last nq cols = memory-conditioned tokens (replace the moment tail in place)
                mem = torch.zeros(Lk, dtype=torch.bool); mem[Lk - nq:] = True
                rec["mass"] = {"image": img_mass, "text": float(cr_all[(~img) & (~mem)].sum()),
                               "memory": float(cr_all[mem].sum())}
            else:
                rec["mass"] = {"image": img_mass, "text": float(cr_all[~img].sum())}
            if ci == 0:
                print(f"  [{('HAMLET' if is_hamlet else 'vanilla')}] cross calls/step={total} -> "
                      f"{per_layer.shape[0]} cross-attn layers x {n_den} denoise | mass={ {k: round(v,3) for k,v in rec['mass'].items()} }")
        if want_memory and is_hamlet and _CAP["mem"]:
            mem = _CAP["mem"][-1][0]; L = mem.shape[0]
            per_key = mem[L - nq:L, :].mean(0)
            ma = per_key.view(K, nq).sum(1).numpy(); ma = ma / (ma.sum() + 1e-8)
            rec["mem_attn"] = ma  # (action->moment mass now lives in rec["mass"], computed above)
        out.append(rec)
    return out, {**pmeta, "instr": instr, "ep_len": ep_len}


def overlay(frame, heat, alpha=0.5):
    import matplotlib.cm as cm
    H, W = frame.shape[:2]
    ht = torch.as_tensor(heat, dtype=torch.float32)[None, None]
    ht = torch.nn.functional.interpolate(ht, size=(H, W), mode="bilinear", align_corners=False)[0, 0].numpy()
    ht = (ht - ht.min()) / (ht.max() - ht.min() + 1e-8)
    rgb = (cm.get_cmap("turbo")(ht)[..., :3] * 255).astype(np.uint8)
    return np.clip(frame.astype(float) * (1 - alpha) + rgb.astype(float) * alpha, 0, 255).astype(np.uint8)


def fig_memory(recs, meta, ep, out_path):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    K, nq, stride, instr = meta["K"], meta["nq"], meta["stride"], meta["instr"]
    rows = [r for r in recs if "mem_attn" in r][-4:]
    if not rows:
        print("fig_memory: no memory rows"); return
    nc = 1 + K + 1  # current + K cached + action->moment bar
    fig, axes = plt.subplots(len(rows), nc, figsize=(2.0 * nc, 2.3 * len(rows)), squeeze=False)
    for ri, r in enumerate(rows):
        ax = axes[ri][0]; ax.imshow(r["frames"]["front"]); ax.set_xticks([]); ax.set_yticks([])
        ax.set_ylabel(f"step {r['step']}", fontsize=10)
        if ri == 0: ax.set_title("current", fontsize=9)
        amax = int(r["mem_attn"].argmax())
        for ki in range(K):
            ax = axes[ri][1 + ki]
            ax.imshow(r["cached"][ki] if ki < len(r["cached"]) else np.zeros_like(r["frames"]["front"]))
            ax.set_xticks([]); ax.set_yticks([])
            w = r["mem_attn"][ki]
            ax.set_title(f"{w:.2f}", fontsize=9, color="red" if ki == amax else "black")
            for sp in ax.spines.values():
                sp.set_color("red" if ki == amax else "0.7"); sp.set_linewidth(3 if ki == amax else 0.5)
            if ri == 0: ax.set_xlabel(f"t-{(K-1-ki)*stride}", fontsize=8); ax.xaxis.set_label_position("top")
        ax = axes[ri][1 + K]
        mass = r.get("mass", {})
        labels = list(mass.keys()); vals = [mass[k] for k in labels]
        cmap = {"image": "#4c72b0", "text": "#55a868", "moment": "#c44e52", "memory": "#8172b3"}
        ax.bar(range(len(vals)), vals, color=[cmap.get(k, "#888") for k in labels])
        ax.set_ylim(0, 1); ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=6, rotation=35, ha="right"); ax.tick_params(labelsize=6)
        if "memory" in mass:
            ax.text(len(labels) - 1, min(mass["memory"] + 0.03, 0.95), f"{mass['memory']:.2f}",
                    fontsize=7, ha="center", color="#8172b3")
        if ri == 0: ax.set_title("action attn mass", fontsize=8)
    fig.suptitle(f"HAMLET memory mechanism — ep{ep}: which past snapshot does the CURRENT step read? "
                 f"(red=peak; cols=K={K} cached, oldest→newest)\n\"{instr[:95]}\"", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.95]); Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight"); print(f"wrote {out_path}")


def fig_compare(rec_h, rec_v, meta, ep, out_path, n_steps=6):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    instr = meta["instr"]
    hm = {r["step"]: r for r in rec_h if r.get("img_heat") is not None}
    vm = {r["step"]: r for r in rec_v if r.get("img_heat") is not None}
    steps = sorted(set(hm) & set(vm))
    if not steps:
        print("fig_compare: no common steps with heat"); return
    idx = np.linspace(0, len(steps) - 1, min(n_steps, len(steps))).astype(int)
    steps = [steps[i] for i in idx]
    fig, axes = plt.subplots(len(steps), 2, figsize=(2.4 * 2, 2.4 * len(steps)), squeeze=False)
    for ri, s in enumerate(steps):
        for ci, (tag, m) in enumerate([("HAMLET", hm), ("vanilla", vm)]):
            ax = axes[ri][ci]; r = m[s]
            ax.imshow(overlay(r["front"], r["img_heat"])); ax.set_xticks([]); ax.set_yticks([])
            if ri == 0: ax.set_title(tag, fontsize=11)
            if ci == 0: ax.set_ylabel(f"step {s}", fontsize=10)
    fig.suptitle(f"action→image attention (front view) — ep{ep}, HAMLET vs vanilla across steps\n"
                 f"\"{instr[:95]}\"", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.95]); Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight"); print(f"wrote {out_path}")


def fig_layers(recs, meta, tag, view_idx, view_name, ep, out_path, every=2):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    have = [r for r in recs if r.get("heat_layers") and view_idx in r["heat_layers"]]
    if not have:
        print(f"fig_layers[{tag} {view_name}]: no heat"); return
    cols = have[::every]
    n_lay = max(len(r["heat_layers"][view_idx]) for r in cols)
    blk = 32 // max(n_lay, 1)  # DiT block stride between cross-attn layers
    nr, nc = n_lay, len(cols)
    fig, axes = plt.subplots(nr, nc, figsize=(1.55 * nc, 1.55 * nr), squeeze=False)
    for ci, r in enumerate(cols):
        frame = r["frames"][view_name]
        hl = r["heat_layers"][view_idx]
        for li in range(n_lay):
            ax = axes[li][ci]
            heat = hl[li] if li < len(hl) else None
            ax.imshow(overlay(frame, heat) if heat is not None else frame)
            ax.set_xticks([]); ax.set_yticks([])
            if li == 0: ax.set_title(f"step {r['step']}", fontsize=8)
            if ci == 0: ax.set_ylabel(f"L{li}\nblk{li*blk}", fontsize=7)
    fig.suptitle(f"{tag} [{view_name} cam] — action→image attention per DiT cross-attn layer "
                 f"(rows L0..L{n_lay-1} = blocks 0,{blk},…) across steps (cols) — ep{ep}\n\"{meta['instr'][:90]}\"", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96]); Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=125, bbox_inches="tight"); print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hamlet", default="/tmp/hamlet_runs/hamlet_e2e_model")
    ap.add_argument("--vanilla", default="/tmp/hamlet_runs/vanilla_model")
    ap.add_argument("--repo", default="/home/users/xuehui/myfile/HAMLET-Isaac-GR00T-N1d6")
    ap.add_argument("--episodes", default="1561,626,864,995",
                    help="comma list of episode indices (default: one per RoboMME suite)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--every", type=int, default=2, help="render every Nth policy call (~stride*N frames per column)")
    ap.add_argument("--outdir", default="/home/users/xuehui/myfile/Markdown/skill-vla")
    args = ap.parse_args()
    episodes = [int(x) for x in str(args.episodes).split(",") if x.strip()]

    install_attn_hooks()
    print("=== HAMLET (load once, loop episodes) ===")
    pol, pmeta = build_policy(args.hamlet, args.device)
    rec_h = {}
    for ep in episodes:
        print(f"-- HAMLET ep{ep}")
        rec_h[ep], meta_ep = replay(pol, pmeta, args.repo, ep, want_memory=True)
        fig_memory(rec_h[ep], meta_ep, ep, f"{args.outdir}/memory_attention_ep{ep}.png")
        for vi, vn in enumerate(pmeta["vkeys"]):
            short = vn.replace("_view", "")
            fig_layers(rec_h[ep], meta_ep, "HAMLET", vi, short, ep,
                       f"{args.outdir}/action_image_layers_HAMLET_{short}_ep{ep}.png", every=args.every)
    del pol; torch.cuda.empty_cache()

    print("=== vanilla (load once, loop episodes) ===")
    pol, pmeta = build_policy(args.vanilla, args.device)
    for ep in episodes:
        print(f"-- vanilla ep{ep}")
        rec_v, meta_ep = replay(pol, pmeta, args.repo, ep, want_memory=False)
        for vi, vn in enumerate(pmeta["vkeys"]):
            short = vn.replace("_view", "")
            fig_layers(rec_v, meta_ep, "vanilla", vi, short, ep,
                       f"{args.outdir}/action_image_layers_vanilla_{short}_ep{ep}.png", every=args.every)
    del pol; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
