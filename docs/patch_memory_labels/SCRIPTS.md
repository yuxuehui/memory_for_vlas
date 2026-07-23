# patch_memory_labels — probe & plotting scripts

Analysis/visualization scripts behind the patch-memory findings in [`README.md`](README.md)
(Chinese write-up).

> **Note for this repo copy:** the interactive per-episode HTML filmstrips
> (`acttail_*.html`, `cmp_navt_*.html`, `specprune_*.html`, ~92 MB) are **not committed** —
> re-generate them by running the script listed for each (below). The committed set is the
> findings write-up, PNG figures, CSV/txt data, and the scripts themselves.
 Each script loads a trained GR00T-N1.6 checkpoint, replays RoboMME
episodes teacher-forced (stride-8), captures attention / features, and emits HTML filmstrips,
PNG figures, CSV tables, or `.txt`/`.npz` data. All share the same capture plumbing:

- **backbone self-attention** (`gen_nonvision_attn.install_self_attn_capture`) — wraps the
  Eagle flash-attn function; gives `nonimage-token → image-patch` maps per layer (the *tail*
  channel: post-image summary tokens → patches).
- **DiT action→patch cross-attention** (`fs_patch_union` sdpa patch) — the *act* channel.
- **backbone image-mask + features** (`visualize_memory_attention.install_imagemask_hook`,
  and a feature hook in `dual_probe.py`) — which KV columns are image patches, and the raw
  per-patch token vectors that become `mem_seq`.

Common run form (pick a spare GPU; not GPU 0 on the shared box):

```bash
cd HAMLET-Isaac-GR00T-N1d6 && CUDA_VISIBLE_DEVICES=<gpu> NO_ALBUMENTATIONS_UPDATE=1 \
  .venv/bin/python /path/to/patch_memory_labels/<script>.py --model /tmp/robomme_eval/model_pu50k
```

Models used: `model_vanilla` (N1.6, no moment tokens), `model_expd60k` (framesamp, moment
unconsumed), `model_B_60k` = HAMLET (moment consumed), `model_pu50k` (deployed patch_union).

---

## Overview

| script | question | outputs | key finding |
|---|---|---|---|
| `gen_patch_labels.py` | td_diff vs attention vs union, per-patch | `summary.csv` | union is the only 3-way-near-best combiner |
| `gen_act_vs_tail_patches.py` | act (DiT) vs tail (backbone) relevance, 3 models | `acttail_*.html` ×15, `acttail_{summary,overlap}.csv` | tail_L13 readout circuit only in the moment-consuming model (0.31/0.31/0.17 overlap) |
| `temporal_persistence.py` | is each channel event-like or plateau-like? | `temporal_persistence.txt` | attention = "permanent kernel + moving rim" (~56%); novelty = event-like |
| `dual_probe.py` | is position in the token? do fragments keep content? | `dualprobe_cache.npz` | (view,y,x) decodable 1.00/0.99/0.95, time 0.61; selection less frame-distinctive than random |
| `compare_nov_act_tail.py` | what does adding tail_L15 change in the kept history? | `cmp_navt_*.html` ×5 | tail pulls budget from action-region close-ups to scene landmarks |
| `spec_prune_viz.py` | SpecPrune's action-aware token, ported to GR00T | `specprune_*.html` ×5 | prefix→vision is causally 0 in GR00T ⇒ the port is tail→vision @ L15 |
| `proxy_vs_real.py` | does the deployed dot-product tail proxy match real L15 attention? | `proxy.log` | NO — spearman 0.058, top-25% overlap 0.101 (< random) |
| `relevance_ablation.py` | *(early, superseded by `gen_act_vs_tail_patches.py`)* | stdout | act/tail top-512 overlap ~0.30 |

---

## Per-script detail

### `gen_patch_labels.py` — the original patch-selection probe
Simulates a running 512-patch heap under three rules (`td_diff` pixel-diff, `attn_L{5,10,13}`
action→patch, `split` = diff∪attn union) on `vanilla` + `expd_framesamp`. **`summary.csv` is
still used by `README.md` §1** (the union comparison table). Per-episode HTML filmstrips were
removed as superseded by the 3-model `acttail_*` set; re-generate with `--smoke` if needed.

### `gen_act_vs_tail_patches.py` — main comparison (act vs tail, 3 models)
The workhorse. In ONE policy call captures both the DiT act→patch cross-attn and the backbone
tail→patch self-attn, over 5 tasks × 3 models. Renders per page: per-channel filmstrips, an
act-vs-tail overlay, and a **"Union comparison"** block (`nov∪act` / `nov∪tail_L15` /
`nov∪act∪tail_L15`, each spanning the whole episode, boxes colored by the channel that paid
for the slot). Also the reusable helpers (`episode_signals`, `select`, `draw_by_channel`,
`span_ts`) that the other scripts import. Outputs `acttail_<Task>_<model>.html` ×15,
`acttail_summary.csv`, `acttail_overlap.csv`, figure `acttail_L13_circuit_ButtonUnmask.png`.

### `temporal_persistence.py` — event vs plateau
Per channel, top-K attended-cell overlap at lags 1/2/4/8 (lag = 8 control steps) + mean run
length once a cell enters the top-K. Answers whether "high dup" is redundancy (delete) or a
temporal trace (keep). Output `temporal_persistence.txt`.

### `dual_probe.py` — position + content probes (H1/H2)
Linear-probes the stored `mem_seq` tokens for (t,y,x,view) [H1: is position present?] and a
same-space own-frame retrieval of the selected vs random vs whole-frame pools [H2: do
fragments keep scene content?]. Motivated the PPE position-RoPE work (note 23). Caches tokens
to `dualprobe_cache.npz`.

### `compare_nov_act_tail.py` — history add/drop diff
Side-by-side of the deployed `nov∪act` vs candidate `nov∪act∪tail_L15` over the whole episode,
with a DELTA row (green = patches tail_L15 inserts, gray = patches it displaces). Output
`cmp_navt_<task>.html` ×5.

### `spec_prune_viz.py` — SpecPrune action-aware token (adapted to GR00T)
Ports SpecPrune-VLA's two signals — text→vision attention at a deep layer + frame-cosine
dynamic region — to GR00T. **Finding baked into the script:** GR00T's instruction is a causal
prefix, so `prefix→vision` mass is exactly 0; the only valid port is `tail(post-image)→vision
@ L15`. Green = top-k action-aware, red = dynamic region, plus a shallow-vs-deep heat panel.
Output `specprune_<task>.html` ×5.

### `proxy_vs_real.py` — is the deployed tail proxy faithful?
Compares the deployed `<patch, frame-summary>` dot-product proxy against the REAL backbone L15
tail→vision attention (Spearman + top-25% selection overlap). Verdict: not faithful (0.058 /
0.101) — a real-attention capture is needed. Output `proxy.log`.

### `relevance_ablation.py` — early act/tail ablation *(superseded)*
First act vs tail overlap measurement; kept for provenance. Use `gen_act_vs_tail_patches.py`
for current results.

---

## Figures (PNG, standalone)

| file | from | shows |
|---|---|---|
| `acttail_L13_circuit_ButtonUnmask.png` | `gen_act_vs_tail_patches.py` | 3 models × front/wrist; tail_L13 wrist-shift only in HAMLET |
| `union_vanilla_{ButtonUnmask,VideoPlaceButton,SwingXtimes}.png` | `compare_nov_act_tail` helpers | 3 union variants over the whole episode, boxes by paying channel |
| `union_f0_act_tail_{ButtonUnmask,VideoPlaceButton}.png` | *(negative result)* | frame-0 anchor can't replace novelty (deprecated f0 scheme) |

`ep300_annotated/` — per-timestep annotated frames from the early dilation ablation (README §7).
