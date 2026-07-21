# Memory for VLAs

**Memory representations for vision-language-action models** — implemented on GR00T-N1.6 and measured on
the **RoboMME** long-horizon benchmark.

VLAs are **single-frame reactive policies**: GR00T-N1.6 and π₀.₅ both take an observation horizon of **1**
(`delta_indices=[0]` — one frame per camera). Long-horizon manipulation, however, is a **POMDP**: the policy
must condition on *history*. Every method here answers the same question differently — **what to remember,
and how to feed it to the action head** — under a *fixed token budget*.

The cost of having no memory is large and measurable: **8.2 %** (no memory) → **17.2 %** (compressed
read-out) overall success on RoboMME. But no single memory wins everywhere — the best method changes with
the *task semantics* (counting vs. occlusion vs. trajectory imitation), which is the main empirical result
of this repo ([§5b](#5b-keyframe-selection-ab-all-60k-16-tasks--50-eps--800-episodes-each)).

**Methods at a glance** — five trained arms, all from one codebase, selected by config flags
([§0](#0-method-index-quick-reference)): `vanilla` · `HAMLET` read-out (compressed) ·
**keyframe selection** in three flavors — `FrameSamp` (uniform) → `TokenDrop` (observation change) →
`Action-conditioned patch memory` (novelty ∪ action-relevance) · `Multi-resolution memory` (read-out fused
into image tokens).

This README catalogs every method and gives a verified, copy-paste command for each. Architecture detail lives in the code (pointers in [Code map](#code-map)).

> Built on [Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) N1.6 and the
> [HAMLET](https://github.com/myungkyuKoo/HAMLET-Isaac-GR00T-N1d6) fork (Apache-2.0); the memory methods,
> keyframe selection, and RoboMME evaluation harness here are this project's contribution.

---

## 0. Method index (quick reference)

The five trained methods, their defining flags, and where they live in the code. All are the **same
codebase** — a method is a combination of config flags, not a branch.

| 方法 | 关键 flag | 实现位置 |
|---|---|---|
| **vanilla** | `--hamlet-mode off` | — （单帧反应式策略；GR00T 的 obs horizon = 1） |
| **HAMLET（压缩记忆）** | `--hamlet-mode finetune --mem-source moment` | `modules/memory.py`（`MemoryTransformer`）· `seq_memory.py`（GRU/SSM/Mamba 变体）· `multiscale_memory.py` |
| **Multi-resolution memory（read-out 融进 image token）** | `--mem-cond-type dual --mem-fs-inject moment` | `gr00t_n1d6.py::_fs_inject_states` + `modules/dit.py`（spatial cross-attn）· 见 §5a |
| **framesamp（均匀关键帧）** | `--mem-source framesamp --mem-fs-select fifo` | `gr00t_n1d6.py::_framesamp_mem_seq` + loader linspace |
| **tokendrop（diff 关键帧）** | `--mem-source framesamp --mem-fs-select diff` | `modules/fs_diff_select.py` + loader `_fs_diff_scores` / `_fs_diff_indices` |
| **Action-conditioned patch memory（patch_union）** | `--mem-source framesamp --mem-fs-select patch_union` | `modules/fs_patch_union.py` + `gr00t_n1d6.py::_patch_union_mem_seq` / `_patch_union_score_pass` / `_pu_commit` |

`--mem-fs-select` 只在 `--mem-source framesamp` 下生效，它决定 **哪些帧/patch 进入 memory**（选择规则），
与 `--mem-cond-type`（记忆怎么注入 DiT）正交。
---

## 1. The idea

A memory is formed from a **history sequence** (raw vision tokens / text tokens / a learned read-out token),
optionally **condensed**, then fed to the action expert. Methods differ in **what** they take as history and
**how** they represent it. Existing methods fall into two families:

| Family | Idea | Spectral view | Implemented here as |
|---|---|---|---|
| **① Raw tokens** | keep past frames as raw, differentiable vision tokens (ContextVLA / **FrameSamp** / EventVLA) | **all-pass** — preserves per-position (high-freq) detail; costly | `--mem-source framesamp` |
| **② Read-out summary** | attention-pool history into a few learned summary tokens (**HAMLET** read-out + TCL) | **low-pass (DC-dominant)** — keeps the invariant background, discards what varies | moment tokens + `MemoryTransformer`, `--mem-cond-type cross_attn` |

**Key empirical finding (RoboMME).** The read-out **wins the aggregate / low-frequency suites**
(Counting, Permanence, Reference) but **loses the position-varying / high-frequency suite (Imitation)** to
raw tokens — because attention-pooling is a convex combination, hence a **low-pass filter** (max gain at
DC, strict attenuation of every other frequency). Scaling the read-out 32× (n_q 4→128) barely moves it:
the ceiling is the **pooling operation**, not the token count.

**Two routes out of the low-pass ceiling** (and how this repo explores them):
1. **Un-pool → keep raw per-position tokens** (`A=I`): the **FrameSamp** path (family ①), and the
   **`h_spatial`** channel of **Multi-resolution memory**.
2. **Replace attention-pooling with a non-low-pass aggregator**: the **`SequenceMemory`** recurrent/SSM
   memories (`--memory-arch gru|ssm|mamba`) — a learned *selective* running state instead of a softmax
   average (e.g. count-/event-tracking), rather than a DC summary.

**Multi-resolution memory** is the **hybrid** that keeps the read-out's useful low-pass summary **and** its
register effect (freeing the action query from image-background patches) **while** adding raw spatial detail
back — and it **fuses** the two resolutions instead of merely concatenating them: each frame's raw image
tokens are injected with that frame's own read-out state (`--mem-fs-inject moment`), so episode-spanning
frames stop being indistinguishable to the policy.

---

## 2. Methods

A memory method = **(1) which memory** (*what to remember*) × **(2) how to integrate it** (*how to inject it
into the action head*). The two axes are **orthogonal** — any memory pairs with any integration. (Baseline:
`--hamlet-mode off` = no memory.) All runs fine-tune from `nvidia/GR00T-N1.6-3B`; flags append to the
[common Stage-2 command](#4-how-to-run). `n_q` = `--n-moment-tokens`, `K` = `--memory-window`.

### (1) Which memory — *what to remember*

| Memory | Proposal family | What it remembers | Distinguishing flags | Status |
|---|---|---|---|---|
| **Read-out (HAMLET)** | ② read-out summary | K past sets of `n_q` learnable moment tokens, compressed by the aggregator → a low-pass (DC) summary of the history | `--n-moment-tokens <n_q>` | ✅ main recipe (~18.4) |
| **FrameSamp (raw tokens)** | ① raw tokens | F episode-spanning frames' **raw vision patch tokens** (≤ budget) — uncompressed spatial/temporal detail | `--mem-source framesamp --mem-framesamp-frames 8 --mem-framesamp-budget 512` | ✅ |
| **Multi-resolution memory** (hybrid) | ①+② the proposed fix | **both at once**: read-out (`h_sem`) **+** raw framesamp (`h_spatial`, zero-init per-block spatial cross-attn), with each frame's tokens **colored by its own memory state** (block-causal, zero-init proj) so repeated frames are no longer aliased | `--mem-cond-type dual --mem-fs-inject moment --mem-framesamp-frames 8` | ✅ @60k **11.1 %** (§5a) |

### (1b) Keyframe selection — *which frames/patches enter the fixed budget*

`--mem-source framesamp` fixes **how much** memory there is (`--mem-framesamp-frames` frames,
`--mem-framesamp-budget` tokens); **`--mem-fs-select` decides what fills it.** Three methods, increasingly
informed — from content-blind uniform sampling, to observation-driven change detection, to
action-conditioned relevance. The rule is applied **identically at train and inference** (except `fifo`,
which is historically mismatched — see caveats):

| # | Selection method | Selection signal | Unit | Train side | Inference side | Flags | Result |
|---|---|---|---|---|---|---|---|
| 1 | **FrameSamp** (uniform) | none — content-blind even coverage | frame | loader `linspace(0, T-1, F)` (acausal) | rolling FIFO of the F most recent frames ⚠️ **mismatched** | `--mem-fs-select fifo` (default) | ✅ **12.4 %** |
| 2 | **TokenDrop** (diff) | **observation change**: mean \|pixel diff\| vs the last scored frame (frame-0 sentinel + top-(F−2) peaks ≤ anchor) | frame | `_fs_diff_scores` / `_fs_diff_indices` (causal, memoized per episode) | `DiffFrameSelector` (incremental heap) | `--mem-fs-select diff [--mem-fs-diff-stride 8]` | ✅ **14.0 %** |
| 3 | **Action-conditioned patch memory** (patch_union) | **novelty ∪ action-relevance**: token-space Δ top-(αM) **∪** DiT action→patch cross-attention top-((1−α)M) | **patch** | `_patch_union_mem_seq` + two-pass `_patch_union_score_pass` (no_grad pass over *all* candidates captures layer-ℓ attention) | read heap → capture attn in the real forward → `_pu_commit` post-action write | `--mem-fs-select patch_union [--mem-fs-attn-layer 13 --mem-fs-diff-share 0.5]` | ⏳ training |

- **1 → 2** trades uniform coverage for event coverage: wins the event-sparse suites (Counting +11.0,
  Permanence +8.5) and loses the continuous-trajectory ones (Reference −6.5, Imitation −6.5). Uniform
  sampling turns out to be the *correct prior for continuous manner*, not a naive baseline (§5b).
- **2 → 3** drops the selection unit from frames to **patches** and adds a second, *task-conditioned*
  channel: the policy's own action queries vote on which patches matter. At the same 512-token budget a
  patch-level union spans ~120 timesteps instead of 8 frames.

- **Read-out size** — set by `--n-moment-tokens` (`n_q`): **4** (light; the ~18.4 baseline) or **128** (wide
  bank; tests token-count vs the pooling ceiling — scaling 32× barely helps, so the limit is the *pooling op*).
- **Read-out aggregator** — how the K·n_q history is compressed, set by `--memory-arch`: `transformer`
  (default, block-causal attention) · `gru` · `ssm` (S4D) · `mamba` (selective SSM). The recurrent/SSM
  variants replace softmax-pooling with a learned running state (count/event tracking — a *non*-low-pass
  alternative). `mamba` (`exp_mamba/mamba_b`, K=16, hidden 512, state 64): **@40k = 11.5 %** — above vanilla
  8.25 / multi-resolution 11.1, but below framesamp 12.4 and the moment bar 18.38 on *every* suite (Counting 17.0/24.5,
  Permanence 10.5/17.0, Reference 13.0/21.5, Imitation 5.5/10.5; best tasks SwingXtimes 32, VideoPlaceOrder 18).
  ⚠️ run cut at **40k of 60k** (VM deleted) — undertrained confound LIVE (cf. multi-resolution jumped +4.9 at 40k→50k);
  ckpts 10k–40k in `gs://…/exp_mamba/mamba_b/`, eval CSVs `/tmp/robomme_eval/out_mamba_40k/`.
- **History coverage** — `--mem-window-mode recent` (recent K-stride window, default) vs `linspace` (causal
  whole-episode coverage). Orthogonal to everything above.

### (2) How to integrate — *how to inject memory into the DiT*

Set by `--mem-cond-type`. Orthogonal to (1).

| Integration | How it conditions the action head | Flag | Note |
|---|---|---|---|
| **cross_attn** | aggregated memory **replaces the moment-token KV tail**; the DiT cross-attends it alongside image+text (0 added keys/params) | `--mem-cond-type cross_attn` | ✅ **default & currently the best** |
| **adaln** | pooled memory → zero-init Linear → **added to the DiT timestep embedding** (moment tail sliced off the KV) | `--mem-cond-type adaln` | alternative |
| **modul (FiLM)** | action tokens cross-attend the full memory sequence and **FiLM-modulate each DiT block** (`--mem-film-layers` picks the depth band); also the path FrameSamp uses | `--mem-cond-type modul [--mem-film-layers all\|mid\|8-20]` | alternative |

---

## 3. The design space (config knobs)

Everything is a flag on `gr00t/experiment/launch_finetune.py` (`gr00t/configs/finetune_config.py`).

| Flag | Default | Choices | Meaning |
|---|---|---|---|
| `--hamlet-mode` | `finetune` | `off` / `tcl` / `finetune` | stage gate: vanilla / Stage-1 TCL pretrain / Stage-2 memory+head |
| `--mem-cond-type` | `cross_attn` | `cross_attn` / `adaln` / `modul` / `dual` | **how** memory conditions the DiT (`dual` = Multi-resolution memory's two-channel wiring) |
| `--memory-arch` | `transformer` | `transformer` / `gru` / `ssm` / `mamba` | **aggregator** over the K·n_q history |
| `--mem-source` | `moment` | `moment` / `framesamp` | compressed moment tokens vs raw per-frame patches (`modul` only) |
| `--mem-window-mode` | `recent` | `recent` / `linspace` | recent K-stride window vs causal whole-episode coverage (`moment` only) |
| `--memory-type` | `moment_token` | `moment_token` / `vision_feature` | learned moment tokens vs pooled primary-view tokens (64/step) |
| `--n-moment-tokens` | `4` | int | n_q — read-out tokens per step |
| `--memory-window` | `4` | int | K — history length (past snapshots) |
| `--memory-stride` | `16` | int | env-steps between snapshots; **must == eval `--n-action-steps`** |
| `--memory-num-layers` | `2` | int | aggregator depth |
| `--memory-hidden` | `512` | int | SequenceMemory bottleneck (`gru/ssm/mamba`) |
| `--memory-state-dim` | `64` | int | SSM state size (`ssm/mamba`) |
| `--mem-framesamp-frames` | `8` | int | episode-spanning frames the loader appends (**required >0** for `framesamp` **and** Multi-resolution memory) |
| `--mem-fs-inject` | `none` | `none`/`te`/`moment` | fuse the read-out state into each frame's image tokens (`moment` = the Multi-resolution recipe; `te` = ordering-only ablation; `none` = unfused two-channel) |
| `--mem-framesamp-budget` | `512` | int | cap on raw vision tokens fed to FiLM/spatial-attn |
| `--mem-fs-select` | `fifo` | `fifo`/`diff`/`patch_union` | **which** frames/patches enter memory (`framesamp` only); stamped into the ckpt so eval matches training |
| `--mem-fs-diff-stride` | `8` | int | scoring cadence for `diff`/`patch_union` (env steps at train, **policy calls** at eval — if the server is hit once per action chunk the effective stride is 8×chunk) |
| `--mem-fs-attn-layer` | `13` | int | `patch_union` only — DiT cross-attn layer whose action→patch attention is the relevance channel |
| `--mem-fs-diff-share` | `0.5` | float | `patch_union` only — novelty-channel share of the patch budget (rest = relevance) |
| `--mem-film-layers` | `all` | `all`/`mid`/`8,10,12`/`8-20` | FiLM injection depth (`modul` only) |
| `--mem-image-side` | `False` | bool | route memory tokens via IMAGE (vs TEXT) cross-attn pathway (`cross_attn` only) |
| `--load-moment-tokens-from` | `None` | path | warm-start moment tokens from a Stage-1 TCL checkpoint |
| `--freeze-moment-tokens` | `False` | bool | freeze moment tokens in Stage-2 |
| `--tune-top-llm-layers` | `4` | int | trainable top Eagle-LLM layers (**TCL must use 0**) |
| `--tcl-tau` / `--tcl-no-projection-head` | `0.07` / `False` | — | InfoNCE temperature / drop the TCL projection head (`tcl` only) |

---

## 4. How to run

Shared setup for every command below: 4×A100, `DATASET_PATH` = RoboMME root. The flags in the first 4 lines
are common to all three; only the **last line** differs per method. (Baseline with no memory:
`bash run_scripts/train_vanilla_n1d6.sh`, i.e. add `--hamlet-mode off`.)

### Method 1 — Read-out (HAMLET)  *(the ~18.4 baseline)*

```bash
torchrun --nproc_per_node=4 --master_port=29500 gr00t/experiment/launch_finetune.py \
  --base-model-path nvidia/GR00T-N1.6-3B --dataset-path "$DATASET_PATH" --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path gr00t/configs/data/robomme_config.py --num-gpus 4 --global-batch-size 32 \
  --hamlet-mode finetune --learning-rate 1e-4 --max-grad-norm 1.0 --tune-top-llm-layers 4 \
  --memory-window 4 --memory-stride 16 --memory-num-layers 2 --memory-type moment_token --no-freeze-moment-tokens \
  --max-steps 60000 --save-steps 10000 --save-total-limit 10 --output-dir runs/robomme/hamlet \
  --n-moment-tokens 4 --mem-cond-type cross_attn
```
Variations (read-out only): wide bank `--n-moment-tokens 128`; aggregator swap `--memory-arch gru|ssm|mamba`
(`+ --memory-state-dim 64` for ssm/mamba); whole-episode history `--mem-window-mode linspace --memory-window 16`;
integration `--mem-cond-type adaln` or `--mem-cond-type modul` instead of cross_attn (cross_attn is best).

**Optional two stages (HAMLET only).** The read-out tokens can be warm-started by TCL (time-contrastive)
pretraining before the finetune above — FrameSamp and Multi-resolution memory have no TCL stage.

> **Note — recommended: skip Stage 1.** Just train Method 1 directly from **random** moment-token init (the
> command above). The read-out summarizes the history nearly identically regardless of
> initialization (random vs TCL), because the attention pooling is dominated by the VLM's pretrained keys
> `k_t`, not by the query `q`. So Stage 1 is not needed in practice; treat the TCL chain below as optional.

```bash
# Stage 1 — TCL pretrain (= run_scripts/train_tcl_stage1.sh). MUST freeze the top LLM layers.
torchrun --nproc_per_node=4 gr00t/experiment/launch_finetune.py ... \
  --hamlet-mode tcl --tune-top-llm-layers 0 --learning-rate 2e-5 --max-grad-norm 0.5 \
  --max-steps 20000 --output-dir runs/robomme/tcl_stage1

# Stage 2 — feed the TCL moment tokens into the Method-1 command above by adding:
--load-moment-tokens-from runs/robomme/tcl_stage1/checkpoint-20000 [--freeze-moment-tokens]
```

### Method 2 — FrameSamp (raw tokens; **selection method 1/3** — uniform)

```bash
torchrun --nproc_per_node=4 --master_port=29500 gr00t/experiment/launch_finetune.py \
  --base-model-path nvidia/GR00T-N1.6-3B --dataset-path "$DATASET_PATH" --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path gr00t/configs/data/robomme_config.py --num-gpus 4 --global-batch-size 32 \
  --hamlet-mode finetune --learning-rate 1e-4 --max-grad-norm 1.0 --tune-top-llm-layers 4 \
  --memory-window 4 --memory-stride 16 --memory-num-layers 2 --memory-type moment_token --no-freeze-moment-tokens \
  --max-steps 60000 --save-steps 10000 --save-total-limit 10 --output-dir runs/robomme/framesamp \
  --mem-cond-type modul --mem-source framesamp --mem-framesamp-frames 8 --mem-framesamp-budget 512
```
FrameSamp is integrated via `modul` (FiLM); `--mem-film-layers all|mid|8-20` picks the injection depth.

### Method 3 — Multi-resolution memory (read-out fused into framesamp image tokens)

```bash
torchrun --nproc_per_node=4 --master_port=29500 gr00t/experiment/launch_finetune.py \
  --base-model-path nvidia/GR00T-N1.6-3B --dataset-path "$DATASET_PATH" --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path gr00t/configs/data/robomme_config.py --num-gpus 4 --global-batch-size 32 \
  --hamlet-mode finetune --learning-rate 1e-4 --max-grad-norm 1.0 --tune-top-llm-layers 4 \
  --memory-window 4 --memory-stride 16 --memory-num-layers 2 --memory-type moment_token --no-freeze-moment-tokens \
  --max-steps 60000 --save-steps 10000 --save-total-limit 10 --output-dir runs/robomme/multires \
  --n-moment-tokens 4 --mem-cond-type dual --mem-fs-inject moment --mem-framesamp-frames 8
```
Multi-resolution memory carries its own integration (h_sem KV-tail + h_spatial spatial-cross-attn);
`--mem-framesamp-frames > 0` is **required** or it degenerates to the read-out baseline.
**`--mem-fs-inject moment` is the fusion** — without it you get the unfused two-channel variant (9.1 %),
which is *worse than either single channel*.

### Method 4 — TokenDrop (**selection method 2/3** — diff keyframes)  *(= `run_scripts/train_tokendrop_n1d6.sh`)*

Same as FrameSamp **except** the selection rule — causal pixel-difference keyframes on **both** sides.
Trained via `cross_attn` here (1:1 with the exp-d framesamp baseline, so the A/B isolates selection only):

```bash
torchrun --nproc_per_node=4 --master_port=29500 gr00t/experiment/launch_finetune.py \
  --base-model-path nvidia/GR00T-N1.6-3B --dataset-path "$DATASET_PATH" --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path gr00t/configs/data/robomme_config.py --num-gpus 4 --global-batch-size 32 \
  --hamlet-mode finetune --learning-rate 1e-4 --max-grad-norm 1.0 --tune-top-llm-layers 4 \
  --n-moment-tokens 4 --memory-window 8 --memory-stride 16 --memory-num-layers 2 \
  --memory-type moment_token --no-freeze-moment-tokens \
  --max-steps 60000 --save-steps 10000 --save-total-limit 10 --output-dir runs/robomme/tokendrop_diff \
  --mem-cond-type cross_attn --mem-source framesamp --mem-framesamp-frames 8 --mem-framesamp-budget 512 \
  --mem-fs-select diff --mem-fs-diff-stride 8
```
Eval needs no extra flag (`mem_fs_select` is in the ckpt config); to override on an *old* ckpt:
`run_gr00t_server.py --mem-fs-select diff`.

### Method 5 — Action-conditioned patch memory (**selection method 3/3** — patch_union)

Selection drops from **frames** to **individual patch tokens**. Keep `--mem-framesamp-frames 8`: the
backbone processes K+F frames, and F=16 **OOMs** on 80 GB (see caveats).

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun --nproc_per_node=4 --master_port=29500 gr00t/experiment/launch_finetune.py \
  --base-model-path nvidia/GR00T-N1.6-3B --dataset-path "$DATASET_PATH" --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path gr00t/configs/data/robomme_config.py --num-gpus 4 \
  --global-batch-size 32 --gradient-accumulation-steps 2 \
  --hamlet-mode finetune --learning-rate 1e-4 --max-grad-norm 1.0 --tune-top-llm-layers 4 \
  --n-moment-tokens 4 --memory-window 8 --memory-stride 16 --memory-num-layers 2 \
  --memory-type moment_token --no-freeze-moment-tokens \
  --max-steps 60000 --save-steps 10000 --save-total-limit 10 --output-dir runs/robomme/patch_union \
  --mem-cond-type cross_attn --mem-source framesamp --mem-framesamp-frames 8 --mem-framesamp-budget 512 \
  --mem-fs-select patch_union --mem-fs-attn-layer 13 --mem-fs-diff-share 0.5 --mem-fs-diff-stride 8
```
`patch_union` currently supports **`--mem-cond-type cross_attn` only** (the sdpa capture counts one
cross-attn call per DiT block; `modul` adds a second and breaks the layer counter — asserted in code).

### Evaluation (RoboMME rollout)

```bash
MODEL_PATH=runs/robomme/<NAME>/checkpoint-60000 \
ROBOMME_PYTHON=/path/to/robomme_venv/bin/python \
bash run_scripts/eval_n1d6.sh
# serves the checkpoint (run_gr00t_server.py) then rolls out 16 RoboMME tasks x 50 episodes,
# and aggregates with gr00t/eval/sim/robomme/aggregate_eval_summary.py
```

**Eval parity (asserted):** `--n-action-steps` at eval **must equal the trained `--memory-stride`** (=16),
and the trained `memory_window` must match — `run_robomme_rollout.py` reads these from `--model-config` and
aborts on mismatch. (There is **no** `--control-mode` flag in this repo.)

---

## 5. Results so far (RoboMME success-rate, %)

### 5a. Multi-resolution memory, fuse read-out token into image token (select by framesamp) 

| memory | Counting (temporal) | Permanence (spatial) | Reference (object) | Imitation (procedural) | **Overall** |
|---|--:|--:|--:|--:|--:|
| GR00T — no memory | 13.0 | 5.0 | 10.5 | 4.5 | **8.25** |
| read-out n_q=4 (HAMLET) | 24.5 | 17.0 | 21.5 | 10.5 | **18.4** |
| read-out n_q=128 | 29.5 | 15.0 | 17.0 | 13.0 | **18.6** |
| raw image tokens (FrameSamp) | 8.5 | 11.0 | 14.5 | **15.5** | 12.4 |
| **Multi-resolution memory** | 7.0 | **17.5** | 14.5 | 5.5 | **11.1** |

*Read-out wins the aggregate suites; raw tokens win Imitation — the low-pass ceiling.*

**Multi-resolution memory** (`--mem-cond-type dual --mem-fs-inject moment`) targets that ceiling by
**fusing the two resolutions instead of running them side by side**: each framesamp frame's raw image
tokens are *colored* by that frame's own read-out state (block-causal, zero-init projection), so the
low-res summary and the high-res detail arrive **bound together** rather than as two channels the DiT must
reconcile. Verdict at 60k: **11.1 %** — it wins **Permanence outright (17.5**, above FrameSamp 11.0 and
read-out 17.0) and beats the unfused two-channel ablation (9.1, which is *worse than either channel alone* —
the classic gradient-competition failure), but it stays under FrameSamp 12.4 / read-out 18.4 overall.
Reading: the fusion fixes **frame aliasing** (episode-spanning frames were previously indistinguishable to
the policy), not the deeper problem that the two channels still compete for the same gradient budget. It
also trains slowly — 5.2 → 10.1 → 11.1 across 40k/50k/60k, i.e. still climbing at the end of the run.
(Reference point: the `mamba` selective-SSM aggregator reaches 11.5 % but only @40k of 60k — undertrained,
not directly comparable.)
(n_q=4 K=16-linspace `mamba` is **not** directly comparable to the K=4-recent 18.4.)

### 5b. Keyframe-selection A/B (all @60k, 16 tasks × 50 eps = 800 episodes each)

Measured on the *same* pipeline; the three memory arms share K=8 / budget 512 / `cross_attn`, so
framesamp↔tokendrop isolates **selection only**. (vanilla = K=4 no-memory control.)

| suite | vanilla | HAMLET (K=8) | FrameSamp (uniform) | TokenDrop (diff) |
|---|--:|--:|--:|--:|
| Counting | 13.0 | **25.5** | 8.5 | 19.5 |
| Permanence | 5.0 | 18.0 | 11.0 | **19.5** |
| Reference | 10.5 | **18.0** | 14.5 | 8.0 |
| Imitation | 4.5 | 7.5 | **15.5** | 9.0 |
| **OVERALL** | **8.2** | **17.2** | **12.4** | **14.0** |
| vs vanilla (z) | — | +5.40 | +2.71 | +3.66 |

Per-task winners: HAMLET 10 · framesamp 5 · tokendrop 2 · vanilla 1. **每个套件的赢家都不同**, and the
win pattern tracks the *instruction semantics*:

- `"repeating X times"` (counting) → **HAMLET** (PickXtimes 38, SwingXtimes 32) — counting is *drift*
  content needing an accumulated state; snapshots don't carry it (note-20 drift-invisibility).
- `"hiding the X cube"` (permanence) → **tokendrop** (VideoUnmask 46 = single best cell in the table) —
  the evidence is one instantaneous event, exactly what a pixel-diff peak captures.
- `"in the same manner / same path"` (continuous imitation) → **framesamp** (RouteStick 34, PatternLock 14) —
  trajectory *shape* needs dense uniform coverage; diff keeps only peaks and drops the manner in between.
- `"right after / the first target"` (ordering) → HAMLET/framesamp; tokendrop collapses (VideoPlaceOrder 24→8)
  because sparse event frames carry no ordinality.

**tokendrop suite split is significant in both directions**: Counting +11.0 (z=+3.17), Permanence +8.5
(z=+2.36), Reference −6.5 (z=−2.06), Imitation −6.5 (z=−1.98); OVERALL +1.6 is **not** significant (z=0.96).
Per-task numbers: [§5c](#5c-per-task-results-with-instructions).

⇒ **No single selection rule dominates**; uniform sampling is the correct prior for *continuous* content and
diff is the correct prior for *event* content — motivating a hybrid budget (diff peaks + uniform + compressed
drift) rather than an either/or.

---

### 5c. Per-task results (with instructions)

Same runs as §5b. The **example instruction** column is what makes the pattern legible: the winning memory
tracks the *semantics of the task*, not a single global ranking.

| suite | task | vanilla | HAMLET | FrameSamp | TokenDrop | example instruction |
|---|---|---:|---:|---:|---:|---|
| Counting | **BinFill** | 8.0 | **24.0** | 6.0 | 18.0 | put one red cube into the bin, then press the button to stop |
| Counting | **PickXtimes** | 20.0 | **38.0** | 4.0 | 24.0 | pick up the green cube and place it on the target, repeating this action **three times**, then press the button to stop |
| Counting | **StopCube** | **10.0** | 8.0 | **10.0** | 8.0 | press the button to stop the cube just as it reaches the target for the **fourth time** |
| Counting | **SwingXtimes** | 14.0 | **32.0** | 14.0 | 28.0 | move the cube right-side → left-side target, repeating **two times**, finally press the button to stop |
| Permanence | **ButtonUnmask** | 4.0 | **20.0** | 6.0 | 12.0 | first press the button, then pick up the container **hiding the red cube** |
| Permanence | **ButtonUnmaskSwap** | 0.0 | **16.0** | 0.0 | 0.0 | press both buttons, then pick up the container hiding the blue cube, finally another hiding the green cube |
| Permanence | **VideoUnmask** | 10.0 | 14.0 | 24.0 | **46.0** | watch the video carefully, then pick up the container hiding the green cube |
| Permanence | **VideoUnmaskSwap** | 6.0 | **22.0** | 14.0 | 20.0 | watch the video, pick up the container hiding the green cube, finally another hiding the blue cube |
| Reference | **PickHighlight** | 6.0 | **16.0** | 6.0 | 8.0 | first press the button, then pick up all cubes that **have been highlighted** with white areas |
| Reference | **VideoPlaceButton** | 16.0 | **26.0** | **26.0** | 16.0 | watch the video, then place the green cube on the target **right after the button was pressed** |
| Reference | **VideoPlaceOrder** | 20.0 | 20.0 | **24.0** | 8.0 | watch the video, then place the red cube on the **first** target it was previously placed on |
| Reference | **VideoRepick** | 0.0 | **10.0** | 2.0 | 0.0 | watch the video, repeatedly pick up/put down the **same block** three times, finally press the button |
| Imitation | **InsertPeg** | 0.0 | **4.0** | 0.0 | 2.0 | watch the video, grasp the **same end of the same peg** and insert into the **same side** of the box |
| Imitation | **MoveCube** | 12.0 | 18.0 | 14.0 | **24.0** | watch the video, then move the cube to the target **in the same manner as before** |
| Imitation | **PatternLock** | 0.0 | 4.0 | **14.0** | 6.0 | watch the video, then use the stick to **retrace the same pattern** |
| Imitation | **RouteStick** | 6.0 | 4.0 | **34.0** | 4.0 | watch the video, then navigate around the sticks **following the same path** |

Per-task winners: HAMLET 10 · FrameSamp 5 · TokenDrop 2 · vanilla 1 (ties counted for each).

---

## 6. Caveats / gotchas

- **Multi-resolution memory and `framesamp` require `--mem-framesamp-frames > 0`.** The loader only appends the
  episode-spanning frames when `mem_source=framesamp` **or** `mem_cond_type=dual`; omit it and the hybrid
  silently degenerates to the moment-only `cross_attn` baseline (its `h_spatial` channel has nothing to read).
- **`--mem-fs-inject` was silently dropped once** (the `setup.py` allowlist bug, §"Checkpoint config stamping"):
  a run labelled multi-resolution actually trained as the unfused two-channel variant. **Verify after launch**: the ckpt config must show
  `mem_fs_inject=moment` **and** the weights must contain `fs_inject` tensors (21 of them).
- **Flag scoping:** `mem_source`/`mem_film_layers` are `modul`-only; `mem_image_side` is `cross_attn`-only;
  `mem_window_mode` is `mem_source=moment`-only. Keep `framesamp` on `--mem-cond-type modul` (or use it as
  the hybrid's `h_spatial`).
- **`run_scripts/train_hamlet_n1d6.sh` only wires the basic HAMLET flags.** To train `gru/ssm/mamba`,
  `framesamp`, multi-resolution, or `linspace` you must **append** `--memory-arch/--mem-source/--mem-framesamp-*/`
  `--mem-window-mode/...` to the Stage-2 torchrun — otherwise the defaults (`transformer`/`moment`/`recent`) apply.
- **SequenceMemory NaN (fixed).** The `gru/ssm/mamba` residual stream was observed to spike to ~10³¹ and NaN
  in bf16. The fix is baked into `seq_memory.py`: **bias-free Linears** (matching `MemoryTransformer`),
  **overflow-safe RMSNorm** (factors the per-row max before squaring), **fp32 sequential scan**,
  **`clamp(±1e4)` before GELU**, **zero-init up-projection** (identity warm-start), init-parity `std=0.02`.
  Keep `--max-grad-norm 1.0`.
- **VRAM.** Larger `K` (e.g. 16), `n_q=128`, or big framesamp budgets lengthen the memory sequence and raise
  VRAM. For `memory_window=16` set the runtime env `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (avoids
  fragmentation OOM — *runtime env, not in any script*); otherwise lower `--global-batch-size`.
- **TCL (Stage-1) must use `--tune-top-llm-layers 0`** (`--learning-rate 2e-5`, `--max-grad-norm 0.5`), else
  the discarded top LLM layers absorb the contrastive task and moment tokens barely move from init.
- **n_q=128 is not a special code path** — just `--n-moment-tokens 128` (KV tail → 128, heavier). It differs
  from `--memory-type vision_feature` (a fixed 64 pooled vision tokens/step — a different token *source*).
- **⚠️ Selection must be TRAINED IN — swapping it at eval only makes things worse.** Measured on the exp-d
  framesamp ckpt with eval-only overrides: `fifo` 12.4 % → `diff` 8.8 % → `patch_union` 7.4 %. The ordering is
  the *distance from the trained `mem_seq` distribution*, not selection quality (that ckpt was trained on
  acausal linspace full frames). Trained in, the same `diff` rule flips to **14.0 %**. Probe metrics
  (coverage/redundancy) do **not** predict eval — only a trained-in run does.
- **`patch_union` OOM: the culprit is `--mem-framesamp-frames`, not the two-pass.** The backbone processes
  **K+F** frames per sample; F=16 → 24 frames = +50 % vs the proven exp-d config and OOMs at 12.8 GiB on
  80 GB cards (the failure is at the *backbone* forward, so shrinking the micro-batch does not help). Use
  **F=8** (backbone cost identical to exp-d), plus `--gradient-accumulation-steps 2` and
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` as headroom for pass A. Cost: ~2× wall-clock
  (~4.0 s/it vs 1.9 s/it) — mostly the halved micro-batch, ~20-25 % the extra no_grad DiT pass.
- **Eval/train consistency for `framesamp` variants**: the ckpt config carries `mem_fs_select`,
  `mem_framesamp_frames`, `mem_framesamp_budget` — **eval must use the same values** (the server reads the
  ckpt; only pass `--mem-fs-select` explicitly when deliberately testing a mismatch).
- **Checkpoint config stamping (fixed 2026-07-20).** `setup.py::_create_model` copies HAMLET overrides onto
  the base-ckpt config through an **explicit whitelist**; a new flag that is missing from it is silently
  dropped from the saved `config.json` (training is unaffected — the loader path reads
  `MODALITY_CONFIGS` — but eval then defaults wrongly). Add every new `mem_*` flag to that tuple.

---

## Code map

| Concern | File |
|---|---|
| All memory flags (CLI / dataclass) | `gr00t/configs/finetune_config.py` |
| Persisted model config | `gr00t/configs/model/gr00t_n1d6.py` |
| Memory wiring + `mem_cond_type` branches (`process_backbone_output`) | `gr00t/model/gr00t_n1d6/gr00t_n1d6.py` |
| Read-out aggregator (block-causal attention) | `gr00t/model/modules/memory.py` (`MemoryTransformer`) |
| Recurrent/SSM aggregator (gru/ssm/mamba) | `gr00t/model/modules/seq_memory.py` (`SequenceMemory`) |
| Multi-scale memory bank (note 10) | `gr00t/model/modules/multiscale_memory.py` |
| **Keyframe selection — diff (tokendrop)** | `gr00t/model/modules/fs_diff_select.py` (`DiffFrameSelector`, eval) + `gr00t/data/dataset/sharded_single_step_dataset.py` (`_fs_diff_scores`/`_fs_diff_indices`, train) |
| **Patch-level selection — patch_union** | `gr00t/model/modules/fs_patch_union.py` (`PatchUnionSelector` + sdpa attn capture) + `gr00t_n1d6.py` (`_patch_union_mem_seq`, `_patch_union_score_pass`, `_pu_commit`) |
| Per-session selector state round-trip (eval) | `gr00t/policy/gr00t_policy.py` (`_fs_session_state`) |
| Eval-time selection override | `gr00t/eval/run_gr00t_server.py` (`--mem-fs-select`, `--mem-fs-attn-layer`) |
| Train entry / loader window+framesamp logic | `gr00t/experiment/launch_finetune.py`, `gr00t/experiment/experiment.py` |
| Train scripts | `run_scripts/{train_vanilla_n1d6,train_tcl_stage1,train_hamlet_n1d6,train_tokendrop_n1d6}.sh` |
| Eval (serve + RoboMME rollout + aggregate) | `run_scripts/eval_n1d6.sh`, `gr00t/eval/sim/robomme/` |

*Selection theory (Bayes-filter view: drift vs. correction, why single-channel scores are blind to the
orthogonal channel, and why deterministic-given-action content cannot be selected at all) is developed in
the project's research notes, kept outside this repo.*
