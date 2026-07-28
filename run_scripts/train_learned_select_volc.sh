#!/usr/bin/env bash
# Learned patch-memory selection (note-24) — 60k training on a VolcEngine dev instance.
#
#   export DATASET_PATH=/path/to/robomme   BASE_MODEL=/path/to/GR00T-N1.6-3B
#   export CUDA_VISIBLE_DEVICES=0,1,2,5    # shared box: only the free cards
#   bash run_scripts/train_learned_select_volc.sh
#
# Same memory geometry as the patch_union arm (framesamp F=8, budget 512, cross_attn, pos-RoPE)
# so the A/B isolates ONE thing: whether selection is a fixed heuristic or trained end-to-end.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

DATASET_PATH="${DATASET_PATH:?set DATASET_PATH to the RoboMME dataset dir (meta/ + data/chunk-*)}"
BASE_MODEL="${BASE_MODEL:-nvidia/GR00T-N1.6-3B}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/robomme/learned_select}"
# note-24 §5.5: warm-start from a trained patch_union V2 checkpoint. Cold start also runs, but
# then loss begins at ~1.2 instead of ~0.01 and the selector has to learn on top of a from-scratch
# memory — the A/B question ("does TRAINING the selection rule help?") needs the trained baseline.
MAX_STEPS="${MAX_STEPS:-60000}"
SAVE_STEPS="${SAVE_STEPS:-2500}"
ANNEAL_STEPS="${ANNEAL_STEPS:-20000}"     # soft-topk temperature/gumbel anneal horizon
MASTER_PORT="${MASTER_PORT:-$(( 20000 + RANDOM % 10000 ))}"

# Only our GPUs on a shared box (counting physical cards would launch onto other people's work).
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  NUM_GPUS="${NUM_GPUS:-$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)}"
else
  NUM_GPUS="${NUM_GPUS:-$(nvidia-smi -L | wc -l)}"
fi
PER_DEVICE="${PER_DEVICE:-2}"             # A100-80G; 40G cards need 1

# effective batch = PER_DEVICE * NUM_GPUS * ACCUM (repo semantics: per_device =
# global_batch_size / num_gpus, and accum MULTIPLIES on top — the other arms ran 32)
TARGET_EFF="${TARGET_EFF:-32}"
GLOBAL_BS=$(( PER_DEVICE * NUM_GPUS ))
ACCUM=$(( TARGET_EFF / GLOBAL_BS )); [ "$ACCUM" -lt 1 ] && ACCUM=1
EFF=$(( GLOBAL_BS * ACCUM ))
[ "$EFF" -ne "$TARGET_EFF" ] && echo "WARNING: effective batch $EFF != $TARGET_EFF (record this deviation)"

[ -f "$DATASET_PATH/meta/modality.json" ] || {
  echo "ERROR: $DATASET_PATH/meta/modality.json missing — copy this repo's data/robomme/meta/ over it."
  exit 1; }
python -c "import torch, flash_attn, gr00t" 2>/dev/null || {
  echo "ERROR: env not ready (torch / flash_attn / gr00t not importable in this venv)."; exit 1; }
python -c "from gr00t.model.modules.fs_score_head import LearnedPatchSelector" 2>/dev/null || {
  echo "ERROR: fs_score_head missing — this repo copy predates note-24. Update the code."; exit 1; }
case "$BASE_MODEL" in
  nvidia/*) echo "NOTE: cold start (no warm-start ckpt). note-24 recommends warm-starting from a"
            echo "      trained patch_union V2 checkpoint; loss will begin at ~1.2, not ~0.01." ;;
esac

echo "== launching: ${NUM_GPUS} GPUs x per-device ${PER_DEVICE} x accum ${ACCUM} = effective ${EFF}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -m torch.distributed.run --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" \
  gr00t/experiment/launch_finetune.py \
  --base-model-path "$BASE_MODEL" --dataset-path "$DATASET_PATH" \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path gr00t/configs/data/robomme_config.py --num-gpus "$NUM_GPUS" \
  --global-batch-size "$GLOBAL_BS" --gradient-accumulation-steps "$ACCUM" \
  --hamlet-mode finetune --learning-rate 1e-4 --max-grad-norm 1.0 --tune-top-llm-layers 4 \
  --n-moment-tokens 4 --memory-window 8 --memory-stride 16 --memory-num-layers 2 \
  --memory-type moment_token --no-freeze-moment-tokens \
  --max-steps "$MAX_STEPS" --save-steps "$SAVE_STEPS" --save-total-limit 10 \
  --output-dir "$OUTPUT_DIR" \
  --mem-cond-type cross_attn --mem-source framesamp --mem-framesamp-frames 8 \
  --mem-framesamp-budget 512 \
  --mem-fs-select patch_union --mem-fs-learned-select --mem-fs-score-residual \
  --mem-fs-anneal-steps "$ANNEAL_STEPS" --mem-fs-pos-rope \
  --no-save-only-model
