#!/usr/bin/env bash
# SMOKE run: learned patch-memory selection (note-24, `mem_fs_learned_select`).
# Single GPU (this box: 4-GPU NaN / 2-GPU hang — PCIe, no NVLink), tiny steps, no ckpt saving.
# Verifies end-to-end: causal patch_union frames -> PatchScoreHead over all F*n candidates ->
# budgeted Soft-TopK alpha -> log(alpha) additive bias on the DiT memory keys -> finite loss
# AND a non-zero gradient on the score head (printed by the head's own hook below).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

DATASET_PATH="${DATASET_PATH:-/home/storage/xuehui/HAMLET-data/robomme}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/robomme/smoke_learned_select}"
BASE_MODEL="${BASE_MODEL:-nvidia/GR00T-N1.6-3B}"   # warm-start: point at model_puv2_60k
MAX_STEPS="${MAX_STEPS:-6}"
MASTER_PORT="${MASTER_PORT:-$(( 20000 + RANDOM % 10000 ))}"

"$REPO_ROOT/.venv/bin/torchrun" --nproc_per_node=1 --master_port="$MASTER_PORT" \
    gr00t/experiment/launch_finetune.py \
    --base-model-path "$BASE_MODEL" \
    --dataset-path "$DATASET_PATH" \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path gr00t/configs/data/robomme_config.py \
    --num-gpus 1 \
    --output-dir "$OUTPUT_DIR" \
    --max-steps "$MAX_STEPS" \
    --global-batch-size 2 \
    --save-steps 100000 \
    --save-total-limit 1 \
    --hamlet-mode finetune \
    --learning-rate 1e-4 \
    --max-grad-norm 1.0 \
    --tune-top-llm-layers 4 \
    --n-moment-tokens 4 \
    --memory-window 8 \
    --memory-stride 16 \
    --memory-num-layers 2 \
    --mem-cond-type cross_attn \
    --mem-source framesamp \
    --mem-fs-select patch_union \
    --mem-fs-learned-select \
    --mem-fs-anneal-steps 4 \
    --mem-fs-score-residual \
    --mem-fs-pos-rope \
    --mem-framesamp-frames 8 \
    --mem-framesamp-budget 512 \
    --memory-type moment_token \
    --no-freeze-moment-tokens
