#!/usr/bin/env bash
# SMOKE run: VAR-pyramid multi-coarseness memory (mem_fs_select='var_pyramid').
# Single GPU (this box: 4-GPU NaN / 2-GPU hang — PCIe, no NVLink), tiny steps, no ckpt saving.
# Verifies end-to-end: loader causal diff-frames -> fs_pixels stash -> frozen VAR encode ->
# selector gates -> mem_seq tail replacement -> aux loss -> finite training loss.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

DATASET_PATH="${DATASET_PATH:-/home/storage/xuehui/HAMLET-data/robomme}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/robomme/smoke_var_pyramid}"
BASE_MODEL="${BASE_MODEL:-nvidia/GR00T-N1.6-3B}"
MAX_STEPS="${MAX_STEPS:-30}"
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
    --memory-window 4 \
    --memory-stride 16 \
    --memory-num-layers 2 \
    --mem-cond-type cross_attn \
    --mem-source framesamp \
    --mem-fs-select var_pyramid \
    --mem-varp-ckpt ckpts/vae_ch160v4096z32.pth \
    --mem-varp-res 128 \
    --mem-varp-budget 512 \
    --mem-varp-budget-lambda 0.05 \
    --mem-varp-gist-scales 4 \
    --mem-fs-pos-rope \
    --mem-framesamp-frames 8 \
    --mem-framesamp-budget 512 \
    --memory-type moment_token \
    --no-freeze-moment-tokens
