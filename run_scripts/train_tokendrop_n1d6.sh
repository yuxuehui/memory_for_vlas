#!/usr/bin/env bash
# GR00T N1.6 + TokenDrop-style framesamp memory (mem_fs_select='diff').
#
# Same recipe as train_hamlet_n1d6.sh EXCEPT the memory channel:
#   --mem-cond-type modul --mem-source framesamp   raw per-frame vision tokens -> MemoryFiLM
#   --mem-fs-select diff                           TokenDrop keyframes at BOTH train & eval:
#     train (loader): CAUSAL frame 0 + top-(F-2) pixel-diff steps <= anchor + anchor
#     eval  (model):  DiffFrameSelector incremental top-K (stamped into the ckpt config)
#   Replaces the old regime: acausal whole-episode linspace at train / recent-FIFO at eval
#   (the note-8 train/eval window mismatch; see Markdown/20_tokendrop.md).
#
# Usage: DATASET_PATH=/path/to/robomme bash run_scripts/train_tokendrop_n1d6.sh
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

DATASET_PATH="${DATASET_PATH:?set DATASET_PATH to your RoboMME dataset directory}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/robomme/tokendrop_n1d6}"
BASE_MODEL="${BASE_MODEL:-nvidia/GR00T-N1.6-3B}"
NUM_GPUS="${NUM_GPUS:-4}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
MAX_STEPS="${MAX_STEPS:-60000}"
SAVE_STEPS="${SAVE_STEPS:-10000}"             # 10k cadence + limit 10 (checkpoint-loss lesson)
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-10}"
MASTER_PORT="${MASTER_PORT:-$(( 20000 + RANDOM % 10000 ))}"

LEARNING_RATE="${LEARNING_RATE:-1e-4}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
TUNE_TOP_LLM_LAYERS="${TUNE_TOP_LLM_LAYERS:-4}"

K="${K:-4}"
MEMORY_STRIDE="${MEMORY_STRIDE:-16}"
N_MOMENT_TOKENS="${N_MOMENT_TOKENS:-4}"
MEM_FRAMESAMP_FRAMES="${MEM_FRAMESAMP_FRAMES:-8}"
MEM_FRAMESAMP_BUDGET="${MEM_FRAMESAMP_BUDGET:-512}"
MEM_FS_DIFF_STRIDE="${MEM_FS_DIFF_STRIDE:-8}"
# 'modul' = faithful MME-VLA TokenDrop configuration (their `modulation`: action tokens
# CROSS-ATTEND the 512 memory tokens inside the per-block FiLM — see note 2; the memory
# read IS attention, only the injection is scale/shift). 'cross_attn' = memory-as-context
# (framesamp tokens ride the DiT KV). Both work with mem_fs_select='diff'.
MEM_COND_TYPE="${MEM_COND_TYPE:-modul}"

torchrun --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" \
    gr00t/experiment/launch_finetune.py \
    --base-model-path "$BASE_MODEL" \
    --dataset-path "$DATASET_PATH" \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path gr00t/configs/data/robomme_config.py \
    --num-gpus "$NUM_GPUS" \
    --output-dir "$OUTPUT_DIR" \
    --max-steps "$MAX_STEPS" \
    --global-batch-size "$GLOBAL_BATCH_SIZE" \
    --save-steps "$SAVE_STEPS" \
    --save-total-limit "$SAVE_TOTAL_LIMIT" \
    --hamlet-mode finetune \
    --learning-rate "$LEARNING_RATE" \
    --max-grad-norm "$MAX_GRAD_NORM" \
    --tune-top-llm-layers "$TUNE_TOP_LLM_LAYERS" \
    --n-moment-tokens "$N_MOMENT_TOKENS" \
    --memory-window "$K" \
    --memory-stride "$MEMORY_STRIDE" \
    --memory-num-layers 2 \
    --mem-cond-type "$MEM_COND_TYPE" \
    --mem-source framesamp \
    --mem-fs-select diff \
    --mem-fs-diff-stride "$MEM_FS_DIFF_STRIDE" \
    --mem-framesamp-frames "$MEM_FRAMESAMP_FRAMES" \
    --mem-framesamp-budget "$MEM_FRAMESAMP_BUDGET" \
    --memory-type moment_token \
    --no-freeze-moment-tokens
