#!/usr/bin/env bash
# GR00T N1.6 + HAMLET -- Stage-1: time-contrastive pretraining of the MOMENT TOKENS (TCL).
#
# This is the Stage-1 config. It differs from Stage-2 (train_hamlet_n1d6.sh) ONLY in the
# knobs set below -- every Stage-1 vs Stage-2 difference lives in these two scripts, not in
# launch_finetune.py:
#   --hamlet-mode tcl            contrastive head instead of the action head
#   --tune-top-llm-layers 0      VLM fully frozen; the moment tokens are the trainable path
#                                (Stage-2 uses 4 -> layers 12-15). If the top LLM layers are
#                                trainable here, they absorb the easy contrastive task and the
#                                tokens barely move from init -- see Markdown/.../moment_token.md.
#   --learning-rate 2e-5         lower LR for InfoNCE stability (Stage-2: 1e-4)
#   --max-grad-norm 0.5          tighter grad clip for InfoNCE stability (Stage-2: 1.0)
#   --tcl-no-projection-head     OPTIONAL (TCL_NO_HEAD=1): drop the projection head so the
#                                moment tokens are the ONLY trainable parameters.
#
# Output: a checkpoint whose `backbone.moment_tokens` you feed to Stage-2 via
#   LOAD_MOMENT_TOKENS_FROM=<ckpt> bash run_scripts/train_hamlet_n1d6.sh
#
# Usage: DATASET_PATH=/path/to/robomme bash run_scripts/train_tcl_stage1.sh
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# config (override via env)
DATASET_PATH="${DATASET_PATH:?set DATASET_PATH to your RoboMME dataset directory}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/robomme/tcl_stage1}"
BASE_MODEL="${BASE_MODEL:-nvidia/GR00T-N1.6-3B}"
NUM_GPUS="${NUM_GPUS:-4}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
MAX_STEPS="${MAX_STEPS:-20000}"
SAVE_STEPS="${SAVE_STEPS:-10000}"
MASTER_PORT="${MASTER_PORT:-$(( 20000 + RANDOM % 10000 ))}"

# Stage-1 (TCL) knobs -- the whole point of a separate config file.
LEARNING_RATE="${LEARNING_RATE:-2e-5}"           # lower than Stage-2 (1e-4) for InfoNCE stability
MAX_GRAD_NORM="${MAX_GRAD_NORM:-0.5}"            # tighter clip than Stage-2 (1.0)
TUNE_TOP_LLM_LAYERS="${TUNE_TOP_LLM_LAYERS:-0}"  # 0 = VLM frozen (Stage-2 uses 4)
N_MOMENT_TOKENS="${N_MOMENT_TOKENS:-4}"          # moment tokens per step (n_q)
TCL_NO_HEAD="${TCL_NO_HEAD:-0}"                  # 1 = drop the projection head (tokens are the only trainable path)

HEAD_ARGS=()
[ "$TCL_NO_HEAD" = "1" ] && HEAD_ARGS+=(--tcl-no-projection-head)

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
    --hamlet-mode tcl \
    --n-moment-tokens "$N_MOMENT_TOKENS" \
    --learning-rate "$LEARNING_RATE" \
    --max-grad-norm "$MAX_GRAD_NORM" \
    --tune-top-llm-layers "$TUNE_TOP_LLM_LAYERS" \
    "${HEAD_ARGS[@]}"
