#!/usr/bin/env bash
# VAR-pyramid 60k training on a VolcEngine ML-platform dev instance (A100-80G).
#
# Usage on the dev machine (web terminal):
#   export DATASET_PATH=/path/to/robomme        # must contain meta/ + data/chunk-*
#   bash run_scripts/train_varp_volc.sh          # defaults: all visible GPUs, per-device 2
#
# Differences from the spot-VM recipe: A100-80G fits per-device 2 (peak ~38G), so this is
# ~2x fewer optimizer-visible micro-steps than the 40G configuration.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

DATASET_PATH="${DATASET_PATH:?set DATASET_PATH to the RoboMME dataset dir (meta/ + data/chunk-*)}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/robomme/var_pyramid}"
BASE_MODEL="${BASE_MODEL:-nvidia/GR00T-N1.6-3B}"
VAE_CKPT="${VAE_CKPT:-ckpts/vae_ch160v4096z32.pth}"
# Respect CUDA_VISIBLE_DEVICES — on a SHARED dev box only some GPUs are ours, and counting
# physical cards would launch ranks onto other people's work.
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  NUM_GPUS="${NUM_GPUS:-$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)}"
else
  NUM_GPUS="${NUM_GPUS:-$(nvidia-smi -L | wc -l)}"
fi
PER_DEVICE="${PER_DEVICE:-2}"                       # 80G card; 40G cards need 1
MAX_STEPS="${MAX_STEPS:-60000}"
SAVE_STEPS="${SAVE_STEPS:-2500}"
MASTER_PORT="${MASTER_PORT:-$(( 20000 + RANDOM % 10000 ))}"

# effective batch = PER_DEVICE * NUM_GPUS * ACCUM  (repo semantics: per_device =
# global_batch_size / num_gpus, and accum MULTIPLIES on top — matched to the other arms at 32)
GLOBAL_BS=$(( PER_DEVICE * NUM_GPUS ))
ACCUM=$(( 32 / GLOBAL_BS ))
[ "$ACCUM" -lt 1 ] && ACCUM=1

# ---- preflight -----------------------------------------------------------------------------
[ -f "$DATASET_PATH/meta/modality.json" ] || {
  echo "ERROR: $DATASET_PATH/meta/modality.json missing — RoboMME meta not in place."
  echo "       The authored meta lives in this repo at data/robomme/meta/ — copy it over the"
  echo "       dataset's meta/ if the dataset came without it."; exit 1; }
ls "$DATASET_PATH"/data/chunk-* >/dev/null 2>&1 || {
  echo "ERROR: no data/chunk-* under $DATASET_PATH — dataset parquet shards missing."; exit 1; }
if [ ! -f "$VAE_CKPT" ]; then
  echo "== fetching VAR vae ckpt (440MB)"
  mkdir -p "$(dirname "$VAE_CKPT")"
  HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}" \
    huggingface-cli download FoundationVision/var vae_ch160v4096z32.pth \
      --local-dir "$(dirname "$VAE_CKPT")"
fi
python -c "import torch, flash_attn, gr00t" 2>/dev/null || {
  echo "ERROR: env not ready. In a fresh venv:  pip install torch==2.7.1 torchvision==0.22.1 &&"
  echo "       MAX_JOBS=16 pip install -e .   (flash-attn builds from source, ~15 min)"; exit 1; }

echo "== launching: ${NUM_GPUS} GPUs x per-device ${PER_DEVICE} x accum ${ACCUM} = effective $(( GLOBAL_BS * ACCUM ))"
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
  --mem-fs-select var_pyramid --mem-varp-ckpt "$VAE_CKPT" \
  --mem-varp-res 128 --mem-varp-budget 512 --mem-varp-budget-lambda 0.05 \
  --mem-varp-gist-scales 4 --mem-fs-pos-rope
