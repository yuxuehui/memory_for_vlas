#!/usr/bin/env bash
# learned-select (note-24) 60k on a 4x A100-80G GCP instance — the recipe that actually works.
#
#   bash run_scripts/train_learned_select_gcp.sh          # or install as the VM startup-script
#
# Everything below that looks like paranoia was paid for once. In order of how much they cost:
#
# 1. FS_GRAD_PROBE MUST BE 0 HERE. The probe reads gradients, and under ZeRO that needs
#    DeepSpeed's safe_get_full_grad — a COLLECTIVE. One call per parameter from inside
#    training_step injects thousands of unscheduled all-reduces into DeepSpeed's own comm
#    sequence and deadlocks every rank before step 1. Cost: 61 crash-loop restarts over two
#    hours at $14.7/h, zero training steps. The probe now refuses to run when world_size > 1
#    (5625672), but leave this at 0 as a second line of defence. Diagnose on ONE GPU.
# 2. The supervisor aborts after 3 zero-progress failures. A loop that only retries turns a
#    hard failure into an invisible money burner — that is how the 61 restarts went unnoticed.
# 3. Dataset on the local NVMe, not the boot disk. Not the cause of (1), but with 8 dataloader
#    workers the boot pd-balanced disk is the next bottleneck, and NVMe is already paid for:
#      sudo mkfs.ext4 -F /dev/nvme0n1 && sudo mkdir -p /mnt/data && sudo mount /dev/nvme0n1 /mnt/data
#      cp -r $W/robomme /mnt/data/robomme     # ~5.5 min for 121G
# 4. --no-save-only-model, or resume silently restarts from scratch (df8453b).
#
# When a multi-GPU job hangs: `py-spy dump --pid <rank pid>` FIRST. It named the offending line
# in seconds. Inferring from symptoms cost two hours and sent me down an I/O dead end.
set -uo pipefail

W="${W:-/opt/ls}"
DATA="${DATA:-/mnt/data/robomme}"; [ -d "$DATA" ] || DATA=$W/robomme
BASE_MODEL="${BASE_MODEL:-$W/model_puv2_50k}"      # patch_union V2 @50k — warm start, loss ~0.01
OUT="${OUT:-$W/runs/learned_select}"
BUCKET="${BUCKET:-gs://promptvla-skill-belief-data/exp_learnedselect/learned_select_60k}"
LOG_DIR=$W/logs; mkdir -p "$LOG_DIR"

exec >> "$LOG_DIR/train_$(date +%m%d_%H%M).log" 2>&1
echo "=== boot $(date -Is) host=$(hostname) data=$DATA"

cd "$W/repo"
. "$W/venv/bin/activate"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export FS_GRAD_PROBE=0            # see (1) above — never non-zero on a multi-GPU run
export FS_SCORE_LR_MULT="${FS_SCORE_LR_MULT:-10}"   # 100 (the DynamicViT number) diverges here: at
                    # base lr 1e-4 that is 1e-2 on a 1.25M MLP with a LayerNorm, and between step
                    # 1.2k and 10k norm.weight went 1 -> 88, alpha saturated, loss parked at 0.34
                    # and grad_norm hit 1e9. DynamicViT pairs 100x with a frozen, 0.01x backbone.

# 60k steps of work should not live only on this disk.
( while true; do sleep 900; gsutil -m -q rsync -r "$OUT" "$BUCKET" 2>/dev/null || true; done ) &

last_step=-1; dry=0
for i in $(seq 1 200); do
  step=$(grep -hoE '[0-9]+/60000' "$LOG_DIR"/train_*.log 2>/dev/null | tail -1 | cut -d/ -f1)
  step=${step:-0}
  if [ "$step" -gt "$last_step" ]; then dry=0; last_step=$step; else dry=$((dry + 1)); fi
  if [ "$dry" -ge 3 ]; then
    echo "=== ABORT: 3 consecutive failures with no progress past step $last_step."
    echo "=== Hard failure, not a transient crash. Read the traceback / py-spy the ranks."
    break
  fi
  echo "=== attempt $i  $(date +%F_%T)  (last step $last_step)"
  torchrun --nproc_per_node=4 --master_port="${MASTER_PORT:-29531}" \
    gr00t/experiment/launch_finetune.py \
    --base-model-path "$BASE_MODEL" --dataset-path "$DATA" \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path gr00t/configs/data/robomme_config.py --num-gpus 4 \
    --global-batch-size 8 --gradient-accumulation-steps 4 \
    --hamlet-mode finetune --learning-rate 1e-4 --max-grad-norm 1.0 --tune-top-llm-layers 4 \
    --n-moment-tokens 4 --memory-window 8 --memory-stride 16 --memory-num-layers 2 \
    --memory-type moment_token --no-freeze-moment-tokens \
    --max-steps 60000 --save-steps 2000 --save-total-limit 10 \
    --output-dir "$OUT" \
    --mem-cond-type cross_attn --mem-source framesamp --mem-framesamp-frames 8 \
    --mem-framesamp-budget 512 \
    --mem-fs-select patch_union --mem-fs-learned-select --mem-fs-score-residual \
    --mem-fs-anneal-steps 20000 --mem-fs-pos-rope \
    --no-save-only-model && break
  echo "=== attempt $i died rc=$? — retrying in 60s"
  sleep 60
done

gsutil -m -q rsync -r "$OUT" "$BUCKET" || true
echo "=== finished $(date -Is)"
