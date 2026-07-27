#!/usr/bin/env bash
# One-command bootstrap for VAR-pyramid training on a VolcEngine ML-platform dev instance.
#
#   bash <(curl -fsSL https://raw.githubusercontent.com/yuxuehui/memory_for_vlas/main/run_scripts/bootstrap_volc.sh)
#
# Does: pick a workspace with enough disk -> clone repo -> fetch RoboMME (HF mirror) -> overlay
# the repo's authored meta -> build env -> fetch VAR vae -> launch 60k training under nohup.
# Re-runnable: every step is skipped if already done. Override anything via env vars.
set -euo pipefail

WORK="${WORK:-}"
NEED_GB="${NEED_GB:-260}"                 # 121G dataset + ~7G model + ckpts (10 x ~7G)
REPO_URL="${REPO_URL:-https://github.com/yuxuehui/memory_for_vlas.git}"
HF_DATASET="${HF_DATASET:-Yinpei/robomme_data_lerobot}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"   # mainland-China mirror

# ---- 0) workspace with enough free space (dev-instance system disks are ~20GiB) ------------
if [ -z "$WORK" ]; then
  best=""; best_avail=0
  for cand in /mnt/* /data /workspace /home/* "$HOME" /opt; do
    [ -d "$cand" ] && [ -w "$cand" ] || continue
    avail=$(df -BG --output=avail "$cand" 2>/dev/null | tail -1 | tr -dc '0-9') || continue
    [ -n "$avail" ] && [ "$avail" -gt "$best_avail" ] && { best_avail=$avail; best=$cand; }
  done
  WORK="$best/varp"
  echo "== workspace: $WORK  (${best_avail}G free on $best)"
  [ "$best_avail" -lt "$NEED_GB" ] && {
    echo "ERROR: no writable mount with >= ${NEED_GB}G free (best: ${best_avail}G on $best)."
    echo "       Attach/point at a data volume and re-run with WORK=/your/volume/varp"; exit 1; }
fi
mkdir -p "$WORK"; cd "$WORK"

# ---- 1) code ------------------------------------------------------------------------------
if [ ! -d repo/.git ]; then git clone --depth 1 "$REPO_URL" repo; else git -C repo pull --ff-only; fi

# ---- 2) env -------------------------------------------------------------------------------
if [ ! -x venv/bin/python ]; then python3 -m venv venv; fi
source venv/bin/activate
pip -q install -U pip wheel setuptools
python -c "import torch" 2>/dev/null || pip -q install torch==2.7.1 torchvision==0.22.1
python -c "import gr00t, flash_attn" 2>/dev/null || {
  echo "== installing repo deps (flash-attn builds from source, ~15 min)"
  ( cd repo && MAX_JOBS="${MAX_JOBS:-16}" pip -q install -e . ) || \
  ( cd repo && MAX_JOBS="${MAX_JOBS:-16}" pip -q install flash-attn==2.7.4.post1 --no-build-isolation && pip -q install -e . ); }

# ---- 3) dataset (121G) + authored meta ----------------------------------------------------
DATA="${DATA:-$WORK/robomme}"
if ! ls "$DATA"/data/chunk-* >/dev/null 2>&1; then
  echo "== downloading RoboMME LeRobot data from $HF_ENDPOINT ($HF_DATASET, ~121G)"
  huggingface-cli download "$HF_DATASET" --repo-type dataset --local-dir "$DATA"
fi
# The repo carries the AUTHORED meta (modality.json etc.); it must win over the dataset's copy.
mkdir -p "$DATA/meta" && cp -f repo/data/robomme/meta/* "$DATA/meta/"

# ---- 4) VAR vae ---------------------------------------------------------------------------
[ -f repo/ckpts/vae_ch160v4096z32.pth ] || \
  huggingface-cli download FoundationVision/var vae_ch160v4096z32.pth --local-dir repo/ckpts

# ---- 5) launch ----------------------------------------------------------------------------
cd repo
mkdir -p "$WORK/logs"
LOG="$WORK/logs/train_$(date -u +%Y%m%d_%H%M%S).log"
echo "== launching training; log: $LOG"
DATASET_PATH="$DATA" OUTPUT_DIR="${OUTPUT_DIR:-$WORK/runs/var_pyramid}" \
  setsid nohup bash run_scripts/train_varp_volc.sh > "$LOG" 2>&1 < /dev/null &
sleep 5
echo "== started. follow with:  tail -f $LOG"
echo "== GPU check:"; nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader
