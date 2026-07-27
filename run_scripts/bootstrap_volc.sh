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
  # /vepfs-* first: on this platform it is the shared parallel FS — hundreds of TB AND it
  # survives instance deletion, so the 121G dataset is downloaded once for every future run.
  for cand in /vepfs-*/*/* /vepfs-*/* /mnt/* /data /workspace /home/* "$HOME" /opt; do
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

# ---- 0b) system deps (the plain CUDA images are minimal: no venv module, no compiler) ------
SUDO=""; [ "$(id -u)" -ne 0 ] && SUDO="sudo"
PYV=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
need=""
python3 -c "import venv, ensurepip" 2>/dev/null || need="$need python${PYV}-venv"
command -v gcc  >/dev/null || need="$need build-essential"
command -v git  >/dev/null || need="$need git"
command -v curl >/dev/null || need="$need curl"
python3 -c "import sysconfig,os;raise SystemExit(0 if os.path.exists(sysconfig.get_paths()['include']+'/Python.h') else 1)" \
  2>/dev/null || need="$need python${PYV}-dev"
if [ -n "$need" ]; then
  echo "== apt-get installing:$need"
  $SUDO apt-get -qq update && $SUDO DEBIAN_FRONTEND=noninteractive apt-get -qq install -y $need
fi

# ---- 1) code ------------------------------------------------------------------------------
# git-over-https to github is heavily throttled from mainland China (observed ~5 KB/s), while the
# codeload tarball and mirrors are usually fine. Try tarball -> mirror tarball -> git clone.
fetch_code() {
  for url in \
      "https://codeload.github.com/yuxuehui/memory_for_vlas/tar.gz/refs/heads/main" \
      "${CODE_MIRROR:-https://ghfast.top/https://github.com/yuxuehui/memory_for_vlas/archive/refs/heads/main.tar.gz}" \
      "https://gh-proxy.com/https://github.com/yuxuehui/memory_for_vlas/archive/refs/heads/main.tar.gz"; do
    echo "== fetching code tarball: ${url%%\?*}"
    if curl -fsSL --connect-timeout 20 --max-time 600 --speed-limit 20000 --speed-time 30 \
         "$url" -o /tmp/varp_code.tgz; then
      rm -rf repo && mkdir -p repo && tar xzf /tmp/varp_code.tgz -C repo --strip-components=1 \
        && rm -f /tmp/varp_code.tgz && return 0
    fi
  done
  # Last resort: git. Two hardenings for the Beijing instances — HTTP/1.1 (the observed failure
  # is the classic "HTTP/2 stream 0 was not closed cleanly: CANCEL") and a sparse checkout that
  # skips docs/ (~20 MB of analysis PNGs, i.e. most of the repo, none of it needed to train).
  echo "== tarball routes failed; git clone over HTTP/1.1, sparse (no docs/)"
  local GITC=(-c http.version=HTTP/1.1 -c http.postBuffer=524288000 -c core.compression=0)
  rm -rf repo
  if git "${GITC[@]}" clone --depth 1 --filter=blob:none --sparse "$REPO_URL" repo; then
    git -C repo sparse-checkout set --no-cone '/*' '!/docs' || git -C repo sparse-checkout disable
  else
    rm -rf repo && git "${GITC[@]}" clone --depth 1 "$REPO_URL" repo
  fi
}
if [ -d repo/.git ]; then git -C repo pull --ff-only || true
elif [ -f repo/run_scripts/train_varp_volc.sh ]; then echo "== code already present (tarball)"
else fetch_code; fi

# ---- 2) env -------------------------------------------------------------------------------
if [ ! -x venv/bin/python ]; then python3 -m venv venv; fi
source venv/bin/activate
pip -q install -U pip wheel setuptools
# torch 2.7.1 wheels exist for cu118/cu126/cu128; pick the one matching the image's toolkit so
# flash-attn's source build (nvcc from CUDA_HOME) links against a compatible runtime.
if ! python -c "import torch" 2>/dev/null; then
  NVCC_VER=$(nvcc --version 2>/dev/null | grep -oE "release [0-9]+\.[0-9]+" | grep -oE "[0-9]+\.[0-9]+" || echo "")
  case "$NVCC_VER" in
    # PyPI's default linux wheel for torch 2.7.1 IS the cu126 build — installing it from the
    # (usually domestically mirrored) PyPI index is far faster than download.pytorch.org, which
    # is slow from mainland China. Only reach for the explicit index when we need a cu128 build.
    12.6|12.7) echo "== nvcc $NVCC_VER -> torch 2.7.1 from PyPI (default = cu126)"
               pip -q install torch==2.7.1 torchvision==0.22.1 ;;
    *)         echo "== nvcc ${NVCC_VER:-unknown} -> torch 2.7.1+cu128 from the pytorch index"
               pip -q install torch==2.7.1 torchvision==0.22.1 \
                 --index-url https://download.pytorch.org/whl/cu128 ;;
  esac
fi
python -c "import gr00t, flash_attn" 2>/dev/null || {
  # flash-attn's setup.py FIRST tries to fetch a prebuilt wheel from github releases, which hangs
  # forever where github is unreachable (the Beijing instances). Force the local build instead.
  export FLASH_ATTENTION_FORCE_BUILD=TRUE
  export MAX_JOBS="${MAX_JOBS:-32}"          # ~2GB RAM per nvcc job; these boxes have ~1TB
  echo "== installing repo deps (flash-attn builds from source, FORCE_BUILD, MAX_JOBS=$MAX_JOBS)"
  ( cd repo && MAX_JOBS="$MAX_JOBS" pip -q install -e . ) || \
  ( cd repo && MAX_JOBS="$MAX_JOBS" pip -q install flash-attn==2.7.4.post1 --no-build-isolation \
      && pip -q install -e . --no-build-isolation ) || {
    echo "ERROR: flash-attn failed to build. Most likely the image's CUDA toolkit is too new for"
    echo "       flash-attn 2.7.4 (CUDA 13 images are the usual cause). Recreate the instance on a"
    echo "       CUDA 12.6/12.8 image, or set CUDA_HOME to a 12.x toolkit and re-run this script."
    nvcc --version 2>/dev/null | tail -2; exit 1; }; }

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
