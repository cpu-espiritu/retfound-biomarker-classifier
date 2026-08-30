#!/bin/bash
#SBATCH --account=wangsu-tennis-ai
#SBATCH --qos=bbgpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=18
#SBATCH --mem=64G
#SBATCH --time=0:40:00
#SBATCH --job-name=amdsd_feat
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
# Repo root. Under sbatch the script is copied to /var/spool, so BASH_SOURCE
# is useless there; SLURM_SUBMIT_DIR is the directory sbatch was run from.
REPO="${RETFOUND_REPO:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}"
if [ ! -f "$REPO/scripts/finetune/train_amdsd.py" ]; then
  echo "cannot locate the repo from $REPO — submit from the repo root, or set RETFOUND_REPO" >&2
  exit 1
fi
source /rds/projects/w/wangsu-tennis-ai/retfound/env.sh
export HF_HUB_OFFLINE=1
cd $PROJECT/retfound/code/RETFound
export PYTHONPATH=$PROJECT/retfound/code/RETFound:${PYTHONPATH:-}

SPLITS=${1:-$PROJECT/retfound/data/amdsd_splits}
SIZE=${2:-224}
CROP=${3:-}

echo "[cfg] splits=$SPLITS size=$SIZE crop='$CROP'"

python $REPO/scripts/features/extract_features.py \
  --manifest   $SPLITS/manifest.csv \
  --images     $PROJECT/retfound/data/amdsd/images \
  --out        $PROJECT/retfound/data/amdsd_features \
  --input-size $SIZE $CROP
