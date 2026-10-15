#!/bin/bash
#SBATCH --account=wangsu-tennis-ai
#SBATCH --qos=bbgpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=18
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --job-name=amdsd_bench
#SBATCH --array=0-2
#SBATCH --output=logs/%x_%A_%a.out

set -euo pipefail
# Task 0 = last-4 blocks, 1 = full fine-tuning, 2 = frozen trunk + attention pooling.
# One array, so all three land on the same GPU model in the same queue state — a
# wall-clock comparison across separate submissions is not one, it is a comparison
# of cluster load.
REPO="${RETFOUND_REPO:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}"
if [ ! -f "$REPO/scripts/finetune/train_amdsd.py" ]; then
  echo "cannot locate the repo from $REPO — submit from the repo root, or set RETFOUND_REPO" >&2
  exit 1
fi
source /rds/projects/w/wangsu-tennis-ai/retfound/env.sh
export HF_HUB_OFFLINE=1
cd $PROJECT/retfound/code/RETFound
export PYTHONPATH=$PROJECT/retfound/code/RETFound:${PYTHONPATH:-}

MODES=(last4 full lp)
POOLS=(mean mean attn)
MODE=${MODES[$SLURM_ARRAY_TASK_ID]}
POOL=${POOLS[$SLURM_ARRAY_TASK_ID]}
FOLD=${FOLD:-0}
SEED=${SEED:-0}
SIZE=${SIZE:-224}

# Writes to data/bench, never data/amdsd_preds: fold 0 seed 0 of both arms already
# exists there and backs the numbers in RESULTS.md. A benchmark must not overwrite
# a result.
OUT=${OUT:-$PROJECT/retfound/data/bench}
mkdir -p "$OUT"

echo "task $SLURM_ARRAY_TASK_ID -> mode=$MODE pool=$POOL fold=$FOLD seed=$SEED size=$SIZE"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python $REPO/scripts/finetune/train_amdsd.py \
  --manifest $PROJECT/retfound/data/amdsd_splits/manifest.csv \
  --images   $PROJECT/retfound/data/amdsd/images \
  --out      "$OUT" \
  --fold $FOLD --seed $SEED --mode $MODE --pool $POOL --input-size $SIZE
