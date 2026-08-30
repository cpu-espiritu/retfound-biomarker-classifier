#!/bin/bash
#SBATCH --account=wangsu-tennis-ai
#SBATCH --qos=bbgpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=18
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --job-name=amdsd_sizecurve
#SBATCH --array=0-44
#SBATCH --output=logs/%x_%A_%a.out

set -euo pipefail
REPO="${RETFOUND_REPO:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}"
if [ ! -f "$REPO/scripts/finetune/train_amdsd.py" ]; then
  echo "cannot locate the repo from $REPO — submit from the repo root, or set RETFOUND_REPO" >&2
  exit 1
fi
source /rds/projects/w/wangsu-tennis-ai/retfound/env.sh
export HF_HUB_OFFLINE=1
cd $PROJECT/retfound/code/RETFound
export PYTHONPATH=$PROJECT/retfound/code/RETFound:${PYTHONPATH:-}

# n=118 is the full pool and, under the default step budget, is exactly the existing
# last4_224_f*_s{0,1,2} runs — it is deliberately not in this grid.
# data/amdsd_splits/ is gitignored, so the subset file does not arrive with the repo;
# it is generated next to the manifest by scripts/prep/make_size_subsets.py
SPLITS=${SPLITS:-$PROJECT/retfound/data/amdsd_splits}
if [ ! -f "$SPLITS/size_subsets.csv" ]; then
  echo "missing $SPLITS/size_subsets.csv — run scripts/prep/make_size_subsets.py first" >&2
  exit 1
fi

SIZES=(${SIZES:-15 30 60})
SEEDS=(${SEEDS:-0 1 2})
MODE=${MODE:-last4}
POOL=${POOL:-mean}

NFOLD=5
NSEED=${#SEEDS[@]}
i=$SLURM_ARRAY_TASK_ID
FOLD=$(( i % NFOLD ))
SEED=${SEEDS[$(( (i / NFOLD) % NSEED ))]}
N=${SIZES[$(( i / (NFOLD * NSEED) ))]}

echo "task $i -> n=$N seed=$SEED fold=$FOLD mode=$MODE pool=$POOL"

python $REPO/scripts/finetune/train_amdsd.py \
  --manifest $SPLITS/manifest.csv \
  --images   $PROJECT/retfound/data/amdsd/images \
  --out      $PROJECT/retfound/data/amdsd_preds \
  --subsets  $SPLITS/size_subsets.csv \
  --n-train-patients $N --seed $SEED --fold $FOLD \
  --mode $MODE --pool $POOL --input-size 224
