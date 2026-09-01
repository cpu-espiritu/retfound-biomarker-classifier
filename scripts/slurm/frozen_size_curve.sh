#!/bin/bash
#SBATCH --account=wangsu-tennis-ai
#SBATCH --qos=bbdefault
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --job-name=amdsd_frozen_curve
#SBATCH --array=0-3
#SBATCH --output=logs/%x_%A_%a.out

set -euo pipefail
REPO="${RETFOUND_REPO:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}"
if [ ! -f "$REPO/scripts/features/frozen_size_curve.py" ]; then
  echo "cannot locate the repo from $REPO — submit from the repo root, or set RETFOUND_REPO" >&2
  exit 1
fi
source /rds/projects/w/wangsu-tennis-ai/retfound/env.sh

# head-only fits on cached tokens: no GPU, but ~5 h of BLAS. The inner ops are
# (n*tokens, d) x (d, L) matmuls, so the CPU request is what buys the wall-clock.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export MKL_NUM_THREADS=$OMP_NUM_THREADS

SPLITS=${SPLITS:-$PROJECT/retfound/data/amdsd_splits}
FEAT=${FEAT:-$PROJECT/retfound/data/amdsd_features}
TAG=${TAG:-RETFound_mae_natureOCT_224}
BUDGET=${BUDGET:-steps}

# One array task per training-set size. Calibration: the existing frozenarms job
# (full pool, 3 seeds) took 3h18m, so the whole grid is ~13 h in one job — past any
# sensible wall. Split by size it is ~3-4 h a task, and they run in parallel.
SIZES=(${SIZES:-15 30 60 118})
N=${SIZES[${SLURM_ARRAY_TASK_ID:-0}]}
echo "task ${SLURM_ARRAY_TASK_ID:-0} -> n=$N budget=$BUDGET"

if [ ! -f "$SPLITS/size_subsets.csv" ]; then
  echo "missing $SPLITS/size_subsets.csv — run scripts/prep/make_size_subsets.py first" >&2
  exit 1
fi

python $REPO/scripts/features/frozen_size_curve.py \
  --tokens   $FEAT/tokens_$TAG.npy \
  --manifest $SPLITS/manifest.csv \
  --subsets  $SPLITS/size_subsets.csv \
  --selected $REPO/results/attn_tuned.csv \
  --preds-dir $PROJECT/retfound/data/amdsd_preds \
  --out      $REPO/results/size_curve_frozen_n$N.csv \
  --sizes    $N \
  --epoch-budget $BUDGET
