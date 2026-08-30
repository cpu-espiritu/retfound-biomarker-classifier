#!/bin/bash
#SBATCH --account=wangsu-tennis-ai
#SBATCH --qos=bbdefault
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --job-name=amdsd_frozen_curve
#SBATCH --output=logs/%x_%j.out

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
  --out      $REPO/results/size_curve_frozen.csv \
  --epoch-budget $BUDGET
