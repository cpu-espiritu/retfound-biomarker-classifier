#!/bin/bash
#SBATCH --account=wangsu-tennis-ai
#SBATCH --qos=bbdefault
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=20:00:00
#SBATCH --job-name=amdsd_norm_sens
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
REPO="${RETFOUND_REPO:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}"
[ -f "$REPO/scripts/features/norm_sensitivity.py" ] || {
  echo "cannot locate the repo from $REPO — submit from the repo root" >&2; exit 1; }
source /rds/projects/w/wangsu-tennis-ai/retfound/env.sh
cd $PROJECT/retfound/code/RETFound          # models_vit, for the sanity check
export PYTHONPATH=$PROJECT/retfound/code/RETFound:${PYTHONPATH:-}

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export MKL_NUM_THREADS=$OMP_NUM_THREADS

SPLITS=${SPLITS:-$PROJECT/retfound/data/amdsd_splits}
FEAT=${FEAT:-$PROJECT/retfound/data/amdsd_features}
TAG=${TAG:-RETFound_mae_natureOCT_224}
# the normed cache is another ~2.4 GB beside the tokens; point WORK elsewhere if tight
WORK=${WORK:-$FEAT}

python $REPO/scripts/features/norm_sensitivity.py \
  --tokens   $FEAT/tokens_$TAG.npy \
  --manifest $SPLITS/manifest.csv \
  --selected $REPO/results/attn_tuned.csv \
  --images   $PROJECT/retfound/data/amdsd/images \
  --work     $WORK \
  --out      $REPO/results/norm_sensitivity.csv
