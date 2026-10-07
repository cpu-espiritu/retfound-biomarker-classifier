#!/bin/bash
#SBATCH --account=wangsu-tennis-ai
#SBATCH --qos=bbdefault
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --job-name=amdsd_attn_rerun
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
REPO="${RETFOUND_REPO:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}"
if [ ! -f "$REPO/scripts/features/tune_attention.py" ]; then
  echo "cannot locate the repo from $REPO — submit from the repo root, or set RETFOUND_REPO" >&2
  exit 1
fi
source /rds/projects/w/wangsu-tennis-ai/retfound/env.sh

# numpy inner ops are BLAS matmuls over (n*tokens, d) x (d, L); the CPU request is
# what buys the wall-clock, so make sure the libraries actually see the cores
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export MKL_NUM_THREADS=$OMP_NUM_THREADS

SPLITS=${SPLITS:-$PROJECT/retfound/data/amdsd_splits}
FEAT=${FEAT:-$PROJECT/retfound/data/amdsd_features}
TAG=${TAG:-RETFound_mae_natureOCT_224}
STAGE=${1:-tune}
TOKENS=$FEAT/tokens_$TAG.npy

[ -f "$TOKENS" ] || { echo "missing $TOKENS" >&2; exit 1; }
echo "stage=$STAGE  tokens=$TOKENS"

case "$STAGE" in
  tune)
    # the previous selection was made while the AttnPool bias gradient was wrong;
    # keep it so the two can be diffed before anything downstream is rerun
    if [ -f "$REPO/results/attn_tuned.csv" ] && [ ! -f "$REPO/results/attn_tuned_prefix.csv" ]; then
      cp "$REPO/results/attn_tuned.csv" "$REPO/results/attn_tuned_prefix.csv"
      echo "kept the pre-fix selection at results/attn_tuned_prefix.csv"
    fi
    python $REPO/scripts/features/tune_attention.py \
      --tokens   $TOKENS \
      --manifest $SPLITS/manifest.csv \
      --out      $REPO/results/attn_tuned.csv
    echo '--- modal configuration, before and after the fix ---'
    python - "$REPO" <<'PY'
import sys, pandas as pd
from pathlib import Path
R = Path(sys.argv[1]) / 'results'
for name, f in (('pre-fix', R/'attn_tuned_prefix.csv'), ('post-fix', R/'attn_tuned.csv')):
    if not f.exists():
        continue
    d = pd.read_csv(f)
    for c in ('IRF', 'SRF', 'PED'):
        m = {k: d[d.cls == c][k].mode().iloc[0] for k in ('L', 'lr', 'wd', 'epochs')}
        print(f'{name:<9}{c:<5}' + '  '.join(f'{k}={v:g}' for k, v in m.items()))
PY
    ;;
  arms)
    python $REPO/scripts/features/frozen_arms.py \
      --tokens   $TOKENS \
      --manifest $SPLITS/manifest.csv \
      --selected $REPO/results/attn_tuned.csv \
      --out      $REPO/results/frozen_arms.csv
    ;;
  *)
    echo "unknown stage '$STAGE' (want: tune, arms)" >&2; exit 1 ;;
esac
