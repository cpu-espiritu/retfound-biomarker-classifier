#!/bin/bash
#SBATCH --account=wangsu-tennis-ai
#SBATCH --qos=bbdefault
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --job-name=amdsd_attn_remeasure
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
REPO="${RETFOUND_REPO:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}"
[ -f "$REPO/scripts/features/probe_pooling.py" ] || {
  echo "cannot locate the repo from $REPO — submit from the repo root" >&2; exit 1; }
source /rds/projects/w/wangsu-tennis-ai/retfound/env.sh

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export MKL_NUM_THREADS=$OMP_NUM_THREADS

SPLITS=${SPLITS:-$PROJECT/retfound/data/amdsd_splits}
FEAT=${FEAT:-$PROJECT/retfound/data/amdsd_features}
TAG=${TAG:-RETFound_mae_natureOCT_224}
TOKENS=$FEAT/tokens_$TAG.npy
SEL=$REPO/results/attn_tuned.csv
R=$REPO/results

[ -f "$TOKENS" ] || { echo "missing $TOKENS" >&2; exit 1; }
[ -f "$SEL" ] || { echo "missing $SEL — the arms stage needs the retuned selection" >&2; exit 1; }

# These four all fit AttnPool and were last measured before the bias-gradient fix.
# Each writes its own CSV; the previous file is kept beside it as *_prefix.csv so the
# old and new numbers can be diffed rather than silently replaced.
keep () { [ -f "$1" ] && [ ! -f "${1%.csv}_prefix.csv" ] && cp "$1" "${1%.csv}_prefix.csv" && echo "kept $(basename "${1%.csv}_prefix.csv")"; true; }
keep $R/pooling_comparison.csv
keep $R/pooling_size_tuned.csv
keep $R/pooling_size_strat.csv
keep $R/attn_final.csv

echo "=============== probe_pooling (pooling_comparison)"
python $REPO/scripts/features/probe_pooling.py \
  --tokens $TOKENS --manifest $SPLITS/manifest.csv --out $R/pooling_comparison.csv

echo "=============== pooling_size_tuned"
python $REPO/scripts/features/pooling_size_tuned.py \
  --tokens $TOKENS --manifest $SPLITS/manifest.csv --selected $SEL \
  --out $R/pooling_size_tuned.csv

echo "=============== pooling_size_strat"
python $REPO/scripts/features/pooling_size_strat.py \
  --tokens $TOKENS --manifest $SPLITS/manifest.csv --out $R/pooling_size_strat.csv

echo "=============== attn_finalise (attn_final)"
python $REPO/scripts/features/attn_finalise.py \
  --tokens $TOKENS --manifest $SPLITS/manifest.csv --selected $SEL \
  --out $R/attn_final.csv
