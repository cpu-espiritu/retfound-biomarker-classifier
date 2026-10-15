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
# probe_pooling resumes from its own output, so the old CSV must be MOVED out of the way,
# not copied, or it reloads the old rows and trains nothing. The first time, the old file
# becomes *_prefix.csv; if that already exists it is moved to a timestamped backup, so
# nothing is ever deleted.
aside () {
  [ -f "$1" ] || return 0
  if [ ! -f "${1%.csv}_prefix.csv" ]; then mv "$1" "${1%.csv}_prefix.csv"
  else mv "$1" "${1%.csv}.bak_$(date +%s).csv"; fi
  echo "moved $(basename "$1") aside"
}
# STEPS selects which to run, e.g. STEPS="comparison final" to redo only those two
STEPS=${STEPS:-"comparison control tuned strat final"}
has () { [[ " $STEPS " == *" $1 "* ]]; }

if has comparison; then
  echo "=============== probe_pooling (pooling_comparison)"
  aside $R/pooling_comparison.csv
  python $REPO/scripts/features/probe_pooling.py \
    --tokens $TOKENS --manifest $SPLITS/manifest.csv --out $R/pooling_comparison.csv
fi
if has control; then
  echo "=============== pooling_control (pooling_control_tunedC)"
  aside $R/pooling_control_tunedC.csv
  aside $R/pooling_control_tests.csv
  python $REPO/scripts/features/pooling_control.py \
    --tokens $TOKENS --manifest $SPLITS/manifest.csv --out $R/pooling_control_tunedC.csv
fi
if has tuned; then
  echo "=============== pooling_size_tuned"
  aside $R/pooling_size_tuned.csv
  python $REPO/scripts/features/pooling_size_tuned.py \
    --tokens $TOKENS --manifest $SPLITS/manifest.csv --selected $SEL \
    --out $R/pooling_size_tuned.csv
fi
if has strat; then
  echo "=============== pooling_size_strat"
  aside $R/pooling_size_strat.csv
  python $REPO/scripts/features/pooling_size_strat.py \
    --tokens $TOKENS --manifest $SPLITS/manifest.csv --out $R/pooling_size_strat.csv
fi
if has final; then
  echo "=============== attn_finalise (attn_final)"
  aside $R/attn_final.csv
  python $REPO/scripts/features/attn_finalise.py \
    --tokens $TOKENS --manifest $SPLITS/manifest.csv --selected $SEL \
    --out $R/attn_final.csv
fi
