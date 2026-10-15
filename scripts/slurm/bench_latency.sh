#!/bin/bash
#SBATCH --account=wangsu-tennis-ai
#SBATCH --qos=bbgpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=18
#SBATCH --mem=64G
#SBATCH --time=0:30:00
#SBATCH --job-name=amdsd_latency
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
# Both devices in one job: the GPU timing needs the A100, and taking the CPU
# timing on the same allocation keeps the comparison on one machine rather than
# across a login node and a compute node.
REPO="${RETFOUND_REPO:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}"
if [ ! -f "$REPO/scripts/explore/bench_latency.py" ]; then
  echo "cannot locate the repo from $REPO — submit from the repo root, or set RETFOUND_REPO" >&2
  exit 1
fi
source /rds/projects/w/wangsu-tennis-ai/retfound/env.sh
export HF_HUB_OFFLINE=1
cd $PROJECT/retfound/code/RETFound
export PYTHONPATH=$PROJECT/retfound/code/RETFound:${PYTHONPATH:-}

# the CPU figure is only meaningful with its thread count pinned and reported
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-18}
export MKL_NUM_THREADS=$OMP_NUM_THREADS

POOL=${POOL:-mean}

echo "=============== GPU"
python $REPO/scripts/explore/bench_latency.py --device cuda --pool $POOL

echo
echo "=============== CPU ($OMP_NUM_THREADS threads)"
python $REPO/scripts/explore/bench_latency.py --device cpu --pool $POOL --reps 5 --warmup 2
