#!/bin/bash
#SBATCH --account=wangsu-tennis-ai
#SBATCH --qos=bbgpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=18
#SBATCH --mem=64G
#SBATCH --time=0:30:00
#SBATCH --job-name=amdsd_ctrl
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
# repo root, derived from this script: <repo>/scripts/slurm/<this>
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source /rds/projects/w/wangsu-tennis-ai/retfound/env.sh
export HF_HUB_OFFLINE=1
cd $PROJECT/retfound/code/RETFound

python $REPO/scripts/features/extract_control.py \
  --manifest $PROJECT/retfound/data/amdsd_splits/manifest.csv \
  --images   $PROJECT/retfound/data/amdsd/images \
  --out      $PROJECT/retfound/data/amdsd_features \
  --encoder  ${1:-mae_in1k} --input-size ${2:-224}
