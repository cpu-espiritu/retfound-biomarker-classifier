#!/bin/bash
#SBATCH --account=wangsu-tennis-ai
#SBATCH --qos=bbgpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=18
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --job-name=amdsd_ft
#SBATCH --array=0-4
#SBATCH --output=logs/%x_%A_%a.out

set -euo pipefail
source /rds/projects/w/wangsu-tennis-ai/retfound/env.sh
export HF_HUB_OFFLINE=1
cd $PROJECT/retfound/code/RETFound
export PYTHONPATH=$PROJECT/retfound/code/RETFound:${PYTHONPATH:-}

MODE=${1:-last4}
SIZE=${2:-224}
ENC=${3:-retfound}

python $PROJECT/retfound/code/scripts/finetune/train_amdsd.py \
  --manifest $PROJECT/retfound/data/amdsd_splits/manifest.csv \
  --images   $PROJECT/retfound/data/amdsd/images \
  --out      $PROJECT/retfound/data/amdsd_preds \
  --fold $SLURM_ARRAY_TASK_ID --mode $MODE --input-size $SIZE --encoder $ENC
