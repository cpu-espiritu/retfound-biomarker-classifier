#!/bin/bash
#SBATCH --job-name=neh_lp
#SBATCH --array=0-4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=02:00:00
#SBATCH --output=/rds/projects/w/wangsu-tennis-ai/retfound/output/neh_lp/logs/%x_%A_%a.out
# <<< COPY these from your Kermany .slurm (site-specific — leave yours as-is):
#SBATCH --account=REPLACE_ME
#SBATCH --qos=REPLACE_ME
#SBATCH --gres=gpu:1
#SBATCH --mem=64G

set -euo pipefail

# --- Environment (same bootstrap as Kermany) ---
source /rds/projects/w/wangsu-tennis-ai/retfound/env.sh
export HF_HUB_OFFLINE=1          # weights already cached; no network on compute nodes

# --- Per-fold parameters, driven by the array index ---
FOLD=$SLURM_ARRAY_TASK_ID
PORT=$((29500 + FOLD))            # unique per fold in case tasks co-locate
TASK=retfound_mae_NEH_lp_fold${FOLD}
DATA=$PROJECT/retfound/data/neh_splits/fold${FOLD}
OUT=$PROJECT/retfound/output/neh_lp

# --- Repo containing main_finetune.py (edit if yours lives elsewhere) ---
REPO=$PROJECT/retfound/code/RETFound_MAE
cd "$REPO"

echo "Fold $FOLD | data=$DATA | task=$TASK | port=$PORT"

torchrun --nproc_per_node=1 --master_port=$PORT main_finetune.py \
  --model RETFound_mae \
  --model_arch retfound_mae \
  --finetune RETFound_mae_natureOCT \
  --adaptation lp \
  --global_pool \
  --nb_classes 3 \
  --input_size 224 \
  --batch_size 64 \
  --epochs 30 \
  --warmup_epochs 5 \
  --lr 1e-3 \
  --weight_decay 0.05 \
  --data_path "$DATA" \
  --task "$TASK" \
  --output_dir "$OUT" \
  --log_dir "$OUT/logs_tb" \
  --num_workers 10 \
  --seed 42
