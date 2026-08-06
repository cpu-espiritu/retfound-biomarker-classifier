#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --qos=bbgpu
#SBATCH --account=wangsu-tennis-ai
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=18
#SBATCH --mem=64G
#SBATCH --array=0-4
#SBATCH --job-name=neh_lp
#SBATCH --output=/rds/projects/w/wangsu-tennis-ai/retfound/logs/%x_%A_%a.out
set -e
source /rds/projects/w/wangsu-tennis-ai/retfound/env.sh
export HF_HUB_OFFLINE=1

FOLD=$SLURM_ARRAY_TASK_ID
TASK=retfound_mae_NEH_lp_fold${FOLD}
DATA=$PROJECT/retfound/data/neh_splits/fold${FOLD}
OUT=$PROJECT/retfound/output/neh_lp

cd $PROJECT/retfound/code/RETFound
echo "Fold $FOLD | data=$DATA | task=$TASK"

torchrun --nproc_per_node=1 --master_port=$((20000 + RANDOM % 20000)) main_finetune.py \
  --model RETFound_mae \
  --model_arch retfound_mae \
  --finetune RETFound_mae_natureOCT \
  --adaptation lp \
  --savemodel --global_pool \
  --batch_size 64 \
  --epochs 30 \
  --warmup_epochs 5 \
  --lr 1e-3 \
  --weight_decay 0.05 \
  --nb_classes 3 \
  --data_path $DATA \
  --input_size 224 \
  --task $TASK \
  --output_dir $OUT \
  --log_dir $OUT/logs_tb \
  --num_workers 18 \
  --seed 42
