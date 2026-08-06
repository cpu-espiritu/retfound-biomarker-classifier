#!/bin/bash
#SBATCH --job-name=retfound_ucsd
#SBATCH --time=06:00:00
#SBATCH --qos=bbgpu
#SBATCH --account=wangsu-tennis-ai
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=18
#SBATCH --mem=64G
#SBATCH --output=/rds/projects/w/wangsu-tennis-ai/retfound/logs/%x_%j.out

set -e
source /rds/projects/w/wangsu-tennis-ai/retfound/env.sh
export HF_HUB_OFFLINE=1

cd $PROJECT/retfound/code/RETFound

torchrun --nproc_per_node=1 --master_port=$((20000 + RANDOM % 20000)) main_finetune.py \
  --model RETFound_mae \
  --model_arch retfound_mae \
  --finetune RETFound_mae_natureOCT \
  --savemodel --global_pool \
  --batch_size 64 \
  --epochs 100 \
  --nb_classes 3 \
  --data_path $OCT \
  --input_size 224 \
  --task retfound_mae_UCSD_lp \
  --output_dir $PROJECT/retfound/outputs \
  --adaptation lp
