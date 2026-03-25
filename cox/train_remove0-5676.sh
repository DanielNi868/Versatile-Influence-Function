#!/bin/bash
#SBATCH --gres=gpu:A40:1
#SBATCH --job-name=clip_captions_train
#SBATCH --output=logs/clip_captions_train_%j.out
#SBATCH --error=logs/clip_captions_train_%j.err
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00

set -euo pipefail

# Run from the directory where `sbatch` was submitted.
# In Slurm, the script itself may execute from a spool path.
cd "${SLURM_SUBMIT_DIR:-$PWD}"

echo "job is starting on $(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-N/A}"

# 如果你的集群没有做 GPU 隔离，可固定到空闲卡（当前建议 2 或 3）
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"

# PyTorch memory management
export PYTORCH_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=0
export PYTHONUNBUFFERED=1

mkdir -p logs

set -e
for i in $(seq 0 5677); do
  echo "[$(date)] training remove_index=$i"
  python train_cox.py --remove_index "$i" --dataset support \
    >> "logs/train_remove_0_1216_metabric.log" 2>&1
done