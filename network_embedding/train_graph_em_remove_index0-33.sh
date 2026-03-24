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

python train_graph_embedding.py --remove_index 0
python train_graph_embedding.py --remove_index 1
python train_graph_embedding.py --remove_index 2
python train_graph_embedding.py --remove_index 3
python train_graph_embedding.py --remove_index 4
python train_graph_embedding.py --remove_index 5
python train_graph_embedding.py --remove_index 6
python train_graph_embedding.py --remove_index 7
python train_graph_embedding.py --remove_index 8
python train_graph_embedding.py --remove_index 9
python train_graph_embedding.py --remove_index 10
python train_graph_embedding.py --remove_index 11
python train_graph_embedding.py --remove_index 12
python train_graph_embedding.py --remove_index 13
python train_graph_embedding.py --remove_index 14
python train_graph_embedding.py --remove_index 15
python train_graph_embedding.py --remove_index 16
python train_graph_embedding.py --remove_index 17
python train_graph_embedding.py --remove_index 18
python train_graph_embedding.py --remove_index 19
python train_graph_embedding.py --remove_index 20
python train_graph_embedding.py --remove_index 21
python train_graph_embedding.py --remove_index 22
python train_graph_embedding.py --remove_index 23
python train_graph_embedding.py --remove_index 24
python train_graph_embedding.py --remove_index 25
python train_graph_embedding.py --remove_index 26
python train_graph_embedding.py --remove_index 27
python train_graph_embedding.py --remove_index 28
python train_graph_embedding.py --remove_index 29
python train_graph_embedding.py --remove_index 30
python train_graph_embedding.py --remove_index 31
python train_graph_embedding.py --remove_index 32
python train_graph_embedding.py --remove_index 33