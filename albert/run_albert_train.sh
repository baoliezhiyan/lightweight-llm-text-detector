#!/bin/bash
#SBATCH -A bcjw-delta-gpu
#SBATCH --partition=gpuA100x4,gpuA40x4
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --mem=64g
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --job-name=albert_train_364
#SBATCH --array=0-2
#SBATCH --output=/u/yliu110/364/albert/logs/train_%a_%j.out
#SBATCH --error=/u/yliu110/364/albert/logs/train_%a_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yl188@illinois.edu

module load miniforge3-python
eval "$(conda shell.bash hook)"
conda activate cs364

export HF_HOME=/scratch/bcjw/yliu110/huggingface
export TRANSFORMERS_CACHE=/scratch/bcjw/yliu110/huggingface
export CUDA_VISIBLE_DEVICES=0

SEEDS=(42 1337 2024)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

echo "========================================"
echo "ALBERT-base-v2 | Seed ${SEED} (task ${SLURM_ARRAY_TASK_ID})"
echo "Node: $(hostname) | GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start: $(date)"
echo "========================================"

mkdir -p /u/yliu110/364/albert/logs
cd /u/yliu110/364/albert

MODE=${MODE:-full}   # full 或 cv（cv 时请把 --time 改为 06:00:00）

python3 albert.py --mode "${MODE}" --seed "${SEED}"

echo ""
echo "========================================"
echo "End: $(date)"
echo "========================================"
