#!/bin/bash
#SBATCH -A bcjw-delta-gpu
#SBATCH --partition=gpuA100x4,gpuA40x4
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --mem=64g
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --job-name=albert_364
#SBATCH --output=/u/yliu110/364/albert/logs/albert_%j.out
#SBATCH --error=/u/yliu110/364/albert/logs/albert_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yl188@illinois.edu

module load miniforge3-python
eval "$(conda shell.bash hook)"
conda activate cs364

export HF_HOME=/scratch/bcjw/yliu110/huggingface
export TRANSFORMERS_CACHE=/scratch/bcjw/yliu110/huggingface
export CUDA_VISIBLE_DEVICES=0

# ── 运行阶段控制 ───────────────────────────────────────────────────────
# 默认只跑 Full（~3h）；CV 需要 ~12h，单独提交时设 RUN_CV=1 RUN_FULL=0
# 并把 --time 改为 14:00:00
RUN_CV=${RUN_CV:-0}
RUN_FULL=${RUN_FULL:-1}
W_ALBERT=${W_ALBERT:-0.37}  # 先用和 ELECTRA 实验相同的起始权重
W_LGBM=${W_LGBM:-0.63}

echo "========================================"
echo "ALBERT-base-v2  |  364"
echo "Node: $(hostname) | GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "RUN_CV=${RUN_CV}  RUN_FULL=${RUN_FULL}"
echo "W_ALBERT=${W_ALBERT}  W_LGBM=${W_LGBM}"
echo "Start: $(date)"
echo "========================================"

mkdir -p /u/yliu110/364/albert/logs
cd /u/yliu110/364/albert

if [[ "${RUN_CV}" == "1" ]]; then
    echo ""
    echo ">>> [CV] 5-fold x 3 seeds  (ALBERT-base-v2)..."
    python3 albert.py --mode cv
fi

if [[ "${RUN_FULL}" == "1" ]]; then
    echo ""
    echo ">>> [FULL] Full-data training x 3 seeds  (ALBERT-base-v2)..."
    python3 albert.py --mode full

    echo ""
    echo ">>> [ENSEMBLE] Generating submission (w_albert=${W_ALBERT}, w_lgbm=${W_LGBM})..."
    python3 ensemble.py --mode predict --w_albert "${W_ALBERT}" --w_lgbm "${W_LGBM}"
fi

echo ""
echo "========================================"
echo "End: $(date)"
echo "========================================"
