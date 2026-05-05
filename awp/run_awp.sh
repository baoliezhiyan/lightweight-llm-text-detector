#!/bin/bash
#SBATCH -A bcjw-delta-gpu
#SBATCH --partition=gpuA100x4,gpuA40x4
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --mem=64g
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --job-name=awp_364
#SBATCH --output=/u/yliu110/364/awp/logs/awp_%j.out
#SBATCH --error=/u/yliu110/364/awp/logs/awp_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yl188@illinois.edu

module load miniforge3-python
eval "$(conda shell.bash hook)"
conda activate cs364

export HF_HOME=/scratch/bcjw/yliu110/huggingface
export TRANSFORMERS_CACHE=/scratch/bcjw/yliu110/huggingface
export CUDA_VISIBLE_DEVICES=0

# ── 运行阶段控制 ───────────────────────────────────────────────────────
# 默认只跑 Full（AWP 约 2x 时间，3 seeds ≈ 6h）
# CV 需要 ~24h，单独提交时设 RUN_CV=1 RUN_FULL=0，并把 --time 改为 26:00:00
RUN_CV=${RUN_CV:-0}
RUN_FULL=${RUN_FULL:-1}
W_ELECTRA=${W_ELECTRA:-0.37}  # 融合权重（先跑 CV → 本地 python ensemble.py --mode search 确认）
W_LGBM=${W_LGBM:-0.63}

echo "========================================"
echo "ELECTRA + AWP  |  364"
echo "Node: $(hostname) | GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "RUN_CV=${RUN_CV}  RUN_FULL=${RUN_FULL}"
echo "W_ELECTRA=${W_ELECTRA}  W_LGBM=${W_LGBM}"
echo "Start: $(date)"
echo "========================================"

mkdir -p /u/yliu110/364/awp/logs
cd /u/yliu110/364/awp

if [[ "${RUN_CV}" == "1" ]]; then
    echo ""
    echo ">>> [CV] 5-fold x 3 seeds with AWP..."
    python3 electra_awp.py --mode cv
fi

if [[ "${RUN_FULL}" == "1" ]]; then
    echo ""
    echo ">>> [FULL] Full-data training x 3 seeds with AWP..."
    python3 electra_awp.py --mode full

    echo ""
    echo ">>> [ENSEMBLE] Generating submission (w_electra=${W_ELECTRA}, w_lgbm=${W_LGBM})..."
    python3 ensemble.py --mode predict --w_electra "${W_ELECTRA}" --w_lgbm "${W_LGBM}"
fi

echo ""
echo "========================================"
echo "End: $(date)"
echo "========================================"
