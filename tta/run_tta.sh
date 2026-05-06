#!/bin/bash
#SBATCH -A bcjw-delta-gpu
#SBATCH --partition=gpuA100x4,gpuA40x4
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --mem=64g
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --job-name=tta_electra_364
#SBATCH --output=/u/yliu110/364/tta/logs/tta_%j.out
#SBATCH --error=/u/yliu110/364/tta/logs/tta_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yl188@illinois.edu

module load miniforge3-python
eval "$(conda shell.bash hook)"
conda activate cs364

export HF_HOME=/scratch/bcjw/yliu110/huggingface
export TRANSFORMERS_CACHE=/scratch/bcjw/yliu110/huggingface
export CUDA_VISIBLE_DEVICES=0

# TEST_ONLY=1 只生成 test 预测，不跑 OOF（OOF 需要额外 ~21h，暂时跳过）
TEST_ONLY=${TEST_ONLY:-1}

echo "========================================"
echo "ELECTRA TTA Inference | 3 strategies"
echo "Node: $(hostname) | GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "TEST_ONLY=${TEST_ONLY}"
echo "Start: $(date)"
echo "========================================"

cd /u/yliu110/364/tta

EXTRA_ARGS=()
[[ "${TEST_ONLY}" == "1" ]] && EXTRA_ARGS+=(--test_only)

python3 tta_inference.py "${EXTRA_ARGS[@]}"

echo ""
echo "========================================"
echo "End: $(date)"
echo "========================================"
