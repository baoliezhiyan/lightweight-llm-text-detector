#!/bin/bash
#SBATCH -A bcjw-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16g
#SBATCH --time=00:10:00
#SBATCH --job-name=ensemble_full_364
#SBATCH --output=/u/yliu110/364/logs/ensemble_full_%j.out
#SBATCH --error=/u/yliu110/364/logs/ensemble_full_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yl188@illinois.edu

module load miniforge3-python
eval "$(conda shell.bash hook)"
conda activate cs364

# ── 权重参数（提交时可用环境变量覆盖）──────────────────────────────
# 示例: W_ELECTRA=0.8 W_LGBM=0.2 sbatch run_ensemble_full.sh
W_ELECTRA=${W_ELECTRA:-0.9}
W_LGBM=${W_LGBM:-0.1}
OUTPUT=${OUTPUT:-}   # 留空则自动命名 prediction_full_we{w_e}_wl{w_l}.csv

# ── 可选：提交时加 RETRAIN=1 重新生成模型预测（需 GPU 分区）─────────
# 此选项仅在 lgbm_full.npz / electra_full.npz 不存在或需更新时使用
# 注意：需将 --partition 改为 gpuA100x4 并添加 --gpus=1
RETRAIN=${RETRAIN:-0}

echo "========================================"
echo "364 Full Ensemble Weight Tuning"
echo "Node: $(hostname)"
echo "W_ELECTRA=${W_ELECTRA}  W_LGBM=${W_LGBM}"
echo "Start: $(date)"
echo "========================================"

cd /u/yliu110/364

# 数据软链（脚本用相对路径 train.csv / test.csv，数据在 data/）
[ -f train.csv ] || ln -s data/train.csv train.csv
[ -f test.csv  ] || ln -s data/test.csv  test.csv

if [[ "${RETRAIN}" == "1" ]]; then
    echo ""
    echo ">>> [1/3] LightGBM full-data training..."
    python3 LGBM_full.py

    echo ""
    echo ">>> [2/3] ELECTRA full-data training..."
    python3 electra_full.py
fi

echo ""
echo ">>> [${RETRAIN:+3/3}${RETRAIN:-1/1}] Running ensemble (w_electra=${W_ELECTRA}, w_lgbm=${W_LGBM})..."

EXTRA_ARGS=()
if [[ -n "${OUTPUT}" ]]; then
    EXTRA_ARGS+=(--output "${OUTPUT}")
fi

python3 ensemble_2full.py \
    --w_electra "${W_ELECTRA}" \
    --w_lgbm    "${W_LGBM}" \
    "${EXTRA_ARGS[@]}"

echo ""
echo "========================================"
echo "End: $(date)"
echo "========================================"
