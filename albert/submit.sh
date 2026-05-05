#!/bin/bash
# 一键提交：3 个并行训练 job + 训练完自动触发 ensemble job
#
# 用法：
#   bash submit.sh                        # 默认 full 模式
#   MODE=cv bash submit.sh                # CV 模式（自动把训练 time 延长到 6h）
#   W_ALBERT=0.4 W_LGBM=0.6 bash submit.sh

MODE=${MODE:-full}
W_ALBERT=${W_ALBERT:-0.37}
W_LGBM=${W_LGBM:-0.63}

cd "$(dirname "$0")"

# CV 模式时延长时间限制
if [[ "${MODE}" == "cv" ]]; then
    sed -i 's/--time=02:00:00/--time=06:00:00/' run_albert_train.sh
    echo "[submit] CV 模式：训练 time 已调整为 06:00:00"
fi

TRAIN_JOB=$(MODE=${MODE} sbatch --parsable run_albert_train.sh)
echo "[submit] 训练 array job 已提交: ${TRAIN_JOB} (seeds 42/1337/2024 并行)"

ENSEMBLE_JOB=$(W_ALBERT=${W_ALBERT} W_LGBM=${W_LGBM} \
    sbatch --parsable --dependency=afterok:${TRAIN_JOB} run_albert_ensemble.sh)
echo "[submit] Ensemble job 已提交: ${ENSEMBLE_JOB} (等待训练完成后自动触发)"
echo ""
echo "查看状态: squeue -u \$USER"
echo "查看日志: tail -f /u/yliu110/364/albert/logs/train_*_*.out"
