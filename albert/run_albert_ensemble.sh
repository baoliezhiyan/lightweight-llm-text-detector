#!/bin/bash
#SBATCH -A bcjw-delta-gpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16g
#SBATCH --time=00:10:00
#SBATCH --job-name=albert_ensemble_364
#SBATCH --output=/u/yliu110/364/albert/logs/ensemble_%j.out
#SBATCH --error=/u/yliu110/364/albert/logs/ensemble_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yl188@illinois.edu

module load miniforge3-python
eval "$(conda shell.bash hook)"
conda activate cs364

W_ALBERT=${W_ALBERT:-0.37}
W_LGBM=${W_LGBM:-0.63}

echo "========================================"
echo "ALBERT Ensemble | w_albert=${W_ALBERT} w_lgbm=${W_LGBM}"
echo "Start: $(date)"
echo "========================================"

cd /u/yliu110/364/albert

python3 ensemble.py --mode predict \
    --w_albert "${W_ALBERT}" \
    --w_lgbm   "${W_LGBM}"

echo ""
echo "========================================"
echo "End: $(date)"
echo "========================================"
