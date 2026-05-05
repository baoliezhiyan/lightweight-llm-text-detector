#!/bin/bash
#SBATCH -A bcjw-delta-gpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16g
#SBATCH --time=00:10:00
#SBATCH --job-name=awp_ensemble_364
#SBATCH --output=/u/yliu110/364/awp/logs/ensemble_%j.out
#SBATCH --error=/u/yliu110/364/awp/logs/ensemble_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yl188@illinois.edu

module load miniforge3-python
eval "$(conda shell.bash hook)"
conda activate cs364

W_ELECTRA=${W_ELECTRA:-0.37}
W_LGBM=${W_LGBM:-0.63}

echo "========================================"
echo "AWP Ensemble | w_electra=${W_ELECTRA} w_lgbm=${W_LGBM}"
echo "Start: $(date)"
echo "========================================"

cd /u/yliu110/364/awp

python3 ensemble.py --mode predict \
    --w_electra "${W_ELECTRA}" \
    --w_lgbm    "${W_LGBM}"

echo ""
echo "========================================"
echo "End: $(date)"
echo "========================================"
