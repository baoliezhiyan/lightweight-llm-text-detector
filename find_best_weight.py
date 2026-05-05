"""
用 CV 阶段的 OOF 预测搜索 ELECTRA-full 与 LGBM-full 的最优融合权重。
全程 CPU，本地秒级跑完。

用法：
    cd /u/yliu110/364
    python find_best_weight.py
    python find_best_weight.py --top 20          # 显示前 20 名
    python find_best_weight.py --step 0.01       # 精度到小数点后 2 位（默认）
    python find_best_weight.py --step 0.005      # 精度到小数点后 3 位
"""

import argparse
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

parser = argparse.ArgumentParser()
parser.add_argument("--step", type=float, default=0.01, help="搜索步长（默认 0.01）")
parser.add_argument("--top",  type=int,   default=10,  help="显示前 N 名")
args = parser.parse_args()

# ── 加载训练集标签 ────────────────────────────────────────────────
train  = pd.read_csv("data/train.csv")
labels = (train["label"] == "positive").astype(int).values
n      = len(labels)

# ── 加载 ELECTRA OOF（三 seed 平均）────────────────────────────────
electra_oof = np.zeros(n)
for seed in [42, 1337, 2024]:
    d   = np.load(f"outputs/oof/electra_seed{seed}.npz")
    buf = np.zeros(n)
    buf[d["idx"]] = d["preds"]
    electra_oof += buf
electra_oof /= 3

# ── 加载 LGBM OOF ────────────────────────────────────────────────
d        = np.load("outputs/oof/lgbm.npz")
lgbm_oof = np.zeros(n)
lgbm_oof[d["idx"]] = d["preds"]

# ── Rank 归一化（与 ensemble_2full.py 保持一致）─────────────────────
def rank_norm(x):
    return rankdata(x) / len(x)

re = rank_norm(electra_oof)
rl = rank_norm(lgbm_oof)

# 单模型基线
auc_e = roc_auc_score(labels, re)
auc_l = roc_auc_score(labels, rl)
print(f"\n基线 AUC:")
print(f"  ELECTRA (3-seed avg OOF) : {auc_e:.6f}")
print(f"  LGBM OOF                 : {auc_l:.6f}")

# ── 网格搜索 ─────────────────────────────────────────────────────
weights = np.arange(0.0, 1.0 + args.step / 2, args.step)
results = []
for w in weights:
    w   = round(w, 4)
    auc = roc_auc_score(labels, w * re + (1 - w) * rl)
    results.append((w, round(1 - w, 4), auc))

results.sort(key=lambda x: -x[2])
best_w_e, best_w_l, best_auc = results[0]

print(f"\n搜索步长 = {args.step}，共 {len(results)} 个候选点")
print(f"\n{'排名':>4}  {'w_electra':>10}  {'w_lgbm':>8}  {'OOF AUC':>12}")
print("─" * 42)
for rank, (w_e, w_l, auc) in enumerate(results[: args.top], 1):
    marker = " ◀ best" if rank == 1 else ""
    print(f"{rank:>4}  {w_e:>10.4f}  {w_l:>8.4f}  {auc:>12.6f}{marker}")

print(f"\n推荐提交命令：")
print(f"  W_ELECTRA={best_w_e} W_LGBM={best_w_l} sbatch run_ensemble_full.sh")
