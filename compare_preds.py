"""
对比两个预测 CSV 文件的相似度。

用法：
    python compare_preds.py file_a.csv file_b.csv
    python compare_preds.py outputs/preds/prediction_full_we0.90_wl0.10.csv \
                            outputs/preds/prediction_full_we0.37_wl0.63.csv
"""

import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

if len(sys.argv) < 3:
    print("用法: python compare_preds.py <file_a.csv> <file_b.csv>")
    sys.exit(1)

path_a, path_b = sys.argv[1], sys.argv[2]

df_a = pd.read_csv(path_a).sort_values("Id").reset_index(drop=True)
df_b = pd.read_csv(path_b).sort_values("Id").reset_index(drop=True)

assert list(df_a["Id"]) == list(df_b["Id"]), "两个文件的 Id 列不一致！"

prob_a = df_a["Prob"].values
prob_b = df_b["Prob"].values

# 二值化（阈值 0.5：概率 > 0.5 判定为正类，否则为负类）
label_a = (prob_a > 0.5).astype(int)
label_b = (prob_b > 0.5).astype(int)

agree      = (label_a == label_b).mean()
disagree_n = (label_a != label_b).sum()

pearson_r,  _ = pearsonr(prob_a, prob_b)
spearman_r, _ = spearmanr(prob_a, prob_b)

# 分歧样本：一个预测正类，另一个预测负类
diff_idx  = np.where(label_a != label_b)[0]
diff_prob = np.abs(prob_a[diff_idx] - prob_b[diff_idx])

print(f"\n文件 A: {path_a}")
print(f"文件 B: {path_b}")
print(f"样本总数: {len(prob_a):,}")
print()
print(f"{'='*45}")
print(f"  二值一致率  (threshold=0.5): {agree:.4%}  ({len(prob_a)-disagree_n:,} 同 / {disagree_n:,} 不同)")
print(f"  Pearson  相关系数          : {pearson_r:.6f}")
print(f"  Spearman 排名相关系数       : {spearman_r:.6f}")
if disagree_n > 0:
    print(f"\n  分歧样本概率差异（A-B）:")
    print(f"    mean  = {diff_prob.mean():.4f}")
    print(f"    max   = {diff_prob.max():.4f}")
    # 列出最大分歧的前5个
    top5 = np.argsort(diff_prob)[::-1][:5]
    print(f"\n  最大分歧 Top-5:")
    print(f"  {'Id':>10}  {'Prob_A':>8}  {'Prob_B':>8}  {'diff':>8}  {'A_label':>8}  {'B_label':>8}")
    for i in top5:
        idx = diff_idx[i]
        print(f"  {df_a['Id'].iloc[idx]:>10}  {prob_a[idx]:>8.4f}  {prob_b[idx]:>8.4f}  "
              f"{diff_prob[i]:>8.4f}  {label_a[idx]:>8}  {label_b[idx]:>8}")
print(f"{'='*45}")
