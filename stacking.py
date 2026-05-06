"""
用已有的 OOF 预测训练 Logistic Regression meta-learner，替代手动 rank-blend。
全程 CPU，本地秒级完成。

用法：
    cd /u/yliu110/364
    python stacking.py                   # 训练 + 生成提交文件（使用 full preds）
    python stacking.py --no_submit       # 只看 OOF AUC，不生成提交文件
"""

import argparse
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

parser = argparse.ArgumentParser()
parser.add_argument("--no_submit", action="store_true", help="只输出 OOF AUC，不生成提交 CSV")
args = parser.parse_args()

# ── 加载训练集标签 ────────────────────────────────────────────────
train  = pd.read_csv("data/train.csv")
labels = (train["label"] == "positive").astype(int).values
n      = len(labels)

# ── 加载 ELECTRA OOF（3 seed 平均）──────────────────────────────
oof_electra = np.zeros(n)
for seed in [42, 1337, 2024]:
    d   = np.load(f"outputs/oof/electra_seed{seed}.npz")
    buf = np.zeros(n)
    buf[d["idx"]] = d["preds"]
    oof_electra += buf
oof_electra /= 3

# ── 加载 LGBM OOF ────────────────────────────────────────────────
d        = np.load("outputs/oof/lgbm.npz")
oof_lgbm = np.zeros(n)
oof_lgbm[d["idx"]] = d["preds"]

# ── Baseline：当前 rank-blend（0.37/0.63）──────────────────────────
def rank_norm(x):
    return rankdata(x) / len(x)

blend_baseline = 0.37 * rank_norm(oof_electra) + 0.63 * rank_norm(oof_lgbm)
auc_baseline   = roc_auc_score(labels, blend_baseline)
print(f"\nBaseline rank-blend (0.37/0.63) OOF AUC : {auc_baseline:.6f}")

# ── Stacking：LogisticRegressionCV meta-learner ───────────────────
# 特征：原始概率（不做 rank norm，让 LR 自己学）
X_oof = np.stack([oof_electra, oof_lgbm], axis=1)

# 5-fold cross_val_score 评估 meta-learner 的 OOF AUC
cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
meta   = LogisticRegressionCV(cv=5, max_iter=1000, scoring="roc_auc", random_state=42)

cv_aucs = cross_val_score(meta, X_oof, labels, cv=cv, scoring="roc_auc")
print(f"Stacking LogReg CV AUC folds       : {[f'{a:.6f}' for a in cv_aucs]}")
print(f"Stacking mean OOF AUC              : {cv_aucs.mean():.6f}  (±{cv_aucs.std():.6f})")
print(f"vs baseline                        : {cv_aucs.mean() - auc_baseline:+.6f}")

if args.no_submit:
    print("\n(--no_submit 模式，跳过生成提交文件)")
else:
    # 用全量数据训练 meta-learner，预测 test
    meta.fit(X_oof, labels)

    # 加载全量 test 预测
    test_electra = np.load("outputs/preds/electra_full.npz")["preds"]
    test_lgbm    = np.load("outputs/preds/lgbm_full.npz")["preds"]
    test_ids     = np.load("outputs/preds/lgbm_full.npz")["ids"]

    X_test    = np.stack([test_electra, test_lgbm], axis=1)
    test_prob = meta.predict_proba(X_test)[:, 1]

    out_path = "outputs/preds/prediction_stacking.csv"
    pd.DataFrame({"Id": test_ids, "Prob": test_prob}).to_csv(out_path, index=False)
    print(f"\nSaved {out_path}  (mean={test_prob.mean():.4f})")
    print(f"\n对比基准文件:")
    print(f"  python compare_preds.py outputs/preds/prediction_full_we0.90_wl0.10.csv {out_path}")
