"""
Full-Route Stacking：
  - 元学习器在 OOF 预测（rank 归一化）上训练，用于学习最优融合系数
  - 测试集预测来自 Full 模型（rank 归一化后同分布），解决 OOF vs Full 的分布偏移

不需要任何 GPU 或新模型训练，全程 CPU，几秒内完成。

用法（在 364/ 目录下执行）：
    python stacking/stacking_full.py
    python stacking/stacking_full.py --no_submit   # 只看 OOF AUC
    python stacking/stacking_full.py --debug       # 打印各阶段详细信息
"""

import argparse
import os
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from scipy.special import logit
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT       = os.path.dirname(SCRIPT_DIR)
OUT_DIR    = os.path.join(SCRIPT_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--no_submit", action="store_true")
parser.add_argument("--debug",     action="store_true")
args = parser.parse_args()

# ── 工具函数 ─────────────────────────────────────────────────────
def rank_norm(x):
    """映射到 (0,1) 均匀分布，消除不同模型间的分布差异"""
    return rankdata(x) / (len(x) + 1)

def safe_logit(p, eps=1e-6):
    """logit 变换，log(p/(1-p))，让特征在 (-inf,+inf) 更线性可分"""
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))

def auc(y, p):
    return roc_auc_score(y, p)

# ── 加载训练集标签 ────────────────────────────────────────────────
train  = pd.read_csv(os.path.join(ROOT, "data", "train.csv"))
labels = (train["label"] == "positive").astype(int).values
n      = len(labels)

# ── 加载 OOF 预测 ────────────────────────────────────────────────
oof_electra = np.zeros(n)
for seed in [42, 1337, 2024]:
    d   = np.load(os.path.join(ROOT, "outputs", "oof", f"electra_seed{seed}.npz"))
    buf = np.zeros(n)
    buf[d["idx"]] = d["preds"]
    oof_electra += buf
oof_electra /= 3

d        = np.load(os.path.join(ROOT, "outputs", "oof", "lgbm.npz"))
oof_lgbm = np.zeros(n)
oof_lgbm[d["idx"]] = d["preds"]

# ── 加载 Full 测试预测 ───────────────────────────────────────────
full_electra = np.load(os.path.join(ROOT, "outputs", "preds", "electra_full.npz"))["preds"]
full_lgbm    = np.load(os.path.join(ROOT, "outputs", "preds", "lgbm_full.npz"))["preds"]
test_ids     = np.load(os.path.join(ROOT, "outputs", "preds", "lgbm_full.npz"))["ids"]

# ── rank 归一化（统一 OOF 和 Full 的分布）───────────────────────
oof_e_rn   = rank_norm(oof_electra)
oof_l_rn   = rank_norm(oof_lgbm)
full_e_rn  = rank_norm(full_electra)
full_l_rn  = rank_norm(full_lgbm)

# ── 构造特征矩阵：logit(rank_norm(p))  ───────────────────────────
# logit 变换让特征更接近线性可分，Ridge 回归效果更稳定
X_oof  = np.stack([safe_logit(oof_e_rn),  safe_logit(oof_l_rn)],  axis=1)
X_full = np.stack([safe_logit(full_e_rn), safe_logit(full_l_rn)], axis=1)

# ── 基准：0.9/0.1 rank-blend ─────────────────────────────────────
blend_baseline = 0.9 * rank_norm(oof_electra) + 0.1 * rank_norm(oof_lgbm)
auc_baseline   = auc(labels, blend_baseline)
print(f"\n基准 rank-blend (0.9/0.1) OOF AUC  : {auc_baseline:.6f}")

# ── Stacking：RidgeCV 元学习器 ───────────────────────────────────
alphas = np.logspace(-4, 4, 40)
meta   = RidgeCV(alphas=alphas, scoring="r2")

cv_obj = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 用 RidgeCV 在 OOF 特征上做 5-fold 交叉验证，手动算 AUC
oof_meta_preds = np.zeros(n)
for fold, (tr_idx, va_idx) in enumerate(cv_obj.split(X_oof, labels)):
    m = RidgeCV(alphas=alphas)
    m.fit(X_oof[tr_idx], labels[tr_idx])
    oof_meta_preds[va_idx] = m.predict(X_oof[va_idx])

auc_meta = auc(labels, oof_meta_preds)
print(f"Stacking RidgeCV   OOF AUC         : {auc_meta:.6f}  ({auc_meta - auc_baseline:+.6f} vs 基准)")

if args.debug:
    meta.fit(X_oof, labels)
    print(f"\n[DEBUG] meta coefficients : {meta.coef_}")
    print(f"[DEBUG] best alpha        : {meta.alpha_}")
    print(f"[DEBUG] oof_electra range : [{oof_electra.min():.4f}, {oof_electra.max():.4f}]")
    print(f"[DEBUG] full_electra range: [{full_electra.min():.4f}, {full_electra.max():.4f}]")
    print(f"[DEBUG] X_oof  mean/std   : {X_oof.mean(0).round(3)} / {X_oof.std(0).round(3)}")
    print(f"[DEBUG] X_full mean/std   : {X_full.mean(0).round(3)} / {X_full.std(0).round(3)}")

if not args.no_submit:
    # 全量训练元学习器，应用到 Full 预测
    meta.fit(X_oof, labels)
    test_scores = meta.predict(X_full)

    # 映射回 [0,1] 概率（rank norm 归一化）
    test_prob = rank_norm(test_scores)

    out_path = os.path.join(OUT_DIR, "prediction_stacking_full.csv")
    pd.DataFrame({"Id": test_ids, "Prob": test_prob}).to_csv(out_path, index=False)
    print(f"\n已保存 → {out_path}  (mean={test_prob.mean():.4f})")
    print(f"\n对比基准：")
    print(f"  python compare_preds.py {os.path.join(ROOT, 'outputs/preds/prediction_full_we0.90_wl0.10.csv')} {out_path}")
