"""
Full-Route Stacking：支持动态增减模型（ELECTRA / LGBM / ALBERT / AWP 等）
  - 元学习器在 OOF 预测（rank 归一化 + logit）上训练
  - 测试集预测优先使用 Full 模型；若未提供 Full 预测，自动回退到 CV 均值
  - 全程 CPU，几秒内完成，不需要任何新模型训练

用法（在 364/ 目录下执行）：
    # 仅 ELECTRA + LGBM（默认）
    python stacking/stacking_full.py

    # 加入 ALBERT（CV 均值作测试集预测）
    python stacking/stacking_full.py --albert_cv

    # 加入 ALBERT（Full 模型作测试集预测，需已跑完 albert full）
    python stacking/stacking_full.py --albert_full

    # 加入 AWP-ELECTRA Full
    python stacking/stacking_full.py --awp_full

    # 组合使用
    python stacking/stacking_full.py --albert_full --awp_full

    # 只看 OOF AUC，不生成提交文件
    python stacking/stacking_full.py --albert_cv --no_submit

    # 打印调试信息
    python stacking/stacking_full.py --debug
"""

import argparse
import os
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT       = os.path.dirname(SCRIPT_DIR)
OUT_DIR    = os.path.join(SCRIPT_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--albert_cv",  action="store_true", help="加入 ALBERT（CV 均值预测）")
parser.add_argument("--albert_full",action="store_true", help="加入 ALBERT（Full 模型预测，更强）")
parser.add_argument("--awp_full",   action="store_true", help="加入 AWP-ELECTRA（Full 模型预测）")
parser.add_argument("--no_submit",  action="store_true", help="只看 OOF AUC，不生成提交文件")
parser.add_argument("--debug",      action="store_true", help="打印调试信息")
args = parser.parse_args()

# ── 工具函数 ─────────────────────────────────────────────────────
def rank_norm(x):
    return rankdata(x) / (len(x) + 1)

def safe_logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))

def load_npz_seeds(pattern, seeds, n, idx_key="idx"):
    """加载多 seed 的 npz 文件并平均，返回长度为 n 的数组"""
    result = np.zeros(n)
    for seed in seeds:
        d   = np.load(pattern.format(seed=seed))
        buf = np.zeros(n)
        buf[d[idx_key]] = d["preds"]
        result += buf
    return result / len(seeds)

def load_npz_cv_test(pattern, seeds, n_test):
    """加载 CV 模式下的测试集预测（多 seed 平均），用于无 Full 预测时回退"""
    result = np.zeros(n_test)
    for seed in seeds:
        d = np.load(pattern.format(seed=seed))
        result += d["preds"]
    return result / len(seeds)

# ── 加载训练集标签 ────────────────────────────────────────────────
train  = pd.read_csv(os.path.join(ROOT, "data", "train.csv"))
labels = (train["label"] == "positive").astype(int).values
n_train = len(labels)

# ── 锚定 test_ids（来自 LGBM Full）──────────────────────────────
lgbm_full_path = os.path.join(ROOT, "outputs", "preds", "lgbm_full.npz")
test_ids = np.load(lgbm_full_path)["ids"]
n_test   = len(test_ids)

# ─────────────────────────────────────────────────────────────────
# 模型注册表：每个模型需提供 oof_loader 和 test_loader
# oof_loader()  → 长度 n_train 的数组（OOF 预测）
# test_loader() → 长度 n_test  的数组（测试集预测）
# ─────────────────────────────────────────────────────────────────
models = {}

# ── ELECTRA（必选）──────────────────────────────────────────────
def load_electra_oof():
    return load_npz_seeds(
        os.path.join(ROOT, "outputs", "oof", "electra_seed{seed}.npz"),
        [42, 1337, 2024], n_train)

def load_electra_full():
    return np.load(os.path.join(ROOT, "outputs", "preds", "electra_full.npz"))["preds"]

models["ELECTRA"] = (load_electra_oof, load_electra_full)

# ── LGBM（必选）─────────────────────────────────────────────────
def load_lgbm_oof():
    d = np.load(os.path.join(ROOT, "outputs", "oof", "lgbm.npz"))
    buf = np.zeros(n_train); buf[d["idx"]] = d["preds"]; return buf

def load_lgbm_full():
    return np.load(lgbm_full_path)["preds"]

models["LGBM"] = (load_lgbm_oof, load_lgbm_full)

# ── ALBERT（可选）───────────────────────────────────────────────
if args.albert_cv or args.albert_full:
    albert_oof_pattern  = os.path.join(ROOT, "albert", "outputs", "oof",   "albert_seed{seed}.npz")
    albert_pred_pattern = os.path.join(ROOT, "albert", "outputs", "preds", "albert_seed{seed}.npz")
    albert_full_pattern = os.path.join(ROOT, "albert", "outputs", "preds", "albert_full_seed{seed}.npz")

    available_seeds = []
    for s in [42, 1337, 2024]:
        if os.path.exists(albert_oof_pattern.format(seed=s)):
            available_seeds.append(s)

    if not available_seeds:
        print("[WARN] 未找到 ALBERT OOF 文件，跳过 ALBERT")
    else:
        print(f"[INFO] 找到 ALBERT OOF seeds: {available_seeds}")

        def load_albert_oof(seeds=available_seeds):
            return load_npz_seeds(albert_oof_pattern, seeds, n_train)

        if args.albert_full:
            full_seeds = [s for s in available_seeds
                          if os.path.exists(albert_full_pattern.format(seed=s))]
            if full_seeds:
                def load_albert_test(seeds=full_seeds):
                    result = np.zeros(n_test)
                    for s in seeds:
                        result += np.load(albert_full_pattern.format(seed=s))["preds"]
                    return result / len(seeds)
                print(f"[INFO] ALBERT Full seeds 可用: {full_seeds}")
            else:
                print("[WARN] albert_full 文件不存在，回退到 CV 均值")
                def load_albert_test(seeds=available_seeds):
                    return load_npz_cv_test(albert_pred_pattern, seeds, n_test)
        else:
            def load_albert_test(seeds=available_seeds):
                return load_npz_cv_test(albert_pred_pattern, seeds, n_test)

        models["ALBERT"] = (load_albert_oof, load_albert_test)

# ── AWP-ELECTRA（可选）──────────────────────────────────────────
if args.awp_full:
    awp_oof_pattern  = os.path.join(ROOT, "awp", "outputs", "oof",   "electra_awp_seed{seed}.npz")
    awp_full_pattern = os.path.join(ROOT, "awp", "outputs", "preds", "electra_awp_full_seed{seed}.npz")
    awp_pred_pattern = os.path.join(ROOT, "awp", "outputs", "preds", "electra_awp_seed{seed}.npz")

    awp_seeds = [s for s in [42, 1337, 2024]
                 if os.path.exists(awp_oof_pattern.format(seed=s))]

    if not awp_seeds:
        print("[WARN] 未找到 AWP OOF 文件，跳过 AWP")
    else:
        print(f"[INFO] 找到 AWP OOF seeds: {awp_seeds}")

        def load_awp_oof(seeds=awp_seeds):
            return load_npz_seeds(awp_oof_pattern, seeds, n_train)

        awp_full_seeds = [s for s in awp_seeds
                          if os.path.exists(awp_full_pattern.format(seed=s))]
        if awp_full_seeds:
            def load_awp_test(seeds=awp_full_seeds):
                result = np.zeros(n_test)
                for s in seeds:
                    result += np.load(awp_full_pattern.format(seed=s))["preds"]
                return result / len(seeds)
            print(f"[INFO] AWP Full seeds 可用: {awp_full_seeds}")
        else:
            def load_awp_test(seeds=awp_seeds):
                return load_npz_cv_test(awp_pred_pattern, seeds, n_test)

        models["AWP"] = (load_awp_oof, load_awp_test)

# ─────────────────────────────────────────────────────────────────
# 构造特征矩阵
# ─────────────────────────────────────────────────────────────────
print(f"\n使用模型: {list(models.keys())}")

oof_feats  = []
test_feats = []
for name, (oof_fn, test_fn) in models.items():
    oof_arr  = oof_fn()
    test_arr = test_fn()
    oof_feats.append(safe_logit(rank_norm(oof_arr)))
    test_feats.append(safe_logit(rank_norm(test_arr)))
    if args.debug:
        print(f"[DEBUG] {name:8s}  OOF range=[{oof_arr.min():.4f},{oof_arr.max():.4f}]"
              f"  test range=[{test_arr.min():.4f},{test_arr.max():.4f}]")

X_oof  = np.stack(oof_feats,  axis=1)
X_full = np.stack(test_feats, axis=1)

# ─────────────────────────────────────────────────────────────────
# 基准：0.9/0.1 rank-blend（仅 ELECTRA + LGBM）
# ─────────────────────────────────────────────────────────────────
oof_e = models["ELECTRA"][0]()
oof_l = models["LGBM"][0]()
blend_baseline = 0.9 * rank_norm(oof_e) + 0.1 * rank_norm(oof_l)
from sklearn.metrics import roc_auc_score
auc_baseline = roc_auc_score(labels, blend_baseline)
print(f"\n基准 rank-blend (0.9/0.1) OOF AUC  : {auc_baseline:.6f}")

# ─────────────────────────────────────────────────────────────────
# RidgeCV 元学习器：5-fold OOF AUC 评估
# ─────────────────────────────────────────────────────────────────
alphas         = np.logspace(-4, 4, 40)
cv_obj         = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_meta_preds = np.zeros(n_train)

for fold, (tr_idx, va_idx) in enumerate(cv_obj.split(X_oof, labels)):
    m = RidgeCV(alphas=alphas)
    m.fit(X_oof[tr_idx], labels[tr_idx])
    oof_meta_preds[va_idx] = m.predict(X_oof[va_idx])

auc_meta = roc_auc_score(labels, oof_meta_preds)
print(f"Stacking RidgeCV   OOF AUC         : {auc_meta:.6f}  ({auc_meta - auc_baseline:+.6f} vs 基准)")

if args.debug:
    meta_full = RidgeCV(alphas=alphas)
    meta_full.fit(X_oof, labels)
    names = list(models.keys())
    print(f"\n[DEBUG] 模型系数: { {n: f'{c:.4f}' for n, c in zip(names, meta_full.coef_)} }")
    print(f"[DEBUG] best alpha: {meta_full.alpha_:.4f}")

# ─────────────────────────────────────────────────────────────────
# 生成提交文件
# ─────────────────────────────────────────────────────────────────
if not args.no_submit:
    meta = RidgeCV(alphas=alphas)
    meta.fit(X_oof, labels)
    test_scores = meta.predict(X_full)
    test_prob   = rank_norm(test_scores)

    model_tag = "_".join(k.lower() for k in models.keys())
    out_path  = os.path.join(OUT_DIR, f"prediction_stacking_{model_tag}.csv")
    pd.DataFrame({"Id": test_ids, "Prob": test_prob}).to_csv(out_path, index=False)
    print(f"\n已保存 → {out_path}  (mean={test_prob.mean():.4f})")
    print(f"\n对比基准：")
    best = os.path.join(ROOT, "outputs/preds/prediction_full_we0.90_wl0.10.csv")
    print(f"  python compare_preds.py {best} {out_path}")
