"""
TTA 跑完后的 ensemble：TTA ELECTRA + LGBM full。
同时打印 OOF AUC 对比，并生成提交文件。

用法（本地运行，无需 GPU）：
    cd /u/yliu110/364/tta
    python ensemble_tta.py --w_electra 0.37 --w_lgbm 0.63
    python ensemble_tta.py --search     # 先搜最优权重
"""

import argparse
import os
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

parser = argparse.ArgumentParser()
parser.add_argument("--w_electra", type=float, default=None)
parser.add_argument("--w_lgbm",    type=float, default=None)
parser.add_argument("--search",    action="store_true", help="先做 OOF 权重搜索")
args = parser.parse_args()

SCRIPT_DIR      = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR      = os.path.join(SCRIPT_DIR, "..")
PARENT_OOF_DIR  = os.path.join(PARENT_DIR, "outputs", "oof")
PARENT_PRED_DIR = os.path.join(PARENT_DIR, "outputs", "preds")
OWN_OOF_DIR     = os.path.join(SCRIPT_DIR, "outputs", "oof")
OWN_PRED_DIR    = os.path.join(SCRIPT_DIR, "outputs", "preds")

def rank_norm(x):
    return rankdata(x) / len(x)

# ── OOF 对比（如果 TTA OOF 存在）────────────────────────────────
oof_tta_path = os.path.join(OWN_OOF_DIR, "electra_tta.npz")
if os.path.exists(oof_tta_path):
    train      = pd.read_csv(os.path.join(PARENT_DIR, "data", "train.csv"))
    labels     = (train["label"] == "positive").astype(int).values
    n          = len(labels)

    oof_tta = np.load(oof_tta_path)["preds"]

    # 对比原始 ELECTRA OOF
    oof_electra_orig = np.zeros(n)
    for seed in [42, 1337, 2024]:
        d   = np.load(os.path.join(PARENT_OOF_DIR, f"electra_seed{seed}.npz"))
        buf = np.zeros(n)
        buf[d["idx"]] = d["preds"]
        oof_electra_orig += buf
    oof_electra_orig /= 3

    d        = np.load(os.path.join(PARENT_OOF_DIR, "lgbm.npz"))
    oof_lgbm = np.zeros(n)
    oof_lgbm[d["idx"]] = d["preds"]

    auc_orig = roc_auc_score(labels, oof_electra_orig)
    auc_tta  = roc_auc_score(labels, oof_tta)
    print(f"\nOOF AUC 对比:")
    print(f"  ELECTRA 原始 (no TTA) : {auc_orig:.6f}")
    print(f"  ELECTRA TTA           : {auc_tta:.6f}  ({auc_tta - auc_orig:+.6f})")

    if args.search:
        re = rank_norm(oof_tta)
        rl = rank_norm(oof_lgbm)
        results = []
        for w in np.arange(0.0, 1.01, 0.01):
            w   = round(w, 2)
            auc = roc_auc_score(labels, w * re + (1 - w) * rl)
            results.append((w, round(1 - w, 2), auc))
        results.sort(key=lambda x: -x[2])
        best_w_e, best_w_l, best_auc = results[0]
        print(f"\n最优融合权重: w_electra={best_w_e}  w_lgbm={best_w_l}  OOF AUC={best_auc:.6f}")
        if args.w_electra is None:
            args.w_electra = best_w_e
            args.w_lgbm    = best_w_l

# ── 生成提交文件 ─────────────────────────────────────────────────
if args.w_electra is not None:
    w_e = args.w_electra
    w_l = args.w_lgbm

    test_tta  = np.load(os.path.join(OWN_PRED_DIR,    "electra_tta_full.npz"))["preds"]
    test_lgbm = np.load(os.path.join(PARENT_PRED_DIR, "lgbm_full.npz"))["preds"]
    test_ids  = np.load(os.path.join(PARENT_PRED_DIR, "lgbm_full.npz"))["ids"]

    blend    = w_e * rank_norm(test_tta) + w_l * rank_norm(test_lgbm)
    out_path = os.path.join(SCRIPT_DIR, f"prediction_tta_we{w_e:.2f}_wl{w_l:.2f}.csv")
    pd.DataFrame({"Id": test_ids, "Prob": blend}).to_csv(out_path, index=False)
    print(f"\nSaved {out_path}")
    print(f"\n对比基准文件:")
    print(f"  python ../compare_preds.py ../outputs/preds/prediction_full_we0.90_wl0.10.csv {out_path}")
else:
    print("\n[提示] TTA test preds 已就绪，运行以下命令生成提交文件：")
    print("  python ensemble_tta.py --w_electra 0.37 --w_lgbm 0.63")
    print("  python ensemble_tta.py --search  # 自动搜最优权重（需要 OOF TTA 文件）")
