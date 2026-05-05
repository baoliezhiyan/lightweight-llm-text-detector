"""
ALBERT 实验的融合脚本。

用法：
    # 本地搜最优权重（需要先跑完 CV）
    python ensemble.py --mode search

    # 生成提交文件（需要先跑完 full training）
    python ensemble.py --mode predict --w_albert 0.37 --w_lgbm 0.63
"""

import argparse
import os
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

parser = argparse.ArgumentParser()
parser.add_argument("--mode",     choices=["search", "predict"], default="search")
parser.add_argument("--w_albert", type=float, default=None)
parser.add_argument("--w_lgbm",   type=float, default=None)
parser.add_argument("--step",     type=float, default=0.01)
parser.add_argument("--top",      type=int,   default=10)
args = parser.parse_args()

SCRIPT_DIR      = os.path.dirname(os.path.abspath(__file__))
PARENT_OOF_DIR  = os.path.join(SCRIPT_DIR, "..", "outputs", "oof")
PARENT_PRED_DIR = os.path.join(SCRIPT_DIR, "..", "outputs", "preds")
OWN_OOF_DIR     = os.path.join(SCRIPT_DIR, "outputs", "oof")
OWN_PRED_DIR    = os.path.join(SCRIPT_DIR, "outputs", "preds")
DATA_DIR        = os.path.join(SCRIPT_DIR, "..", "data")


def rank_norm(x):
    return rankdata(x) / len(x)


if args.mode == "search":
    train  = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    labels = (train["label"] == "positive").astype(int).values
    n      = len(labels)

    # ALBERT OOF: 3 seeds averaged
    oof_albert = np.zeros(n)
    for seed in [42, 1337, 2024]:
        d   = np.load(os.path.join(OWN_OOF_DIR, f"albert_seed{seed}.npz"))
        buf = np.zeros(n)
        buf[d["idx"]] = d["preds"]
        oof_albert += buf
    oof_albert /= 3

    # Parent LGBM OOF
    d        = np.load(os.path.join(PARENT_OOF_DIR, "lgbm.npz"))
    lgbm_oof = np.zeros(n)
    lgbm_oof[d["idx"]] = d["preds"]

    auc_a = roc_auc_score(labels, oof_albert)
    auc_l = roc_auc_score(labels, lgbm_oof)
    print(f"\n基线 AUC:")
    print(f"  ALBERT-base-v2 (3-seed avg OOF) : {auc_a:.6f}")
    print(f"  LGBM OOF                        : {auc_l:.6f}")

    ra = rank_norm(oof_albert)
    rl = rank_norm(lgbm_oof)

    weights = np.arange(0.0, 1.0 + args.step / 2, args.step)
    results = []
    for w in weights:
        w   = round(w, 4)
        auc = roc_auc_score(labels, w * ra + (1 - w) * rl)
        results.append((w, round(1 - w, 4), auc))
    results.sort(key=lambda x: -x[2])

    best_w_a, best_w_l, best_auc = results[0]
    print(f"\n搜索步长={args.step}，共 {len(results)} 个候选")
    print(f"\n{'排名':>4}  {'w_albert':>10}  {'w_lgbm':>8}  {'OOF AUC':>12}")
    print("─" * 42)
    for rank, (w_a, w_l, auc) in enumerate(results[:args.top], 1):
        marker = " ◀ best" if rank == 1 else ""
        print(f"{rank:>4}  {w_a:>10.4f}  {w_l:>8.4f}  {auc:>12.6f}{marker}")

    print(f"\n推荐提交命令：")
    print(f"  python ensemble.py --mode predict --w_albert {best_w_a} --w_lgbm {best_w_l}")

else:  # predict
    w_a = args.w_albert
    w_l = args.w_lgbm
    if w_a is None or w_l is None:
        raise ValueError("predict 模式需要指定 --w_albert 和 --w_lgbm")

    # 优先加载 per-seed 文件（array job 产出），不存在时回退到合并文件
    seeds = [42, 1337, 2024]
    per_seed_files = [os.path.join(OWN_PRED_DIR, f"albert_full_seed{s}.npz") for s in seeds]
    if all(os.path.exists(f) for f in per_seed_files):
        test_albert = np.mean([np.load(f)["preds"] for f in per_seed_files], axis=0)
        test_ids    = np.load(per_seed_files[0])["ids"]
        print(f"Loaded per-seed files and averaged: {[os.path.basename(f) for f in per_seed_files]}")
    else:
        d           = np.load(os.path.join(OWN_PRED_DIR, "albert_full.npz"))
        test_albert = d["preds"]
        test_ids    = d["ids"]
        print("Loaded albert_full.npz")

    test_lgbm = np.load(os.path.join(PARENT_PRED_DIR, "lgbm_full.npz"))["preds"]

    blend    = w_a * rank_norm(test_albert) + w_l * rank_norm(test_lgbm)
    out_path = os.path.join(SCRIPT_DIR, f"prediction_albert_wa{w_a:.2f}_wl{w_l:.2f}.csv")
    pd.DataFrame({"Id": test_ids, "Prob": blend}).to_csv(out_path, index=False)
    print(f"Saved {out_path}  (mean={blend.mean():.4f})")
