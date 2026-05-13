import os
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
OOF_DIR = os.path.join(HERE, "outputs", "oof")
DATA_DIR = os.path.join(HERE, "data")

def rank_norm(x):
    return rankdata(x) / (len(x) + 1)

train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
labels = (train["label"] == "positive").astype(int).values
n = len(labels)

def load_albert_oof(seeds=(42, 1337, 2024)):
    buf = np.zeros(n)
    for s in seeds:
        d = np.load(os.path.join(OOF_DIR, f"albert_seed{s}.npz"))
        tmp = np.zeros(n)
        tmp[d["idx"]] = d["preds"]
        buf += tmp
    return buf / len(seeds)

oof_a = load_albert_oof()

d_l = np.load(os.path.join(OOF_DIR, "lgbm.npz"))
oof_l = np.zeros(n)
oof_l[d_l["idx"]] = d_l["preds"]

ra = rank_norm(oof_a)
rl = rank_norm(oof_l)

best_auc, best_wa, best_wl = 0.0, 0.0, 1.0
for wa_int in range(0, 1001):
    wa = wa_int / 1000
    wl = 1.0 - wa
    auc = roc_auc_score(labels, wa * ra + wl * rl)
    if auc > best_auc:
        best_auc, best_wa, best_wl = auc, wa, wl

print(f"Best ALBERT={best_wa:.3f} LGBM={best_wl:.3f} OOF AUC={best_auc:.6f}")
