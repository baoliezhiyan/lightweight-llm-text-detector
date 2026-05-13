import os
import numpy as np
import pandas as pd
from scipy.stats import rankdata

HERE = os.path.dirname(os.path.abspath(__file__))
PRED_DIR = os.path.join(HERE, "outputs", "preds")

W_ALBERT = 0.733
W_LGBM = 0.267

def rank_norm(x):
    return rankdata(x) / (len(x) + 1)

lgbm_npz = np.load(os.path.join(PRED_DIR, "lgbm_full.npz"))
lgbm_full = lgbm_npz["preds"]
test_ids = lgbm_npz["ids"]

albert_full = np.mean(
    [np.load(os.path.join(PRED_DIR, f"albert_full_seed{s}.npz"))["preds"] for s in [42, 1337, 2024]],
    axis=0,
)

test_prob = W_ALBERT * rank_norm(albert_full) + W_LGBM * rank_norm(lgbm_full)

out_path = os.path.join(HERE,f"prediction_albert_lgbm_blend_wa{W_ALBERT:.3f}_wl{W_LGBM:.3f}.csv")
pd.DataFrame({"Id": test_ids, "Prob": test_prob}).to_csv(out_path, index=False)
print(f"Saved to {out_path}")
