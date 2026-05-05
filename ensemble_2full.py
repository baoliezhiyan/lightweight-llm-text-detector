import argparse
import numpy as np
import pandas as pd
from scipy.stats import rankdata

parser = argparse.ArgumentParser()
parser.add_argument("--w_electra", type=float, default=0.9, help="rank weight for ELECTRA full (0~1)")
parser.add_argument("--w_lgbm",    type=float, default=0.1, help="rank weight for LGBM full (0~1)")
parser.add_argument("--output",    type=str,   default=None, help="output csv path; auto-named if omitted")
args = parser.parse_args()

w_e = args.w_electra
w_l = args.w_lgbm
assert abs(w_e + w_l - 1.0) < 1e-6, f"weights must sum to 1.0, got {w_e} + {w_l} = {w_e + w_l}"

output_path = args.output or f"outputs/preds/prediction_full_we{w_e:.2f}_wl{w_l:.2f}.csv"

print(f"w_electra={w_e}  w_lgbm={w_l}  -> {output_path}")

test_full_electra = np.load("outputs/preds/electra_full.npz")["preds"]
test_full_lgbm    = np.load("outputs/preds/lgbm_full.npz")["preds"]
test_ids          = np.load("outputs/preds/lgbm_full.npz")["ids"]


def rank_normalize(x):
    return rankdata(x) / len(x)


re_rank = rank_normalize(test_full_electra)
rl_rank = rank_normalize(test_full_lgbm)

blend = w_e * re_rank + w_l * rl_rank

out = pd.DataFrame({"Id": test_ids, "Prob": blend})
out.to_csv(output_path, index=False)
print(f"Saved {output_path}  (min={blend.min():.4f}, max={blend.max():.4f}, mean={blend.mean():.4f})")