import os
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

# ─────────────── Load all OOFs and test preds ───────────────
print("Loading OOFs and test predictions...")

# ELECTRA seeds
oof_e42   = np.load("outputs/oof/electra_seed42.npz")["preds"]
oof_e1337 = np.load("outputs/oof/electra_seed1337.npz")["preds"]
oof_e2024 = np.load("outputs/oof/electra_seed2024.npz")["preds"]

# LightGBM
oof_lgbm  = np.load("outputs/oof/lgbm.npz")["preds"]

# Test predictions
test_e42   = np.load("outputs/preds/electra_seed42.npz")["preds"]
test_e1337 = np.load("outputs/preds/electra_seed1337.npz")["preds"]
test_e2024 = np.load("outputs/preds/electra_seed2024.npz")["preds"]
test_lgbm  = np.load("outputs/preds/lgbm.npz")["preds"]
test_ids   = np.load("outputs/preds/lgbm.npz")["ids"]

# Ground truth labels (for OOF scoring)
import pandas as pd
train = pd.read_csv("train.csv")
y = (train["label"] == "positive").astype(int).values

# ─────────────── Step 1: Average ELECTRA seeds ───────────────
print("\nStep 1: Average ELECTRA across seeds...")
oof_electra  = (oof_e42 + oof_e1337 + oof_e2024) / 3
test_electra = (test_e42 + test_e1337 + test_e2024) / 3

print(f"  ELECTRA seed 42 OOF AUC:   {roc_auc_score(y, oof_e42):.5f}")
print(f"  ELECTRA seed 1337 OOF AUC: {roc_auc_score(y, oof_e1337):.5f}")
print(f"  ELECTRA seed 2024 OOF AUC: {roc_auc_score(y, oof_e2024):.5f}")
print(f"  ELECTRA averaged OOF AUC:  {roc_auc_score(y, oof_electra):.5f}")
print(f"  LightGBM OOF AUC:          {roc_auc_score(y, oof_lgbm):.5f}")


# ─────────────── Step 2: Rank normalization ───────────────
def rank_normalize(x):
    return rankdata(x) / len(x)

print("\nStep 2: Rank normalization...")
oof_electra_rank  = rank_normalize(oof_electra)
oof_lgbm_rank     = rank_normalize(oof_lgbm)
test_electra_rank = rank_normalize(test_electra)
test_lgbm_rank    = rank_normalize(test_lgbm)


# ─────────────── Step 3: Grid search blend weight ───────────────
print("\nStep 3: Grid search blend weight...")
best_auc = 0
best_w   = 0.5

for w in np.arange(0.0, 1.01, 0.01):
    blend = w * oof_electra_rank + (1 - w) * oof_lgbm_rank
    auc = roc_auc_score(y, blend)
    if auc > best_auc:
        best_auc = auc
        best_w   = w

print(f"  Best weight: ELECTRA={best_w:.2f}, LightGBM={1-best_w:.2f}")
print(f"  Best OOF AUC: {best_auc:.5f}")


# ─────────────── Step 4: Final blend on test ───────────────
print("\nStep 4: Apply best weights to test...")
test_final = best_w * test_electra_rank + (1 - best_w) * test_lgbm_rank


# ─────────────── Step 5: Save prediction.csv ───────────────
out = pd.DataFrame({"Id": test_ids, "Prob": test_final})
out.to_csv("prediction.csv", index=False)
print(f"\nSaved prediction.csv: {out.shape}")
print(out.head())