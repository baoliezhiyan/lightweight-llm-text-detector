# Essay binary classification: ALBERT + LightGBM + rank blend

Judge whether a short essay is AI-generated or not. 

Pipeline: cross-validate to get OOF -> search blend weights -> train on full data -> rank-blend full test preds and export CSV.

## Run order

### Step 1: Cross-validation, OOF (and CV-stage test preds)

**GPU and CUDA are required**

```bash
python train_albert_cv.py
python train_lgbm_cv.py
```

- ALBERT runs three random seeds (42, 1337, 2024) in order.
- OOF: `outputs/oof/albert_seed*.npz`, `outputs/oof/lgbm.npz`.
- Fold-averaged test preds: `outputs/preds/albert_seed*.npz`, `outputs/preds/lgbm.npz` (only for checks / backup)

### Step 2: Search rank-blend weights on OOF

```bash
python blend_fit.py
```

The terminal prints the best ALBERT & LightGBM weights and the OOF AUC. Copy them into `W_ALBERT` and `W_LGBM` in `blend_predict_full.py`.

### Step 3: Full-data training, final test preds

```bash
python train_albert_full.py
python train_lgbm_full.py
```

- ALBERT: `outputs/preds/albert_full_seed*.npz`
- LightGBM: `outputs/preds/lgbm_full.npz`  

### Step 4: Blend and write submission CSV

```bash
python blend_predict_full.py
```

Writes the final submission data `prediction_albert_lgbm_blend_wa*.csv`

---

Data and paths:

data/train.csv: training set with id, text, label

data/test.csv: test set with id, text

outputs/oof/: OOF predictions from CV, npz (preds, idx, etc.)

outputs/preds/: test predictions as npz (written by CV or full-data scripts)

outputs/models/: ALBERT CV fold checkpoints as .pt

Final outputs:

prediction_albert_lgbm_blend_*.csv: final submission from step 4; wa/wl in the filename are the blend weights

## Dependencies

- Common: `numpy`, `pandas`, `scikit-learn`, `scipy`, `tqdm`
- ALBERT: `torch`, `transformers` (CUDA)
- LightGBM: `lightgbm`

