"""
对已有的 ELECTRA checkpoints 做 Test-Time Augmentation (TTA)。
每个模型用 3 种截断策略推理，取平均，替代原来的单一 head+tail 预测。

三种策略：
  head_tail : 前 256 + 后 256 tokens（原始策略）
  head_only : 前 512 tokens
  tail_only : 后 512 tokens

用法：
    python tta_inference.py           # 同时生成 OOF TTA + Test TTA
    python tta_inference.py --test_only  # 只生成 Test TTA（不需要验证集推理）
"""

import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--test_only", action="store_true")
args = parser.parse_args()

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR   = os.path.join(SCRIPT_DIR, "..")
MODEL_DIR    = os.path.join(PARENT_DIR, "outputs", "models")
DATA_DIR     = os.path.join(PARENT_DIR, "data")
OUT_OOF      = os.path.join(SCRIPT_DIR, "outputs", "oof")
OUT_PREDS    = os.path.join(SCRIPT_DIR, "outputs", "preds")

MODEL_NAME  = "google/electra-small-discriminator"
MAX_LEN     = 512
HEAD_TOKENS = 256
TAIL_TOKENS = 256
BATCH_SIZE  = 128   # 推理无需梯度，显存允许大 batch，减少推理时间
SEEDS       = [42, 1337, 2024]
N_FOLDS     = 5
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

train    = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test     = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
train_texts  = train["text"].tolist()
train_labels = (train["label"] == "positive").astype(int).tolist()
test_texts   = test["text"].tolist()
test_ids     = test["id"].tolist()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


# ── Dataset：支持三种截断策略 ─────────────────────────────────────
class EssayDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, strategy="head_tail"):
        self.texts     = texts
        self.labels    = labels
        self.tokenizer = tokenizer
        self.strategy  = strategy  # "head_tail" | "head_only" | "tail_only"

    def __len__(self):
        return len(self.texts)

    def _encode(self, text):
        tokens         = self.tokenizer(text, add_special_tokens=False, truncation=False)
        ids            = tokens["input_ids"]
        mask           = tokens["attention_mask"]

        if self.strategy == "head_tail":
            if len(ids) > HEAD_TOKENS + TAIL_TOKENS:
                ids  = ids[:HEAD_TOKENS]  + ids[-TAIL_TOKENS:]
                mask = mask[:HEAD_TOKENS] + mask[-TAIL_TOKENS:]
        elif self.strategy == "head_only":
            ids  = ids[:MAX_LEN - 2]
            mask = mask[:MAX_LEN - 2]
        elif self.strategy == "tail_only":
            ids  = ids[-(MAX_LEN - 2):]
            mask = mask[-(MAX_LEN - 2):]

        cls_id = self.tokenizer.cls_token_id
        sep_id = self.tokenizer.sep_token_id
        ids    = [cls_id] + ids  + [sep_id]
        mask   = [1]      + mask + [1]

        pad_len = MAX_LEN - len(ids)
        if pad_len > 0:
            ids  += [self.tokenizer.pad_token_id] * pad_len
            mask += [0] * pad_len

        return {
            "input_ids":      torch.tensor(ids[:MAX_LEN],  dtype=torch.long),
            "attention_mask": torch.tensor(mask[:MAX_LEN], dtype=torch.long),
        }

    def __getitem__(self, idx):
        enc = self._encode(self.texts[idx])
        if self.labels is not None:
            enc["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)
        return enc


# ── Model ─────────────────────────────────────────────────────────
class ElectraClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone   = AutoModel.from_pretrained(MODEL_NAME)
        hidden_size     = self.backbone.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256), nn.GELU(),
            nn.Dropout(0.3), nn.Linear(256, 1),
        )

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier(out.last_hidden_state[:, 0, :]).squeeze(-1)


@torch.no_grad()
def predict_tta(model, texts, labels=None):
    """对 texts 用三种策略推理并平均，返回 sigmoid 概率。"""
    all_preds = []
    for strategy in ["head_tail", "head_only", "tail_only"]:
        ds     = EssayDataset(texts, labels, tokenizer, strategy=strategy)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=4, pin_memory=True)
        probs  = []
        model.eval()
        for batch in tqdm(loader, desc=f"  {strategy}", leave=False):
            with autocast():
                logits = model(batch["input_ids"].to(DEVICE),
                               batch["attention_mask"].to(DEVICE))
            probs.extend(torch.sigmoid(logits).float().cpu().numpy())
        all_preds.append(np.array(probs))
    return np.mean(all_preds, axis=0)


# ── TTA on test set ───────────────────────────────────────────────
print("\n[1/2] TTA inference on TEST set...")
test_preds_all = np.zeros((len(SEEDS), N_FOLDS, len(test_texts)))

for si, seed in enumerate(SEEDS):
    seed_preds = np.zeros((N_FOLDS, len(test_texts)))
    for fold in range(N_FOLDS):
        ckpt = os.path.join(MODEL_DIR, f"electra_s{seed}_f{fold}.pt")
        print(f"\n  Seed {seed} Fold {fold+1}/{N_FOLDS}  ({ckpt})")
        model = ElectraClassifier().to(DEVICE)
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        seed_preds[fold] = predict_tta(model, test_texts)
        del model
        torch.cuda.empty_cache()
    test_preds_all[si] = seed_preds

# 先对 fold 取平均，再对 seed 取平均
test_preds_avg = test_preds_all.mean(axis=(0, 1))
np.savez(os.path.join(OUT_PREDS, "electra_tta_full.npz"),
         preds=test_preds_avg, ids=np.array(test_ids))
print(f"\nSaved test TTA preds  "
      f"(min={test_preds_avg.min():.4f} max={test_preds_avg.max():.4f} mean={test_preds_avg.mean():.4f})")

# ── TTA OOF（用于本地评估）────────────────────────────────────────
if not args.test_only:
    print("\n[2/2] TTA inference on TRAIN (OOF)...")
    oof_preds_all = np.zeros((len(SEEDS), len(train_texts)))

    for si, seed in enumerate(SEEDS):
        skf      = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        oof_seed = np.zeros(len(train_texts))
        for fold, (_, val_idx) in enumerate(skf.split(train_texts, train_labels)):
            ckpt = os.path.join(MODEL_DIR, f"electra_s{seed}_f{fold}.pt")
            print(f"\n  Seed {seed} Fold {fold+1}/{N_FOLDS}  (OOF inference)")
            val_texts = [train_texts[i] for i in val_idx]
            model     = ElectraClassifier().to(DEVICE)
            model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
            oof_seed[val_idx] = predict_tta(model, val_texts)
            del model
            torch.cuda.empty_cache()
        oof_preds_all[si] = oof_seed
        oof_auc = roc_auc_score(train_labels, oof_seed)
        print(f"\n  Seed {seed} TTA OOF AUC: {oof_auc:.6f}")

    oof_avg     = oof_preds_all.mean(axis=0)
    oof_auc_avg = roc_auc_score(train_labels, oof_avg)
    print(f"\nTTA averaged (3-seed) OOF AUC: {oof_auc_avg:.6f}")
    print(f"Baseline (no TTA) ELECTRA OOF: 0.999179")

    np.savez(os.path.join(OUT_OOF, "electra_tta.npz"),
             preds=oof_avg, idx=np.arange(len(train_texts)))
    print(f"Saved OOF TTA preds")
