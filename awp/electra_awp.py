"""
ELECTRA-small with Adversarial Weight Perturbation (AWP).

Usage:
    python electra_awp.py --mode cv    # 5-fold CV x 3 seeds -> OOF + per-fold test preds
    python electra_awp.py --mode full  # full-data training x 3 seeds -> test preds
"""

import argparse
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["cv", "full"], default="cv")
parser.add_argument("--seed", type=int, default=None,
                    help="单独训练指定 seed（为空则顺序跑全部 3 个）")
args = parser.parse_args()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(SCRIPT_DIR, "..", "data")
OUT_MODELS = os.path.join(SCRIPT_DIR, "outputs", "models")
OUT_OOF    = os.path.join(SCRIPT_DIR, "outputs", "oof")
OUT_PREDS  = os.path.join(SCRIPT_DIR, "outputs", "preds")
for d in [OUT_MODELS, OUT_OOF, OUT_PREDS]:
    os.makedirs(d, exist_ok=True)

# ── Config (identical to parent electra.py) ────────────────────────────
MODEL_NAME   = "google/electra-small-discriminator"
MAX_LEN      = 512
HEAD_TOKENS  = 256
TAIL_TOKENS  = 256
BATCH_SIZE   = 16
EPOCHS       = 4
LR           = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
N_FOLDS      = 5
SEEDS        = [42, 1337, 2024]
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── AWP hyperparameters ────────────────────────────────────────────────
AWP_LR          = 1e-4   # step size for weight perturbation
AWP_EPS         = 1e-4   # max perturbation magnitude (relative to weight norm)
AWP_START_EPOCH = 2      # warm-up: enable AWP only from this epoch onward


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


print(f"Mode: {args.mode} | Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ── Data ───────────────────────────────────────────────────────────────
train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

train_texts  = train["text"].tolist()
train_labels = (train["label"] == "positive").astype(int).tolist()
test_texts   = test["text"].tolist()
test_ids     = test["id"].tolist()

print(f"Train: {len(train_texts):,} | Test: {len(test_texts):,}")
print(f"Positive: {sum(train_labels):,} | Negative: {len(train_labels) - sum(train_labels):,}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


# ── Dataset ────────────────────────────────────────────────────────────
class EssayDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts     = texts
        self.labels    = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def head_tail_encode(self, text):
        tokens         = self.tokenizer(text, add_special_tokens=False, truncation=False)
        input_ids      = tokens["input_ids"]
        attention_mask = tokens["attention_mask"]

        if len(input_ids) > HEAD_TOKENS + TAIL_TOKENS:
            input_ids      = input_ids[:HEAD_TOKENS] + input_ids[-TAIL_TOKENS:]
            attention_mask = attention_mask[:HEAD_TOKENS] + attention_mask[-TAIL_TOKENS:]

        cls_id = self.tokenizer.cls_token_id
        sep_id = self.tokenizer.sep_token_id
        input_ids      = [cls_id] + input_ids + [sep_id]
        attention_mask = [1]      + attention_mask + [1]

        pad_len = MAX_LEN - len(input_ids)
        if pad_len > 0:
            input_ids      += [self.tokenizer.pad_token_id] * pad_len
            attention_mask += [0] * pad_len

        return {
            "input_ids":      torch.tensor(input_ids[:MAX_LEN],      dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask[:MAX_LEN], dtype=torch.long),
        }

    def __getitem__(self, idx):
        encoded = self.head_tail_encode(self.texts[idx])
        if self.labels is not None:
            encoded["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)
        return encoded


# ── Model ──────────────────────────────────────────────────────────────
class ElectraClassifier(nn.Module):
    def __init__(self, model_name=MODEL_NAME, dropout=0.3):
        super().__init__()
        self.backbone   = AutoModel.from_pretrained(model_name)
        hidden_size     = self.backbone.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def forward(self, input_ids, attention_mask):
        outputs    = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls_hidden = outputs.last_hidden_state[:, 0, :]
        return self.classifier(cls_hidden).squeeze(-1)

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


# ── AWP ────────────────────────────────────────────────────────────────
class AWP:
    """
    Adversarial Weight Perturbation.
    After the normal backward pass, perturbs model 'weight' parameters,
    re-computes adversarial gradients, then restores the original weights.
    The subsequent optimizer.step() uses the adversarial gradients.
    """

    def __init__(self, model, optimizer, criterion,
                 adv_param="weight", adv_lr=AWP_LR, adv_eps=AWP_EPS):
        self.model     = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.adv_param = adv_param
        self.adv_lr    = adv_lr
        self.adv_eps   = adv_eps
        self._backup     = {}
        self._backup_eps = {}

    def attack_backward(self, input_ids, attention_mask, labels):
        self._save()
        self._perturb()
        self.optimizer.zero_grad()
        adv_loss = self.criterion(self.model(input_ids, attention_mask), labels)
        adv_loss.backward()
        self._restore()

    def _save(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                if name not in self._backup:
                    self._backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    self._backup_eps[name] = self.adv_eps * param.data.detach().abs().mean()

    def _perturb(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm) and name in self._backup_eps:
                    r = self.adv_lr * param.grad / (norm + 1e-8)
                    param.data.add_(r)
                    param.data.clamp_(
                        self._backup[name] - self._backup_eps[name],
                        self._backup[name] + self._backup_eps[name],
                    )

    def _restore(self):
        for name, param in self.model.named_parameters():
            if name in self._backup:
                param.data = self._backup[name]
        self._backup.clear()
        self._backup_eps.clear()


# ── Training functions ─────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, scheduler, criterion, awp=None):
    model.train()
    total_loss = 0
    all_logits, all_labels = [], []

    for batch in tqdm(loader, desc="train", leave=False):
        input_ids      = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels         = batch["labels"].to(DEVICE)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss   = criterion(logits, labels)
        loss.backward()

        if awp is not None:
            awp.attack_backward(input_ids, attention_mask, labels)

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        all_logits.extend(logits.detach().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return total_loss / len(loader), roc_auc_score(all_labels, all_logits)


@torch.no_grad()
def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss = 0
    all_logits, all_labels = [], []

    for batch in tqdm(loader, desc="eval", leave=False):
        input_ids      = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels         = batch["labels"].to(DEVICE)
        logits         = model(input_ids, attention_mask)
        loss           = criterion(logits, labels)
        total_loss    += loss.item()
        all_logits.extend(logits.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    val_probs = 1 / (1 + np.exp(-np.array(all_logits)))
    return total_loss / len(loader), roc_auc_score(all_labels, all_logits), val_probs


@torch.no_grad()
def predict(model, loader):
    model.eval()
    all_probs = []
    for batch in tqdm(loader, desc="predict", leave=False):
        logits = model(batch["input_ids"].to(DEVICE), batch["attention_mask"].to(DEVICE))
        all_probs.extend(torch.sigmoid(logits).cpu().numpy())
    return np.array(all_probs)


# ── CV mode ────────────────────────────────────────────────────────────
def run_cv(seed):
    set_seed(seed)
    skf        = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    oof_preds  = np.zeros(len(train_texts))
    test_preds = np.zeros(len(test_texts))
    fold_aucs  = []

    test_loader = DataLoader(
        EssayDataset(test_texts, None, tokenizer),
        batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=0, pin_memory=True,
    )

    for fold, (tr_idx, val_idx) in enumerate(skf.split(train_texts, train_labels)):
        print(f"\n{'='*55}\nSeed {seed} | Fold {fold+1}/{N_FOLDS}\n{'='*55}")

        train_loader = DataLoader(
            EssayDataset([train_texts[i] for i in tr_idx],
                         [train_labels[i] for i in tr_idx], tokenizer),
            batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True,
        )
        val_loader = DataLoader(
            EssayDataset([train_texts[i] for i in val_idx],
                         [train_labels[i] for i in val_idx], tokenizer),
            batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=0, pin_memory=True,
        )

        model     = ElectraClassifier().to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        total_steps  = len(train_loader) * EPOCHS
        scheduler    = get_cosine_schedule_with_warmup(
            optimizer, int(total_steps * WARMUP_RATIO), total_steps)
        criterion    = nn.BCEWithLogitsLoss()
        awp          = AWP(model, optimizer, criterion)

        if fold == 0 and seed == SEEDS[0]:
            print(f"  Params: {model.count_params():,}  AWP: lr={AWP_LR} eps={AWP_EPS} start_epoch={AWP_START_EPOCH}")

        best_auc  = 0.0
        best_path = os.path.join(OUT_MODELS, f"electra_awp_s{seed}_f{fold}.pt")

        for epoch in range(1, EPOCHS + 1):
            current_awp = awp if epoch >= AWP_START_EPOCH else None
            tr_loss, tr_auc          = train_epoch(model, train_loader, optimizer, scheduler, criterion, current_awp)
            val_loss, val_auc, _     = eval_epoch(model, val_loader, criterion)
            awp_tag = "(AWP)" if current_awp else "     "
            print(f"  Epoch {epoch} {awp_tag}: tr_loss={tr_loss:.4f} tr_auc={tr_auc:.4f} | val_loss={val_loss:.4f} val_auc={val_auc:.4f}")

            if val_auc > best_auc:
                best_auc = val_auc
                torch.save(model.state_dict(), best_path)

        model.load_state_dict(torch.load(best_path))
        _, final_auc, final_probs = eval_epoch(model, val_loader, criterion)
        fold_aucs.append(final_auc)
        oof_preds[val_idx] = final_probs
        test_preds += predict(model, test_loader) / N_FOLDS

        del model
        torch.cuda.empty_cache()

    oof_auc = roc_auc_score(train_labels, oof_preds)
    print(f"\n{'='*55}")
    print(f"Seed {seed} fold AUCs: {[f'{a:.4f}' for a in fold_aucs]}")
    print(f"Seed {seed} OOF AUC:   {oof_auc:.6f}")

    np.savez(os.path.join(OUT_OOF,   f"electra_awp_seed{seed}.npz"),
             preds=oof_preds, idx=np.arange(len(train_texts)))
    np.savez(os.path.join(OUT_PREDS, f"electra_awp_seed{seed}.npz"),
             preds=test_preds, ids=np.array(test_ids))
    print(f"Saved OOF and test predictions for seed {seed}")
    return oof_preds, test_preds, oof_auc


# ── Full mode ──────────────────────────────────────────────────────────
def run_full(seed):
    print(f"\n{'='*55}\nFull-data training | Seed {seed}\n{'='*55}")
    set_seed(seed)

    test_loader = DataLoader(
        EssayDataset(test_texts, None, tokenizer),
        batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=0, pin_memory=True,
    )
    train_loader = DataLoader(
        EssayDataset(train_texts, train_labels, tokenizer),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True,
    )
    model     = ElectraClassifier().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_loader) * EPOCHS
    scheduler   = get_cosine_schedule_with_warmup(
        optimizer, int(total_steps * WARMUP_RATIO), total_steps)
    criterion   = nn.BCEWithLogitsLoss()
    awp         = AWP(model, optimizer, criterion)

    for epoch in range(1, EPOCHS + 1):
        current_awp = awp if epoch >= AWP_START_EPOCH else None
        tr_loss, tr_auc = train_epoch(model, train_loader, optimizer, scheduler, criterion, current_awp)
        awp_tag = "(AWP)" if current_awp else "     "
        print(f"  Epoch {epoch} {awp_tag}: tr_loss={tr_loss:.4f} tr_auc={tr_auc:.4f}")

    test_preds = predict(model, test_loader)
    print(f"  preds: min={test_preds.min():.4f} max={test_preds.max():.4f} mean={test_preds.mean():.4f}")

    del model
    torch.cuda.empty_cache()

    out_file = os.path.join(OUT_PREDS, f"electra_awp_full_seed{seed}.npz")
    np.savez(out_file, preds=test_preds, ids=np.array(test_ids))
    print(f"Saved -> {out_file}")


# ── Entry ──────────────────────────────────────────────────────────────
target_seeds = [args.seed] if args.seed is not None else SEEDS

if args.mode == "cv":
    for seed in target_seeds:
        run_cv(seed)
else:
    for seed in target_seeds:
        run_full(seed)
