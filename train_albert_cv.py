import os
import random
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

if not torch.cuda.is_available():
    print("Error: No CUDA", file=sys.stderr)
    sys.exit(1)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
OUT_MODELS = os.path.join(HERE, "outputs", "models")
OUT_OOF = os.path.join(HERE, "outputs", "oof")
OUT_PREDS = os.path.join(HERE, "outputs", "preds")
for d in [OUT_MODELS, OUT_OOF, OUT_PREDS]:
    os.makedirs(d, exist_ok=True)

# config
MODEL_NAME = "albert-base-v2"
MAX_LEN = 512
HEAD_TOKENS = 256
TAIL_TOKENS = 256
BATCH_SIZE = 16
EPOCHS = 4
LR = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
N_FOLDS = 5
SEEDS = [42, 1337, 2024]
DEVICE = torch.device("cuda")

print(f"GPU: {torch.cuda.get_device_name(0)}")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Data
train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
train_texts = train["text"].tolist()
train_labels = (train["label"] == "positive").astype(int).tolist()
test_texts = test["text"].tolist()
test_ids = test["id"].tolist()
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class EssayDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def head_tail_encode(self, text):
        tokens = self.tokenizer(text, add_special_tokens=False, truncation=False)
        input_ids = tokens["input_ids"]
        attention_mask = tokens["attention_mask"]
        if len(input_ids) > HEAD_TOKENS + TAIL_TOKENS:
            input_ids = input_ids[:HEAD_TOKENS] + input_ids[-TAIL_TOKENS:]
            attention_mask = attention_mask[:HEAD_TOKENS] + attention_mask[-TAIL_TOKENS:]
        cls_id = self.tokenizer.cls_token_id
        sep_id = self.tokenizer.sep_token_id
        input_ids = [cls_id] + input_ids + [sep_id]
        attention_mask = [1] + attention_mask + [1]
        pad_len = MAX_LEN - len(input_ids)
        if pad_len > 0:
            input_ids += [self.tokenizer.pad_token_id] * pad_len
            attention_mask += [0] * pad_len
        return {
            "input_ids": torch.tensor(input_ids[:MAX_LEN], dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask[:MAX_LEN], dtype=torch.long),
        }

    def __getitem__(self, idx):
        enc = self.head_tail_encode(self.texts[idx])
        if self.labels is not None:
            enc["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)
        return enc


# Model
class TextClassifier(nn.Module):
    def __init__(self, model_name=MODEL_NAME, dropout=0.3):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls_hidden = outputs.last_hidden_state[:, 0, :]
        return self.classifier(cls_hidden).squeeze(-1)

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


# Training
def train_epoch(model, loader, optimizer, scheduler, criterion, scaler):
    model.train()
    total_loss = 0
    all_logits, all_labels = [], []
    for batch in tqdm(loader, desc="train", leave=False):
        ids = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        labs = batch["labels"].to(DEVICE)
        optimizer.zero_grad()
        with autocast("cuda", enabled=True):
            logits = model(ids, mask)
            loss = criterion(logits, labs)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        total_loss += loss.item()
        all_logits.extend(logits.detach().float().cpu().numpy())
        all_labels.extend(labs.cpu().numpy())
    return total_loss / len(loader), roc_auc_score(all_labels, all_logits)


@torch.no_grad()
def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss = 0
    all_logits, all_labels = [], []
    for batch in tqdm(loader, desc="eval", leave=False):
        ids = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        labs = batch["labels"].to(DEVICE)
        with autocast("cuda", enabled=True):
            logits = model(ids, mask)
            loss = criterion(logits, labs)
        total_loss += loss.item()
        all_logits.extend(logits.float().cpu().numpy())
        all_labels.extend(labs.cpu().numpy())
    val_probs = 1 / (1 + np.exp(-np.array(all_logits)))
    return total_loss / len(loader), roc_auc_score(all_labels, all_logits), val_probs


@torch.no_grad()
def predict(model, loader):
    model.eval()
    all_probs = []
    for batch in tqdm(loader, desc="predict", leave=False):
        with autocast("cuda", enabled=True):
            logits = model(batch["input_ids"].to(DEVICE), batch["attention_mask"].to(DEVICE))
        all_probs.extend(torch.sigmoid(logits).float().cpu().numpy())
    return np.array(all_probs)


# CV
def run_cv(seed):
    set_seed(seed)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    oof_preds = np.zeros(len(train_texts))
    test_preds = np.zeros(len(test_texts))
    fold_aucs = []

    test_loader = DataLoader(
        EssayDataset(test_texts, None, tokenizer),
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    for fold, (tr_idx, val_idx) in enumerate(skf.split(train_texts, train_labels)):
        tr_loader = DataLoader(
            EssayDataset([train_texts[i] for i in tr_idx], [train_labels[i] for i in tr_idx], tokenizer),
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )
        val_loader = DataLoader(
            EssayDataset([train_texts[i] for i in val_idx], [train_labels[i] for i in val_idx], tokenizer),
            batch_size=BATCH_SIZE * 2,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

        model = TextClassifier().to(DEVICE)
        total_steps = len(tr_loader) * EPOCHS
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, int(total_steps * WARMUP_RATIO), total_steps
        )
        criterion = nn.BCEWithLogitsLoss()
        scaler = GradScaler("cuda", enabled=True)

        best_auc = 0.0
        best_path = os.path.join(OUT_MODELS, f"albert_s{seed}_f{fold}.pt")

        for epoch in range(1, EPOCHS + 1):
            tr_loss, tr_auc = train_epoch(model, tr_loader, optimizer, scheduler, criterion, scaler)
            val_loss, val_auc, _ = eval_epoch(model, val_loader, criterion)
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
    print(f"Seed {seed} fold AUCs: {[f'{a:.4f}' for a in fold_aucs]}")
    print(f"Seed {seed} OOF AUC: {oof_auc:.6f}")

    np.savez(os.path.join(OUT_OOF, f"albert_seed{seed}.npz"), preds=oof_preds, idx=np.arange(len(train_texts)))
    np.savez(os.path.join(OUT_PREDS, f"albert_seed{seed}.npz"), preds=test_preds, ids=np.array(test_ids))
    print(f"Saved OOF to outputs/oof/albert_seed{seed}.npz")
    print(f"Saved pred to outputs/preds/albert_seed{seed}.npz")


# entry
for seed in SEEDS:
    run_cv(seed)
