import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

# ─────────────── Config ───────────────
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
SEED         = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────── Reproducibility ───────────────
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)
print("Device:", DEVICE)
print("GPU:   ", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no GPU")

# ─────────────── Load data ───────────────
train = pd.read_csv("train.csv")
test  = pd.read_csv("test.csv")

train_texts  = train["text"].tolist()
train_labels = (train["label"] == "positive").astype(int).tolist()
test_texts   = test["text"].tolist()
test_ids     = test["id"].tolist()

print(f"Train: {len(train_texts):,} | Test: {len(test_texts):,}")
print(f"Positive: {sum(train_labels):,} | Negative: {len(train_labels)-sum(train_labels):,}")

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class EssayDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts     = texts
        self.labels    = labels  # 可以是 None(test set)
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.texts)
    
    def head_tail_encode(self, text):
        # 1. 不加特殊 token,直接 tokenize 全文
        tokens = self.tokenizer(
            text,
            add_special_tokens=False,
            truncation=False,
        )
        input_ids      = tokens["input_ids"]
        attention_mask = tokens["attention_mask"]
        
        # 2. Head+Tail 截断
        if len(input_ids) > HEAD_TOKENS + TAIL_TOKENS:
            input_ids      = input_ids[:HEAD_TOKENS] + input_ids[-TAIL_TOKENS:]
            attention_mask = attention_mask[:HEAD_TOKENS] + attention_mask[-TAIL_TOKENS:]
        
        # 3. 加 [CLS] 和 [SEP]
        cls_id = self.tokenizer.cls_token_id
        sep_id = self.tokenizer.sep_token_id
        input_ids      = [cls_id] + input_ids + [sep_id]
        attention_mask = [1]      + attention_mask + [1]
        
        # 4. Pad 到 MAX_LEN
        pad_len = MAX_LEN - len(input_ids)
        if pad_len > 0:
            input_ids      += [self.tokenizer.pad_token_id] * pad_len
            attention_mask += [0] * pad_len
        
        # 5. 截到 MAX_LEN(防极端情况超长)
        return {
            "input_ids":      torch.tensor(input_ids[:MAX_LEN],      dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask[:MAX_LEN], dtype=torch.long),
        }
    
    def __getitem__(self, idx):
        encoded = self.head_tail_encode(self.texts[idx])
        if self.labels is not None:
            encoded["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)
        return encoded
    
class ElectraClassifier(nn.Module):
    def __init__(self, model_name=MODEL_NAME, dropout=0.3):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size  # ELECTRA-small = 256
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )
    
    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        cls_hidden = outputs.last_hidden_state[:, 0, :]  # [CLS] 位置的输出向量
        logit = self.classifier(cls_hidden).squeeze(-1)   # (batch,) 的 logit
        return logit
    
    def count_params(self):
        return sum(p.numel() for p in self.parameters())
    
def train_epoch(model, loader, optimizer, scheduler, criterion):
    model.train()
    total_loss = 0
    all_logits = []
    all_labels = []
    
    for batch in tqdm(loader, desc="train", leave=False):
        input_ids      = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels         = batch["labels"].to(DEVICE)
        
        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss   = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        all_logits.extend(logits.detach().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(loader)
    train_auc = roc_auc_score(all_labels, all_logits)
    return avg_loss, train_auc


@torch.no_grad()
def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss = 0
    all_logits = []
    all_labels = []
    
    for batch in tqdm(loader, desc="eval", leave=False):
        input_ids      = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels         = batch["labels"].to(DEVICE)
        
        logits = model(input_ids, attention_mask)
        loss   = criterion(logits, labels)
        
        total_loss += loss.item()
        all_logits.extend(logits.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(loader)
    val_auc  = roc_auc_score(all_labels, all_logits)
    val_probs = 1 / (1 + np.exp(-np.array(all_logits)))   # sigmoid
    return avg_loss, val_auc, val_probs


@torch.no_grad()
def predict(model, loader):
    model.eval()
    all_probs = []
    
    for batch in tqdm(loader, desc="predict", leave=False):
        input_ids      = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        
        logits = model(input_ids, attention_mask)
        probs  = torch.sigmoid(logits)
        all_probs.extend(probs.cpu().numpy())
    
    return np.array(all_probs)

os.makedirs("outputs/oof", exist_ok=True)
os.makedirs("outputs/preds", exist_ok=True)
os.makedirs("outputs/models", exist_ok=True)


def run_cv(seed):
    set_seed(seed)
    
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    oof_preds  = np.zeros(len(train_texts))
    test_preds = np.zeros(len(test_texts))
    fold_aucs  = []
    
    test_dataset = EssayDataset(test_texts, None, tokenizer)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE * 2,
                              shuffle=False, num_workers=0, pin_memory=True)
    
    for fold, (tr_idx, val_idx) in enumerate(skf.split(train_texts, train_labels)):
        print(f"\n{'='*50}\nSeed {seed} | Fold {fold+1}/{N_FOLDS}\n{'='*50}")
        
        fold_train_texts  = [train_texts[i]  for i in tr_idx]
        fold_train_labels = [train_labels[i] for i in tr_idx]
        fold_val_texts    = [train_texts[i]  for i in val_idx]
        fold_val_labels   = [train_labels[i] for i in val_idx]
        
        train_dataset = EssayDataset(fold_train_texts, fold_train_labels, tokenizer)
        val_dataset   = EssayDataset(fold_val_texts,   fold_val_labels,   tokenizer)
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                  shuffle=True, num_workers=0, pin_memory=True)
        val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE * 2,
                                  shuffle=False, num_workers=0, pin_memory=True)
        
        model = ElectraClassifier().to(DEVICE)
        if fold == 0 and seed == SEED:
            n = model.count_params()
            print(f"Model parameters: {n:,} ({n/1e6:.2f}M)")
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        total_steps  = len(train_loader) * EPOCHS
        warmup_steps = int(total_steps * WARMUP_RATIO)
        scheduler    = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        criterion    = nn.BCEWithLogitsLoss()
        
        best_auc  = 0
        best_path = f"outputs/models/electra_s{seed}_f{fold}.pt"
        
        for epoch in range(1, EPOCHS + 1):
            train_loss, train_auc = train_epoch(model, train_loader, optimizer, scheduler, criterion)
            val_loss, val_auc, val_probs = eval_epoch(model, val_loader, criterion)
            print(f"  Epoch {epoch}: train_loss={train_loss:.4f} train_auc={train_auc:.4f} | val_loss={val_loss:.4f} val_auc={val_auc:.4f}")
            
            if val_auc > best_auc:
                best_auc = val_auc
                torch.save(model.state_dict(), best_path)
        
        model.load_state_dict(torch.load(best_path))
        _, final_auc, final_val_probs = eval_epoch(model, val_loader, criterion)
        fold_aucs.append(final_auc)
        oof_preds[val_idx] = final_val_probs
        
        test_preds += predict(model, test_loader) / N_FOLDS
        
        del model
        torch.cuda.empty_cache()
    
    oof_auc = roc_auc_score(train_labels, oof_preds)
    print(f"\n{'='*50}")
    print(f"Seed {seed} fold AUCs: {[f'{a:.4f}' for a in fold_aucs]}")
    print(f"Seed {seed} mean fold: {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")
    print(f"Seed {seed} OOF AUC:   {oof_auc:.4f}")
    
    np.savez(f"outputs/oof/electra_seed{seed}.npz", preds=oof_preds, idx=np.arange(len(train_texts)))
    np.savez(f"outputs/preds/electra_seed{seed}.npz", preds=test_preds, ids=np.array(test_ids))
    print(f"Saved OOF and test predictions for seed {seed}")
    
    return oof_preds, test_preds, oof_auc


# 跑 3 个种子
for seed in [42, 1337, 2024]:
    run_cv(seed)