import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup
from tqdm import tqdm

# config
MODEL_NAME   = "google/electra-small-discriminator"
MAX_LEN      = 512
HEAD_TOKENS  = 256
TAIL_TOKENS  = 256
BATCH_SIZE   = 16
EPOCHS       = 4
LR           = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEEDS = [42, 1337, 2024]

os.makedirs("outputs/preds", exist_ok=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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

        # keep head and tail if too long
        if len(input_ids) > HEAD_TOKENS + TAIL_TOKENS:
            input_ids = input_ids[:HEAD_TOKENS] + input_ids[-TAIL_TOKENS:]
            attention_mask = attention_mask[:HEAD_TOKENS] + attention_mask[-TAIL_TOKENS:]

        # add CLS and SEP
        cls_id = self.tokenizer.cls_token_id
        sep_id = self.tokenizer.sep_token_id
        input_ids = [cls_id] + input_ids + [sep_id]
        attention_mask = [1] + attention_mask + [1]

        # pad
        pad_len = MAX_LEN - len(input_ids)
        if pad_len > 0:
            input_ids += [self.tokenizer.pad_token_id] * pad_len
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


class ElectraClassifier(nn.Module):
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


def train_epoch(model, loader, optimizer, scheduler, criterion):
    model.train()
    total_loss = 0
    for batch in tqdm(loader, desc="train", leave=False):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def predict(model, loader):
    model.eval()
    all_probs = []
    for batch in tqdm(loader, desc="predict", leave=False):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        logits = model(input_ids, attention_mask)
        probs = torch.sigmoid(logits)
        all_probs.extend(probs.cpu().numpy())
    return np.array(all_probs)


def main():
    print("Loading data...")
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")

    train_texts = train["text"].tolist()
    train_labels = (train["label"] == "positive").astype(int).tolist()
    test_texts = test["text"].tolist()
    test_ids = test["id"].tolist()

    print(f"Train: {len(train_texts):,} | Test: {len(test_texts):,}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    test_dataset = EssayDataset(test_texts, None, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE * 2,
                             shuffle=False, num_workers=0, pin_memory=True)

    # average predictions across seeds
    test_preds_avg = np.zeros(len(test_texts))

    for seed in SEEDS:
        print(f"\n{'='*60}\nFull-data training | Seed {seed}\n{'='*60}")
        set_seed(seed)

        train_dataset = EssayDataset(train_texts, train_labels, tokenizer)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                  shuffle=True, num_workers=0, pin_memory=True)

        model = ElectraClassifier().to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        total_steps = len(train_loader) * EPOCHS
        warmup_steps = int(total_steps * WARMUP_RATIO)
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        criterion = nn.BCEWithLogitsLoss()

        for epoch in range(1, EPOCHS + 1):
            train_loss = train_epoch(model, train_loader, optimizer, scheduler, criterion)
            print(f"  Seed {seed} Epoch {epoch}: train_loss={train_loss:.4f}")

        print(f"  Predicting test for seed {seed}...")
        test_preds = predict(model, test_loader)
        print(f"    preds: min={test_preds.min():.4f}, max={test_preds.max():.4f}, mean={test_preds.mean():.4f}")

        test_preds_avg += test_preds / len(SEEDS)

        del model
        torch.cuda.empty_cache()

    np.savez("outputs/preds/electra_full.npz", preds=test_preds_avg, ids=np.array(test_ids))
    print(f"\nSaved averaged full-data predictions to outputs/preds/electra_full.npz")
    print(f"Final preds: min={test_preds_avg.min():.4f}, max={test_preds_avg.max():.4f}, mean={test_preds_avg.mean():.4f}")


if __name__ == "__main__":
    main()