import os
import re
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

HERE = os.path.dirname(os.path.abspath(__file__))

# config
N_FOLDS = 5
SEED = 42

WORD_NGRAM_RANGE = (1, 2)
WORD_MAX_FEATURES = 100000

CHAR_NGRAM_RANGE = (3, 5)
CHAR_MAX_FEATURES = 100000

LGBM_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.05,
    "num_leaves": 64,
    "max_depth": -1,
    "min_child_samples": 20,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.7,
    "bagging_freq": 5,
    "verbose": -1,
    "seed": SEED,
    "n_jobs": -1,
}

os.makedirs(os.path.join(HERE, "outputs", "oof"), exist_ok=True)
os.makedirs(os.path.join(HERE, "outputs", "preds"), exist_ok=True)

np.random.seed(SEED)

# Data
train = pd.read_csv(os.path.join(HERE, "data", "train.csv"))
test = pd.read_csv(os.path.join(HERE, "data", "test.csv"))

train_texts = train["text"].tolist()
train_labels = (train["label"] == "positive").astype(int).values
test_texts = test["text"].tolist()
test_ids = test["id"].tolist()

word_vectorizer = TfidfVectorizer(
    analyzer="word",
    ngram_range=WORD_NGRAM_RANGE,
    max_features=WORD_MAX_FEATURES,
    sublinear_tf=True,
    min_df=3,
    lowercase=True,
    strip_accents="unicode",
)
X_word_train = word_vectorizer.fit_transform(train_texts)
X_word_test = word_vectorizer.transform(test_texts)

char_vectorizer = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=CHAR_NGRAM_RANGE,
    max_features=CHAR_MAX_FEATURES,
    sublinear_tf=True,
    min_df=3,
    lowercase=True,
    strip_accents="unicode",
)
X_char_train = char_vectorizer.fit_transform(train_texts)
X_char_test = char_vectorizer.transform(test_texts)

def feat_n_chars(text):
    return len(text)

def feat_n_words(text):
    return len(text.split())

def feat_n_sents(text):
    sentences = re.split(r'[.!?]+', text)
    sentences = [s for s in sentences if len(s.strip()) > 3]
    return len(sentences)

def feat_avg_word_len(text):
    words = text.split()
    if not words:
        return 0
    return np.mean([len(w) for w in words])

def feat_sent_std(text):
    sentences = re.split(r'[.!?]+', text)
    sentences = [s for s in sentences if len(s.strip()) > 3]
    word_counts = [len(s.split()) for s in sentences]
    if len(word_counts) < 2:
        return 0
    return np.std(word_counts)

def feat_punct_ratio(text):
    if len(text) == 0:
        return 0
    return sum(c in ',.;:!?' for c in text) / len(text)

def feat_upper_ratio(text):
    if len(text) == 0:
        return 0
    return sum(c.isupper() for c in text) / len(text)

def feat_ttr(text):
    words = text.lower().split()
    if not words:
        return 0
    return len(set(words)) / len(words)

def feat_comma_ratio(text):
    if len(text) == 0:
        return 0
    return text.count(',') / len(text)

def feat_paragraph_count(text):
    return len([p for p in text.split('\n\n') if p.strip()])

HAND_FEATURES = [
    feat_n_chars,
    feat_n_words,
    feat_n_sents,
    feat_avg_word_len,
    feat_sent_std,
    feat_punct_ratio,
    feat_upper_ratio,
    feat_ttr,
    feat_comma_ratio,
    feat_paragraph_count,
]

def extract_handcrafted_features(texts):
    n = len(texts)
    n_feats = len(HAND_FEATURES)
    X = np.zeros((n, n_feats), dtype=np.float32)
    for i, text in enumerate(texts):
        for j, fn in enumerate(HAND_FEATURES):
            X[i, j] = fn(text)
    return X

X_hand_train = extract_handcrafted_features(train_texts)
X_hand_test = extract_handcrafted_features(test_texts)

X_train = sp.hstack([X_word_train, X_char_train, sp.csr_matrix(X_hand_train)]).tocsr()
X_test = sp.hstack([X_word_test, X_char_test, sp.csr_matrix(X_hand_test)]).tocsr()

def run_lgbm_cv():
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_preds = np.zeros(len(train_texts))
    test_preds = np.zeros(len(test_texts))

    for _, (tr_idx, val_idx) in enumerate(skf.split(X_train, train_labels)):
        X_tr, X_val = X_train[tr_idx], X_train[val_idx]
        y_tr, y_val = train_labels[tr_idx], train_labels[val_idx]

        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

        model = lgb.train(
            LGBM_PARAMS,
            dtrain,
            num_boost_round=2000,
            valid_sets=[dval],
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(200),
            ],
        )

        val_probs = model.predict(X_val, num_iteration=model.best_iteration)

        oof_preds[val_idx] = val_probs
        test_preds += model.predict(X_test, num_iteration=model.best_iteration) / N_FOLDS

    oof_auc = roc_auc_score(train_labels, oof_preds)
    print(f"OOF AUC: {oof_auc:.4f}")

    np.savez(os.path.join(HERE, "outputs", "oof", "lgbm.npz"), preds=oof_preds, idx=np.arange(len(train_texts)))
    np.savez(os.path.join(HERE, "outputs", "preds", "lgbm.npz"), preds=test_preds, ids=np.array(test_ids))
    print("Saved to outputs/oof/lgbm.npz and outputs/preds/lgbm.npz")

    return oof_preds, test_preds, oof_auc


run_lgbm_cv()
