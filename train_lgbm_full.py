import os
import re
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
import lightgbm as lgb

HERE = os.path.dirname(os.path.abspath(__file__))

# config
WORD_NGRAM_RANGE = (1, 2)
WORD_MAX_FEATURES = 100000
CHAR_NGRAM_RANGE = (3, 5)
CHAR_MAX_FEATURES = 100000

NUM_BOOST_ROUND = 600

LGBM_PARAMS_BASE = {
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
    "n_jobs": -1,
}

SEEDS = [42]

os.makedirs(os.path.join(HERE, "outputs", "preds"), exist_ok=True)

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
    return np.mean([len(w) for w in words]) if words else 0


def feat_sent_std(text):
    sentences = re.split(r'[.!?]+', text)
    sentences = [s for s in sentences if len(s.strip()) > 3]
    word_counts = [len(s.split()) for s in sentences]
    return np.std(word_counts) if len(word_counts) >= 2 else 0


def feat_punct_ratio(text):
    return sum(c in ',.;:!?' for c in text) / len(text) if len(text) > 0 else 0


def feat_upper_ratio(text):
    return sum(c.isupper() for c in text) / len(text) if len(text) > 0 else 0


def feat_ttr(text):
    words = text.lower().split()
    return len(set(words)) / len(words) if words else 0


def feat_comma_ratio(text):
    return text.count(',') / len(text) if len(text) > 0 else 0


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


def run_lgbm_full():
    test_preds_avg = np.zeros(len(test_texts))

    for seed in SEEDS:
        params = {**LGBM_PARAMS_BASE, "seed": seed}
        dtrain = lgb.Dataset(X_train, label=train_labels)
        model = lgb.train(
            params,
            dtrain,
            num_boost_round=NUM_BOOST_ROUND,
        )
        test_preds = model.predict(X_test, num_iteration=NUM_BOOST_ROUND)
        test_preds_avg += test_preds / len(SEEDS)

    np.savez(
        os.path.join(HERE, "outputs", "preds", "lgbm_full.npz"),
        preds=test_preds_avg,
        ids=np.array(test_ids),
    )
    print("Saved to outputs/preds/lgbm_full.npz")


run_lgbm_full()
