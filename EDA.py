import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

print("Train shape:", train.shape)
print("Test shape:", test.shape)
print()
print(train['label'].value_counts())
print()
print(train['label'].value_counts(normalize=True))

# count chars and words
train['n_chars'] = train['text'].str.len()
train['n_words'] = train['text'].str.split().str.len()

print(train.groupby('label')[['n_chars', 'n_words']].describe())
print("\nRows with 0 words:", (train['n_words']==0).sum())

# sentence length std
def sentence_length_std(text):
    sentences = re.split(r'[.!?]+', text)
    sentences = [s for s in sentences if len(s.strip()) > 3]
    word_counts = [len(s.split()) for s in sentences]
    if len(word_counts) < 2:
        return 0
    return np.std(word_counts)

train['sent_std'] = train['text'].apply(sentence_length_std)
print(train.groupby('label')['sent_std'].describe())

# punctuation and uppercase ratios
train['punct_ratio'] = train['text'].str.count('[.,;:!?]') / train['n_chars']

def upper_count(text):
    count = 0
    for c in text:
        if c.isupper():
            count += 1
    return count

train['upper_ratio'] = train['text'].apply(upper_count) / train['n_chars']

print(train.groupby('label')[['punct_ratio', 'upper_ratio']].mean())

# type-token ratio
def type_token_ratio(text):
    words = text.lower().split()
    if not words:
        return 0
    return len(set(words)) / len(words)

train['ttr'] = train['text'].apply(type_token_ratio)
print(train.groupby('label')['ttr'].describe())

# token count using electra tokenizer
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google/electra-small-discriminator")

def n_tokens(text):
    return len(tokenizer(text, add_special_tokens=False)['input_ids'])

train['n_tokens'] = train['text'].apply(n_tokens)

print(train['n_tokens'].describe())
print()
print(f"Essays > 512 tokens: {(train['n_tokens'] > 512).sum()} ({(train['n_tokens'] > 512).mean():.1%})")
print(f"95th percentile: {train['n_tokens'].quantile(0.95):.0f}")
print(f"Max: {train['n_tokens'].max()}")

# check for issues in the data
print("NaN in train text:", train['text'].isna().sum())
print("NaN in train label:", train['label'].isna().sum())
print("NaN in test text:", test['text'].isna().sum())

print("\nDuplicated text in train:", train['text'].duplicated().sum())
print("Duplicated text in test:", test['text'].duplicated().sum())

print("\nUnique labels:", train['label'].unique())

print("\nTrain id unique?", train['id'].is_unique)
print("Test id unique?", test['id'].is_unique)

# look at shortest essays
print("\nShortest 5 essays in train:")
print(train.nsmallest(5, 'n_words')[['n_words', 'text']])