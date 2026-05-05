import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

train = pd.read_csv("train.csv")
test  = pd.read_csv("test.csv")

print("Train shape:", train.shape)
print("Test shape: ", test.shape)
print()
print(train['label'].value_counts())
print()
print(train['label'].value_counts(normalize=True))

# 算每篇文章的字符数和单词数
train['n_chars'] = train['text'].str.len()
train['n_words'] = train['text'].str.split().str.len()

# 按 label 分组看统计量
print(train.groupby('label')[['n_chars', 'n_words']].describe())

# Sanity check:有没有空文本
print("\nRows with 0 words:", (train['n_words']==0).sum())

#计算每个句子的长度算出标准差
def sentence_length_std(text):
    sentences = re.split(r'[.!?]+', text)
    sentences = [s for s in sentences if len(s.strip()) > 3]
    word_counts = [len(s.split()) for s in sentences]
    if len(word_counts) < 2:
        return 0
    return np.std(word_counts)

train['sent_std'] = train['text'].apply(sentence_length_std)
print(train.groupby('label')['sent_std'].describe())

#计算标点比例和大小写比例
train['punct_ratio'] = train['text'].str.count(r'[.,;:!?]') / train['n_chars']
train['upper_ratio'] = train['text'].apply(lambda t: sum(c.isupper() for c in t)) / train['n_chars']

print(train.groupby('label')[['punct_ratio', 'upper_ratio']].mean())

# 词汇多样性
def type_token_ratio(text):
    words = text.lower().split()
    if not words:
        return 0
    return len(set(words)) / len(words)

train['ttr'] = train['text'].apply(type_token_ratio)
print(train.groupby('label')['ttr'].describe())

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

# 1. 缺失值
print("NaN in train text:", train['text'].isna().sum())
print("NaN in train label:", train['label'].isna().sum())
print("NaN in test text: ", test['text'].isna().sum())

# 2. 重复行
print("\nDuplicated text in train:", train['text'].duplicated().sum())
print("Duplicated text in test: ", test['text'].duplicated().sum())

# 3. Label 取值
print("\nUnique labels:", train['label'].unique())

# 4. ID 唯一性
print("\nTrain id unique?", train['id'].is_unique)
print("Test id unique? ", test['id'].is_unique)

# 5. 极端短文本(可能是脏数据)
print("\nShortest 5 essays in train:")
print(train.nsmallest(5, 'n_words')[['n_words', 'text']])