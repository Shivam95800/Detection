import pandas as pd
import numpy as np
import os

data_dir = r"C:\Users\Asus\.gemini\antigravity-ide\scratch\kestrel-warranty\data"

train = pd.read_csv(os.path.join(data_dir, "train.csv"))
test = pd.read_csv(os.path.join(data_dir, "test_unlabelled.csv"))
partners = pd.read_csv(os.path.join(data_dir, "partners.csv"))
products = pd.read_csv(os.path.join(data_dir, "products.csv"))
sample_sub = pd.read_csv(os.path.join(data_dir, "sample_submission.csv"))

print("=== TRAIN DATA INFO ===")
print("Train shape:", train.shape)
print("Submitted_at min/max:", train['submitted_at'].min(), "to", train['submitted_at'].max())
print("\nTrain is_fraud value counts (including NaN):")
print(train['is_fraud'].value_counts(dropna=False))

print("\nTrain source value counts:")
print(train['source'].value_counts(dropna=False))

print("\nis_fraud by source:")
print(pd.crosstab(train['source'], train['is_fraud'].fillna(-1)))

print("\n=== TEST DATA INFO ===")
print("Test shape:", test.shape)
print("Submitted_at min/max:", test['submitted_at'].min(), "to", test['submitted_at'].max())
print("\nTest source value counts:")
print(test['source'].value_counts(dropna=False))

print("\n=== MISSING VALUES IN TRAIN ===")
print(train.isnull().sum())

print("\n=== MISSING VALUES IN TEST ===")
print(test.isnull().sum())

print("\n=== DUPLICATE CLAIM IDS ===")
print("Train claim_id nunique:", train['claim_id'].nunique(), "vs total len:", len(train))
print("Test claim_id nunique:", test['claim_id'].nunique(), "vs total len:", len(test))
overlap = set(train['claim_id']).intersection(set(test['claim_id']))
print("Overlap between train and test claim_id:", len(overlap))

# Inspect duplicates within train
train_dup_ids = train[train['claim_id'].duplicated(keep=False)].sort_values('claim_id')
print(f"Number of rows with duplicated claim_id in train: {len(train_dup_ids)}")
if len(train_dup_ids) > 0:
    print("Sample duplicate claim_ids in train:")
    print(train_dup_ids[['claim_id', 'submitted_at', 'partner_id', 'sku', 'claim_amount_inr', 'is_fraud']].head(10))

# Inspect duplicates within test
test_dup_ids = test[test['claim_id'].duplicated(keep=False)].sort_values('claim_id')
print(f"Number of rows with duplicated claim_id in test: {len(test_dup_ids)}")

print("\n=== MONTHLY DISTRIBUTION & FRAUD RATE IN TRAIN ===")
train['month'] = pd.to_datetime(train['submitted_at']).dt.to_period('M')
monthly = train.groupby('month').agg(
    total=('claim_id', 'count'),
    fraud=('is_fraud', lambda x: (x == 1).sum()),
    non_fraud=('is_fraud', lambda x: (x == 0).sum()),
    nan_count=('is_fraud', lambda x: x.isnull().sum()),
    fraud_rate_valid=('is_fraud', lambda x: x.mean())
)
print(monthly)

test['month'] = pd.to_datetime(test['submitted_at']).dt.to_period('M')
print("\n=== TEST MONTHLY DISTRIBUTION ===")
print(test['month'].value_counts().sort_index())
