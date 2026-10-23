import pandas as pd
import numpy as np
import os

data_dir = r"C:\Users\Asus\.gemini\antigravity-ide\scratch\kestrel-warranty\data"
train = pd.read_csv(os.path.join(data_dir, "train.csv"))
train['submitted_dt'] = pd.to_datetime(train['submitted_at'])
train_valid = train.dropna(subset=['is_fraud']).copy()

# Pre-May 2026 vs Post-May 2026 in train
pre_may = train_valid[train_valid['submitted_dt'] < '2026-05-01']
post_may = train_valid[train_valid['submitted_dt'] >= '2026-05-01']

print("=== FRAUD RATE BY CLAIM AMOUNT PRE-MAY vs POST-MAY ===")
print("Pre-May (< 2026-05-01):")
print("Under 2000 INR:", pre_may[pre_may['claim_amount_inr'] < 2000]['is_fraud'].agg(['count', 'sum', 'mean']))
print("2000+ INR:     ", pre_may[pre_may['claim_amount_inr'] >= 2000]['is_fraud'].agg(['count', 'sum', 'mean']))

print("\nPost-May (>= 2026-05-01):")
print("Under 2000 INR:", post_may[post_may['claim_amount_inr'] < 2000]['is_fraud'].agg(['count', 'sum', 'mean']))
print("2000+ INR:     ", post_may[post_may['claim_amount_inr'] >= 2000]['is_fraud'].agg(['count', 'sum', 'mean']))

# Customer prior claims relationship
print("\n=== CUSTOMER PRIOR CLAIMS FRAUD RATE PRE vs POST MAY ===")
print("Pre-May:")
print(pre_may.groupby('customer_prior_claims')['is_fraud'].agg(['count', 'sum', 'mean']))
print("\nPost-May:")
print(post_may.groupby('customer_prior_claims')['is_fraud'].agg(['count', 'sum', 'mean']))
