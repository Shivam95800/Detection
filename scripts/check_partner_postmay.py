import pandas as pd
import numpy as np
import os

data_dir = r"C:\Users\Asus\.gemini\antigravity-ide\scratch\kestrel-warranty\data"
train = pd.read_csv(os.path.join(data_dir, "train.csv"))
train['submitted_dt'] = pd.to_datetime(train['submitted_at'])
train_valid = train.dropna(subset=['is_fraud']).copy()

pre_may = train_valid[train_valid['submitted_dt'] < '2026-05-01']
post_may = train_valid[train_valid['submitted_dt'] >= '2026-05-01']

print("Pre-May by partner_inspected:")
print(pre_may.groupby('partner_inspected')['is_fraud'].agg(['count', 'sum', 'mean']))

print("\nPost-May by partner_inspected:")
print(post_may.groupby('partner_inspected')['is_fraud'].agg(['count', 'sum', 'mean']))

# What about partner history?
# For post-May claims, are the frauds coming from specific partners?
print("\nPost-May frauds by partner_id:")
pm_fraud_partners = post_may[post_may['is_fraud'] == 1]['partner_id'].value_counts()
print(pm_fraud_partners)

# How many of these partners had prior claims or frauds in pre-May?
for pid in pm_fraud_partners.index[:10]:
    pre_claims = len(pre_may[pre_may['partner_id'] == pid])
    pre_frauds = pre_may[pre_may['partner_id'] == pid]['is_fraud'].sum()
    print(f"Partner {pid}: pre-May claims = {pre_claims}, pre-May frauds = {pre_frauds}, post-May frauds = {pm_fraud_partners[pid]}")
