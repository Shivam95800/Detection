import pandas as pd
import numpy as np
import os

data_dir = r"C:\Users\Asus\.gemini\antigravity-ide\scratch\kestrel-warranty\data"

train = pd.read_csv(os.path.join(data_dir, "train.csv"))
test = pd.read_csv(os.path.join(data_dir, "test_unlabelled.csv"))
partners = pd.read_csv(os.path.join(data_dir, "partners.csv"))
products = pd.read_csv(os.path.join(data_dir, "products.csv"))

print("=== INSPECTOR NOTE ANALYSIS ===")
print("Train inspector_note non-null count:", train['inspector_note'].notnull().sum())
print("Test inspector_note non-null count:", test['inspector_note'].notnull().sum())
print("Sample inspector_notes from train (fraud == 1):")
print(train[train['is_fraud'] == 1]['inspector_note'].dropna().head(10).tolist())
print("\nSample inspector_notes from train (fraud == 0):")
print(train[train['is_fraud'] == 0]['inspector_note'].dropna().head(10).tolist())

print("\nValue counts of inspector_note (top 20):")
print(train['inspector_note'].value_counts().head(20))

print("\n=== PARTNER INSPECTED & PHOTO ATTACHED ===")
print("Train partner_inspected:")
print(pd.crosstab(train['source'], train['partner_inspected'], dropna=False))
print("Test partner_inspected:")
print(test['partner_inspected'].value_counts(dropna=False))

print("Train photo_attached:")
print(pd.crosstab(train['source'], train['photo_attached'], dropna=False))
print("Test photo_attached:")
print(test['photo_attached'].value_counts(dropna=False))

print("\n=== PARTNERS & PRODUCTS JOIN CHECK ===")
print("Partners shape:", partners.shape)
print("Unique partner_id in partners:", partners['partner_id'].nunique())
train_unmatched_partners = set(train['partner_id']) - set(partners['partner_id'])
test_unmatched_partners = set(test['partner_id']) - set(partners['partner_id'])
print(f"Unmatched partner_id in train: {len(train_unmatched_partners)}")
print(f"Unmatched partner_id in test: {len(test_unmatched_partners)}")

print("Products shape:", products.shape)
print("Unique sku in products:", products['sku'].nunique())
train_unmatched_skus = set(train['sku']) - set(products['sku'])
test_unmatched_skus = set(test['sku']) - set(products['sku'])
print(f"Unmatched sku in train: {len(train_unmatched_skus)}")
print(f"Unmatched sku in test: {len(test_unmatched_skus)}")

print("\n=== PARTNER ONBOARDING DATES & TENURE ===")
print("Partners onboarded_date min/max:", partners['onboarded_date'].min(), "to", partners['onboarded_date'].max())
print("Partners by partner_type:")
print(partners['partner_type'].value_counts())
print("Partners by city (top 10):")
print(partners['city'].value_counts().head(10))

print("\n=== CLAIM AMOUNT vs LIST PRICE ===")
train_merged = train.merge(products, on='sku', how='left')
test_merged = test.merge(products, on='sku', how='left')

train_merged['claim_ratio'] = train_merged['claim_amount_inr'] / train_merged['list_price_inr']
test_merged['claim_ratio'] = test_merged['claim_amount_inr'] / test_merged['list_price_inr']

print("Train claim_ratio describe:")
print(train_merged['claim_ratio'].describe())
print("Test claim_ratio describe:")
print(test_merged['claim_ratio'].describe())

print("Train claims exceeding list price (>1.0):", (train_merged['claim_ratio'] > 1.0).sum())
print("Test claims exceeding list price (>1.0):", (test_merged['claim_ratio'] > 1.0).sum())

print("\n=== DAYS SINCE PURCHASE vs WARRANTY MONTHS ===")
train_merged['warranty_days'] = train_merged['warranty_months'] * 30.5  # approx
train_merged['past_warranty'] = train_merged['days_since_purchase'] > (train_merged['warranty_months'] * 30.5)
print("Train claims past warranty (approx standard):")
print(pd.crosstab(train_merged['past_warranty'], train_merged['is_fraud'], dropna=False))

print("\n=== NEGATIVE DAYS SINCE PURCHASE ===")
print("Train days_since_purchase < 0:", (train['days_since_purchase'] < 0).sum())
print("Test days_since_purchase < 0:", (test['days_since_purchase'] < 0).sum())
