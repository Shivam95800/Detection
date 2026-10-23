import pandas as pd
import numpy as np
import os

data_dir = r"C:\Users\Asus\.gemini\antigravity-ide\scratch\kestrel-warranty\data"
train = pd.read_csv(os.path.join(data_dir, "train.csv"))
test = pd.read_csv(os.path.join(data_dir, "test_unlabelled.csv"))
partners = pd.read_csv(os.path.join(data_dir, "partners.csv"))
products = pd.read_csv(os.path.join(data_dir, "products.csv"))

# Convert dates
train['submitted_dt'] = pd.to_datetime(train['submitted_at'])
test['submitted_dt'] = pd.to_datetime(test['submitted_at'])
partners['onboarded_dt'] = pd.to_datetime(partners['onboarded_date'])

train = train.merge(partners, on='partner_id', how='left').merge(products, on='sku', how='left')
test = test.merge(partners, on='partner_id', how='left').merge(products, on='sku', how='left')

train['partner_tenure_days'] = (train['submitted_dt'] - train['onboarded_dt']).dt.days
test['partner_tenure_days'] = (test['submitted_dt'] - test['onboarded_dt']).dt.days

print("=== PARTNER TENURE CHECKS ===")
print("Train partner tenure min/max:", train['partner_tenure_days'].min(), train['partner_tenure_days'].max())
print("Test partner tenure min/max:", test['partner_tenure_days'].min(), test['partner_tenure_days'].max())
print("Claims before partner onboarding (tenure < 0):")
print("Train:", (train['partner_tenure_days'] < 0).sum())
print("Test:", (test['partner_tenure_days'] < 0).sum())

print("\n=== DUPLICATE CLAIM_ID INVESTIGATION ===")
dup_claims = train[train.duplicated(subset=['claim_id'], keep=False)].sort_values(['claim_id', 'submitted_dt'])
print(f"Total duplicate rows: {len(dup_claims)} across {dup_claims['claim_id'].nunique()} unique claim_ids")
# Look at differences between first and second submission
dup_pivot = dup_claims.groupby('claim_id').agg(
    n_records=('submitted_at', 'count'),
    min_date=('submitted_dt', 'min'),
    max_date=('submitted_dt', 'max'),
    date_diff_hours=('submitted_dt', lambda x: (x.max() - x.min()).total_seconds() / 3600),
    is_fraud_set=('is_fraud', lambda x: set(x.dropna())),
    same_serial=('product_serial', lambda x: x.nunique() == 1),
    same_amount=('claim_amount_inr', lambda x: x.nunique() == 1),
    same_partner=('partner_id', lambda x: x.nunique() == 1),
)
print("Duplicate claims summary:")
print(dup_pivot.head(10))
print("Are fraud labels consistent for duplicate claim_ids?")
print(dup_pivot['is_fraud_set'].value_counts())
print("Date diff hours describe:")
print(dup_pivot['date_diff_hours'].describe())

print("\n=== SERIAL NUMBER ANALYSIS ===")
print("Train product_serial nunique:", train['product_serial'].nunique(), "total:", len(train))
print("Test product_serial nunique:", test['product_serial'].nunique(), "total:", len(test))
train_serial_counts = train.groupby('product_serial').size()
print("Train serials appearing > 1 time:", (train_serial_counts > 1).sum())
print("Train serials appearing > 2 times:", (train_serial_counts > 2).sum())
print("Top repeating serials in train:")
print(train_serial_counts.sort_values(ascending=False).head(10))

# Serial overlap between train and test
serial_overlap = set(train['product_serial']).intersection(set(test['product_serial']))
print("Product serials appearing in BOTH train and test:", len(serial_overlap))

print("\n=== CLAIM DESCRIPTION ANALYSIS ===")
print("Train claim descriptions nunique:", train['claim_description'].nunique())
print("Top descriptions in train:")
print(train['claim_description'].value_counts().head(10))

print("\n=== CUSTOMER PRIOR CLAIMS ===")
print(pd.crosstab(train['customer_prior_claims'], train['is_fraud'], dropna=False, normalize='index'))

print("\n=== AUTO-APPROVAL & MAY 2026 RULE (claim_amount < 2000) ===")
train['under_2000'] = train['claim_amount_inr'] < 2000
test['under_2000'] = test['claim_amount_inr'] < 2000
train['post_may_2026'] = train['submitted_dt'] >= '2026-05-01'
test['post_may_2026'] = test['submitted_dt'] >= '2026-05-01'

print("Train cross-tab post_may vs under_2000 vs partner_inspected:")
print(pd.crosstab([train['post_may_2026'], train['under_2000']], train['partner_inspected']))
print("Test cross-tab under_2000 vs partner_inspected:")
print(pd.crosstab(test['under_2000'], test['partner_inspected']))

print("\nFraud rate by under_2000 and post_may_2026:")
print(train.groupby(['post_may_2026', 'under_2000']).agg(
    total=('is_fraud', 'count'),
    fraud=('is_fraud', lambda x: (x == 1).sum()),
    rate=('is_fraud', 'mean')
))
