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

print("=== 1. LABEL MATURITY & MISSING LABELS ===")
train['month'] = train['submitted_dt'].dt.to_period('M')
maturity = train.groupby('month').agg(
    total=('claim_id', 'count'),
    unresolved=('is_fraud', lambda x: x.isnull().sum()),
    fraud=('is_fraud', lambda x: (x == 1).sum()),
    genuine=('is_fraud', lambda x: (x == 0).sum()),
    unresolved_pct=('is_fraud', lambda x: x.isnull().mean() * 100),
    fraud_rate_resolved=('is_fraud', lambda x: x.dropna().mean() * 100)
)
print(maturity)

print("\n=== 2. POLICY VIOLATIONS AUDIT ===")
# A. Claim after warranty expiry
# standard warranty is products.warranty_months (12 or 24 months)
# days_since_purchase:
# Does warranty expire at warranty_months * 30 or 365 days?
train['warranty_max_days'] = train['warranty_months'] * 30.5  # 366 or 732
train['days_past_standard_warranty'] = train['days_since_purchase'] - (train['warranty_months'] * 365 / 12)
print("Claims filed with days_since_purchase > warranty_months * 30.5:")
print(train[train['days_since_purchase'] > train['warranty_months'] * 30.5][['days_since_purchase', 'warranty_months', 'is_fraud']].shape)

# What are the max days_since_purchase by warranty_months?
print("\nMax days_since_purchase by warranty_months in train:")
print(train.groupby('warranty_months')['days_since_purchase'].agg(['min', 'median', 'max']))
print("Max days_since_purchase by warranty_months in test:")
print(test.groupby('warranty_months')['days_since_purchase'].agg(['min', 'median', 'max']))

# Are there claims where days_since_purchase > 365 for 12 month products?
train_12m_late = train[(train['warranty_months'] == 12) & (train['days_since_purchase'] > 365)]
print(f"Train 12-month claims with days_since_purchase > 365: {len(train_12m_late)}")
if len(train_12m_late) > 0:
    print("Fraud rate for 12m claims past 365 days:", train_12m_late['is_fraud'].mean())
    print(train_12m_late[['sku', 'days_since_purchase', 'is_fraud']].head())

# B. Negative days since purchase
print("Train negative days_since_purchase:", (train['days_since_purchase'] < 0).sum())
print("Test negative days_since_purchase:", (test['days_since_purchase'] < 0).sum())

# C. Claim amount vs List Price
train['claim_ratio'] = train['claim_amount_inr'] / train['list_price_inr']
test['claim_ratio'] = test['claim_amount_inr'] / test['list_price_inr']
print("Train claim amount > list price:", (train['claim_amount_inr'] > train['list_price_inr']).sum())
print("Test claim amount > list price:", (test['claim_amount_inr'] > test['list_price_inr']).sum())

# D. Claims submitted before partner onboarding
print("Train claims submitted before partner onboarding:", (train['partner_tenure_days'] < 0).sum())
print("Test claims submitted before partner onboarding:", (test['partner_tenure_days'] < 0).sum())

print("\n=== 3. SERIAL NUMBER AUDIT ===")
# Frequency of serials in train
serial_train_counts = train.groupby('product_serial').agg(
    n_claims=('claim_id', 'count'),
    n_fraud=('is_fraud', lambda x: (x == 1).sum()),
    min_date=('submitted_dt', 'min'),
    max_date=('submitted_dt', 'max'),
    n_partners=('partner_id', 'nunique'),
    n_skus=('sku', 'nunique')
)
print("Serials with multiple claims in train:", (serial_train_counts['n_claims'] > 1).sum())
print("Serials with claims across multiple partners in train:", (serial_train_counts['n_partners'] > 1).sum())
print("Serials with claims across multiple SKUs in train:", (serial_train_counts['n_skus'] > 1).sum())

# Serial fraud rate vs repeat count
print("\nFraud rate by serial repeat count in train:")
serial_claim_map = train['product_serial'].value_counts()
train['serial_count_train'] = train['product_serial'].map(serial_claim_map)
print(train.groupby('serial_count_train')['is_fraud'].agg(['count', 'sum', 'mean']))

print("\n=== 4. DISTRIBUTION SHIFTS (TRAIN vs TEST) ===")
print("Product Family distribution:")
p_fam = pd.DataFrame({
    'train_pct': train['family'].value_counts(normalize=True) * 100,
    'test_pct': test['family'].value_counts(normalize=True) * 100
})
print(p_fam)

print("\nPartner Type distribution:")
p_type = pd.DataFrame({
    'train_pct': train['partner_type'].value_counts(normalize=True) * 100,
    'test_pct': test['partner_type'].value_counts(normalize=True) * 100
})
print(p_type)

print("\nPartner Inspected distribution:")
p_insp = pd.DataFrame({
    'train_pct': train['partner_inspected'].value_counts(normalize=True) * 100,
    'test_pct': test['partner_inspected'].value_counts(normalize=True) * 100
})
print(p_insp)

print("\nPhoto Attached distribution:")
p_photo = pd.DataFrame({
    'train_pct': train['photo_attached'].value_counts(normalize=True) * 100,
    'test_pct': test['photo_attached'].value_counts(normalize=True) * 100
})
print(p_photo)

print("\nClaim Amount INR Summary:")
print("Train:", train['claim_amount_inr'].describe())
print("Test:", test['claim_amount_inr'].describe())

print("\nDays Since Purchase Summary:")
print("Train:", train['days_since_purchase'].describe())
print("Test:", test['days_since_purchase'].describe())

print("\nCustomer Prior Claims:")
print("Train:", train['customer_prior_claims'].value_counts(normalize=True).sort_index())
print("Test:", test['customer_prior_claims'].value_counts(normalize=True).sort_index())
