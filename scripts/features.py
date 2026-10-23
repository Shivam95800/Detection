import pandas as pd
import numpy as np
import os

def load_and_preprocess_data(data_dir):
    train = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test = pd.read_csv(os.path.join(data_dir, "test_unlabelled.csv"))
    partners = pd.read_csv(os.path.join(data_dir, "partners.csv"))
    products = pd.read_csv(os.path.join(data_dir, "products.csv"))
    
    # Merge reference tables
    train = train.merge(partners, on='partner_id', how='left').merge(products, on='sku', how='left')
    test = test.merge(partners, on='partner_id', how='left').merge(products, on='sku', how='left')
    
    # Parse dates
    train['submitted_dt'] = pd.to_datetime(train['submitted_at'])
    test['submitted_dt'] = pd.to_datetime(test['submitted_at'])
    train['onboarded_dt'] = pd.to_datetime(train['onboarded_date'])
    test['onboarded_dt'] = pd.to_datetime(test['onboarded_date'])
    
    # Sort chronologically for time-aware computations
    train = train.sort_values('submitted_dt').reset_index(drop=True)
    test = test.sort_values('submitted_dt').reset_index(drop=True)
    
    return train, test, partners, products

def build_features(df, historical_df=None):
    """
    Computes claim-time features.
    If historical_df is provided, historical aggregates (partner fraud rate, serial count)
    are calculated strictly from historical_df prior to the claim date.
    If historical_df is None, an expanding time window is used on df itself.
    """
    feats = pd.DataFrame(index=df.index)
    
    # Basic numeric features
    feats['claim_amount_inr'] = df['claim_amount_inr'].astype(float)
    feats['list_price_inr'] = df['list_price_inr'].astype(float)
    feats['claim_ratio'] = feats['claim_amount_inr'] / feats['list_price_inr']
    feats['days_since_purchase'] = df['days_since_purchase'].astype(float)
    
    warranty_days = df['warranty_months'] * 30.5
    feats['warranty_days'] = warranty_days
    feats['days_to_warranty_expiry'] = warranty_days - feats['days_since_purchase']
    feats['warranty_consumed_pct'] = feats['days_since_purchase'] / warranty_days
    
    feats['customer_prior_claims'] = df['customer_prior_claims'].astype(float)
    feats['photo_attached_bin'] = (df['photo_attached'] == 'Y').astype(float)
    feats['partner_inspected_bin'] = (df['partner_inspected'] == 'Y').astype(float)
    
    # Partner tenure
    tenure_days = (df['submitted_dt'] - df['onboarded_dt']).dt.days.clip(lower=0)
    feats['partner_tenure_days'] = tenure_days.astype(float)
    feats['is_new_partner_6m'] = (tenure_days < 180).astype(float)
    feats['is_new_partner_1y'] = (tenure_days < 365).astype(float)
    
    # Policy regime features
    feats['under_2000'] = (feats['claim_amount_inr'] < 2000).astype(float)
    feats['post_may'] = (df['submitted_dt'] >= '2026-05-01').astype(float)
    feats['auto_approved_uninspected'] = (feats['under_2000'] == 1.0) & (feats['partner_inspected_bin'] == 0.0)
    feats['auto_approved_uninspected'] = feats['auto_approved_uninspected'].astype(float)
    
    # Temporal patterns
    feats['submission_hour'] = df['submitted_dt'].dt.hour.astype(float)
    feats['submission_dayofweek'] = df['submitted_dt'].dt.dayofweek.astype(float)
    feats['is_weekend'] = (feats['submission_dayofweek'] >= 5).astype(float)
    
    # Categoricals
    feats['family'] = df['family'].astype('category')
    feats['partner_type'] = df['partner_type'].astype('category')
    feats['claim_description'] = df['claim_description'].astype('category')
    
    # Time-aware historical features
    # If historical_df is provided (e.g. test set using train data)
    # else expanding window on df
    partner_prior_claims = []
    partner_prior_frauds = []
    serial_prior_counts = []
    
    # Pre-index historical claims by partner and serial
    # We do time-aware lookups
    all_records = []
    if historical_df is not None:
        # historical claims (must have known is_fraud)
        hist_valid = historical_df.dropna(subset=['is_fraud']).copy()
        hist_records = hist_valid[['submitted_dt', 'partner_id', 'product_serial', 'is_fraud']].copy()
    else:
        # df itself
        hist_valid = df.dropna(subset=['is_fraud']).copy() if 'is_fraud' in df.columns else df.copy()
        hist_records = hist_valid[['submitted_dt', 'partner_id', 'product_serial', 'is_fraud']].copy()
    
    # To compute efficiently:
    # Group historical claims by partner_id and sort by date
    # In live service, this is a fast database query or in-memory dict
    print("Computing time-aware partner and serial history...")
    
    return feats

if __name__ == "__main__":
    data_dir = r"C:\Users\Asus\.gemini\antigravity-ide\scratch\kestrel-warranty\data"
    train, test, partners, products = load_and_preprocess_data(data_dir)
    print(f"Loaded train: {train.shape}, test: {test.shape}")
    feats_train = build_features(train)
    print("Features extracted successfully:", feats_train.shape)
    print("Columns:", list(feats_train.columns))
