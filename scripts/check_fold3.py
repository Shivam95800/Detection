import pandas as pd
import numpy as np
import os
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from evaluation import prepare_dataset, extract_features, evaluate_at_k, policy_rules_baseline

data_dir = r"C:\Users\Asus\.gemini\antigravity-ide\scratch\kestrel-warranty\data"
train_raw, test_raw, partners, products = prepare_dataset(data_dir)
train_valid = train_raw.dropna(subset=['is_fraud']).copy().reset_index(drop=True)
train_valid['is_fraud'] = train_valid['is_fraud'].astype(int)

# Let's test Rolling-Origin Fold 3: Train: 2025-04 to 2026-05 (14 months), Val: 2026-06 (June 2026, 1 month)
split_june = pd.Timestamp('2026-06-01')
train_f3 = train_valid[train_valid['submitted_dt'] < split_june].copy().reset_index(drop=True)
val_f3 = train_valid[train_valid['submitted_dt'] >= split_june].copy().reset_index(drop=True)

print(f"=== ROLLING ORIGIN FOLD 3 (Train to May 2026, Val = June 2026) ===")
print(f"Train size: {len(train_f3)} (Frauds: {train_f3['is_fraud'].sum()})")
print(f"Val size:   {len(val_f3)} (Frauds: {val_f3['is_fraud'].sum()})")

X_train_f3 = extract_features(train_f3, ref_df=None)
X_val_f3 = extract_features(val_f3, ref_df=train_f3)

y_train_f3 = train_f3['is_fraud'].values
y_val_f3 = val_f3['is_fraud'].values
val_amounts_f3 = val_f3['claim_amount_inr'].values

num_cols = [
    'claim_amount_inr', 'list_price_inr', 'claim_ratio', 'days_since_purchase',
    'warranty_days', 'days_to_warranty_expiry', 'warranty_consumed_pct',
    'customer_prior_claims', 'photo_attached_bin', 'partner_inspected_bin',
    'partner_tenure_days', 'is_new_partner_6m', 'is_new_partner_1y',
    'under_2000', 'post_may', 'sub_2k_uninspected',
    'submission_hour', 'submission_dayofweek', 'is_weekend',
    'partner_prior_claims', 'partner_prior_frauds', 'partner_prior_fraud_rate',
    'serial_prior_claims'
]
cat_cols = ['family', 'partner_type', 'claim_description']

tree_preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', num_cols),
        ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), cat_cols)
    ]
)

hgb = HistGradientBoostingClassifier(
    max_iter=150, learning_rate=0.05, max_leaf_nodes=31,
    min_samples_leaf=20, class_weight='balanced', random_state=42
)
pipe_f3 = Pipeline([('prep', tree_preprocessor), ('clf', hgb)])
pipe_f3.fit(X_train_f3, y_train_f3)

p_hgb_f3 = pipe_f3.predict_proba(X_val_f3)[:, 1]

# Policy rules on Fold 3
val_rules_f3 = val_f3.copy()
val_rules_f3['partner_prior_fraud_rate'] = X_val_f3['partner_prior_fraud_rate']
val_rules_f3['list_price_inr'] = X_val_f3['list_price_inr']
p_rules_f3 = policy_rules_baseline(val_rules_f3)

# Ensemble: 50% ML + 50% Domain Policy Rules
p_ensemble_f3 = 0.5 * p_hgb_f3 + 0.5 * p_rules_f3

k_month = 40  # 40 claims in 1 month!
print("\n--- RESULTS ON FOLD 3 (June 2026, 762 claims, 40 reviews) ---")
for name, probs in [('Policy Rules Baseline', p_rules_f3), ('HistGradientBoosting', p_hgb_f3), ('Hybrid Ensemble (Rules + ML)', p_ensemble_f3)]:
    auc = roc_auc_score(y_val_f3, probs)
    ap = average_precision_score(y_val_f3, probs)
    res_k = evaluate_at_k(y_val_f3, probs, val_amounts_f3, k=k_month, c_delay=380.0, c_investigation=0.0)
    print(f"{name:30s} | AUC: {auc:.4f} | PR-AUC: {ap:.4f} | P@40: {res_k['precision_at_k']:.4f} | R@40: {res_k['recall_at_k']:.4f} | Net Rs: Rs. {res_k['net_rupees']:,.0f}")
