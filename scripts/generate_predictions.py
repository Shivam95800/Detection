import os
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from evaluation import prepare_dataset, extract_features, policy_rules_baseline

def generate_predictions():
    base_dir = r"C:\Users\Asus\.gemini\antigravity-ide\scratch\kestrel-warranty"
    data_dir = os.path.join(base_dir, "data")
    
    print("Loading data...")
    train_raw, test_raw, partners, products = prepare_dataset(data_dir)
    
    # Exclude unresolved NaNs from training set
    train_valid = train_raw.dropna(subset=['is_fraud']).copy().reset_index(drop=True)
    train_valid['is_fraud'] = train_valid['is_fraud'].astype(int)
    
    sample_sub = pd.read_csv(os.path.join(data_dir, "sample_submission.csv"))
    print(f"Sample submission shape: {sample_sub.shape}")
    
    # Feature columns
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
    
    tree_prep = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', num_cols),
            ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), cat_cols)
        ]
    )
    
    print("Extracting features for full train set...")
    X_train_full = extract_features(train_valid, ref_df=None)
    y_train_full = train_valid['is_fraud'].values
    
    print("Extracting features for unlabelled test set (using full train as historical reference)...")
    X_test = extract_features(test_raw, ref_df=train_valid)
    
    print("Training final HistGradientBoosting model on all 11,814 resolved training claims...")
    hgb = HistGradientBoostingClassifier(
        max_iter=150, learning_rate=0.05, max_leaf_nodes=31,
        min_samples_leaf=20, class_weight='balanced', random_state=42
    )
    pipe = Pipeline([('prep', tree_prep), ('clf', hgb)])
    pipe.fit(X_train_full, y_train_full)
    
    print("Generating ML probability predictions on test...")
    p_ml = pipe.predict_proba(X_test)[:, 1]
    
    print("Generating domain policy rules predictions on test...")
    test_rules_df = test_raw.copy()
    test_rules_df['partner_prior_fraud_rate'] = X_test['partner_prior_fraud_rate']
    test_rules_df['list_price_inr'] = X_test['list_price_inr']
    p_rules = policy_rules_baseline(test_rules_df)
    
    # Hybrid ensemble: 50% ML + 50% Domain Policy Rules
    p_final = 0.5 * p_ml + 0.5 * p_rules
    
    # Clip and round to 4 decimal places for clean formatting
    p_final = np.clip(p_final, 0.0001, 0.9999).round(4)
    
    # Build predictions dataframe
    pred_df = pd.DataFrame({
        'claim_id': test_raw['claim_id'],
        'score': p_final
    })
    
    # Align order exactly with sample_submission.csv
    pred_df = sample_sub[['claim_id']].merge(pred_df, on='claim_id', how='left')
    
    out_path = os.path.join(base_dir, "predictions.csv")
    pred_df.to_csv(out_path, index=False)
    print(f"\nWrote predictions to: {out_path}")
    print(f"Predictions shape: {pred_df.shape}")
    print("Sample predictions:")
    print(pred_df.head(10))
    print("\nScore summary statistics:")
    print(pred_df['score'].describe())

if __name__ == "__main__":
    generate_predictions()
