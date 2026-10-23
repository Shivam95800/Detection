import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, roc_curve, brier_score_loss, confusion_matrix
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from evaluation import (
    prepare_dataset, extract_features, compute_rupee_metrics,
    evaluate_at_k, bootstrap_ci, policy_rules_baseline
)

def run_experiment():
    base_dir = r"C:\Users\Asus\.gemini\antigravity-ide\scratch\kestrel-warranty"
    data_dir = os.path.join(base_dir, "data")
    reports_dir = os.path.join(base_dir, "reports")
    fig_dir = os.path.join(reports_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    
    print("--- 1. Loading and Preparing Data ---")
    train_raw, test_raw, partners, products = prepare_dataset(data_dir)
    
    # Exclude undecided NaNs from training/validation
    train_valid = train_raw.dropna(subset=['is_fraud']).copy().reset_index(drop=True)
    train_valid['is_fraud'] = train_valid['is_fraud'].astype(int)
    print(f"Total resolved training records: {len(train_valid)} (Fraud: {train_valid['is_fraud'].sum()}, Non-Fraud: {(train_valid['is_fraud']==0).sum()})")
    
    # Define primary time-based split: Train: < 2026-04-01 (12 months), Val: >= 2026-04-01 (3 months)
    split_date = pd.Timestamp('2026-04-01')
    train_split = train_valid[train_valid['submitted_dt'] < split_date].copy().reset_index(drop=True)
    val_split = train_valid[train_valid['submitted_dt'] >= split_date].copy().reset_index(drop=True)
    
    print(f"\n--- 2. Time-Based Holdout Split ---")
    print(f"Train window: {train_split['submitted_dt'].min().date()} to {train_split['submitted_dt'].max().date()} ({len(train_split)} rows, {train_split['is_fraud'].sum()} frauds, base rate: {train_split['is_fraud'].mean()*100:.2f}%)")
    print(f"Val window:   {val_split['submitted_dt'].min().date()} to {val_split['submitted_dt'].max().date()} ({len(val_split)} rows, {val_split['is_fraud'].sum()} frauds, base rate: {val_split['is_fraud'].mean()*100:.2f}%)")
    
    # Extract features strictly time-awarely
    print("\nExtracting time-aware features for train split...")
    X_train_df = extract_features(train_split, ref_df=None)
    print("Extracting time-aware features for val split (using train_split as historical reference)...")
    X_val_df = extract_features(val_split, ref_df=train_split)
    
    y_train = train_split['is_fraud'].values
    y_val = val_split['is_fraud'].values
    claim_amounts_val = val_split['claim_amount_inr'].values
    
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
    
    # Preprocessor for Linear Models
    linear_preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_cols)
        ]
    )
    
    # Preprocessor for Tree Models
    tree_preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', num_cols),
            ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), cat_cols)
        ]
    )
    
    # Baseline A: Policy Rules Baseline
    print("\nEvaluating Baseline A: Policy Rules...")
    val_rules_df = val_split.copy()
    val_rules_df['partner_prior_fraud_rate'] = X_val_df['partner_prior_fraud_rate']
    val_rules_df['list_price_inr'] = X_val_df['list_price_inr']
    y_pred_prob_rules = policy_rules_baseline(val_rules_df)
    
    # Baseline B: Logistic Regression (Balanced)
    print("Training Baseline B: Logistic Regression...")
    lr_pipe = Pipeline([
        ('prep', linear_preprocessor),
        ('clf', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42, C=0.5))
    ])
    lr_pipe.fit(X_train_df, y_train)
    y_pred_prob_lr = lr_pipe.predict_proba(X_val_df)[:, 1]
    
    # Model C: HistGradientBoostingClassifier (Class Weighted)
    print("Training Model C: HistGradientBoosting...")
    hgb_clf = HistGradientBoostingClassifier(
        max_iter=150,
        learning_rate=0.05,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        class_weight='balanced',
        random_state=42
    )
    hgb_pipe = Pipeline([
        ('prep', tree_preprocessor),
        ('clf', hgb_clf)
    ])
    hgb_pipe.fit(X_train_df, y_train)
    y_pred_prob_hgb = hgb_pipe.predict_proba(X_val_df)[:, 1]
    
    # Model D: Calibrated HistGradientBoosting
    print("Calibrating Model D (Sigmoid Calibration)...")
    calibrated_hgb = CalibratedClassifierCV(estimator=hgb_pipe, method='sigmoid', cv=3)
    calibrated_hgb.fit(X_train_df, y_train)
    y_pred_prob_cal_hgb = calibrated_hgb.predict_proba(X_val_df)[:, 1]
    
    models = {
        'Policy Rules Baseline': y_pred_prob_rules,
        'Logistic Regression (Balanced)': y_pred_prob_lr,
        'HistGradientBoosting (Balanced)': y_pred_prob_hgb,
        'Calibrated HistGB': y_pred_prob_cal_hgb
    }
    
    # Evaluate All Models at k = 120 (40 claims/month * 3 months)
    k_eval = 120
    results_summary = {}
    
    print(f"\n{'='*75}")
    print(f"{'Model':32s} | {'ROC-AUC':7s} | {'PR-AUC':7s} | {'P@120':7s} | {'R@120':7s} | {'Net Rs. @ 120':15s}")
    print(f"{'='*75}")
    
    for name, probs in models.items():
        auc = roc_auc_score(y_val, probs)
        ap = average_precision_score(y_val, probs)
        k_res = evaluate_at_k(y_val, probs, claim_amounts_val, k=k_eval, c_delay=380.0, c_investigation=0.0)
        
        results_summary[name] = {
            'roc_auc': float(auc),
            'pr_auc': float(ap),
            'precision_at_k': k_res['precision_at_k'],
            'recall_at_k': k_res['recall_at_k'],
            'tp_at_k': k_res['tp_at_k'],
            'fp_at_k': k_res['fp_at_k'],
            'threshold_at_k': k_res['threshold_at_k'],
            'fraud_rupees_saved': k_res['fraud_rupees_saved'],
            'goodwill_loss': k_res['goodwill_loss'],
            'net_rupees': k_res['net_rupees']
        }
        print(f"{name:32s} | {auc:.4f}  | {ap:.4f}  | {k_res['precision_at_k']:.4f}  | {k_res['recall_at_k']:.4f}  | Rs. {k_res['net_rupees']:,.0f}")
    print(f"{'='*75}")
    
    # Compute Bootstrap Confidence Intervals for the Best Model (Calibrated HistGB)
    print("\nComputing 1,000-sample Bootstrap Confidence Intervals for Calibrated HistGB...")
    boot_ci = bootstrap_ci(y_val, y_pred_prob_cal_hgb, claim_amounts_val, k=k_eval, n_iter=1000, seed=42)
    print(f"ROC-AUC:       {boot_ci['roc_auc']['mean']:.4f} (95% CI: [{boot_ci['roc_auc']['ci_lower']:.4f}, {boot_ci['roc_auc']['ci_upper']:.4f}])")
    print(f"PR-AUC:        {boot_ci['pr_auc']['mean']:.4f} (95% CI: [{boot_ci['pr_auc']['ci_lower']:.4f}, {boot_ci['pr_auc']['ci_upper']:.4f}])")
    print(f"Precision@120: {boot_ci['precision_at_k']['mean']:.4f} (95% CI: [{boot_ci['precision_at_k']['ci_lower']:.4f}, {boot_ci['precision_at_k']['ci_upper']:.4f}])")
    print(f"Recall@120:    {boot_ci['recall_at_k']['mean']:.4f} (95% CI: [{boot_ci['recall_at_k']['ci_lower']:.4f}, {boot_ci['recall_at_k']['ci_upper']:.4f}])")
    print(f"Net Rupees:    Rs. {boot_ci['net_rupees_at_k']['mean']:,.0f} (95% CI: [Rs. {boot_ci['net_rupees_at_k']['ci_lower']:,.0f}, Rs. {boot_ci['net_rupees_at_k']['ci_upper']:,.0f}])")
    
    # Save Model
    model_path = os.path.join(base_dir, "model.joblib")
    joblib.dump({
        'pipeline': calibrated_hgb,
        'feature_names': list(X_train_df.columns),
        'num_cols': num_cols,
        'cat_cols': cat_cols,
        'threshold_at_k120': results_summary['Calibrated HistGB']['threshold_at_k'],
        'train_date_range': (str(train_split['submitted_dt'].min().date()), str(train_split['submitted_dt'].max().date())),
        'val_date_range': (str(val_split['submitted_dt'].min().date()), str(val_split['submitted_dt'].max().date()))
    }, model_path)
    print(f"\nSaved calibrated model artifact to: {model_path}")
    
    # --- GENERATE FIGURES ---
    print("\nGenerating report charts and curves...")
    # 1. ROC Curves
    plt.figure(figsize=(8, 6))
    for name, probs in models.items():
        fpr, tpr, _ = roc_curve(y_val, probs)
        plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {results_summary[name]['roc_auc']:.3f})")
    plt.plot([0, 1], [0, 1], 'k--', lw=1, label="Chance")
    plt.xlabel("False Positive Rate", fontsize=11)
    plt.ylabel("True Positive Rate (Recall)", fontsize=11)
    plt.title("ROC Curves on Time-Based Holdout Split (2026-04 to 2026-06)", fontsize=12, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "roc_curve_comparison.png"), dpi=300)
    plt.close()
    
    # 2. Precision-Recall Curves
    plt.figure(figsize=(8, 6))
    for name, probs in models.items():
        p, r, _ = precision_recall_curve(y_val, probs)
        plt.plot(r, p, lw=2, label=f"{name} (PR-AUC = {results_summary[name]['pr_auc']:.3f})")
    plt.axhline(y_val.mean(), color='k', linestyle='--', lw=1, label=f"Base Rate ({y_val.mean()*100:.1f}%)")
    plt.xlabel("Recall", fontsize=11)
    plt.ylabel("Precision", fontsize=11)
    plt.title("Precision-Recall Curves (Time-Based Holdout)", fontsize=12, fontweight='bold')
    plt.legend(loc="upper right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "pr_curve_comparison.png"), dpi=300)
    plt.close()
    
    # 3. Calibration Curve
    plt.figure(figsize=(8, 6))
    prob_true_uncal, prob_pred_uncal = calibration_curve(y_val, y_pred_prob_hgb, n_bins=10)
    prob_true_cal, prob_pred_cal = calibration_curve(y_val, y_pred_prob_cal_hgb, n_bins=10)
    plt.plot(prob_pred_uncal, prob_true_uncal, "s-", label="HistGB (Uncalibrated)")
    plt.plot(prob_pred_cal, prob_true_cal, "o-", label="HistGB (Calibrated - Sigmoid)")
    plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    plt.xlabel("Predicted Probability", fontsize=11)
    plt.ylabel("Observed Fraud Fraction", fontsize=11)
    plt.title("Reliability Diagram / Calibration Curve", fontsize=12, fontweight='bold')
    plt.legend(loc="upper left")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "calibration_curve.png"), dpi=300)
    plt.close()
    
    # 4. Rupee Impact vs Threshold Curve
    thresholds = np.linspace(0.01, 0.95, 100)
    rupee_curves = []
    for th in thresholds:
        m = compute_rupee_metrics(y_val, y_pred_prob_cal_hgb, claim_amounts_val, threshold=th, c_delay=380.0, c_investigation=0.0)
        rupee_curves.append(m)
    r_df = pd.DataFrame(rupee_curves)
    
    plt.figure(figsize=(9, 5.5))
    plt.plot(r_df['threshold'], r_df['net_rupee_impact'] / 1e5, lw=2.5, color='#1f77b4', label="Net Rupee Savings (Lakhs INR)")
    plt.axvline(results_summary['Calibrated HistGB']['threshold_at_k'], color='red', linestyle='--', lw=1.5, label=f"Operating Threshold @ k=120 ({results_summary['Calibrated HistGB']['threshold_at_k']:.3f})")
    plt.xlabel("Probability Decision Threshold", fontsize=11)
    plt.ylabel("Net Benefit (Lakhs Rs.)", fontsize=11)
    plt.title("Net Business Rupee Impact vs Operating Threshold", fontsize=12, fontweight='bold')
    plt.legend(loc="upper right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "rupee_impact_vs_threshold.png"), dpi=300)
    plt.close()
    
    # 5. Sensitivity Analysis
    sens_results = {}
    inv_scenarios = [0.0, 182.0, 260.0, 338.0]
    for c_inv in inv_scenarios:
        eval_c = evaluate_at_k(y_val, y_pred_prob_cal_hgb, claim_amounts_val, k=k_eval, c_delay=380.0, c_investigation=c_inv)
        sens_results[f"Inv_Cost_Rs_{int(c_inv)}"] = eval_c['net_rupees']
        
    # Error analysis by segment
    val_analysis = val_split.copy()
    val_analysis['score'] = y_pred_prob_cal_hgb
    val_analysis['flagged'] = (val_analysis['score'] >= results_summary['Calibrated HistGB']['threshold_at_k'])
    val_analysis['tenure_days'] = X_val_df['partner_tenure_days']
    val_analysis['partner_age_band'] = pd.cut(val_analysis['tenure_days'], bins=[-1, 180, 365, 730, 2000], labels=['<6m', '6-12m', '1-2y', '>2y'])
    val_analysis['claim_band'] = pd.cut(val_analysis['claim_amount_inr'], bins=[0, 1000, 2000, 5000, 30000], labels=['<1k', '1k-2k', '2k-5k', '>5k'])
    val_analysis['month'] = val_analysis['submitted_dt'].dt.to_period('M').astype(str)
    
    segments = {}
    for seg_col in ['family', 'partner_type', 'partner_age_band', 'claim_band', 'month']:
        seg_dict = {}
        for seg_val, grp in val_analysis.groupby(seg_col, observed=False):
            tp = int(((grp['flagged']) & (grp['is_fraud'] == 1)).sum())
            fp = int(((grp['flagged']) & (grp['is_fraud'] == 0)).sum())
            fn = int(((~grp['flagged']) & (grp['is_fraud'] == 1)).sum())
            seg_dict[str(seg_val)] = {
                'total': len(grp),
                'fraud_actual': int(grp['is_fraud'].sum()),
                'flagged': int(grp['flagged'].sum()),
                'tp': tp,
                'fp': fp,
                'fn': fn,
                'precision': float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0,
                'recall': float(tp / grp['is_fraud'].sum()) if grp['is_fraud'].sum() > 0 else 0.0
            }
        segments[seg_col] = seg_dict
        
    # Extract 10 False Positives and 10 False Negatives
    fps = val_analysis[(val_analysis['flagged']) & (val_analysis['is_fraud'] == 0)].sort_values('score', ascending=False).head(10)
    fns = val_analysis[(~val_analysis['flagged']) & (val_analysis['is_fraud'] == 1)].sort_values('score', ascending=True).head(10)
    
    fp_cases = fps[['claim_id', 'submitted_at', 'partner_id', 'sku', 'family', 'claim_amount_inr', 'customer_prior_claims', 'score']].to_dict(orient='records')
    fn_cases = fns[['claim_id', 'submitted_at', 'partner_id', 'sku', 'family', 'claim_amount_inr', 'customer_prior_claims', 'score']].to_dict(orient='records')
    
    # Save complete metrics.json
    metrics_payload = {
        'split_info': {
            'train_range': [str(train_split['submitted_dt'].min().date()), str(train_split['submitted_dt'].max().date())],
            'val_range': [str(val_split['submitted_dt'].min().date()), str(val_split['submitted_dt'].max().date())],
            'train_size': len(train_split),
            'val_size': len(val_split),
            'val_frauds': int(y_val.sum()),
            'val_base_rate_pct': float(y_val.mean() * 100),
            'do_nothing_accuracy_pct': float((1 - y_val.mean()) * 100)
        },
        'models_comparison': results_summary,
        'best_model_name': 'Calibrated HistGB',
        'bootstrap_ci_95': boot_ci,
        'sensitivity_investigation_cost': sens_results,
        'segments': segments,
        'fp_cases': fp_cases,
        'fn_cases': fn_cases
    }
    
    with open(os.path.join(reports_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)
        
    print(f"\nSuccessfully wrote comprehensive metrics to {os.path.join(reports_dir, 'metrics.json')}")

if __name__ == "__main__":
    run_experiment()
