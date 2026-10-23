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

def run_comprehensive_validation():
    base_dir = r"C:\Users\Asus\.gemini\antigravity-ide\scratch\kestrel-warranty"
    data_dir = os.path.join(base_dir, "data")
    reports_dir = os.path.join(base_dir, "reports")
    fig_dir = os.path.join(reports_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    
    print("=== Loading Data and Setting Up Validation ===")
    train_raw, test_raw, partners, products = prepare_dataset(data_dir)
    
    # Exclude undecided NaNs
    train_valid = train_raw.dropna(subset=['is_fraud']).copy().reset_index(drop=True)
    train_valid['is_fraud'] = train_valid['is_fraud'].astype(int)
    
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
    
    def fit_and_score(train_df, val_df, k_target=None):
        X_tr = extract_features(train_df, ref_df=None)
        X_va = extract_features(val_df, ref_df=train_df)
        y_tr = train_df['is_fraud'].values
        y_va = val_df['is_fraud'].values
        amounts_va = val_df['claim_amount_inr'].values
        
        # Policy rules
        val_rules_df = val_df.copy()
        val_rules_df['partner_prior_fraud_rate'] = X_va['partner_prior_fraud_rate']
        val_rules_df['list_price_inr'] = X_va['list_price_inr']
        p_rules = policy_rules_baseline(val_rules_df)
        
        # HistGradientBoosting
        hgb = HistGradientBoostingClassifier(
            max_iter=150, learning_rate=0.05, max_leaf_nodes=31,
            min_samples_leaf=20, class_weight='balanced', random_state=42
        )
        pipe = Pipeline([('prep', tree_prep), ('clf', hgb)])
        pipe.fit(X_tr, y_tr)
        p_hgb = pipe.predict_proba(X_va)[:, 1]
        
        # Hybrid Ensemble
        p_hybrid = 0.5 * p_hgb + 0.5 * p_rules
        
        if k_target is None:
            # 40 claims per month
            months_count = len(val_df['submitted_dt'].dt.to_period('M').unique())
            k_target = int(40 * months_count)
            
        res = {}
        for m_name, probs in [('Policy Rules Baseline', p_rules), ('HistGradientBoosting', p_hgb), ('Hybrid Ensemble', p_hybrid)]:
            auc = roc_auc_score(y_va, probs)
            ap = average_precision_score(y_va, probs)
            k_metrics = evaluate_at_k(y_va, probs, amounts_va, k=k_target, c_delay=380.0, c_investigation=0.0)
            res[m_name] = {
                'roc_auc': float(auc),
                'pr_auc': float(ap),
                'k': k_target,
                'precision_at_k': k_metrics['precision_at_k'],
                'recall_at_k': k_metrics['recall_at_k'],
                'tp_at_k': k_metrics['tp_at_k'],
                'fp_at_k': k_metrics['fp_at_k'],
                'threshold_at_k': k_metrics['threshold_at_k'],
                'fraud_rupees_saved': k_metrics['fraud_rupees_saved'],
                'goodwill_loss': k_metrics['goodwill_loss'],
                'net_rupees': k_metrics['net_rupees'],
                'probs': probs
            }
        return res, y_va, amounts_va, pipe
        
    print("\n--- 1. ROLLING ORIGIN VALIDATION ---")
    rolling_folds = [
        ("Fold 1 (Pre-May: Dec 2025 - Jan 2026)", '2025-12-01', '2026-02-01', 80),
        ("Fold 2 (Pre-May: Mar 2026 - Apr 2026)", '2026-03-01', '2026-05-01', 80),
        ("Fold 3 (Post-May: June 2026)", '2026-06-01', '2026-07-01', 40),
    ]
    
    rolling_results = {}
    for fold_name, start_val, end_val, k_val in rolling_folds:
        tr = train_valid[train_valid['submitted_dt'] < start_val].copy().reset_index(drop=True)
        va = train_valid[(train_valid['submitted_dt'] >= start_val) & (train_valid['submitted_dt'] < end_val)].copy().reset_index(drop=True)
        
        f_res, _, _, _ = fit_and_score(tr, va, k_target=k_val)
        rolling_results[fold_name] = {m: {k: v for k, v in f_res[m].items() if k != 'probs'} for m in f_res}
        print(f"\n{fold_name}: (Train {len(tr)} rows, Val {len(va)} rows, Val Frauds: {va['is_fraud'].sum()})")
        for m in f_res:
            print(f"  {m:25s} | AUC: {f_res[m]['roc_auc']:.4f} | PR-AUC: {f_res[m]['pr_auc']:.4f} | P@{k_val}: {f_res[m]['precision_at_k']:.4f} | R@{k_val}: {f_res[m]['recall_at_k']:.4f} | Net Rs: Rs. {f_res[m]['net_rupees']:,.0f}")
            
    # Headline Validation: Fold 3 (Most recent post-May month June 2026, directly mimicking production transition)
    # And 3-Month Holdout Split (April - June 2026)
    print("\n--- 2. HEADLINE VALIDATION: RECENT WINDOW (June 2026, Fold 3) ---")
    tr_f3 = train_valid[train_valid['submitted_dt'] < '2026-06-01'].copy().reset_index(drop=True)
    va_f3 = train_valid[train_valid['submitted_dt'] >= '2026-06-01'].copy().reset_index(drop=True)
    f3_res, y_f3, amounts_f3, final_pipe_f3 = fit_and_score(tr_f3, va_f3, k_target=40)
    
    # 1000-iteration Bootstrap CI on Fold 3
    print("\nComputing 1,000-sample Bootstrap Confidence Intervals on Fold 3...")
    boot_ci_f3 = bootstrap_ci(y_f3, f3_res['Hybrid Ensemble']['probs'], amounts_f3, k=40, n_iter=1000, seed=42)
    print(f"Hybrid Ensemble Bootstrap 95% CIs (Fold 3):")
    print(f"  ROC-AUC:       {boot_ci_f3['roc_auc']['mean']:.4f} [{boot_ci_f3['roc_auc']['ci_lower']:.4f}, {boot_ci_f3['roc_auc']['ci_upper']:.4f}]")
    print(f"  PR-AUC:        {boot_ci_f3['pr_auc']['mean']:.4f} [{boot_ci_f3['pr_auc']['ci_lower']:.4f}, {boot_ci_f3['pr_auc']['ci_upper']:.4f}]")
    print(f"  Precision@40:  {boot_ci_f3['precision_at_k']['mean']:.4f} [{boot_ci_f3['precision_at_k']['ci_lower']:.4f}, {boot_ci_f3['precision_at_k']['ci_upper']:.4f}]")
    print(f"  Recall@40:     {boot_ci_f3['recall_at_k']['mean']:.4f} [{boot_ci_f3['recall_at_k']['ci_lower']:.4f}, {boot_ci_f3['recall_at_k']['ci_upper']:.4f}]")
    print(f"  Net Rupees:    Rs. {boot_ci_f3['net_rupees_at_k']['mean']:,.0f} [{boot_ci_f3['net_rupees_at_k']['ci_lower']:,.0f}, {boot_ci_f3['net_rupees_at_k']['ci_upper']:,.0f}]")
    
    # Sensitivity Analysis on Fold 3
    sens_f3 = {}
    for c_inv in [0.0, 182.0, 260.0, 338.0]:
        e = evaluate_at_k(y_f3, f3_res['Hybrid Ensemble']['probs'], amounts_f3, k=40, c_delay=380.0, c_investigation=c_inv)
        sens_f3[f"Inv_Cost_Rs_{int(c_inv)}"] = e['net_rupees']
        
    # --- GENERATE FIGURES ---
    print("\n--- 3. Generating Visual Evidence Charts ---")
    
    # 1. ROC Curves (Fold 3)
    plt.figure(figsize=(8, 6))
    for m in f3_res:
        fpr, tpr, _ = roc_curve(y_f3, f3_res[m]['probs'])
        plt.plot(fpr, tpr, lw=2.5, label=f"{m} (AUC = {f3_res[m]['roc_auc']:.3f})")
    plt.plot([0, 1], [0, 1], 'k--', lw=1, label="Chance")
    plt.xlabel("False Positive Rate", fontsize=11)
    plt.ylabel("True Positive Rate (Recall)", fontsize=11)
    plt.title("ROC Curves — Out-of-Time Validation (June 2026)", fontsize=12, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "roc_curve_comparison.png"), dpi=300)
    plt.close()
    
    # 2. Precision-Recall Curves (Fold 3)
    plt.figure(figsize=(8, 6))
    for m in f3_res:
        p, r, _ = precision_recall_curve(y_f3, f3_res[m]['probs'])
        plt.plot(r, p, lw=2.5, label=f"{m} (PR-AUC = {f3_res[m]['pr_auc']:.3f})")
    plt.axhline(y_f3.mean(), color='k', linestyle='--', lw=1, label=f"Base Rate ({y_f3.mean()*100:.1f}%)")
    plt.xlabel("Recall", fontsize=11)
    plt.ylabel("Precision", fontsize=11)
    plt.title("Precision-Recall Curves — Out-of-Time Validation (June 2026)", fontsize=12, fontweight='bold')
    plt.legend(loc="upper right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "pr_curve_comparison.png"), dpi=300)
    plt.close()
    
    # 3. Calibration Curve
    plt.figure(figsize=(8, 6))
    prob_true, prob_pred = calibration_curve(y_f3, f3_res['Hybrid Ensemble']['probs'], n_bins=10)
    plt.plot(prob_pred, prob_true, "o-", lw=2, color='#2ca02c', label="Hybrid Ensemble (Rules + ML)")
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
        m = compute_rupee_metrics(y_f3, f3_res['Hybrid Ensemble']['probs'], amounts_f3, threshold=th, c_delay=380.0, c_investigation=0.0)
        rupee_curves.append(m)
    r_df = pd.DataFrame(rupee_curves)
    
    plt.figure(figsize=(9, 5.5))
    plt.plot(r_df['threshold'], r_df['net_rupee_impact'], lw=2.5, color='#1f77b4', label="Net Rupee Savings (INR)")
    plt.axvline(f3_res['Hybrid Ensemble']['threshold_at_k'], color='red', linestyle='--', lw=1.5, label=f"Operating Threshold @ k=40 ({f3_res['Hybrid Ensemble']['threshold_at_k']:.3f})")
    plt.xlabel("Probability Decision Threshold", fontsize=11)
    plt.ylabel("Net Benefit (Rs.)", fontsize=11)
    plt.title("Net Business Rupee Impact vs Operating Threshold (June 2026)", fontsize=12, fontweight='bold')
    plt.legend(loc="upper right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "rupee_impact_vs_threshold.png"), dpi=300)
    plt.close()
    
    # 5. Error Analysis by Segments
    val_analysis = va_f3.copy()
    val_analysis['score'] = f3_res['Hybrid Ensemble']['probs']
    order = np.argsort(-val_analysis['score'].values)
    flagged = np.zeros(len(val_analysis), dtype=bool)
    flagged[order[:40]] = True
    val_analysis['flagged'] = flagged
    
    X_val_f3 = extract_features(va_f3, ref_df=tr_f3)
    val_analysis['partner_tenure_days'] = X_val_f3['partner_tenure_days']
    val_analysis['partner_age_band'] = pd.cut(val_analysis['partner_tenure_days'], bins=[-1, 180, 365, 730, 2500], labels=['<6m', '6-12m', '1-2y', '>2y'])
    val_analysis['claim_band'] = pd.cut(val_analysis['claim_amount_inr'], bins=[0, 1000, 2000, 5000, 30000], labels=['<1k', '1k-2k', '2k-5k', '>5k'])
    
    segments = {}
    for seg_col in ['family', 'partner_type', 'partner_age_band', 'claim_band']:
        seg_dict = {}
        for seg_val, grp in val_analysis.groupby(seg_col, observed=False):
            tp = int(((grp['flagged']) & (grp['is_fraud'] == 1)).sum())
            fp = int(((grp['flagged']) & (grp['is_fraud'] == 0)).sum())
            fn = int(((~grp['flagged']) & (grp['is_fraud'] == 1)).sum())
            n_fraud = int(grp['is_fraud'].sum())
            seg_dict[str(seg_val)] = {
                'total_claims': len(grp),
                'fraud_claims': n_fraud,
                'flagged': int(grp['flagged'].sum()),
                'tp': tp,
                'fp': fp,
                'fn': fn,
                'precision': float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0,
                'recall': float(tp / n_fraud) if n_fraud > 0 else 0.0
            }
        segments[seg_col] = seg_dict
        
    # 10 False Positives and 10 False Negatives
    fps = val_analysis[(val_analysis['flagged']) & (val_analysis['is_fraud'] == 0)].sort_values('score', ascending=False).head(10)
    fns = val_analysis[(~val_analysis['flagged']) & (val_analysis['is_fraud'] == 1)].sort_values('score', ascending=True).head(10)
    
    fp_cases = fps[['claim_id', 'submitted_at', 'partner_id', 'sku', 'family', 'claim_amount_inr', 'customer_prior_claims', 'score']].to_dict(orient='records')
    fn_cases = fns[['claim_id', 'submitted_at', 'partner_id', 'sku', 'family', 'claim_amount_inr', 'customer_prior_claims', 'score']].to_dict(orient='records')
    
    # Save Full Metrics JSON
    metrics_payload = {
        'headline_validation': {
            'window': 'June 2026 (Fold 3, Post-May Transition)',
            'train_size': len(tr_f3),
            'val_size': len(va_f3),
            'val_frauds': int(y_f3.sum()),
            'val_base_rate_pct': float(y_f3.mean() * 100),
            'do_nothing_accuracy_pct': float((1 - y_f3.mean()) * 100),
            'k_eval': 40,
            'models': {m: {k: v for k, v in f3_res[m].items() if k != 'probs'} for m in f3_res},
            'bootstrap_ci_95': boot_ci_f3,
            'sensitivity_investigation_cost': sens_f3
        },
        'rolling_folds': rolling_results,
        'segments': segments,
        'fp_cases': fp_cases,
        'fn_cases': fn_cases
    }
    
    with open(os.path.join(reports_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)
        
    print(f"\nWrote comprehensive metrics to {os.path.join(reports_dir, 'metrics.json')}")
    
    # Save Production Pipeline
    # Train full pipeline on ALL 11,814 resolved records in train.csv!
    print("\n--- 4. Training Final Full-Dataset Pipeline ---")
    X_full = extract_features(train_valid, ref_df=None)
    y_full = train_valid['is_fraud'].values
    
    full_hgb = HistGradientBoostingClassifier(
        max_iter=150, learning_rate=0.05, max_leaf_nodes=31,
        min_samples_leaf=20, class_weight='balanced', random_state=42
    )
    full_pipe = Pipeline([('prep', tree_prep), ('clf', full_hgb)])
    full_pipe.fit(X_full, y_full)
    
    full_model_artifact = {
        'pipeline': full_pipe,
        'feature_names': list(X_full.columns),
        'num_cols': num_cols,
        'cat_cols': cat_cols,
        'threshold_at_k40': f3_res['Hybrid Ensemble']['threshold_at_k'],
        'train_records': len(train_valid),
        'train_frauds': int(y_full.sum()),
        'training_window': ("2025-04-01", "2026-06-30")
    }
    joblib.dump(full_model_artifact, os.path.join(base_dir, "model.joblib"))
    print(f"Saved final full-dataset model to {os.path.join(base_dir, 'model.joblib')}")

if __name__ == "__main__":
    run_comprehensive_validation()
