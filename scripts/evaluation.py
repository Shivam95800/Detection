import os
import json
import joblib
import bisect
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, roc_curve, brier_score_loss, confusion_matrix

def prepare_dataset(data_dir):
    train = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test = pd.read_csv(os.path.join(data_dir, "test_unlabelled.csv"))
    partners = pd.read_csv(os.path.join(data_dir, "partners.csv"))
    products = pd.read_csv(os.path.join(data_dir, "products.csv"))
    
    train = train.merge(partners, on='partner_id', how='left').merge(products, on='sku', how='left')
    test = test.merge(partners, on='partner_id', how='left').merge(products, on='sku', how='left')
    
    train['submitted_dt'] = pd.to_datetime(train['submitted_at'])
    test['submitted_dt'] = pd.to_datetime(test['submitted_at'])
    train['onboarded_dt'] = pd.to_datetime(train['onboarded_date'])
    test['onboarded_dt'] = pd.to_datetime(test['onboarded_date'])
    
    train = train.sort_values('submitted_dt').reset_index(drop=True)
    test = test.sort_values('submitted_dt').reset_index(drop=True)
    
    return train, test, partners, products

def compute_historical_features(df, ref_df=None):
    if ref_df is not None:
        source_df = ref_df.dropna(subset=['is_fraud']).copy()
    else:
        source_df = df.dropna(subset=['is_fraud']).copy() if 'is_fraud' in df.columns else df.copy()

    partner_times = defaultdict(list)
    partner_is_fraud = defaultdict(list)
    serial_times = defaultdict(list)
    
    for _, row in source_df.sort_values('submitted_dt').iterrows():
        t = row['submitted_dt'].value
        pid = row['partner_id']
        s_num = row['product_serial']
        fraud = row['is_fraud'] if 'is_fraud' in row and pd.notnull(row['is_fraud']) else 0
        
        partner_times[pid].append(t)
        partner_is_fraud[pid].append(fraud)
        serial_times[s_num].append(t)
    
    partner_fraud_cumsum = {}
    for pid, frauds in partner_is_fraud.items():
        partner_fraud_cumsum[pid] = np.cumsum([0] + frauds)
    
    prior_claims = []
    prior_frauds = []
    prior_fraud_rates = []
    serial_priors = []
    
    alpha = 0.25
    beta = 20.0
    
    for _, row in df.iterrows():
        t = row['submitted_dt'].value
        pid = row['partner_id']
        s_num = row['product_serial']
        
        p_list = partner_times.get(pid, [])
        idx = bisect.bisect_left(p_list, t)
        
        n_p_claims = idx
        n_p_fraud = partner_fraud_cumsum[pid][idx] if pid in partner_fraud_cumsum else 0
        rate = (n_p_fraud + alpha) / (n_p_claims + alpha + beta)
        
        s_list = serial_times.get(s_num, [])
        s_idx = bisect.bisect_left(s_list, t)
        
        prior_claims.append(n_p_claims)
        prior_frauds.append(n_p_fraud)
        prior_fraud_rates.append(rate)
        serial_priors.append(s_idx)
        
    return pd.DataFrame({
        'partner_prior_claims': prior_claims,
        'partner_prior_frauds': prior_frauds,
        'partner_prior_fraud_rate': prior_fraud_rates,
        'serial_prior_claims': serial_priors
    }, index=df.index)

def extract_features(df, ref_df=None):
    hist_feats = compute_historical_features(df, ref_df)
    
    feats = pd.DataFrame(index=df.index)
    feats['claim_amount_inr'] = df['claim_amount_inr'].astype(float)
    feats['list_price_inr'] = df['list_price_inr'].astype(float)
    feats['claim_ratio'] = feats['claim_amount_inr'] / feats['list_price_inr']
    feats['days_since_purchase'] = df['days_since_purchase'].astype(float)
    
    warranty_days = df['warranty_months'] * 30.5
    feats['warranty_days'] = warranty_days
    feats['days_to_warranty_expiry'] = (warranty_days - feats['days_since_purchase']).clip(lower=0)
    feats['warranty_consumed_pct'] = (feats['days_since_purchase'] / warranty_days).clip(upper=1.0)
    
    feats['customer_prior_claims'] = df['customer_prior_claims'].astype(float)
    feats['photo_attached_bin'] = (df['photo_attached'] == 'Y').astype(float)
    feats['partner_inspected_bin'] = (df['partner_inspected'] == 'Y').astype(float)
    
    tenure_days = (df['submitted_dt'] - df['onboarded_dt']).dt.days.clip(lower=0)
    feats['partner_tenure_days'] = tenure_days.astype(float)
    feats['is_new_partner_6m'] = (tenure_days < 180).astype(float)
    feats['is_new_partner_1y'] = (tenure_days < 365).astype(float)
    
    feats['under_2000'] = (feats['claim_amount_inr'] < 2000).astype(float)
    feats['post_may'] = (df['submitted_dt'] >= '2026-05-01').astype(float)
    feats['sub_2k_uninspected'] = ((feats['under_2000'] == 1.0) & (feats['partner_inspected_bin'] == 0.0)).astype(float)
    
    feats['submission_hour'] = df['submitted_dt'].dt.hour.astype(float)
    feats['submission_dayofweek'] = df['submitted_dt'].dt.dayofweek.astype(float)
    feats['is_weekend'] = (feats['submission_dayofweek'] >= 5).astype(float)
    
    feats['family'] = df['family'].astype(str)
    feats['partner_type'] = df['partner_type'].astype(str)
    feats['claim_description'] = df['claim_description'].astype(str)
    
    for c in hist_feats.columns:
        feats[c] = hist_feats[c]
        
    return feats

def compute_rupee_metrics(y_true, y_prob, claim_amounts, threshold=0.5, c_delay=380.0, c_investigation=0.0):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    claim_amounts = np.asarray(claim_amounts)
    
    flagged = (y_prob >= threshold)
    tp = (flagged & (y_true == 1))
    fp = (flagged & (y_true == 0))
    fn = ((~flagged) & (y_true == 1))
    tn = ((~flagged) & (y_true == 0))
    
    fraud_rupees_prevented = claim_amounts[tp].sum()
    fraud_rupees_leaked = claim_amounts[fn].sum()
    goodwill_cost = fp.sum() * c_delay
    investigation_cost = flagged.sum() * c_investigation
    net_rupee_impact = fraud_rupees_prevented - goodwill_cost - investigation_cost
    
    return {
        'threshold': float(threshold),
        'flagged_count': int(flagged.sum()),
        'tp_count': int(tp.sum()),
        'fp_count': int(fp.sum()),
        'fn_count': int(fn.sum()),
        'tn_count': int(tn.sum()),
        'precision': float(tp.sum() / flagged.sum()) if flagged.sum() > 0 else 0.0,
        'recall': float(tp.sum() / y_true.sum()) if y_true.sum() > 0 else 0.0,
        'fraud_rupees_prevented': float(fraud_rupees_prevented),
        'fraud_rupees_leaked': float(fraud_rupees_leaked),
        'goodwill_cost': float(goodwill_cost),
        'investigation_cost': float(investigation_cost),
        'net_rupee_impact': float(net_rupee_impact)
    }

def evaluate_at_k(y_true, y_prob, claim_amounts, k=120, c_delay=380.0, c_investigation=0.0):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    claim_amounts = np.asarray(claim_amounts)
    
    order = np.argsort(-y_prob)
    top_k_indices = order[:k]
    
    flagged = np.zeros(len(y_true), dtype=bool)
    flagged[top_k_indices] = True
    
    tp = (flagged & (y_true == 1))
    fp = (flagged & (y_true == 0))
    
    precision_k = tp.sum() / k if k > 0 else 0.0
    recall_k = tp.sum() / y_true.sum() if y_true.sum() > 0 else 0.0
    fraud_rupees_saved = claim_amounts[tp].sum()
    goodwill_loss = fp.sum() * c_delay
    inv_cost = k * c_investigation
    net_rupees = fraud_rupees_saved - goodwill_loss - inv_cost
    
    return {
        'k': int(k),
        'precision_at_k': float(precision_k),
        'recall_at_k': float(recall_k),
        'tp_at_k': int(tp.sum()),
        'fp_at_k': int(fp.sum()),
        'threshold_at_k': float(y_prob[order[k-1]]) if k > 0 and len(order) >= k else 1.0,
        'fraud_rupees_saved': float(fraud_rupees_saved),
        'goodwill_loss': float(goodwill_loss),
        'net_rupees': float(net_rupees)
    }

def bootstrap_ci(y_true, y_prob, claim_amounts, k=120, n_iter=1000, seed=42):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    claim_amounts = np.asarray(claim_amounts)
    
    np.random.seed(seed)
    n = len(y_true)
    auc_list = []
    ap_list = []
    prec_k_list = []
    rec_k_list = []
    rupee_k_list = []
    
    for _ in range(n_iter):
        idx = np.random.choice(n, size=n, replace=True)
        yt_boot = y_true[idx]
        if len(np.unique(yt_boot)) < 2:
            continue
        yp_boot = y_prob[idx]
        ca_boot = claim_amounts[idx]
        
        auc_list.append(roc_auc_score(yt_boot, yp_boot))
        ap_list.append(average_precision_score(yt_boot, yp_boot))
        
        eval_k = evaluate_at_k(yt_boot, yp_boot, ca_boot, k=k)
        prec_k_list.append(eval_k['precision_at_k'])
        rec_k_list.append(eval_k['recall_at_k'])
        rupee_k_list.append(eval_k['net_rupees'])
        
    def get_ci(arr):
        return {
            'mean': float(np.mean(arr)),
            'std': float(np.std(arr)),
            'ci_lower': float(np.percentile(arr, 2.5)),
            'ci_upper': float(np.percentile(arr, 97.5))
        }
        
    return {
        'roc_auc': get_ci(auc_list),
        'pr_auc': get_ci(ap_list),
        'precision_at_k': get_ci(prec_k_list),
        'recall_at_k': get_ci(rec_k_list),
        'net_rupees_at_k': get_ci(rupee_k_list)
    }

def policy_rules_baseline(df):
    score = np.zeros(len(df), dtype=float)
    score += (df['customer_prior_claims'] >= 2).astype(float) * 0.40
    score += (df['customer_prior_claims'] >= 4).astype(float) * 0.30
    
    post_may = (pd.to_datetime(df['submitted_at']) >= '2026-05-01')
    under_2k = (df['claim_amount_inr'] < 2000)
    uninspected = (df['partner_inspected'] == 'N')
    score += (post_may & under_2k & uninspected).astype(float) * 0.25
    
    if 'partner_prior_fraud_rate' in df.columns:
        score += (df['partner_prior_fraud_rate'] > 0.05).astype(float) * 0.20
    
    if 'list_price_inr' in df.columns:
        ratio = df['claim_amount_inr'] / df['list_price_inr']
        score += (ratio > 0.70).astype(float) * 0.15
        
    return np.clip(score, 0.0, 1.0)
