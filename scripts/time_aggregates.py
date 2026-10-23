import pandas as pd
import numpy as np
import os
import bisect
from collections import defaultdict

def compute_time_aware_features(df, reference_df=None):
    """
    Computes time-aware historical aggregates:
    - partner_prior_claims: number of claims filed by this partner strictly BEFORE current submitted_dt
    - partner_prior_frauds: number of fraud claims filed by this partner strictly BEFORE current submitted_dt
    - partner_prior_fraud_rate: smoothed Bayesian prior fraud rate
    - serial_prior_claims: number of claims with the same serial strictly BEFORE current submitted_dt
    
    If reference_df is provided (e.g. for test set or validation fold), historical events
    are drawn from reference_df. If reference_df is None, an expanding window over df is used.
    """
    if reference_df is not None:
        source_df = reference_df.dropna(subset=['is_fraud']).copy()
    else:
        source_df = df.dropna(subset=['is_fraud']).copy() if 'is_fraud' in df.columns else df.copy()

    # Build sorted timestamp arrays per partner and per serial
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
    
    # Pre-calculate prefix sums of frauds per partner
    partner_fraud_cumsum = {}
    for pid, frauds in partner_is_fraud.items():
        partner_fraud_cumsum[pid] = np.cumsum([0] + frauds)  # 0 at index 0
    
    prior_claims = []
    prior_frauds = []
    prior_fraud_rates = []
    serial_priors = []
    
    # Prior hyperparameters (Empirical Bayes prior: mean = 0.0123, strength = 20)
    alpha = 0.25
    beta = 20.0
    
    for _, row in df.iterrows():
        t = row['submitted_dt'].value
        pid = row['partner_id']
        s_num = row['product_serial']
        
        # Binary search for events strictly before t
        p_list = partner_times.get(pid, [])
        idx = bisect.bisect_left(p_list, t)
        
        # In self-expanding window mode where row is in source_df:
        # bisect_left finds count of events strictly earlier than t.
        n_p_claims = idx
        if pid in partner_fraud_cumsum:
            n_p_fraud = partner_fraud_cumsum[pid][idx]
        else:
            n_p_fraud = 0
            
        rate = (n_p_fraud + alpha) / (n_p_claims + alpha + beta)
        
        s_list = serial_times.get(s_num, [])
        s_idx = bisect.bisect_left(s_list, t)
        
        prior_claims.append(n_p_claims)
        prior_frauds.append(n_p_fraud)
        prior_fraud_rates.append(rate)
        serial_priors.append(s_idx)
        
    res = pd.DataFrame({
        'partner_prior_claims': prior_claims,
        'partner_prior_frauds': prior_frauds,
        'partner_prior_fraud_rate': prior_fraud_rates,
        'serial_prior_claims': serial_priors
    }, index=df.index)
    
    return res

if __name__ == "__main__":
    data_dir = r"C:\Users\Asus\.gemini\antigravity-ide\scratch\kestrel-warranty\data"
    from features import load_and_preprocess_data
    train, test, partners, products = load_and_preprocess_data(data_dir)
    time_feats = compute_time_aware_features(train)
    print("Computed time-aware features for train:")
    print(time_feats.describe())
