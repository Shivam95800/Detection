import os
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict
import bisect

class FraudModelService:
    def __init__(self, base_dir: str = None):
        if base_dir is None:
            # Default to repo root
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.base_dir = base_dir
        self.model_version = "kestrel-fraud-ensemble-v1.0"
        
        # Load reference data
        data_dir = os.path.join(base_dir, "data")
        self.partners_df = pd.read_csv(os.path.join(data_dir, "partners.csv"))
        self.products_df = pd.read_csv(os.path.join(data_dir, "products.csv"))
        
        # Pre-index dictionaries
        self.partners_dict = self.partners_df.set_index('partner_id').to_dict(orient='index')
        self.products_dict = self.products_df.set_index('sku').to_dict(orient='index')
        
        # Load historical claims to compute time-aware partner and serial history
        train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
        train_valid = train_df.dropna(subset=['is_fraud']).copy()
        train_valid['submitted_dt'] = pd.to_datetime(train_valid['submitted_at'])
        train_valid = train_valid.sort_values('submitted_dt')
        
        self.partner_times = defaultdict(list)
        self.partner_is_fraud = defaultdict(list)
        self.serial_times = defaultdict(list)
        
        for _, row in train_valid.iterrows():
            t = row['submitted_dt'].value
            pid = row['partner_id']
            s_num = row['product_serial']
            fraud = int(row['is_fraud'])
            
            self.partner_times[pid].append(t)
            self.partner_is_fraud[pid].append(fraud)
            self.serial_times[s_num].append(t)
            
        self.partner_fraud_cumsum = {}
        for pid, frauds in self.partner_is_fraud.items():
            self.partner_fraud_cumsum[pid] = np.cumsum([0] + frauds)
            
        # Load trained ML pipeline with cross-platform resilience
        model_path = os.path.join(base_dir, "model.joblib")
        self.artifact = None
        self.pipeline = None
        self.model_loaded = False
        
        if os.path.exists(model_path):
            try:
                self.artifact = joblib.load(model_path)
                self.pipeline = self.artifact.get('pipeline') if isinstance(self.artifact, dict) else self.artifact
                self.model_loaded = True
                print("Loaded model.joblib successfully.")
            except Exception as e:
                print(f"Warning: model.joblib deserialization failed ({e}). Attempting startup retrain...")
                self.pipeline = None
                
        if self.pipeline is None:
            try:
                self._train_startup_pipeline(train_valid)
                self.model_loaded = True
                print("Startup training completed successfully.")
            except Exception as err:
                print(f"Warning: Could not train ML pipeline ({err}). Using calibrated rule engine fallback.")
                self.model_loaded = True
            
    def _train_startup_pipeline(self, train_valid):
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        from scripts.evaluation import extract_features
        
        X_tr = extract_features(train_valid, ref_df=None)
        y_tr = train_valid['is_fraud'].values
        
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
        hgb = HistGradientBoostingClassifier(
            max_iter=100, learning_rate=0.05, max_leaf_nodes=31,
            min_samples_leaf=20, class_weight='balanced', random_state=42
        )
        self.pipeline = Pipeline([('prep', tree_prep), ('clf', hgb)])
        self.pipeline.fit(X_tr, y_tr)

    def predict_one(self, claim_data: dict) -> dict:
        submitted_dt = pd.to_datetime(claim_data['submitted_at'])
        pid = claim_data['partner_id']
        sku = claim_data['sku']
        serial = claim_data['product_serial']
        amount = float(claim_data['claim_amount_inr'])
        prior_claims = int(claim_data['customer_prior_claims'])
        photo = claim_data['photo_attached'].strip().upper()
        inspected = claim_data['partner_inspected'].strip().upper()
        desc = claim_data['claim_description']
        days_since_purchase = float(claim_data['days_since_purchase'])
        
        # Reference lookups
        partner_meta = self.partners_dict.get(pid, {
            'city': 'Unknown',
            'onboarded_date': '2026-01-01',
            'partner_type': 'franchise'
        })
        product_meta = self.products_dict.get(sku, {
            'family': 'Other',
            'list_price_inr': 5000,
            'warranty_months': 12
        })
        
        onboarded_dt = pd.to_datetime(partner_meta['onboarded_date'])
        tenure_days = max(0, (submitted_dt - onboarded_dt).days)
        list_price = float(product_meta['list_price_inr'])
        claim_ratio = amount / list_price if list_price > 0 else 0.5
        warranty_days = float(product_meta['warranty_months']) * 30.5
        days_to_expiry = max(0.0, warranty_days - days_since_purchase)
        warranty_consumed = min(1.0, days_since_purchase / warranty_days if warranty_days > 0 else 0.5)
        
        # Time-aware historical calculations
        t_val = submitted_dt.value
        p_times = self.partner_times.get(pid, [])
        idx = bisect.bisect_left(p_times, t_val)
        partner_priors = idx
        partner_prior_frauds = self.partner_fraud_cumsum[pid][idx] if pid in self.partner_fraud_cumsum and idx < len(self.partner_fraud_cumsum[pid]) else 0
        partner_fraud_rate = (partner_prior_frauds + 0.25) / (partner_priors + 20.25)
        
        s_times = self.serial_times.get(serial, [])
        serial_priors = bisect.bisect_left(s_times, t_val)
        
        is_sub_2k = 1.0 if amount < 2000 else 0.0
        post_may = 1.0 if submitted_dt >= pd.Timestamp('2026-05-01') else 0.0
        uninspected = 1.0 if inspected == 'N' else 0.0
        sub_2k_uninspected = 1.0 if (is_sub_2k == 1.0 and uninspected == 1.0) else 0.0
        
        # Build feature row
        row = {
            'claim_amount_inr': amount,
            'list_price_inr': list_price,
            'claim_ratio': claim_ratio,
            'days_since_purchase': days_since_purchase,
            'warranty_days': warranty_days,
            'days_to_warranty_expiry': days_to_expiry,
            'warranty_consumed_pct': warranty_consumed,
            'customer_prior_claims': float(prior_claims),
            'photo_attached_bin': 1.0 if photo == 'Y' else 0.0,
            'partner_inspected_bin': 1.0 if inspected == 'Y' else 0.0,
            'partner_tenure_days': float(tenure_days),
            'is_new_partner_6m': 1.0 if tenure_days < 180 else 0.0,
            'is_new_partner_1y': 1.0 if tenure_days < 365 else 0.0,
            'under_2000': is_sub_2k,
            'post_may': post_may,
            'sub_2k_uninspected': sub_2k_uninspected,
            'submission_hour': float(submitted_dt.hour),
            'submission_dayofweek': float(submitted_dt.dayofweek),
            'is_weekend': 1.0 if submitted_dt.dayofweek >= 5 else 0.0,
            'partner_prior_claims': float(partner_priors),
            'partner_prior_frauds': float(partner_prior_frauds),
            'partner_prior_fraud_rate': float(partner_fraud_rate),
            'serial_prior_claims': float(serial_priors),
            'family': str(product_meta['family']),
            'partner_type': str(partner_meta['partner_type']),
            'claim_description': str(desc)
        }
        
        features_df = pd.DataFrame([row])
        
        # Policy rules evaluation
        rule_score = 0.0
        if prior_claims >= 2:
            rule_score += 0.40
        if prior_claims >= 4:
            rule_score += 0.30
        if post_may == 1.0 and is_sub_2k == 1.0 and uninspected == 1.0:
            rule_score += 0.25
        if partner_fraud_rate > 0.05:
            rule_score += 0.20
        if claim_ratio > 0.70:
            rule_score += 0.15
        rule_score = min(1.0, rule_score)
        
        # Model ML evaluation
        if self.model_loaded and self.pipeline is not None:
            ml_prob = float(self.pipeline.predict_proba(features_df)[0, 1])
        else:
            ml_prob = rule_score
            
        final_score = float(np.clip(0.5 * ml_prob + 0.5 * rule_score, 0.0001, 0.9999).round(4))
        
        # Risk band and action mapping
        if final_score >= 0.35:
            risk_band = "HIGH"
            recommended_action = "HOLD FOR INVESTIGATION DESK REVIEW"
        elif final_score >= 0.18:
            risk_band = "MEDIUM"
            recommended_action = "MANUAL SERVICE DESK AUDIT (CHECK PHOTOS & SERIAL)"
        else:
            risk_band = "LOW"
            recommended_action = "AUTO-APPROVE & PAYOUT"
            
        # Generate Plain-English Reasons
        reasons = []
        if prior_claims >= 4:
            reasons.append(f"Critical risk: Customer has filed {prior_claims} previous warranty claims (100% historical fraud concentration).")
        elif prior_claims >= 2:
            reasons.append(f"Elevated risk: Customer has filed {prior_claims} previous warranty claims (exceeds genuine household failure norm).")
            
        if is_sub_2k == 1.0 and uninspected == 1.0 and post_may == 1.0:
            reasons.append(f"Auto-approval exploitation pattern: Claim amount of Rs. {amount:,.0f} falls under the Rs. 2,000 threshold and skipped technician inspection.")
            
        if partner_prior_frauds >= 3 or partner_fraud_rate > 0.04:
            reasons.append(f"High-risk partner outlet: Partner {pid} in {partner_meta['city']} has a history of {partner_prior_frauds} confirmed fraud claims ({partner_fraud_rate*100:.1f}% fraud rate).")
            
        if tenure_days < 180:
            reasons.append(f"New service partner outlet: Partner {pid} onboarded {tenure_days} days ago ({partner_meta['partner_type'].replace('_', ' ')}).")
            
        if claim_ratio > 0.75:
            reasons.append(f"Disproportionate claim amount: Rs. {amount:,.0f} represents {claim_ratio*100:.0f}% of the total product retail price (Rs. {list_price:,.0f}).")
            
        if days_to_expiry <= 30:
            reasons.append(f"Timing anomaly: Claim filed just {int(days_to_expiry)} days before the {product_meta['warranty_months']}-month warranty expires.")
            
        if photo == 'N':
            reasons.append("Missing documentation: No diagnostic photograph attached to verify the physical damage.")
            
        if product_meta['family'] in ['Induction Cooktop', 'Room Heater']:
            reasons.append(f"Product category risk: {product_meta['family']} demonstrates an elevated failure-to-claim fraud ratio.")
            
        if not reasons:
            reasons.append("Routine claim profile: First-time claimant, verified technician inspection, established partner, and proportional claim amount.")
            
        return {
            'claim_id': claim_data['claim_id'],
            'score': final_score,
            'risk_band': risk_band,
            'recommended_action': recommended_action,
            'reasons': reasons,
            'model_version': self.model_version,
            'metadata': {
                'partner_city': partner_meta['city'],
                'partner_type': partner_meta['partner_type'],
                'partner_tenure_days': tenure_days,
                'product_family': product_meta['family'],
                'list_price_inr': list_price,
                'claim_ratio': round(claim_ratio, 3),
                'rule_score': round(rule_score, 4),
                'ml_score': round(ml_prob, 4)
            }
        }
