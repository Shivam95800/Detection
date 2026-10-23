# Model Card — Kestrel Home Warranty Fraud Review (Variant C)

## Model Overview
- **Model Name:** Kestrel Warranty Claim Review Hybrid Ensemble
- **Model Version:** `kestrel-fraud-ensemble-v1.0`
- **Model Architecture:** Blended ensemble of `HistGradientBoostingClassifier` (50%) and Calibrated Domain Policy Rules (50%).
- **Primary Task:** Binary classification and risk-ranking of incoming warranty claims at submission time (`1` = fraudulent, `0` = genuine).
- **Target Audience:** D2C Operations and Claims Investigation Desk at Kestrel Home Appliances (Pune, India).

## Training Data & Horizons
- **Source:** Historical warranty claims database migrated from `legacy_zoho` and `crm` (`train.csv`).
- **Training Window:** 2025-04-01 to 2026-06-30 (15 continuous months).
- **Total Training Records:** 11,814 resolved claims (145 confirmed frauds, 11,669 genuine claims).
- **Data Exclusion:** 215 unresolved claims (`NaN` in `is_fraud`) representing active investigations in CRM were excluded from training targets to avoid label noise.
- **Base Rate:** 1.227% across the full 15-month dataset (rising to 2.87% in June 2026 post-May policy change).

## Feature Engineering & Whitelist
All features are strictly computed at claim submission time without look-ahead target leakage:
1. `claim_amount_inr`: Claimed reimbursement value.
2. `list_price_inr`: Retail product catalog price from `products.csv`.
3. `claim_ratio`: Ratio of claim amount to list price (`claim_amount_inr / list_price_inr`).
4. `days_since_purchase`: Elapsed days from sale to failure.
5. `warranty_days`: Standard warranty duration in days (`warranty_months * 30.5`).
6. `days_to_warranty_expiry`: Remaining warranty validity (`warranty_days - days_since_purchase`).
7. `warranty_consumed_pct`: Fraction of warranty life consumed.
8. `customer_prior_claims`: Number of previous claims filed by the same customer (ranges 0 to 6).
9. `photo_attached_bin`: Binary indicator (1 if 'Y', 0 if 'N').
10. `partner_inspected_bin`: Partner sign-off indicator (1 if 'Y', 0 if 'N').
11. `partner_tenure_days`: Operating tenure of partner at submission date.
12. `is_new_partner_6m`: Binary flag (1 if tenure < 180 days).
13. `is_new_partner_1y`: Binary flag (1 if tenure < 365 days).
14. `under_2000`: Binary indicator of claim value < ₹2,000 (the auto-approval threshold).
15. `post_may`: Binary flag for submission date $\ge$ 2026-05-01.
16. `sub_2k_uninspected`: Interaction flag for uninspected sub-₹2,000 claims post-May.
17. `submission_hour`, `submission_dayofweek`, `is_weekend`: Temporal filing features.
18. `partner_prior_claims`: Historical claim volume filed by partner strictly prior to current claim timestamp.
19. `partner_prior_frauds`: Confirmed frauds for this partner strictly prior to current claim timestamp.
20. `partner_prior_fraud_rate`: Empirical Bayes smoothed historical fraud rate.
21. `serial_prior_claims`: Repeat claim occurrences for the typed serial number prior to timestamp.
22. Categoricals (one-hot encoded): `family` (7 product lines), `partner_type` (3 types), `claim_description` (16 categories).

## Excluded Leakage Columns
- `inspector_note`: Excluded. Contains post-inspection technician notes; missing in 81.2% of test claims due to auto-approval.
- `source`: Excluded. Identifies platform migration origin (`legacy_zoho` vs `crm`). Not present in test (100% CRM).
- `claim_id`: Excluded. Sequential ID containing duplicate re-submission artifacts.

## Out-of-Time Validation Metrics (Headline: June 2026 Post-May Window, $N=767$, $k=40$)
- **Operating Capacity:** $k = 40$ reviews per month (5.2% review capacity).
- **ROC-AUC:** **0.9084** (95% Bootstrap CI: $[0.7987, 0.9735]$)
- **PR-AUC:** **0.4660** (95% Bootstrap CI: $[0.2802, 0.6964]$)
- **Precision@40:** **37.50%** (15 TP / 40 flagged; 95% Bootstrap CI: $[17.50\%, 50.00\%]$)
- **Recall@40:** **68.18%** (15 of 22 total frauds stopped; 95% Bootstrap CI: $[39.98\%, 86.36\%]$)
- **Net Rupee Impact:** **+₹11,512 / month** (95% Bootstrap CI: $[-₹2,462, +₹20,175]$)
- **Naive Baseline Accuracy:** 97.13% (achieved by approving everything and catching ₹0 fraud).

## Operational Limits & Known Failure Modes
1. **Uninspected High-Value Flagship Claims:** Sophisticated frauds submitted on expensive products (e.g. Robot Vacuums > ₹15,000) with first-time customer accounts and forged partner sign-offs can slip below the triage threshold.
2. **First-Time Rogue Partners:** If a brand-new partner commits fraud on their very first claim, historical partner fraud features will be at baseline prior levels ($\sim 1.2\%$). The model will rely entirely on claim-level characteristics (customer prior claims, uninspected sub-₹2k flag, category).
3. **Typo Serial Numbers:** Service partners type serial numbers manually, creating high false-positive matching on generic sequences (e.g., repeating digits). Serial features are strictly treated as weak secondary signals.
