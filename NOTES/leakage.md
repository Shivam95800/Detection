# Data Leakage and Target Contamination Audit — Kestrel Home

## 1. Executive Summary

In fraud detection and warranty review systems, leakage occurs when information unavailable at the moment of claim submission is used to predict the claim's validity. If a model relies on post-submission artifacts, investigator notes, or system migration quirks, it will demonstrate artificially inflated validation scores but fail in live deployment.

This document identifies all potential leakage vectors, explains why each represents a risk, and establishes strict feature-eligibility boundaries.

---

## 2. Column-by-Column Leakage Analysis

| Column | Present in Train? | Present in Test? | Available at Claim Time? | Verdict / Action | Technical Rationale |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`claim_id`** | Yes | Yes | Yes (identifier) | **Exclude from features** | Arbitrary sequential identifier (`WC700000` to `WC713600`). In train, 681 claim IDs are duplicated due to partner re-submissions. Must not be used as a numerical or categorical predictor. |
| **`submitted_at`** | Yes | Yes | Yes | **Transform only** | Raw timestamp is non-stationary. Allowed to extract time-aware features: day of week, hour of day, elapsed days since partner onboarding, and indicator of post-May 2026 policy change. |
| **`partner_id`** | Yes | Yes | Yes | **Transform with time awareness** | Partner ID is known at submission. However, raw partner target encoding using full train data would leak future fraud outcomes. Any partner-level fraud aggregation MUST use expanding window or historical claims prior to the claim timestamp. |
| **`sku`** | Yes | Yes | Yes | **Allowed** | Known at purchase and submission. |
| **`product_serial`** | Yes | Yes | Yes | **Transform carefully** | As noted by IT Admin Tanmay Kulkarni, serials are manually typed by service partners. Audit shows 1,987 serials appear across multiple different SKUs (e.g., generic placeholders like sequential digits). Cannot be used as unique product IDs; repeat serial frequency can be tracked historically. |
| **`days_since_purchase`** | Yes | Yes | Yes | **Allowed** | Elapsed days from sale to failure. |
| **`claim_amount_inr`** | Yes | Yes | Yes | **Allowed** | Total amount claimed. Interacts strongly with the ₹2,000 auto-approval threshold. |
| **`photo_attached`** | Yes | Yes | Yes | **Allowed** | Binary flag indicating if evidence was provided at submission. |
| **`partner_inspected`** | Yes | Yes | Policy-Dependent | **Handle with caution / Include with interaction** | Sign-off by partner. Crucially, before 1 May 2026, 83.7% of claims were inspected. After 1 May 2026, all claims under ₹2,000 are auto-approved without inspection, dropping inspection rates to 21.3% in test. This is an operational policy shift rather than true fraud signal. |
| **`claim_description`** | Yes | Yes | Yes | **Allowed** | Fault reason chosen from 16 standard options at submission. |
| **`inspector_note`** | Yes (73.1%) | Yes (18.8%) | No / Post-Submission Sign-off | **EXCLUDE FROM FEATURES** | Contains canned operational notes entered by partner technicians after physical bench inspection (e.g., "PCB replaced under warranty", "Customer has bill, serial verified"). Missing in 81.2% of test claims due to auto-approval. Text distributions between fraud and non-fraud are virtually indistinguishable. Using presence/absence would leak the post-May policy shift. |
| **`customer_prior_claims`**| Yes | Yes | Yes | **Allowed** | Historical claims count tracked by CRM at submission time. Extremely strong, legitimate fraud signal. |
| **`source`** | Yes (`crm` / `legacy_zoho`) | Yes (100% `crm`) | System Flag | **EXCLUDE FROM FEATURES** | In train, 4,875 claims are `legacy_zoho` and 7,154 are `crm`. In test, 100% of claims are `crm`. Furthermore, Zoho legacy migration forced unresolved claims to 0. Using `source` introduces severe train-test covariate shift and migration noise. |
| **`is_fraud`** | Yes (145 ones, 11,669 zeros, 215 NaNs) | No | No (Target) | **TARGET ONLY** | Ground truth investigation outcome. All rows with `NaN` (undecided claims) must be excluded from model training and validation targets. |

---

## 3. Policy & System Contamination Vectors

### A. The Zoho Zero-Coercion Trap (Label Noise)
- **Mechanism:** Tanmay Kulkarni confirmed: *"Cases still under investigation are blank in the CRM - but Zoho couldn't store blanks, so the old ones came across as 0."*
- **Impact:** In `legacy_zoho` (April to September 2025), there are 0 undecided claims. Claims that were unresolved or disputed at migration were falsely marked as genuine (`0.0`).
- **Mitigation:** The model must be evaluated on later out-of-time CRM periods (rolling origin validation from October 2025 to June 2026) where true labels and unresolved `NaN`s are cleanly distinguished.

### B. Undecided Claims in CRM (Label Censoring)
- **Mechanism:** In CRM (October 2025 to June 2026), 215 claims are `NaN` (undecided at export).
- **Risk:** Treating `NaN` as 0 (genuine) would contaminate the negative class with unconfirmed frauds. Treating `NaN` as 1 would assume guilt without investigation.
- **Mitigation:** Exclude `NaN` rows from training labels. Do not impute.

### C. Future Information in Partner Performance Metrics
- **Mechanism:** If we calculate `partner_fraud_rate = train.groupby('partner_id')['is_fraud'].mean()` across all 15 months, a claim submitted in May 2025 would be scored using fraud committed in June 2026!
- **Mitigation:** Strict time-aware feature engineering. For any claim at timestamp $T$, partner statistics must only aggregate claims where $\text{submitted\_at} < T$. In out-of-fold cross-validation, partner aggregates must only use training folds.

---

## 4. Final Feature Whitelist

Only the following raw and derived features are permitted for modeling:
1. `claim_amount_inr`
2. `claim_to_list_price_ratio` (`claim_amount_inr / list_price_inr`)
3. `days_since_purchase`
4. `days_to_warranty_expiry` (`warranty_months * 30.5 - days_since_purchase`)
5. `warranty_consumed_pct` (`days_since_purchase / (warranty_months * 30.5)`)
6. `customer_prior_claims`
7. `photo_attached_bin` (`1` if 'Y' else `0`)
8. `partner_tenure_days` (`(submitted_at - onboarded_date).days`)
9. `is_new_partner_6m` (`partner_tenure_days < 180`)
10. `is_new_partner_1y` (`partner_tenure_days < 365`)
11. `under_2000_threshold` (`claim_amount_inr < 2000`)
12. `post_may_policy_regime` (`submitted_at >= '2026-05-01'`)
13. `auto_approved_risk_interaction` (`under_2000 * post_may_policy_regime`)
14. `partner_type_cat` (One-hot encoded)
15. `product_family_cat` (One-hot encoded)
16. `claim_description_cat` (Categorical frequency / target encoded strictly out-of-time)
17. `time_aware_partner_prior_claims` (Count of prior claims filed by partner before $T$)
18. `time_aware_partner_prior_fraud_rate` (Smoothed empirical Bayes prior fraud rate of partner before $T$)
19. `serial_repeat_prior_count` (Prior occurrences of this serial before $T$)
20. `submission_hour` & `submission_dayofweek` (Temporal filing patterns)
