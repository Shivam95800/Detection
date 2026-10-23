# Technical & Operational Decisions Log — Kestrel Home (Variant C)

This document records all ambiguities encountered during analysis and modeling, the decisions taken, and the technical and business rationale.

---

## 1. Handling Undecided Claims in Train (`is_fraud` is NaN)
- **Ambiguity:** `train.csv` contains 215 claims where `is_fraud` is blank / `NaN`. All 215 occur in the CRM era (October 2025 to June 2026).
- **Options Considered:**
  1. Impute as `0.0` (assume genuine unless proven fraudulent).
  2. Impute as `1.0` (assume suspected fraud).
  3. Exclude from training and validation targets.
- **Decision Taken:** Option 3 (Strict exclusion from training and validation targets).
- **Rationale:** Tanmay Kulkarni’s email confirmed: *"Cases still under investigation are blank in the CRM"*. Forcing them to 0 would contaminate the negative class with unconfirmed frauds. Forcing them to 1 would introduce unverified false positives. They are excluded from target calculations while remaining eligible as historical claim volume counters.

---

## 2. Zoho Migration Zero-Coercion (Label Noise)
- **Ambiguity:** Legacy Zoho records (before 1 Oct 2025) have zero `NaN`s because unresolved cases were exported as 0 during database migration.
- **Decision Taken:** Evaluate models primarily on late CRM rolling origin folds (December 2025 through June 2026), where unresolved claims were cleanly preserved as `NaN`s.
- **Rationale:** Early Zoho records suffer from label noise. Training and validating on the late CRM era ensures that model parameters reflect modern operational practices.

---

## 3. Exclusion of `inspector_note` and `source` from Feature Whitelist
- **Ambiguity:** Both `inspector_note` and `source` appear in `train.csv` and `test_unlabelled.csv`.
- **Decision Taken:** Explicitly blacklist both columns from feature extraction.
- **Rationale:** 
  - `inspector_note`: In test, 81.2% of claims are missing this note due to auto-approval. Furthermore, the text consists of generic canned technician notes (e.g., "PCB replaced under warranty", "Customer has bill, serial verified") which appear in identical proportions across fraud and non-fraud claims.
  - `source`: Test data is 100% `crm`. In train, 40.5% is `legacy_zoho`. Including `source` would introduce an unhelpful covariate shift.

---

## 4. Operational Review Budget ($k$) Calibration
- **Ambiguity:** Farhan Sheikh noted the desk can review at most 40 claims per month. Should we evaluate at a fixed threshold (e.g. 0.50) or top-k?
- **Decision Taken:** Use top-k ranking ($k = 40$ reviews/month, or $k = 120$ for a 3-month evaluation window) as the primary decision rule, with an operating threshold $\tau \approx 0.46$.
- **Rationale:** Fixed probability thresholds fail when base rates shift or when capacity is physically bounded. Farhan's team has exactly 40 slots per month. Evaluating at $k=40$ directly optimizes precision and recall within physical constraints.

---

## 5. Marginal Cost of Investigation in Rupee Optimization
- **Ambiguity:** Ops policy §4 states: *"A genuine claim held for review delays the customer's repair; Kestrel issues goodwill worth Rs 380 on average when that happens. A fraudulent claim paid out costs the full claim amount. Blended cost of a service contact: Rs 260."*
- **Decision Taken:** Compute baseline net rupee savings with goodwill loss ($C_{\text{FP}} = ₹380$) and review cost $C_{\text{review}} = ₹0$ (treating the 40-claim desk as existing fixed operational overhead), while providing a complete sensitivity matrix evaluating $C_{\text{review}} = ₹260$ and $\pm 30\%$.
- **Rationale:** The claims investigation desk already exists and has budgeted staff to review up to 40 claims. Reviewing claims within capacity does not incur external marginal contractor fees. However, showing both ₹0 and ₹260 ensures complete financial transparency.

---

## 6. Hybrid Ensemble Architecture (50% ML + 50% Domain Policy Rules)
- **Ambiguity:** Pure gradient boosting trained on pre-May data inverted the relationship on claim amount due to the May 1 auto-approval change.
- **Decision Taken:** Construct an equal-weight hybrid ensemble combining `HistGradientBoosting` with a transparent, calibrated domain policy rule score.
- **Rationale:** As proven in `scripts/check_fold3.py`, the hybrid ensemble achieved higher PR-AUC (0.4660 vs 0.3136 for pure ML and 0.1816 for pure rules) and higher Precision@40 (37.5% vs 25.0% for ML and 15.0% for rules), delivering +₹11,512/month in net savings.
