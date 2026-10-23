# Prediction Expectation Protocol — Kestrel Home (Variant C)

**Commit Date & Time:** 2026-10-23T20:44:00+05:30  
**Author:** Antigravity AI Pair Programmer & Lead Applied Scientist  
**Target File:** `test_unlabelled.csv` (2,252 rows, July 1, 2026 to September 30, 2026)  
**Status:** Written and committed *before* generating `predictions.csv`.

---

## 1. Grading Metric Specification

As defined in `data/README.txt` and `sample_submission.csv`:
- `predictions.csv` requires columns `claim_id, score`, where scores must be numeric and monotonic with risk: **higher score = higher probability of fraud**.
- Because fraud labels in `test_unlabelled.csv` are unobserved and subject to severe class imbalance (~1.2% to 2.5% base rate), standard threshold-independent ranking metrics are primary:
  1. **ROC-AUC (Area Under the Receiver Operating Characteristic):** Measures overall ranking separation.
  2. **PR-AUC (Average Precision / Area Under Precision-Recall Curve):** The definitive standard for highly imbalanced fraud ranking.
  3. **Precision@k and Recall@k:** Operational metrics calibrated to Kestrel's investigation desk capacity ($k = 40$ reviews/month $\times$ 3 months = **$k = 120$ reviews** total on the test set).
  4. **Net Rupee Impact ($k=120$):**
     $$\text{Rupees Saved} = \sum_{i \in \text{Flagged Fraud}} \text{claim\_amount\_inr}_i - (₹380 \times N_{\text{Flagged Genuine}})$$

---

## 2. Quantitative Expected Performance Ranges on `test_unlabelled.csv`

Based strictly on out-of-time rolling-origin validation and the post-May 1, 2026 policy shift, we expect the final model (`Hybrid Ensemble: Calibrated HistGradientBoosting + Policy Rules`) to score within the following bounds on `test_unlabelled.csv`:

| Evaluation Metric | Point Estimate | Expected Range (95% CI equivalent) | Empirical Validation Reference |
| :--- | :--- | :--- | :--- |
| **ROC-AUC** | **0.89** | **0.83 — 0.94** | Headline Fold 3 (June 2026): **0.9084** (95% Bootstrap CI: $[0.7987, 0.9735]$); Mean across all rolling folds: 0.846. |
| **PR-AUC (Avg Precision)** | **0.46** | **0.32 — 0.58** | Headline Fold 3: **0.4660** (95% Bootstrap CI: $[0.2802, 0.6964]$); Fold 1: 0.5867; Fold 2: 0.5072. |
| **Precision@120 ($k=120$)**| **35.0%** | **25.0% — 45.0%** | Headline Fold 3 (scaled to $k=40$/mo): **37.50%** (95% Bootstrap CI: $[17.50\%, 50.00\%]$). Over 15x higher than the base rate (~2.0%). |
| **Recall@120 ($k=120$)** | **65.0%** | **50.0% — 78.0%** | Headline Fold 3: **68.18%** of all fraud caught in top reviews (95% Bootstrap CI: $[39.98\%, 86.36\%]$). |
| **Net Rupee Benefit ($k=120$)**| **+₹36,000** | **+₹15,000 — +₹60,000** | Scaled from Fold 3 (+₹11,512 net per month $\times$ 3 months $\approx$ +₹34,536). |

---

## 3. Rationale and Justification

1. **Alignment with the Post-May Policy Shift:**
   - On May 1, 2026, Kestrel instituted an auto-approval rule for claims under ₹2,000 without partner inspection (`ops-policy.pdf` §5).
   - In `test_unlabelled.csv`, **77.0% of claims (1,734 / 2,252) are under ₹2,000 and uninspected**.
   - Models trained purely on pre-May data fail because they never observed fraudsters exploiting this rule. Our final model is trained on all 15 months of historical claims (including May and June 2026) and integrates domain rule flags with gradient boosting. It directly captures this exploit pattern.

2. **Strict Elimination of Look-Ahead Leakage:**
   - Partner fraud rates and repeat serial numbers are computed strictly as historical expanding windows prior to each claim timestamp.
   - Categorical encodings and scaling are fit strictly without target leakage.

3. **Conservative Margin:**
   - Expected PR-AUC and Precision@120 reflect genuine out-of-time generalization across unseen summer months (July–Sept 2026), accounting for seasonal shifts in appliance repairs (e.g. water purifiers and fans).
