# Comprehensive Data Audit — Kestrel Home (Variant C)

## 1. Ground Truth Base Rate & The "Do-Nothing" Accuracy Trap

In `train.csv`:
- Total rows: **12,029**
- Resolved non-fraud (`0.0`): **11,669**
- Resolved fraud (`1.0`): **145**
- Undecided / open (`NaN`): **215**
- **Observed Fraud Base Rate (on resolved claims):** **1.227%** (145 / 11,814).

### The Mathematical Flaw of the 97% Accuracy Target
Client Ritu Deshpande noted:
> *"The KPI is accuracy, the board wants a number above 97%, and I'd like it live soon."*

Consider a trivial "Do-Nothing" baseline model that automatically approves every claim without flagging a single one:
$$\text{Accuracy}_{\text{Do-Nothing}} = \frac{11,669 \text{ true negatives} + 0 \text{ true positives}}{11,814 \text{ total claims}} = \mathbf{98.77\%}$$

A model that catches **zero fraud** and prevents **zero rupees in losses** already achieves **98.77% accuracy**—well above the Board's 97% threshold. Conversely, an aggressive fraud model that flags 5% of claims and catches 80% of fraud would flag ~600 claims (mostly false positives given the 1.23% base rate), lowering raw accuracy to ~95%, despite saving lakhs of rupees for the business!

**Conclusion for Operations:** Accuracy is a mathematically deceptive KPI in imbalanced fraud environments. The primary operational objectives must be:
1. **Precision@k and Recall@k** calibrated to the investigation desk's 40 claims/month capacity.
2. **Net Rupee Impact** ($\text{Rupees Saved} - \text{False Positive Goodwill Costs} - \text{Investigation Costs}$).
3. **PR-AUC and ROC-AUC** for ranking quality.

---

## 2. Data Leakage and Target Contamination

As detailed in `NOTES/leakage.md`:
1. **`inspector_note`:** Post-inspection sign-off field. Missing in 81.2% of test claims due to auto-approval. Contains generic templated phrases. **Excluded**.
2. **`source`:** Platform flag (`legacy_zoho` vs `crm`). 100% of test records are `crm`. Zoho records suffered zero-coercion during migration. **Excluded**.
3. **`is_fraud` NaNs (215 rows):** Represents active CRM investigations. Coercing to 0 would contaminate the negative set; coercing to 1 would introduce unverified false positives. **Excluded from training targets**.
4. **Partner Historical Aggregates:** Full-dataset target encoding would leak future outcomes. All partner historical rates must be computed strictly with past claims before the timestamp of each claim.

---

## 3. Data Integrity, Duplicates, and Joins

### A. Identifier Overlap & Duplicate Claims
- **Train vs Test Overlap:** Exactly **0** overlapping `claim_id`s between `train.csv` and `test_unlabelled.csv`.
- **Duplicates in Train:** 1,362 rows contain duplicated `claim_id`s (681 pairs).
  - Matches Tanmay Kulkarni's explanation: *"Partners re-submit claims when they're bounced, so the same claim number can show up twice."*
  - Re-submissions occur 8.4 to 120 hours apart (median: 72.0 hours).
  - Ground truth labels are 100% consistent across all duplicate pairs (664 pairs are both 0.0, 4 pairs are both 1.0, 13 pairs are unresolved NaNs).
- **Duplicates in Test:** Exactly **0** duplicate `claim_id`s in `test_unlabelled.csv`.

### B. Foreign Key Joins
- **`partners.csv`:** 380 unique partners. All 12,029 train rows and 2,252 test rows join with **0 unmatched keys**.
- **`products.csv`:** 21 unique SKUs across 7 families. All train and test rows join with **0 unmatched keys**.

### C. Policy Rule Violations Check
- **Warranty Expiry:** Standard warranties are 12 months (max allowed 365 days) or 24 months (max allowed 730 days). In train, max `days_since_purchase` is 359 for 12m products and 719 for 24m products. Zero claims violate warranty duration.
- **Purchase Date Anachronism:** Zero claims have `days_since_purchase < 0`.
- **Claim Value vs List Price:** All claims are less than 100% of list price (max observed ratio: 0.998).
- **Partner Onboarding vs Claim Date:** Zero claims occur before partner onboarding date.

### D. Product Serial Number Messiness
- 8,835 unique serial numbers across 12,029 train rows.
- 2,051 serials appear across multiple distinct service partners.
- 1,987 serials appear across multiple completely different product SKUs (e.g., generic numeric sequences typed in by technicians).
- Serial repeat frequency does **not** correlate positively with fraud (repeat serial fraud rate is 1.05% vs 1.38% for single-occurrence serials). Serials cannot be treated as reliable global hardware identifiers.

---

## 4. Distribution Shifts: Train vs Test

| Metric / Feature | Train (`2025-04` to `2026-06`) | Test (`2026-07` to `2026-09`) | Operational Cause & Impact |
| :--- | :--- | :--- | :--- |
| **Source Platform** | 40.5% `legacy_zoho`, 59.5% `crm` | 100.0% `crm` | Zoho retired 30 Sep 2025. |
| **Partner Inspected** | **83.7% 'Y'**, 16.3% 'N' | **21.3% 'Y'**, **78.7% 'N'** | **Massive Shift:** Policy change on 1 May 2026 auto-approved all claims under ₹2,000 without inspection. |
| **Inspector Note Populated** | 73.1% populated | 18.8% populated | Uninspected claims do not generate inspector notes. |
| **Claims Under ₹2,000** | 62.0% of claims | 77.0% of claims | Volume of small claims expanded rapidly post-May 1. |
| **Mean Claim Amount** | ₹2,620 (median ₹1,437) | ₹2,156 (median ₹1,396) | Higher proportion of sub-₹2,000 claims in test period. |
| **Product Families** | Even split (~14% each) | Even split (~13% - 16% each) | Stable product catalog. |
| **Partner Types** | ASC: 45.9%, Fran: 34.3%, Freelance: 19.8% | ASC: 48.1%, Fran: 33.1%, Freelance: 18.8% | Partner mix remains stable. |

---

## 5. Statistical Test of Ritu's Hypothesis: "Are Newer Partners the Problem?"

### Client Belief
> *"I'm fairly sure it's the newer partners, but let's have the data say it."* — Ritu Deshpande

### Methodology
- Partner tenure is defined strictly at **claim submission time**:
  $$\text{tenure\_days} = (\text{submitted\_at} - \text{onboarded\_date}).\text{days}$$
- We test tenure both continuously and using operational cutoffs (< 6 months, < 1 year).
- We test both bivariate relationships (Chi-squared test, 1,000-sample bootstrap confidence intervals) and multivariate logistic regression controlling for claim value, product family, customer prior claims, partner type, and the post-May auto-approval policy shift.

### Findings

#### 1. Bivariate / Raw Breakdown
- **Tenure < 6 Months:**
  - Total claims: 731 | Fraud claims: 28 | **Fraud Rate: 3.83%**
- **Tenure $\ge$ 6 Months:**
  - Total claims: 11,083 | Fraud claims: 117 | **Fraud Rate: 1.06%**
- **Difference:** $+2.78\%$ higher fraud in new partners ($\chi^2 = 41.29$, $p = 1.31 \times 10^{-10}$).
- **Bootstrap 95% Confidence Interval for Difference:** **$[+1.45\%, +4.25\%]$**.

#### 2. The Granular Reality: Partner Concentration & Simpson's Paradox
While the raw difference is statistically significant, looking at partner-level concentration reveals that partner age is **not** the underlying root cause:
- Of 366 partners who submitted claims in train, **315 partners (86.1%) have ZERO fraud claims**.
- Only **51 partners (13.9%)** have ever submitted a fraudulent claim.
- **Top 5 Fraud Partners by Volume:**
  1. `SP3207` (Franchise, Hyderabad): 14 frauds (tenure = 1,350 days / **3.7 years old!**)
  2. `SP3376` (Franchise, Mysuru): 12 frauds (tenure = 930 days / **2.5 years old!**)
  3. `SP3095` (Franchise, Bhopal): 11 frauds (tenure = 1,468 days / **4.0 years old!**)
  4. `SP3160` (ASC, Pune): 10 frauds (tenure = 94 days / 3 months old)
  5. `SP3103` (Franchise, Jaipur): 9 frauds (tenure = 1,196 days / **3.3 years old!**)
- **Key Insight:** More than half of the worst offending partners are long-tenured veteran franchises!
- Meenal Joshi's observation was verified: *"my team sees the same few outlets again and again. But please don't paint all the new partners with one brush, most of the new ones are fine and we need them in the smaller cities."*

#### 3. Multivariate Regression Results (Controlling for Confounders)
When controlling for other claim variables:
- `customer_prior_claims`: Coefficient $+0.578$ (Odds Ratio: **1.78**, $p < 0.001$). Fraud probability rises exponentially with customer claim history (100% fraud for $\ge 5$ prior claims).
- `post_may_policy`: Coefficient $+0.354$ (Odds Ratio: **1.43**). The May 1 auto-approval change created a major systemic exploit.
- Product Family: Induction Cooktops (OR: **2.55**), Room Heaters (OR: **2.48**), and Mixer Grinders (OR: **1.92**) have far higher inherent fraud risk than Robot Vacuums (OR: **0.28**).
- Partner Type: Franchises have significantly higher fraud rates (OR: **1.82**) than Freelance Technicians (OR: **0.51**).

### Verdict on Ritu's Hypothesis
**Partially Confirmed in Raw Statistics, but Structurally Misleading:**
- While newer partners on average file higher fraud claims (3.83% vs 1.06%), this is driven by **a handful of specific bad actor outlets onboarded in late 2025/early 2026**, rather than newness itself.
- Crucially, veteran partners operating for 3–4 years account for the largest individual share of fraudulent claims.
- Treating all new partners with suspicion would severely penalize honest service providers in Tier-2/3 cities while ignoring rampant fraud at established franchise outlets.
- The model must target **partner behavioral history and claim-level risk attributes**, not blanket partner age.
