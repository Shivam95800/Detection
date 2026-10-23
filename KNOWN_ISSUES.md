# Known Issues and Technical Limitations — Kestrel Home (Variant C)

This document provides a candid disclosure of specific technical imperfections, operational edge cases, and known limitations in the current deliverable.

---

## 1. Serial Number Typos & Multi-SKU Duplication
- **Issue:** Product serial numbers are entered manually by partner technicians without barcode scanning or format validation.
- **Observed Artifact:** In `train.csv`, 1,987 serial numbers appear across completely different product SKUs (e.g. generic numbers like `123456789` or `KH111111111` typed for both an air fryer and a water purifier).
- **Impact:** Duplicate serial counts cannot be trusted as definitive global hardware identifiers. A genuine customer with a typo serial could be falsely flagged as a repeat serial claimant.
- **Mitigation:** The model weights serial repetition very lightly as a weak secondary signal.

---

## 2. Cold-Start Problem for Newly Onboarded Service Partners
- **Issue:** Partner historical fraud rate features (`partner_prior_fraud_rate`) use an empirical Bayes prior with pseudo-counts ($\alpha = 0.25, \beta = 20.0$).
- **Impact:** When a brand-new partner submits their very first claim, their prior claim count is 0 and their estimated fraud rate defaults to the baseline prior ($\sim 1.2\%$). If that partner commits fraud on Day 1, the model cannot detect it via partner history and must rely strictly on claim-level characteristics (customer prior claims, uninspected sub-₹2k flag, category risk).

---

## 3. High-Value Sophisticated Single-Claim Frauds
- **Issue:** The model relies heavily on behavioral patterns (repeat claimant history, auto-approval exploitation, partner velocity).
- **Impact:** A sophisticated one-off fraud on a high-value product (e.g. ₹18,000 Robot Vacuum motherboard replacement) submitted by an established partner with 0 customer prior claims and forged photo documentation will receive a low fraud score ($\sim 0.05$ to $0.15$), escaping the top-40 monthly review queue.

---

## 4. In-Memory Historical Lookup State in FastAPI Service
- **Issue:** The API loads historical claim timestamps and partner fraud counts into in-memory Python dictionaries upon startup from `train.csv`.
- **Impact:** If Kestrel's operations expand to hundreds of thousands of claims across multi-server horizontal pods, maintaining in-memory state across workers would cause desynchronization.
- **Production Recommendation:** Migrate the historical lookups to an operational database (e.g., PostgreSQL with index on `partner_id, submitted_at` or a Redis cache).

---

## 5. Potential Covariate Drift if the ₹2,000 Auto-Approval Rule Changes
- **Issue:** The model currently embeds domain rules tracking the May 1 auto-approval threshold (`claim_amount < 2000` & `not inspected`).
- **Impact:** If Ritu Deshpande modifies operations policy next month (e.g. lowering the threshold to ₹1,000 or eliminating auto-approval), the hardcoded rule threshold in `app/model_service.py` must be updated to match the new policy parameter.
