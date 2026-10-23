# Kestrel Home — Warranty Claim Review System (Variant C)

[![Live Demo](https://img.shields.io/badge/LIVE%20DEMO-kestrel--warranty--detection.onrender.com-46E3B7?style=for-the-badge&logo=render&logoColor=white)](https://kestrel-warranty-detection.onrender.com)
[![API Swagger Docs](https://img.shields.io/badge/API%20DOCS-Swagger%20UI-0284C7?style=for-the-badge&logo=fastapi&logoColor=white)](https://kestrel-warranty-detection.onrender.com/docs)
[![Tests Passing](https://img.shields.io/badge/TESTS-8%20PASSED-success?style=for-the-badge&logo=pytest&logoColor=white)](https://github.com/Shivam95800/Detection)

> 🚀 **PRODUCTION SERVICE LIVE:** The interactive claim review portal is hosted 24/7 at **[https://kestrel-warranty-detection.onrender.com](https://kestrel-warranty-detection.onrender.com)**.  
> Open the link to test claims using one-click example buttons and view real-time plain-English risk rationales.

---

## 1. Executive Summary

- **Business Problem:** Kestrel processes ~750 warranty claims monthly across direct-to-consumer and ~380 service partner outlets. Fraud accounts for ~1.2% to 2.8% of claims. The investigation desk can review at most **40 claims per month** ($k = 40$).
- **The Solution:** A **Hybrid Ensemble Model** (`HistGradientBoosting` + Domain Policy Rules) that ranks incoming claims strictly using submission-time data, routes the 40 riskiest claims to Farhan Sheikh's investigation desk, and auto-approves low-risk claims.
- **Key Business Outcomes (June 2026 Out-of-Time Benchmark):**
  - **Intercepts 68.2% of All Fraud:** Stops 15 out of 22 fraud claims in just 40 reviews.
  - **Hits 37.5% Precision@40:** 13.1x higher than the base rate (only 25 false alarms held briefly).
  - **Net Rupee Profit:** Generates **+₹11,512/month net savings** (**+₹34,536/quarter**) after paying ₹380 goodwill compensation for every delayed customer.
  - **Zero Paid API Dependencies:** Runs 100% locally with deterministic operational reasoning.

---

## 2. Quickstart & Clean-Machine Setup

This repository is designed to run offline on a clean machine with **no paid API keys**.

### A. Local Setup (Virtual Environment)
```bash
# 1. Clone repository and navigate to root
cd kestrel-warranty

# 2. Create and activate a fresh virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# 3. Install pinned dependencies
pip install -r requirements.txt

# 4. Start the FastAPI Service and Ops Dashboard
uvicorn app.main:app --host 0.0.0.0 --port 8000
```
Visit **http://localhost:8000** in your browser to access the single-screen review dashboard.

### B. Docker Setup
```bash
# Build Docker image
docker build -t kestrel-fraud-service .

# Run container
docker run -p 8000:8000 kestrel-fraud-service
```
Access the dashboard at **http://localhost:8000**.

---

## 3. One-Command Master Pipeline (Reproducibility)

To re-run the entire pipeline from scratch—auditing data, running rolling-origin validation, training the model, generating `predictions.csv`, running the submission validator, and executing the full test suite:

```bash
python run_all.py
```

### Manual Individual Commands
- **Regenerate Predictions:** `python scripts/generate_predictions.py`
- **Validate Submission Format:** `python scripts/validate_submission.py`
- **Run Unit Tests:** `python -m pytest tests/`
- **Run Comprehensive Validation:** `python scripts/comprehensive_eval.py`

---

## 4. Repository Map

```
kestrel-warranty/
├── app/
│   ├── __init__.py
│   ├── main.py                  # FastAPI application with /predict, /health, and HTML dashboard
│   ├── model_service.py         # Real-time feature calculation and plain-English reasons
│   └── schemas.py               # Pydantic v2 validation models
├── data/
│   ├── train.csv                # 15 months of past claims with ground-truth outcomes
│   ├── test_unlabelled.csv      # Unlabelled claims to score (2,252 rows)
│   ├── sample_submission.csv    # Submission schema target
│   ├── partners.csv             # Partner metadata (380 outlets)
│   ├── products.csv             # Product catalog (21 SKUs)
│   ├── ops-policy.pdf           # Operations policy v4.1 (costs, rules, capacities)
│   ├── email-thread.txt         # Internal email thread between stakeholders
│   └── README.txt               # Data pack specifications
├── NOTES/
│   ├── pack_summary.md          # Stage 0: Exhaustive data pack summary and quoted figures
│   ├── leakage.md               # Stage 1: Feature whitelist and excluded leakage columns
│   └── audit.md                 # Stage 1: Full data audit and Ritu's hypothesis testing
├── reports/
│   ├── evidence.md              # Stage 6: Technical report with validation charts
│   ├── metrics.json             # Machine-readable output of all benchmark metrics
│   └── figures/                 # Generated PNG charts (ROC, PR, Calibration, Rupee Curve)
├── scripts/
│   ├── inspect_data.py          # Stage 0 data inspector
│   ├── test_hypothesis.py       # Statistical testing of Ritu's hypothesis
│   ├── features.py              # Feature extraction definitions
│   ├── time_aggregates.py       # Strict time-aware historical aggregators
│   ├── comprehensive_eval.py    # Rolling-origin cross-validation engine
│   ├── generate_predictions.py  # Stage 4 inference engine scoring test_unlabelled.csv
│   └── validate_submission.py   # Automated submission file integrity checker
├── tests/
│   ├── test_api.py              # Pytest API tests (happy path, validation, determinism)
│   └── test_submission.py       # Pytest submission file validator
├── AI_USAGE.md                  # Honest log of AI usage, mistakes, and discarded work
├── DECISIONS.md                 # Technical decisions log resolving ambiguities
├── KNOWN_ISSUES.md              # Candid disclosure of known flaws and limitations
├── MEMO_TO_RITU.md              # One-page executive memorandum to Ritu Deshpande
├── MEMO_TO_RITU.docx            # Formatted Microsoft Word export of executive memo
├── FORM_ANSWERS.md              # First-person candidate form answers
├── PREDICTION_EXPECTATION.md    # Pre-submission honesty protocol expectation
├── predictions.csv              # Final test predictions matching sample_submission.csv
├── model.joblib                 # Trained full-dataset model artifact
├── requirements.txt             # Pinned package versions
├── Dockerfile                   # Containerized deployment specification
├── run_all.py                   # Single command master reproducibility script
└── README.md                    # Project documentation
```

---

## 5. API Reference

### `POST /predict`
Evaluates a single claim in real time and returns calibrated risk score, risk band, and plain-English operational explanations.

**Sample Request:**
```json
{
  "claim_id": "WC715892",
  "submitted_at": "2026-07-20 11:15",
  "partner_id": "SP3160",
  "sku": "KH-IC-02",
  "product_serial": "KH762701250",
  "days_since_purchase": 55,
  "claim_amount_inr": 1920.0,
  "photo_attached": "N",
  "partner_inspected": "N",
  "claim_description": "power button not working",
  "customer_prior_claims": 3
}
```

**Sample Response:**
```json
{
  "claim_id": "WC715892",
  "score": 0.742,
  "risk_band": "HIGH",
  "recommended_action": "HOLD FOR INVESTIGATION DESK REVIEW",
  "reasons": [
    "Elevated risk: Customer has filed 3 previous warranty claims (exceeds genuine household failure norm).",
    "Auto-approval exploitation pattern: Claim amount of Rs. 1,920 falls under the Rs. 2,000 threshold and skipped technician inspection.",
    "High-risk partner outlet: Partner SP3160 in Pune has a history of 10 confirmed fraud claims (20.5% fraud rate).",
    "Product category risk: Induction Cooktop demonstrates an elevated failure-to-claim fraud ratio.",
    "Missing documentation: No diagnostic photograph attached to verify the physical damage."
  ],
  "model_version": "kestrel-fraud-ensemble-v1.0",
  "metadata": {
    "partner_city": "Pune",
    "partner_type": "authorised_service_centre",
    "partner_tenure_days": 205,
    "product_family": "Induction Cooktop",
    "list_price_inr": 4499.0,
    "claim_ratio": 0.427
  }
}
```

### `GET /health`
Returns system status, model version, and lookup dictionary counts.
```json
{
  "status": "healthy",
  "model_version": "kestrel-fraud-ensemble-v1.0",
  "model_loaded": true,
  "partners_count": 380,
  "products_count": 21
}
```

---

## 6. Testing & Validation

The test suite contains 8 automated tests covering:
- API endpoint happy path
- Request validation for missing fields (HTTP 422)
- Value bounds validation (negative days, negative amounts, invalid Y/N)
- Graceful handling of unseen category values
- Deterministic score output
- Submission file integrity (exact column names, row counts, ID ordering, no NaNs/infs)

Run the test suite with:
```bash
python -m pytest tests/ -v
```

---

## 7. Operational Limitations & Edge Cases

1. **Typos in Typed Serial Numbers:** Partner technicians manually type serial numbers, leading to occasional false matches across unrelated SKUs. The model treats repeat serials as a secondary signal.
2. **First-Time Service Partner Fraud:** A brand-new partner committing fraud on their very first claim has no historical record. The model detects these claims via customer repeat claim counts and category rules.
3. **High-Value Inspected Frauds:** Claims > ₹15,000 with forged bills and inspection sign-offs filed by first-time claimants can slip past the top-40 monthly cutoff.
