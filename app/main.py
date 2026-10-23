import os
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime

from app.schemas import ClaimInput, PredictionResponse, HealthResponse
from app.model_service import FraudModelService

app = FastAPI(
    title="Kestrel Home — Warranty Fraud Prediction Service",
    description="Operational API for real-time warranty claim scoring, risk banding, and plain-English operational reasoning.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize service singleton
service = FraudModelService()

@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health_check():
    return HealthResponse(
        status="healthy",
        model_version=service.model_version,
        model_loaded=service.model_loaded,
        partners_count=len(service.partners_dict),
        products_count=len(service.products_dict)
    )

@app.post("/predict", response_model=PredictionResponse, tags=["Scoring"])
def predict_claim(claim: ClaimInput):
    try:
        result = service.predict_one(claim.model_dump())
        return PredictionResponse(**result)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}"
        )

@app.get("/", response_class=HTMLResponse, tags=["UI"])
def dashboard():
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Kestrel Home — Warranty Claim Review Portal</title>
  <style>
    :root {
      --bg: #0f172a;
      --card-bg: rgba(30, 41, 59, 0.7);
      --card-border: rgba(255, 255, 255, 0.1);
      --text: #f8fafc;
      --text-muted: #94a3b8;
      --primary: #38bdf8;
      --primary-hover: #0ea5e9;
      --danger: #ef4444;
      --warning: #f59e0b;
      --success: #10b981;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
    body { background-color: var(--bg); color: var(--text); min-height: 100vh; padding: 2rem 1rem; }
    .container { max-width: 1000px; margin: 0 auto; }
    header { margin-bottom: 2rem; border-bottom: 1px solid var(--card-border); padding-bottom: 1rem; }
    header h1 { font-size: 1.8rem; color: #fff; font-weight: 700; }
    header p { color: var(--text-muted); font-size: 0.95rem; margin-top: 0.25rem; }
    .grid { display: grid; grid-template-columns: 1.2fr 1fr; gap: 1.5rem; }
    @media (max-width: 768px) { .grid { grid-template-columns: 1fr; } }
    .card { background: var(--card-bg); backdrop-filter: blur(12px); border: 1px solid var(--card-border); border-radius: 12px; padding: 1.5rem; box-shadow: 0 10px 25px rgba(0,0,0,0.3); }
    .card h2 { font-size: 1.2rem; margin-bottom: 1rem; color: #38bdf8; }
    .form-group { margin-bottom: 0.85rem; }
    label { display: block; font-size: 0.82rem; font-weight: 600; color: var(--text-muted); margin-bottom: 0.3rem; }
    input, select { width: 100%; padding: 0.6rem 0.8rem; background: #0f172a; border: 1px solid var(--card-border); border-radius: 6px; color: #fff; font-size: 0.9rem; transition: border-color 0.2s; }
    input:focus, select:focus { outline: none; border-color: var(--primary); }
    .row { display: flex; gap: 0.75rem; }
    .row .form-group { flex: 1; }
    .btn-group { display: flex; gap: 0.5rem; margin-bottom: 1.2rem; }
    button { padding: 0.65rem 1rem; border-radius: 6px; border: none; font-weight: 600; font-size: 0.85rem; cursor: pointer; transition: all 0.2s; }
    .btn-primary { background: var(--primary); color: #0f172a; width: 100%; font-size: 1rem; padding: 0.8rem; }
    .btn-primary:hover { background: var(--primary-hover); }
    .btn-example { background: rgba(255, 255, 255, 0.08); color: var(--text); border: 1px solid var(--card-border); flex: 1; }
    .btn-example:hover { background: rgba(255, 255, 255, 0.15); }
    .badge { display: inline-block; padding: 0.3rem 0.8rem; border-radius: 20px; font-weight: 700; font-size: 0.85rem; text-transform: uppercase; }
    .badge-high { background: rgba(239, 68, 68, 0.2); color: var(--danger); border: 1px solid var(--danger); }
    .badge-medium { background: rgba(245, 158, 11, 0.2); color: var(--warning); border: 1px solid var(--warning); }
    .badge-low { background: rgba(16, 185, 129, 0.2); color: var(--success); border: 1px solid var(--success); }
    .score-display { text-align: center; margin: 1.5rem 0; padding: 1.5rem; background: rgba(15, 23, 42, 0.6); border-radius: 10px; border: 1px solid var(--card-border); }
    .score-val { font-size: 2.8rem; font-weight: 800; letter-spacing: -1px; }
    .action-box { padding: 1rem; border-radius: 8px; margin-bottom: 1.5rem; font-weight: 600; font-size: 0.95rem; text-align: center; }
    .reasons-list { list-style: none; }
    .reasons-list li { margin-bottom: 0.6rem; padding-left: 1.2rem; position: relative; font-size: 0.9rem; line-height: 1.4; color: #cbd5e1; }
    .reasons-list li::before { content: "•"; position: absolute; left: 0; color: var(--primary); font-size: 1.2rem; top: -2px; }
    .error-box { background: rgba(239, 68, 68, 0.15); border: 1px solid var(--danger); color: #fca5a5; padding: 0.8rem; border-radius: 6px; margin-bottom: 1rem; display: none; font-size: 0.85rem; }
    .empty-state { text-align: center; color: var(--text-muted); padding: 3rem 1rem; }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <h1>Kestrel Home — Warranty Review System</h1>
      <p>Operational Desk Triage & Fraud Risk Scoring (Variant C)</p>
    </header>

    <div class="grid">
      <!-- Input Panel -->
      <div class="card">
        <h2>Enter Claim Details</h2>
        <div class="btn-group">
          <button type="button" class="btn-example" onclick="loadFraudExample()">Load Fraud Example</button>
          <button type="button" class="btn-example" onclick="loadNormalExample()">Load Normal Example</button>
        </div>

        <div id="errorBanner" class="error-box"></div>

        <form id="claimForm" onsubmit="submitClaim(event)">
          <div class="row">
            <div class="form-group">
              <label for="claim_id">Claim ID</label>
              <input type="text" id="claim_id" required value="WC715001">
            </div>
            <div class="form-group">
              <label for="submitted_at">Submission Timestamp (IST)</label>
              <input type="text" id="submitted_at" required value="2026-07-15 14:30">
            </div>
          </div>

          <div class="row">
            <div class="form-group">
              <label for="partner_id">Service Partner ID</label>
              <input type="text" id="partner_id" required value="SP3160">
            </div>
            <div class="form-group">
              <label for="sku">Product SKU</label>
              <input type="text" id="sku" required value="KH-IC-02">
            </div>
          </div>

          <div class="row">
            <div class="form-group">
              <label for="product_serial">Serial Number</label>
              <input type="text" id="product_serial" required value="KH918237461">
            </div>
            <div class="form-group">
              <label for="days_since_purchase">Days Since Purchase</label>
              <input type="number" id="days_since_purchase" min="0" required value="45">
            </div>
          </div>

          <div class="row">
            <div class="form-group">
              <label for="claim_amount_inr">Claim Amount (Rs.)</label>
              <input type="number" id="claim_amount_inr" step="1" min="1" required value="1850">
            </div>
            <div class="form-group">
              <label for="customer_prior_claims">Customer Prior Claims</label>
              <input type="number" id="customer_prior_claims" min="0" max="20" required value="3">
            </div>
          </div>

          <div class="row">
            <div class="form-group">
              <label for="photo_attached">Photo Attached?</label>
              <select id="photo_attached">
                <option value="N">No (N)</option>
                <option value="Y">Yes (Y)</option>
              </select>
            </div>
            <div class="form-group">
              <label for="partner_inspected">Partner Inspected?</label>
              <select id="partner_inspected">
                <option value="N">No (N - Auto-approval route)</option>
                <option value="Y">Yes (Y - Technician sign-off)</option>
              </select>
            </div>
          </div>

          <div class="form-group">
            <label for="claim_description">Reported Fault</label>
            <select id="claim_description">
              <option value="power button not working">power button not working</option>
              <option value="motor not running">motor not running</option>
              <option value="tripping mcb">tripping mcb</option>
              <option value="water leaking">water leaking</option>
              <option value="blade jammed">blade jammed</option>
              <option value="loud noise while running">loud noise while running</option>
              <option value="filter indicator stuck">filter indicator stuck</option>
            </select>
          </div>

          <button type="submit" class="btn-primary" id="submitBtn">Score Claim & Check Risk</button>
        </form>
      </div>

      <!-- Result Panel -->
      <div class="card">
        <h2>Triage & Risk Assessment</h2>
        
        <div id="resultContent" style="display: none;">
          <div class="score-display">
            <span id="riskBadge" class="badge">HIGH RISK</span>
            <div class="score-val" id="scoreVal">0.7420</div>
            <div style="font-size: 0.85rem; color: var(--text-muted);">Calibrated Probability of Fraud</div>
          </div>

          <div id="actionBox" class="action-box">
            Action: <span id="actionVal">HOLD FOR INVESTIGATION DESK REVIEW</span>
          </div>

          <h3 style="font-size: 0.95rem; margin-bottom: 0.8rem; color: #fff;">Operational Explanations</h3>
          <ul id="reasonsList" class="reasons-list"></ul>

          <div style="margin-top: 1.5rem; padding-top: 1rem; border-top: 1px solid var(--card-border); font-size: 0.78rem; color: var(--text-muted);">
            <div><strong>Model:</strong> <span id="modelVer"></span></div>
            <div><strong>Partner:</strong> <span id="partnerInfo"></span></div>
            <div><strong>Product:</strong> <span id="productInfo"></span></div>
          </div>
        </div>

        <div id="emptyState" class="empty-state">
          <p>Submit claim details or select an example above to generate risk scoring and plain-English operational reasons.</p>
        </div>
      </div>
    </div>
  </div>

  <script>
    function loadFraudExample() {
      document.getElementById('claim_id').value = "WC715892";
      document.getElementById('submitted_at').value = "2026-07-20 11:15";
      document.getElementById('partner_id').value = "SP3160"; // Pune new partner, 10 frauds post-May
      document.getElementById('sku').value = "KH-IC-02"; // Induction cooktop
      document.getElementById('product_serial').value = "KH762701250";
      document.getElementById('days_since_purchase').value = "55";
      document.getElementById('claim_amount_inr').value = "1920"; // Sub-2000 auto-approval exploit
      document.getElementById('customer_prior_claims').value = "3"; // Chronic claimant
      document.getElementById('photo_attached').value = "N";
      document.getElementById('partner_inspected').value = "N"; // Skipped inspection
      document.getElementById('claim_description').value = "power button not working";
      hideError();
    }

    function loadNormalExample() {
      document.getElementById('claim_id').value = "WC715104";
      document.getElementById('submitted_at').value = "2026-07-18 09:40";
      document.getElementById('partner_id').value = "SP3033"; // Old ASC, 0 frauds
      document.getElementById('sku').value = "KH-AF-01"; // Air Fryer
      document.getElementById('product_serial').value = "KH192837465";
      document.getElementById('days_since_purchase').value = "180";
      document.getElementById('claim_amount_inr').value = "2400";
      document.getElementById('customer_prior_claims').value = "0"; // First-time claimant
      document.getElementById('photo_attached').value = "Y";
      document.getElementById('partner_inspected').value = "Y"; // Verified inspection
      document.getElementById('claim_description').value = "blade jammed";
      hideError();
    }

    function showError(msg) {
      const banner = document.getElementById('errorBanner');
      banner.innerText = msg;
      banner.style.display = 'block';
    }

    function hideError() {
      document.getElementById('errorBanner').style.display = 'none';
    }

    async function submitClaim(e) {
      e.preventDefault();
      hideError();
      const btn = document.getElementById('submitBtn');
      btn.innerText = "Analyzing Claim...";
      btn.disabled = true;

      const payload = {
        claim_id: document.getElementById('claim_id').value,
        submitted_at: document.getElementById('submitted_at').value,
        partner_id: document.getElementById('partner_id').value,
        sku: document.getElementById('sku').value,
        product_serial: document.getElementById('product_serial').value,
        days_since_purchase: parseInt(document.getElementById('days_since_purchase').value, 10),
        claim_amount_inr: parseFloat(document.getElementById('claim_amount_inr').value),
        photo_attached: document.getElementById('photo_attached').value,
        partner_inspected: document.getElementById('partner_inspected').value,
        claim_description: document.getElementById('claim_description').value,
        customer_prior_claims: parseInt(document.getElementById('customer_prior_claims').value, 10)
      };

      try {
        const res = await fetch('/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });

        if (!res.ok) {
          const errData = await res.json();
          let detail = "Error processing claim.";
          if (errData && errData.detail) {
            if (Array.isArray(errData.detail)) {
              detail = errData.detail.map(d => `${d.loc ? d.loc.join('.') : ''}: ${d.msg}`).join(" | ");
            } else {
              detail = errData.detail;
            }
          }
          showError(detail);
          return;
        }

        const data = await res.json();
        renderResult(data);
      } catch (err) {
        showError("Network or server connection failed. Please ensure backend is running.");
      } finally {
        btn.innerText = "Score Claim & Check Risk";
        btn.disabled = false;
      }
    }

    function renderResult(data) {
      document.getElementById('emptyState').style.display = 'none';
      document.getElementById('resultContent').style.display = 'block';

      const scoreEl = document.getElementById('scoreVal');
      scoreEl.innerText = data.score.toFixed(4);

      const badge = document.getElementById('riskBadge');
      badge.innerText = `${data.risk_band} RISK`;
      badge.className = `badge badge-${data.risk_band.toLowerCase()}`;

      const actionBox = document.getElementById('actionBox');
      actionBox.innerText = `Recommendation: ${data.recommended_action}`;
      if (data.risk_band === 'HIGH') {
        actionBox.style.background = 'rgba(239, 68, 68, 0.2)';
        actionBox.style.color = '#fca5a5';
      } else if (data.risk_band === 'MEDIUM') {
        actionBox.style.background = 'rgba(245, 158, 11, 0.2)';
        actionBox.style.color = '#fcd34d';
      } else {
        actionBox.style.background = 'rgba(16, 185, 129, 0.2)';
        actionBox.style.color = '#6ee7b7';
      }

      const reasonsList = document.getElementById('reasonsList');
      reasonsList.innerHTML = '';
      data.reasons.forEach(r => {
        const li = document.createElement('li');
        li.innerText = r;
        reasonsList.appendChild(li);
      });

      document.getElementById('modelVer').innerText = data.model_version;
      document.getElementById('partnerInfo').innerText = `${data.metadata.partner_city || 'N/A'} (${data.metadata.partner_type || 'N/A'}, ${data.metadata.partner_tenure_days || 0}d tenure)`;
      document.getElementById('productInfo').innerText = `${data.metadata.product_family || 'N/A'} (List Price: Rs. ${data.metadata.list_price_inr || 'N/A'})`;
    }
  </script>
</body>
</html>
"""
    return HTMLResponse(content=html_content)
