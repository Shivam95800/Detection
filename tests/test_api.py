import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert "model_version" in data

def test_predict_happy_path():
    payload = {
        "claim_id": "WC715001",
        "submitted_at": "2026-07-15 14:30",
        "partner_id": "SP3160",
        "sku": "KH-IC-02",
        "product_serial": "KH918237461",
        "days_since_purchase": 45,
        "claim_amount_inr": 1850.0,
        "photo_attached": "N",
        "partner_inspected": "N",
        "claim_description": "power button not working",
        "customer_prior_claims": 3
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["claim_id"] == "WC715001"
    assert 0.0 <= data["score"] <= 1.0
    assert data["risk_band"] in ["LOW", "MEDIUM", "HIGH"]
    assert "recommended_action" in data
    assert isinstance(data["reasons"], list)
    assert len(data["reasons"]) > 0

def test_predict_validation_error_missing_field():
    # Missing partner_id and claim_amount_inr
    payload = {
        "claim_id": "WC715002",
        "submitted_at": "2026-07-15 14:30",
        "sku": "KH-IC-02"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data

def test_predict_validation_error_invalid_value():
    payload = {
        "claim_id": "WC715003",
        "submitted_at": "2026-07-15 14:30",
        "partner_id": "SP3160",
        "sku": "KH-IC-02",
        "product_serial": "KH918237461",
        "days_since_purchase": -10,  # Invalid: negative days
        "claim_amount_inr": -500.0,  # Invalid: negative amount
        "photo_attached": "MAYBE",   # Invalid: not Y/N
        "partner_inspected": "Y",
        "claim_description": "motor not running",
        "customer_prior_claims": 0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

def test_predict_unseen_categories():
    # Unseen partner, unseen sku, unseen description
    payload = {
        "claim_id": "WC799999",
        "submitted_at": "2026-08-01 10:00",
        "partner_id": "SP9999",
        "sku": "KH-UNSEEN-99",
        "product_serial": "UNKNOWN_SERIAL_123",
        "days_since_purchase": 100,
        "claim_amount_inr": 2500.0,
        "photo_attached": "Y",
        "partner_inspected": "Y",
        "claim_description": "strange unknown buzzing sound",
        "customer_prior_claims": 0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert 0.0 <= data["score"] <= 1.0
    assert data["risk_band"] in ["LOW", "MEDIUM", "HIGH"]

def test_predict_deterministic_output():
    payload = {
        "claim_id": "WC715004",
        "submitted_at": "2026-07-16 12:00",
        "partner_id": "SP3033",
        "sku": "KH-AF-01",
        "product_serial": "KH555666777",
        "days_since_purchase": 120,
        "claim_amount_inr": 1200.0,
        "photo_attached": "Y",
        "partner_inspected": "Y",
        "claim_description": "blade jammed",
        "customer_prior_claims": 1
    }
    res1 = client.post("/predict", json=payload).json()
    res2 = client.post("/predict", json=payload).json()
    assert res1["score"] == res2["score"]
    assert res1["risk_band"] == res2["risk_band"]
    assert res1["reasons"] == res2["reasons"]

def test_dashboard_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert "Kestrel Home" in response.text
    assert "Load Fraud Example" in response.text
