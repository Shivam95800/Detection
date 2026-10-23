from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Any

class ClaimInput(BaseModel):
    claim_id: str = Field(..., description="Unique claim reference (e.g. WC701234)")
    submitted_at: str = Field(..., description="Timestamp in IST (YYYY-MM-DD HH:MM or ISO)")
    partner_id: str = Field(..., description="Service partner ID (e.g. SP3160)")
    sku: str = Field(..., description="Product SKU (e.g. KH-MG-03)")
    product_serial: str = Field(..., description="Product serial number entered by partner")
    days_since_purchase: int = Field(..., ge=0, le=1500, description="Days elapsed since purchase")
    claim_amount_inr: float = Field(..., gt=0, le=100000, description="Claim amount in INR")
    photo_attached: str = Field(..., description="'Y' or 'N'")
    partner_inspected: str = Field(..., description="'Y' or 'N'")
    claim_description: str = Field(..., description="Reported fault description")
    customer_prior_claims: int = Field(..., ge=0, le=20, description="Customer previous warranty claims")
    
    @field_validator('photo_attached', 'partner_inspected')
    @classmethod
    def validate_yn(cls, v: str) -> str:
        v_upper = v.strip().upper()
        if v_upper not in ['Y', 'N']:
            raise ValueError(f"Value must be 'Y' or 'N', got '{v}'")
        return v_upper

class PredictionResponse(BaseModel):
    claim_id: str
    score: float
    risk_band: str
    recommended_action: str
    reasons: List[str]
    model_version: str
    metadata: Dict[str, Any]

class HealthResponse(BaseModel):
    status: str
    model_version: str
    model_loaded: bool
    partners_count: int
    products_count: int
