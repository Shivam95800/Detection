import os
import sys
import pandas as pd
import numpy as np

def validate_submission(pred_path, sample_path):
    print("="*60)
    print("SUBMISSION INTEGRITY VALIDATION")
    print("="*60)
    
    if not os.path.exists(pred_path):
        print(f"[FAIL] Submission file not found at: {pred_path}")
        return False
        
    pred = pd.read_csv(pred_path)
    sample = pd.read_csv(sample_path)
    
    passed = True
    
    # 1. Column names check
    expected_cols = list(sample.columns)
    actual_cols = list(pred.columns)
    if expected_cols == actual_cols:
        print(f"[PASS] Columns match expected: {actual_cols}")
    else:
        print(f"[FAIL] Column mismatch! Expected {expected_cols}, got {actual_cols}")
        passed = False
        
    # 2. Row count check
    if len(pred) == len(sample):
        print(f"[PASS] Row count matches: {len(pred)} rows")
    else:
        print(f"[FAIL] Row count mismatch! Expected {len(sample)}, got {len(pred)}")
        passed = False
        
    # 3. ID coverage & exact order
    if (pred['claim_id'] == sample['claim_id']).all():
        print("[PASS] claim_id coverage and row-for-row ordering perfectly match sample_submission.csv")
    else:
        print("[FAIL] claim_id values or ordering do not match sample_submission.csv!")
        passed = False
        
    # 4. No NaNs, nulls, or infs
    null_count = pred.isnull().sum().sum()
    if null_count == 0:
        print("[PASS] Zero null or NaN values found")
    else:
        print(f"[FAIL] Found {null_count} null/NaN values!")
        passed = False
        
    inf_count = np.isinf(pred['score']).sum()
    if inf_count == 0:
        print("[PASS] Zero infinite values found")
    else:
        print(f"[FAIL] Found {inf_count} inf values!")
        passed = False
        
    # 5. Numeric scores
    if pd.api.types.is_numeric_dtype(pred['score']):
        print(f"[PASS] Score column is numeric (min: {pred['score'].min():.4f}, max: {pred['score'].max():.4f})")
    else:
        print("[FAIL] Score column is not numeric!")
        passed = False
        
    # 6. Score range sanity
    if (pred['score'] >= 0.0).all() and (pred['score'] <= 1.0).all():
        print("[PASS] All scores are strictly bounded within [0.0, 1.0]")
    else:
        print("[FAIL] Scores found outside [0, 1] range!")
        passed = False
        
    print("="*60)
    if passed:
        print("ALL SUBMISSION VALIDATION CHECKS PASSED SUCCESSFULLY!")
    else:
        print("SUBMISSION VALIDATION FAILED!")
    print("="*60)
    return passed

if __name__ == "__main__":
    base_dir = r"C:\Users\Asus\.gemini\antigravity-ide\scratch\kestrel-warranty"
    pred_path = os.path.join(base_dir, "predictions.csv")
    sample_path = os.path.join(base_dir, "data", "sample_submission.csv")
    
    success = validate_submission(pred_path, sample_path)
    if not success:
        sys.exit(1)
