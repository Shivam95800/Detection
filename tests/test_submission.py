import os
import pytest
from scripts.validate_submission import validate_submission

def test_submission_file_validation():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    pred_path = os.path.join(base_dir, "predictions.csv")
    sample_path = os.path.join(base_dir, "data", "sample_submission.csv")
    
    assert os.path.exists(pred_path), "predictions.csv must exist in repo root"
    assert os.path.exists(sample_path), "sample_submission.csv must exist in data/"
    
    passed = validate_submission(pred_path, sample_path)
    assert passed is True, "validate_submission must return True for predictions.csv"
