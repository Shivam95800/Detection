import sys
import subprocess
import os
import time

def run_step(step_name, command):
    print("\n" + "=" * 70)
    print(f"RUNNING STEP: {step_name}")
    print(f"Command: {command}")
    print("=" * 70)
    start = time.time()
    res = subprocess.run(command, shell=True)
    duration = time.time() - start
    if res.returncode != 0:
        print(f"\n[ERROR] Step '{step_name}' failed with return code {res.returncode}")
        sys.exit(res.returncode)
    print(f"[SUCCESS] Step '{step_name}' completed in {duration:.2f}s")

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(base_dir)
    print("=" * 70)
    print("KESTREL HOME — REPRODUCIBILITY PIPELINE (VARIANT C)")
    print(f"Working Directory: {base_dir}")
    print("=" * 70)
    
    # Step 1: Data validation & extraction
    run_step("Data Pack Inspection", f"{sys.executable} scripts/inspect_data.py")
    
    # Step 2: Time-based validation, evaluation, metrics, and chart generation
    run_step("Model Training & Comprehensive Validation", f"{sys.executable} scripts/comprehensive_eval.py")
    
    # Step 3: Out-of-time inference on test_unlabelled.csv
    run_step("Generate Final Predictions", f"{sys.executable} scripts/generate_predictions.py")
    
    # Step 4: Submission integrity verification
    run_step("Validate Submission File", f"{sys.executable} scripts/validate_submission.py")
    
    # Step 5: Test suite execution
    run_step("Run Automated Test Suite", f"{sys.executable} -m pytest tests/")
    
    print("\n" + "=" * 70)
    print("ALL PIPELINE STAGES COMPLETED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    main()
