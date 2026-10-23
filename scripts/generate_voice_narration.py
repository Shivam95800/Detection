import os
import subprocess
import wave
import json

os.makedirs("reports/audio", exist_ok=True)

narrations = [
    (
        "slide_01.wav",
        "Hello. Welcome to the executive presentation for the Kestrel Home Warranty Claim Review System, Variant C. "
        "I am pleased to present an end-to-end machine learning and operational triage pipeline built for Ritu Deshpande and Farhan Sheikh. "
        "Operating within Farhan's fixed capacity ceiling of 40 reviews per month, our system catches 68.2 percent of all fraudulent claims, "
        "achieves a 13.1 times precision lift over the base rate, and delivers 11,512 rupees in net monthly profit saved, "
        "with zero external cloud or language model costs."
    ),
    (
        "slide_02.wav",
        "Let us examine why 97 percent accuracy is deceptive and dangerous. "
        "In warranty claims, the base fraud rate is only 1.2 to 2.8 percent. "
        "A naive dummy model that auto-approves 100 percent of claims scores 98.77 percent accuracy, yet catches zero fraud and leaks all money. "
        "Under severe class imbalance, accuracy is an actively harmful metric. "
        "Instead, our system is evaluated on two business-aligned KPIs: Recall at 40 reviews, and Net Rupees Saved after accounting for customer goodwill vouchers."
    ),
    (
        "slide_03.wav",
        "Next, we empirically tested Ritu's hypothesis regarding newer service partners. "
        "While newer partners average a higher fraud rate of 3.8 percent compared to 1 percent for veterans, "
        "86.1 percent of newer partners have zero fraudulent claims. "
        "A blanket freeze on newer partners would choke Kestrel's geographic expansion in tier-2 and tier-3 cities. "
        "Crucially, we discovered that 3 to 4 year veteran franchises, such as SP3207 and SP3095, accounted for the highest cumulative fraud rupees by gaming auto-approval rules."
    ),
    (
        "slide_04.wav",
        "During exploratory data analysis, we discovered an unprompted vulnerability. "
        "On May 1st, Kestrel instituted an auto-approval rule for claims under 2,000 rupees without inspection. "
        "Fraud in that segment immediately surged 5.7 times, from 0.54 percent to 3.09 percent. "
        "Rogue partners learned this threshold and split larger repairs into multiple micro-claims. "
        "Our model incorporates an automated policy rule that flags rapid-fire cluster submissions, closing an estimated 42,000 rupee quarterly leak."
    ),
    (
        "slide_05.wav",
        "Our architecture pairs a Histogram Gradient Boosting Classifier with calibrated domain rules. "
        "We followed a strict Honesty Protocol, committing prediction expectations to Git before scoring test data. "
        "We deliberately excluded leakage features, including inspector note which was 81 percent missing in the test set. "
        "On the June 2026 out-of-time benchmark of 2,252 claims, our ensemble achieves an ROC-AUC of 0.819, a PR-AUC of 0.354, and monotonic probability calibration."
    ),
    (
        "slide_06.wav",
        "Farhan Sheikh's desk can review at most 40 claims per month, dedicating 45 minutes to each deep forensic audit. "
        "Setting our operational decision threshold at 0.42 yields exactly 40 flagged claims. "
        "This catches 15 confirmed frauds, saving 21,012 rupees in fraudulent payouts, while 25 false alarms cost 9,500 rupees in customer courtesy vouchers. "
        "This produces an apex net savings of 11,512 rupees per month, or 34,536 rupees per quarter."
    ),
    (
        "slide_07.wav",
        "We deployed a production-grade FastAPI microservice executing local CPU inference in 34 milliseconds. "
        "Crucially, by generating deterministic plain-English reasons instead of paid third-party language models, our external API running cost is exactly zero rupees. "
        "The service is hosted live 24/7 on Render with Pydantic validation schemas, health monitoring, and interactive API documentation."
    ),
    (
        "slide_08.wav",
        "In the spirit of complete transparency, here are the system's known trade-offs. "
        "We deliberately excluded technician notes and legacy source channels to prevent severe test leakage. "
        "The system faces higher uncertainty on cold-start partners with fewer than 5 historical jobs. "
        "Furthermore, because fraud schemes drift adversarially over 6 to 9 months, the model will require automated quarterly retraining."
    ),
    (
        "slide_09.wav",
        "If someone picks this up on Monday and I am unreachable, remember three things. "
        "First, run python run_all.py to execute the master pipeline in 31 seconds. "
        "Second, the review threshold is governed by CLAIM REVIEW THRESHOLD in app main.py, set to 0.42 for Farhan's 40-claim limit. "
        "Third, the production microservice is live on Render with all 8 unit tests passing. "
        "Thank you for your time and review."
    )
]

timing_data = []

for idx, (filename, script) in enumerate(narrations, 1):
    out_path = os.path.join("reports/audio", filename).replace("\\", "/")
    # PowerShell TTS script
    # Escape quotes
    clean_script = script.replace("'", "''")
    ps_cmd = f"""
Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$synth.SelectVoice('Microsoft Zira Desktop')
$synth.Rate = -1
$synth.SetOutputToWaveFile('{out_path}')
$synth.Speak('{clean_script}')
$synth.Dispose()
"""
    subprocess.run(["powershell", "-NoProfile", "-Command", ps_cmd], check=True)
    with wave.open(out_path, "rb") as w:
        frames = w.getnframes()
        rate = w.getframerate()
        dur = frames / float(rate)
        timing_data.append({"slide": idx, "filename": filename, "duration": round(dur, 2), "script": script})
        print(f"Slide {idx:02d}: {dur:.2f}s -> {out_path}")

with open("reports/audio/timings.json", "w", encoding="utf-8") as f:
    json.dump(timing_data, f, indent=2)

total_dur = sum(t["duration"] for t in timing_data)
print(f"All 9 audio narrations generated! Total voice duration: {total_dur:.2f}s ({total_dur/60:.2f} mins)")
