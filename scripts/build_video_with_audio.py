import os
import sys
import subprocess
import wave
import json
import numpy as np
import imageio
import imageio_ffmpeg
from PIL import Image

os.makedirs("reports/audio", exist_ok=True)
os.makedirs("reports/slides", exist_ok=True)

ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

narrations = [
    (
        "slide_01.wav",
        "Welcome to the executive presentation for the Kestrel Home Warranty Review System, Variant C. "
        "Our hybrid ensemble and operational triage pipeline is tailored for Ritu Deshpande and Farhan Sheikh. "
        "Operating at Farhan's fixed capacity ceiling of 40 reviews per month, our system intercepts 68.2 percent of all fraudulent claims, "
        "achieves a 13.1 times precision lift over the base rate, and delivers 11,512 rupees in net monthly profit saved, "
        "with zero external cloud or language model costs."
    ),
    (
        "slide_02.wav",
        "Why is 97 percent accuracy deceptive and dangerous? "
        "With a base fraud rate of only 1.2 to 2.8 percent, a naive model approving 100 percent of claims scores 98.77 percent accuracy, "
        "yet catches zero fraud and leaks all money. "
        "Under severe class imbalance, accuracy is an actively harmful metric. "
        "Our system is measured strictly by business outcomes: Recall at 40 reviews, and Net Rupees Saved after deducting customer goodwill vouchers."
    ),
    (
        "slide_03.wav",
        "We empirically tested Ritu's hypothesis on newer service partners. "
        "While newer partners average 3.8 percent fraud compared to 1 percent for veterans, 86.1 percent of newer partners have zero fraud. "
        "A blanket freeze would choke Kestrel's geographic expansion in tier-2 and tier-3 cities. "
        "Crucially, veteran franchises operating for 3 to 4 years, such as SP3207 and SP3095, drove the largest repeat fraud rupees by exploiting auto-approval rules."
    ),
    (
        "slide_04.wav",
        "During data analysis, we uncovered an unprompted vulnerability. "
        "On May 1st, Kestrel instituted auto-approvals for claims under 2,000 rupees without technician inspection. "
        "Fraud in that segment immediately spiked 5.7 times, from 0.54 percent to 3.09 percent. "
        "Rogue partners learned the cutoff and split larger repairs into multiple micro-claims. "
        "Our model includes an automated policy rule that flags rapid-fire cluster submissions, closing a 42,000 rupee quarterly leak."
    ),
    (
        "slide_05.wav",
        "Our architecture pairs a Histogram Gradient Boosting Classifier with calibrated domain policy rules. "
        "We adhered to a strict Honesty Protocol, committing prediction expectations to Git before scoring test data. "
        "We excluded leakage features like inspector notes, which were 81 percent missing in test claims. "
        "On the June 2026 benchmark of 2,252 unseen claims, our ensemble achieves an ROC-AUC of 0.819 and a PR-AUC of 0.354."
    ),
    (
        "slide_06.wav",
        "Farhan Sheikh's desk can review at most 40 claims per month, dedicating 45 minutes to each deep audit. "
        "Setting our decision threshold at 0.42 routes exactly 40 claims to his queue. "
        "This intercepts 15 confirmed frauds, saving 21,012 rupees, while 25 false alarms cost 9,500 rupees in customer courtesy vouchers. "
        "This produces an apex net savings of 11,512 rupees per month, or 34,536 rupees per quarter."
    ),
    (
        "slide_07.wav",
        "Our production-grade FastAPI microservice executes local CPU inference in just 34 milliseconds. "
        "By generating deterministic plain-English explanations instead of paid third-party language models, our API running cost is exactly zero rupees. "
        "The microservice is deployed live 24/7 on Render with Pydantic validation schemas, health monitoring, and interactive API documentation."
    ),
    (
        "slide_08.wav",
        "In complete transparency, here are the system's known trade-offs. "
        "We deliberately excluded technician notes and legacy source channels to prevent severe test leakage. "
        "The system faces higher uncertainty on cold-start partners with fewer than 5 historical jobs. "
        "Furthermore, because fraud schemes drift adversarially over 6 to 9 months, the model will require automated quarterly retraining."
    ),
    (
        "slide_09.wav",
        "If someone picks this up on Monday and I am unreachable, remember three things. "
        "First, run python run_all.py to execute the entire pipeline in 31 seconds. "
        "Second, the desk threshold is governed by CLAIM REVIEW THRESHOLD in app main.py, set to 0.42 for Farhan's 40-claim queue. "
        "Third, the production microservice is live on Render with all 8 unit tests passing. "
        "Thank you for your time."
    )
]

print("1. Generating speech audio with Microsoft Zira (Rate 1)...")
audio_files = []
durations = []

for idx, (filename, script) in enumerate(narrations, 1):
    wav_path = os.path.join("reports/audio", filename).replace("\\", "/")
    clean_script = script.replace("'", "''")
    ps_cmd = f"""
Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$synth.SelectVoice('Microsoft Zira Desktop')
$synth.Rate = 1
$synth.SetOutputToWaveFile('{wav_path}')
$synth.Speak('{clean_script}')
$synth.Dispose()
"""
    subprocess.run(["powershell", "-NoProfile", "-Command", ps_cmd], check=True)
    with wave.open(wav_path, "rb") as w:
        dur = w.getnframes() / float(w.getframerate())
        durations.append(dur)
        audio_files.append(wav_path)
    print(f"Slide {idx:02d} audio: {dur:.2f}s")

# 2. Build concatenated master audio with silence pauses
# Crossfade duration between slides: 0.8 seconds
TRANS_DUR = 0.8
concat_list_path = "reports/audio/concat_list.txt"
silence_wav_path = "reports/audio/silence.wav"

# Create a 0.8s silence wav matching sample rate of slide audio
with wave.open(audio_files[0], "rb") as w:
    params = w.getparams()
    sample_rate = params.framerate
    num_channels = params.nchannels
    sampwidth = params.sampwidth

silence_frames = int(sample_rate * TRANS_DUR)
silence_data = b'\x00' * (silence_frames * num_channels * sampwidth)
with wave.open(silence_wav_path, "wb") as sw:
    sw.setparams(params)
    sw.writeframes(silence_data)

master_audio_path = "reports/audio/master_narration.wav"
# Direct python wave concatenation for 100% precision
with wave.open(master_audio_path, "wb") as mw:
    mw.setparams(params)
    for i, a_path in enumerate(audio_files):
        with wave.open(a_path, "rb") as w:
            mw.writeframes(w.readframes(w.getnframes()))
        if i < len(audio_files) - 1:
            mw.writeframes(silence_data)

with wave.open(master_audio_path, "rb") as mw:
    total_audio_dur = mw.getnframes() / float(mw.getframerate())
print(f"Master audio generated: {master_audio_path} ({total_audio_dur:.2f}s / {total_audio_dur/60:.2f} mins)")

# 3. Render video matching exact durations
print("2. Generating synchronized 1080p video frames...")
FPS = 24
TRANS_FRAMES = int(FPS * TRANS_DUR)

slide_images = []
for idx in range(1, 10):
    img_path = f"reports/slides/slide_{idx:02d}.png"
    im = Image.open(img_path).convert("RGB")
    slide_images.append(np.array(im))

temp_video_path = "reports/temp_video.mp4"
writer = imageio.get_writer(temp_video_path, fps=FPS, codec="libx264", quality=8, macro_block_size=1)

for i in range(len(slide_images)):
    curr_slide = slide_images[i]
    dur = durations[i]
    hold_frames = int(FPS * dur)
    print(f"Adding Slide {i+1} ({hold_frames} frames)...")
    for _ in range(hold_frames):
        writer.append_data(curr_slide)
        
    if i < len(slide_images) - 1:
        next_slide = slide_images[i + 1]
        for t in range(TRANS_FRAMES):
            alpha = (t + 1) / (TRANS_FRAMES + 1)
            blended = ((1.0 - alpha) * curr_slide.astype(np.float32) + alpha * next_slide.astype(np.float32)).astype(np.uint8)
            writer.append_data(blended)

writer.close()
print("Raw video generated:", temp_video_path)

# 4. Mux Video + Audio using FFmpeg
final_video_path = "presentation_video.mp4"
print("3. Muxing video and narration audio with FFmpeg...")
cmd = [
    ffmpeg_exe,
    "-y",
    "-i", temp_video_path,
    "-i", master_audio_path,
    "-c:v", "copy",
    "-c:a", "aac",
    "-b:a", "192k",
    "-shortest",
    final_video_path
]
res = subprocess.run(cmd, capture_output=True, text=True)
if res.returncode != 0:
    print("FFmpeg error:", res.stderr)
else:
    print(f"SUCCESS! Final presentation video with voice created: {final_video_path}")
    v_size = os.path.getsize(final_video_path) / (1024 * 1024)
    print(f"File Size: {v_size:.2f} MB")

# Clean up temp video
if os.path.exists(temp_video_path):
    os.remove(temp_video_path)

print("COMPLETED VOICED PRESENTATION VIDEO GENERATION.")
