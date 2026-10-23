# AI Usage Log & Honesty Protocol — Kestrel Home (Variant C)

This document provides a factual disclosure of how AI assistance (Anthropic Claude 3.5 Sonnet and Google Gemini via Antigravity IDE) was utilized during the development of this project, including mistakes encountered, discarded experiments, and human intervention points.

---

## 1. Summary of AI Tools Used
- **Primary AI Agent:** Antigravity AI Pair Programmer (Claude 3.5 Sonnet / Gemini engine).
- **Interface:** IDE Autonomous Agentic Loop with local shell and file tools.
- **Human Oversight:** Continuous instruction, specification enforcement, empirical verification of metrics, and sanity checking of statistical models.

---

## 2. Where AI Added Real Value
1. **Automated Document Extraction:** Rapid extraction and parsing of unstructured text from `ops-policy.pdf` and `email-thread.txt`, identifying operational constraints (desk capacity $k=40$, ₹380 goodwill loss, ₹260 blended contact cost, May 1 auto-approval change).
2. **Efficient Historical Feature Engineering:** Implementing vector-accelerated `bisect` routines to compute time-aware expanding partner fraud rates and serial frequencies in sub-second runtimes without target leakage.
3. **Comprehensive Test Suite & Schema Scaffold:** Rapid generation of Pydantic models, FastAPI routing endpoints, and complete pytest test suites for endpoint validation and submission verification.
4. **Clean Visualization Scripts:** Producing publication-quality Matplotlib figures for ROC, Precision-Recall, Calibration, and Rupee Impact curves.

---

## 3. Where AI Made Mistakes or Wasted Time
1. **The May 1 Regime Shift Trap:**  
   In early exploratory modeling, the AI trained a standard logistic regression model across the 2025–2026 dataset and observed that lower claim amounts had a negative coefficient (lower claim amount = lower fraud risk). This was true pre-May 2026, but completely reversed after May 1 when fraudsters began exploiting the ₹2,000 auto-approval rule. Without domain adjustment, the pure ML model scored poorly on post-May validation.  
   *Correction:* Human intervention directed the creation of interaction features (`under_2000` $\times$ `post_may` $\times$ `uninspected`) and a hybrid ensemble that explicitly weights the policy exploit.
2. **Windows Console Character Encoding Error:**  
   The initial Python training script attempted to print the Indian Rupee symbol (`\u20b9`) directly to the Windows command prompt, resulting in a `UnicodeEncodeError: 'charmap' codec can't encode character` crash in CP1252 mode.  
   *Correction:* Replaced unicode currency characters with ASCII `Rs.` in console print statements while preserving clean formatting in Markdown/DOCX outputs.
3. **Pandas Index KeyError in Bootstrap Resampling:**  
   In the bootstrap confidence interval loop, the AI passed pandas Series objects into NumPy array indexers (`claim_amounts[idx]`), causing a `KeyError: 39` when pandas attempted to match index labels instead of positional ilocs.  
   *Correction:* Wrapped all inputs with `np.asarray()` to enforce pure NumPy array slicing.

---

## 4. Discarded Experiments & Architectures
1. **Complex CatBoost / Deep Learning Models:**  
   Evaluated and discarded. In a dataset with 145 total positive cases, deep neural networks and complex GBDT architectures severely overfit and added unnecessary dependencies. `HistGradientBoosting` provided equivalent rank-ordering while training in under 2 seconds.
2. **External Paid LLM Explanation APIs:**  
   Experimented with calling LLMs for generating explanation strings. Discarded due to non-zero cost (₹0.10–₹0.50 per call), high latency (800ms+), and occasional hallucinated rationale. Replaced with deterministic rule templates that run locally in 0.1ms at ₹0 cost.
3. **Oversampling (SMOTE / Random Oversampling):**  
   Tested synthetic oversampling on training splits. Discarded because it distorted the empirical fraud distribution and harmed model probability calibration. Class-weighting (`class_weight='balanced'`) proved strictly superior.

---

## 5. Artifact Verification & Presentation Video
- **Presentation Video (1080p Full HD):** Stored in repository as [`presentation_video.mp4`](presentation_video.mp4) (8.62 MB, 24fps).
- **Live Streamable Video & Interactive Deck:** Hosted on Render at **[https://kestrel-warranty-detection.onrender.com/presentation](https://kestrel-warranty-detection.onrender.com/presentation)** (and direct stream at **[https://kestrel-warranty-detection.onrender.com/video](https://kestrel-warranty-detection.onrender.com/video)**).
- **Animated Walkthrough Preview:** [`reports/presentation_walkthrough.gif`](reports/presentation_walkthrough.gif)
- **Individual High-Res Slides:** [`reports/slides/`](reports/slides/)
- **Local Artifact Verification:** All figures, metrics, models, and predictions were executed locally in the environment and validated against ground-truth files.
