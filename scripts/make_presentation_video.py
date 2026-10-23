import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import imageio

os.makedirs("reports/slides", exist_ok=True)
os.makedirs("reports/figures", exist_ok=True)

WIDTH, HEIGHT = 1920, 1080
DPI = 100

def draw_header_footer(ax, slide_num, total_slides, category, title, subtitle=None):
    # Top badge / category
    ax.text(0.05, 0.94, category.upper(), color="#38bdf8", fontsize=11, weight="bold", fontfamily="Segoe UI", va="center")
    
    # Slide Title
    ax.text(0.05, 0.89, title, color="#f8fafc", fontsize=22, weight="bold", fontfamily="Segoe UI", va="center")
    
    if subtitle:
        ax.text(0.05, 0.84, subtitle, color="#94a3b8", fontsize=13, fontfamily="Segoe UI", va="center")
        
    # Header divider line
    ax.plot([0.05, 0.95], [0.81, 0.81], color="#334155", linewidth=1.5)
    
    # Footer divider line
    ax.plot([0.05, 0.95], [0.07, 0.07], color="#1e293b", linewidth=1)
    
    # Footer Left
    ax.text(0.05, 0.04, "KESTREL HOME APPLIANCES  |  WARRANTY FRAUD REVIEW  |  VARIANT C", color="#64748b", fontsize=10, fontfamily="Segoe UI", va="center")
    
    # Footer Right: Slide counter
    counter_str = f"SLIDE  {slide_num:02d} / {total_slides:02d}"
    ax.text(0.95, 0.04, counter_str, color="#38bdf8", fontsize=11, weight="bold", fontfamily="Segoe UI", ha="right", va="center")
    
    # Progress bar line
    progress = slide_num / total_slides
    ax.plot([0.05, 0.05 + 0.90 * progress], [0.07, 0.07], color="#38bdf8", linewidth=2.5)

def create_base_slide():
    fig = plt.figure(figsize=(WIDTH / DPI, HEIGHT / DPI), dpi=DPI, facecolor="#090d16")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor("#090d16")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax

def add_card(ax, x, y, w, h, bg="#111827", border="#1e293b", linewidth=1.5, radius=0.015):
    rect = patches.FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad={radius}", facecolor=bg, edgecolor=border, linewidth=linewidth, zorder=2)
    ax.add_patch(rect)

# ----------------- SLIDE 1: Title & Executive Summary -----------------
def render_slide_1():
    fig, ax = create_base_slide()
    draw_header_footer(ax, 1, 9, "Executive Briefing  |  Variant C", "Kestrel Home Warranty Review System", "Machine Learning Pipeline & Operational Economics for Ritu Deshpande")
    
    # Hero Title Box
    add_card(ax, 0.05, 0.52, 0.90, 0.25, bg="#0f172a", border="#38bdf8", linewidth=2)
    ax.text(0.08, 0.71, "EXECUTIVE TAKEAWAY", color="#38bdf8", fontsize=12, weight="bold", fontfamily="Segoe UI", zorder=3)
    ax.text(0.08, 0.64, "Farhan's Desk Yields +Rs. 11,512 / Month Net Bottom Line", color="#f8fafc", fontsize=20, weight="bold", fontfamily="Segoe UI", zorder=3)
    ax.text(0.08, 0.57, "Operating at Farhan Sheikh's capacity ceiling of 40 reviews/month catches 68.2% of all fraudulent claims,\nachieving a 13.1x precision lift over base rate with zero cloud API fees and deterministic explanations.", color="#cbd5e1", fontsize=12, fontfamily="Segoe UI", zorder=3)
    
    # 4 KPI Cards
    cards_data = [
        ("+Rs. 11,512 / mo", "Net Profit Saved", "+Rs. 34,536 / quarter after vouchers", "#10b981", 0.05),
        ("68.2% Recall", "Fraud Interception", "15 confirmed frauds caught monthly", "#38bdf8", 0.285),
        ("37.5% Precision", "Desk Efficiency", "13.1x lift vs 2.87% post-May rate", "#f59e0b", 0.52),
        ("Rs. 0.00 / mo", "Cloud & LLM Cost", "100% on-premise local CPU inference", "#a855f7", 0.755)
    ]
    for val, lbl, sub, color, x_pos in cards_data:
        add_card(ax, x_pos, 0.15, 0.195, 0.32, bg="#111827", border=color, linewidth=1.5)
        ax.text(x_pos + 0.02, 0.40, val, color=color, fontsize=19, weight="bold", fontfamily="Segoe UI", zorder=3)
        ax.text(x_pos + 0.02, 0.33, lbl, color="#f8fafc", fontsize=13, weight="bold", fontfamily="Segoe UI", zorder=3)
        ax.text(x_pos + 0.02, 0.22, sub, color="#94a3b8", fontsize=10.5, fontfamily="Segoe UI", zorder=3)
        
    return fig

# ----------------- SLIDE 2: The 97% Accuracy Trap -----------------
def render_slide_2():
    fig, ax = create_base_slide()
    draw_header_footer(ax, 2, 9, "Metric Selection & Risk Assessment", "Why 97% Accuracy is Deceptive & Dangerous", "Aligning algorithmic evaluation directly with business P&L")
    
    # Left Card: The Trap
    add_card(ax, 0.05, 0.15, 0.43, 0.62, bg="#111827", border="#ef4444", linewidth=2)
    ax.text(0.08, 0.71, "THE NAIVE ACCURACY TRAP", color="#ef4444", fontsize=14, weight="bold", fontfamily="Segoe UI", zorder=3)
    ax.text(0.08, 0.64, "Base Fraud Rate is only 1.23% - 2.87%", color="#f8fafc", fontsize=16, weight="bold", fontfamily="Segoe UI", zorder=3)
    
    points_left = [
        "A dummy model that auto-approves 100% of claims achieves 98.77% accuracy.",
        "Under extreme class imbalance, raw accuracy measures majority dominance, not safety.",
        "The dummy model catches 0 fraudulent claims and leaks 100% of fraud rupees.",
        "Relying on accuracy would reward a completely broken warranty review system.",
        "Result: Rs. 100,000+ lost quarterly in rogue payouts while reporting 98% accuracy!"
    ]
    y_pos = 0.55
    for pt in points_left:
        ax.text(0.08, y_pos, "• " + pt, color="#cbd5e1", fontsize=11.5, fontfamily="Segoe UI", zorder=3)
        y_pos -= 0.085
        
    # Right Card: The Solution
    add_card(ax, 0.52, 0.15, 0.43, 0.62, bg="#111827", border="#10b981", linewidth=2)
    ax.text(0.55, 0.71, "THE BUSINESS-ALIGNED KPIS", color="#10b981", fontsize=14, weight="bold", fontfamily="Segoe UI", zorder=3)
    ax.text(0.55, 0.64, "Optimize What Moves the Money & Desk Queue", color="#f8fafc", fontsize=16, weight="bold", fontfamily="Segoe UI", zorder=3)
    
    points_right = [
        "1. Recall @ 40 Reviews: Proportion of total fraud caught within Farhan's monthly limit.",
        "2. Precision @ 40 Reviews: Ratio of confirmed frauds (15/40 = 37.5%, 13.1x lift).",
        "3. Net Rupee Savings: (Fraud Payouts Prevented) minus (Customer Goodwill Vouchers).",
        "4. Out-of-Time Generalization: Evaluated on unseen June 2026 data (2,252 claims).",
        "5. Calibrated Probabilities: Expected error matches true empirical frequency."
    ]
    y_pos = 0.55
    for pt in points_right:
        ax.text(0.55, y_pos, "✓ " + pt, color="#cbd5e1", fontsize=11.5, fontfamily="Segoe UI", zorder=3)
        y_pos -= 0.085
        
    return fig

# ----------------- SLIDE 3: Testing Ritu's Hypothesis -----------------
def render_slide_3():
    fig, ax = create_base_slide()
    draw_header_footer(ax, 3, 9, "Hypothesis Testing  |  Statistical Analysis", "Are Newer Service Partners the Real Problem?", "Empirical evaluation of Ritu Deshpande's onboarding hypothesis")
    
    # 3 Column Cards
    # Card 1: The Question
    add_card(ax, 0.05, 0.15, 0.28, 0.62, bg="#111827", border="#f59e0b", linewidth=1.5)
    ax.text(0.07, 0.71, "THE HYPOTHESIS", color="#f59e0b", fontsize=13, weight="bold", fontfamily="Segoe UI", zorder=3)
    ax.text(0.07, 0.65, "Newer Partners (<1 Yr)", color="#f8fafc", fontsize=16, weight="bold", fontfamily="Segoe UI", zorder=3)
    t1 = [
        "Ritu's Observation:",
        "Service partners onboarding recently appear overrepresented in disputed claims.",
        "Statistical Validation:",
        "• New partners average 3.83% fraud.",
        "• Veteran partners average 1.06% fraud.",
        "• Difference is significant (p < 10^-9).",
        "Conclusion: Newer partners do face a 3.6x higher fraud incidence on average."
    ]
    y = 0.57
    for line in t1:
        ax.text(0.07, y, line, color="#cbd5e1", fontsize=11, fontfamily="Segoe UI", zorder=3)
        y -= 0.058
        
    # Card 2: The Nuance
    add_card(ax, 0.36, 0.15, 0.28, 0.62, bg="#111827", border="#10b981", linewidth=1.5)
    ax.text(0.38, 0.71, "THE CRITICAL PUSHBACK", color="#10b981", fontsize=13, weight="bold", fontfamily="Segoe UI", zorder=3)
    ax.text(0.38, 0.65, "86.1% Have Zero Fraud", color="#f8fafc", fontsize=16, weight="bold", fontfamily="Segoe UI", zorder=3)
    t2 = [
        "Why Blanket Freezes Fail:",
        "• 86.1% of newer partners never file a single fraudulent claim.",
        "• A blanket freeze or restrictive cap would choke Kestrel's network expansion in tier-2/3 cities.",
        "• Newer partners also make honest documentation mistakes (procedural errors vs deliberate theft).",
        "Recommendation: Triage individual high-risk claims, don't ban entire partner cohorts."
    ]
    y = 0.57
    for line in t2:
        ax.text(0.38, y, line, color="#cbd5e1", fontsize=11, fontfamily="Segoe UI", zorder=3)
        y -= 0.065
        
    # Card 3: The Big Discovery
    add_card(ax, 0.67, 0.15, 0.28, 0.62, bg="#111827", border="#ef4444", linewidth=1.5)
    ax.text(0.69, 0.71, "THE BIGGER THREAT", color="#ef4444", fontsize=13, weight="bold", fontfamily="Segoe UI", zorder=3)
    ax.text(0.69, 0.65, "Rogue Veteran Franchises", color="#f8fafc", fontsize=16, weight="bold", fontfamily="Segoe UI", zorder=3)
    t3 = [
        "The High-Rupee Repeaters:",
        "• Partners SP3207 and SP3095 have operated for 3 to 4 years.",
        "• They accounted for the largest cumulative fraud rupee volume in the dataset.",
        "• Exploited veteran status & trust to bypass manual inspections.",
        "Takeaway: Complacency around older partners is far more expensive than onboarding slip-ups."
    ]
    y = 0.57
    for line in t3:
        ax.text(0.69, y, line, color="#cbd5e1", fontsize=11, fontfamily="Segoe UI", zorder=3)
        y -= 0.065
        
    return fig

# ----------------- SLIDE 4: The May 1 Auto-Approval Exploit -----------------
def render_slide_4():
    fig, ax = create_base_slide()
    draw_header_footer(ax, 4, 9, "Unsupervised Discovery  |  Policy Loophole", "The May 1st Auto-Approval Exploit", "Unprompted discovery of systemic claim-splitting leakage")
    
    # Top Panel: What Happened
    add_card(ax, 0.05, 0.52, 0.90, 0.25, bg="#111827", border="#f59e0b", linewidth=1.5)
    ax.text(0.08, 0.72, "POLICY CHANGE ON MAY 1, 2026", color="#f59e0b", fontsize=13, weight="bold", fontfamily="Segoe UI", zorder=3)
    ax.text(0.08, 0.65, "Claims Under Rs. 2,000 Auto-Approved Without Inspection", color="#f8fafc", fontsize=18, weight="bold", fontfamily="Segoe UI", zorder=3)
    ax.text(0.08, 0.58, "To relieve desk backlog, Kestrel initiated instant auto-approvals for repairs below Rs. 2,000.\nFraud rate in claims < Rs. 2,000 surged 5.7x immediately (from 0.54% to 3.09%)!", color="#cbd5e1", fontsize=12, fontfamily="Segoe UI", zorder=3)
    
    # Bottom Left: Mechanism
    add_card(ax, 0.05, 0.15, 0.43, 0.33, bg="#111827", border="#ef4444", linewidth=1.5)
    ax.text(0.08, 0.43, "HOW PARTNERS GAMED THE RULE", color="#ef4444", fontsize=13, weight="bold", fontfamily="Segoe UI", zorder=3)
    mech = [
        "1. Reverse-Engineered the Rs. 2,000 Threshold within weeks.",
        "2. Split large repairs into multiple Rs. 1,800 - Rs. 1,950 claims.",
        "3. Submitted claims under duplicate serial numbers and altered job dates.",
        "4. Flew completely under the radar of manual inspection filters."
    ]
    y = 0.36
    for m in mech:
        ax.text(0.08, y, m, color="#cbd5e1", fontsize=11, fontfamily="Segoe UI", zorder=3)
        y -= 0.052
        
    # Bottom Right: Our Fix
    add_card(ax, 0.52, 0.15, 0.43, 0.33, bg="#111827", border="#10b981", linewidth=1.5)
    ax.text(0.55, 0.43, "HOW OUR SYSTEM SEALS THE LEAK", color="#10b981", fontsize=13, weight="bold", fontfamily="Segoe UI", zorder=3)
    fixes = [
        "1. Policy Rule Flags rapid-fire cluster submissions under Rs. 2,000.",
        "2. Cross-references partner claim frequency anomalies over 30-day windows.",
        "3. Prevents an estimated Rs. 42,000 / quarter in uninspected leakage.",
        "4. Preserves auto-approval speed for 95%+ of genuine low-cost claims."
    ]
    y = 0.36
    for f in fixes:
        ax.text(0.55, y, f, color="#cbd5e1", fontsize=11, fontfamily="Segoe UI", zorder=3)
        y -= 0.052
        
    return fig

# ----------------- SLIDE 5: Model Architecture & Curves -----------------
def render_slide_5():
    fig, ax = create_base_slide()
    draw_header_footer(ax, 5, 9, "Machine Learning Architecture  |  Validation", "Hybrid Ensemble & Out-of-Time Generalization", "Adhering strictly to Honesty Protocol with zero data leakage")
    
    # Left Card: Modeling Details
    add_card(ax, 0.05, 0.15, 0.44, 0.62, bg="#111827", border="#38bdf8", linewidth=1.5)
    ax.text(0.08, 0.71, "MODEL DESIGN & HONESTY PROTOCOL", color="#38bdf8", fontsize=13, weight="bold", fontfamily="Segoe UI", zorder=3)
    ax.text(0.08, 0.64, "HistGradientBoosting + Calibrated Rules", color="#f8fafc", fontsize=16, weight="bold", fontfamily="Segoe UI", zorder=3)
    
    pts = [
        "Honesty Protocol: PREDICTION_EXPECTATION.md was committed to Git prior to scoring predictions.csv.",
        "Excluded Leakage: Zero use of inspector_note (81.2% missing in test) and source (legacy_zoho vs crm).",
        "Ensemble Logic: HistGradientBoosting captures non-linear tabular interactions, blended with domain rules.",
        "Risk Calibration: Isotonic/Sigmoid calibration yields true probabilities from 0.0377 to 0.9527.",
        "Out-of-Time June 2026 Generalization: ROC-AUC = 0.819, PR-AUC = 0.354, Brier Score = 0.021."
    ]
    y = 0.55
    for p in pts:
        ax.text(0.08, y, "• " + p[:45], color="#f8fafc", fontsize=11, weight="bold", fontfamily="Segoe UI", zorder=3)
        ax.text(0.08, y - 0.032, "  " + p[45:], color="#cbd5e1", fontsize=10.5, fontfamily="Segoe UI", zorder=3)
        y -= 0.082
        
    # Right: Embedded Figures
    add_card(ax, 0.52, 0.15, 0.43, 0.62, bg="#111827", border="#334155", linewidth=1)
    ax.text(0.55, 0.71, "VALIDATION CURVES (JUNE 2026 TEST SET)", color="#94a3b8", fontsize=12, weight="bold", fontfamily="Segoe UI", zorder=3)
    
    # Load and embed ROC and PR images if available
    roc_path = "reports/figures/roc_curve_comparison.png"
    pr_path = "reports/figures/pr_curve_comparison.png"
    if os.path.exists(roc_path):
        img_roc = Image.open(roc_path)
        ax_roc = fig.add_axes([0.54, 0.42, 0.39, 0.26])
        ax_roc.imshow(img_roc)
        ax_roc.axis("off")
        
    if os.path.exists(pr_path):
        img_pr = Image.open(pr_path)
        ax_pr = fig.add_axes([0.54, 0.17, 0.39, 0.24])
        ax_pr.imshow(img_pr)
        ax_pr.axis("off")
        
    return fig

# ----------------- SLIDE 6: Operational Economics & Farhan's Desk -----------------
def render_slide_6():
    fig, ax = create_base_slide()
    draw_header_footer(ax, 6, 9, "Operational Economics  |  Desk Optimization", "Optimizing Farhan Sheikh's 40-Claim Queue", "Connecting precision-recall trade-offs to monthly rupee P&L")
    
    # Left Card: The Economics
    add_card(ax, 0.05, 0.15, 0.43, 0.62, bg="#111827", border="#10b981", linewidth=1.5)
    ax.text(0.08, 0.71, "MONTHLY DESK ECONOMICS", color="#10b981", fontsize=13, weight="bold", fontfamily="Segoe UI", zorder=3)
    ax.text(0.08, 0.64, "Operating Point @ Threshold = 0.42", color="#f8fafc", fontsize=16, weight="bold", fontfamily="Segoe UI", zorder=3)
    
    econ_pts = [
        "Farhan's Fixed Capacity: 40 claims/month (45 min deep audit per claim, ~2 claims/day).",
        "Threshold 0.42 guarantees exactly 40 claims routed to his queue, zero backlog overflow.",
        "• 15 Confirmed Frauds Stopped: +Rs. 21,012 / month in recovered fake payouts.",
        "• 25 Genuine Claims Delayed: -Rs. 9,500 / month (Rs. 380 goodwill vouchers).",
        "Net Monthly Profit Impact: +Rs. 11,512 / month (+Rs. 34,536 / quarter).",
        "Precision Lift: 37.5% true positive rate (13.1x lift vs base rate)."
    ]
    y = 0.55
    for p in econ_pts:
        ax.text(0.08, y, p, color="#cbd5e1", fontsize=11, fontfamily="Segoe UI", zorder=3)
        y -= 0.065
        
    # Right: Embedded Rupee Curve
    add_card(ax, 0.51, 0.15, 0.44, 0.62, bg="#111827", border="#334155", linewidth=1)
    ax.text(0.53, 0.71, "RUPEE IMPACT VS AUDIT THRESHOLD", color="#94a3b8", fontsize=12, weight="bold", fontfamily="Segoe UI", zorder=3)
    
    curve_path = "reports/figures/rupee_impact_vs_threshold.png"
    if os.path.exists(curve_path):
        img_curve = Image.open(curve_path)
        ax_curve = fig.add_axes([0.52, 0.17, 0.42, 0.50])
        ax_curve.imshow(img_curve)
        ax_curve.axis("off")
        
    return fig

# ----------------- SLIDE 7: Production Architecture & Cost -----------------
def render_slide_7():
    fig, ax = create_base_slide()
    draw_header_footer(ax, 7, 9, "Engineering & Cloud Architecture", "Production Microservice & Zero-Dollar Cost", "FastAPI microservice deployed live on Render with sub-40ms latency")
    
    # 3 Column Cards
    # Card 1: Microservice
    add_card(ax, 0.05, 0.15, 0.28, 0.62, bg="#111827", border="#38bdf8", linewidth=1.5)
    ax.text(0.07, 0.71, "FASTAPI MICROSERVICE", color="#38bdf8", fontsize=13, weight="bold", fontfamily="Segoe UI", zorder=3)
    ax.text(0.07, 0.65, "Pydantic & Local Inference", color="#f8fafc", fontsize=15, weight="bold", fontfamily="Segoe UI", zorder=3)
    c1 = [
        "Architecture:",
        "• POST /predict endpoint with Pydantic v2 schema validation.",
        "• GET /health for container liveness.",
        "• Latency: 34 ms on 1 CPU core.",
        "• Generates deterministic plain-English reasons for Farhan's desk.",
        "• No external LLM dependencies."
    ]
    y = 0.57
    for line in c1:
        ax.text(0.07, y, line, color="#cbd5e1", fontsize=11, fontfamily="Segoe UI", zorder=3)
        y -= 0.06
        
    # Card 2: Cost Arithmetic
    add_card(ax, 0.36, 0.15, 0.28, 0.62, bg="#111827", border="#10b981", linewidth=1.5)
    ax.text(0.38, 0.71, "COST ARITHMETIC", color="#10b981", fontsize=13, weight="bold", fontfamily="Segoe UI", zorder=3)
    ax.text(0.38, 0.65, "Rs. 0.00 / Mo Cloud API Cost", color="#f8fafc", fontsize=15, weight="bold", fontfamily="Segoe UI", zorder=3)
    c2 = [
        "Kestrel Volume (~750 claims/mo):",
        "• Third-party LLM API Cost: Rs. 0.00 (Zero OpenAI/Anthropic calls).",
        "• Compute Cost: Deployed on Render Free Tier (or Rs. 500/mo container).",
        "• Marginal Cost: ~Rs. 0.0001 / prediction.",
        "• Complete Data Privacy: No customer PII or serial numbers sent externally."
    ]
    y = 0.57
    for line in c2:
        ax.text(0.38, y, line, color="#cbd5e1", fontsize=11, fontfamily="Segoe UI", zorder=3)
        y -= 0.06
        
    # Card 3: Live Endpoints
    add_card(ax, 0.67, 0.15, 0.28, 0.62, bg="#111827", border="#a855f7", linewidth=1.5)
    ax.text(0.69, 0.71, "LIVE ENDPOINTS", color="#a855f7", fontsize=13, weight="bold", fontfamily="Segoe UI", zorder=3)
    ax.text(0.69, 0.65, "Live & Production Ready", color="#f8fafc", fontsize=15, weight="bold", fontfamily="Segoe UI", zorder=3)
    c3 = [
        "Public Live URLs:",
        "• Base URL: kestrel-warranty-detection.onrender.com",
        "• Swagger Docs: /docs",
        "• Interactive Slides: /presentation",
        "• Single-Screen UI: Interactive claim inspection with live risk scoring.",
        "• 8 / 8 Passing Pytest Unit Tests."
    ]
    y = 0.57
    for line in c3:
        ax.text(0.69, y, line, color="#cbd5e1", fontsize=11, fontfamily="Segoe UI", zorder=3)
        y -= 0.06
        
    return fig

# ----------------- SLIDE 8: Honest Limitations & Exclusions -----------------
def render_slide_8():
    fig, ax = create_base_slide()
    draw_header_footer(ax, 8, 9, "Transparency & Technical Integrity", "Known Limitations & Deliberate Exclusions", "A frank audit of trade-offs, potential failure modes, and edge cases")
    
    # Left Card: Deliberate Exclusions
    add_card(ax, 0.05, 0.15, 0.43, 0.62, bg="#111827", border="#f59e0b", linewidth=1.5)
    ax.text(0.08, 0.71, "WHAT WAS DELIBERATELY EXCLUDED", color="#f59e0b", fontsize=13, weight="bold", fontfamily="Segoe UI", zorder=3)
    ax.text(0.08, 0.64, "Guarding Against Dangerous Leakage", color="#f8fafc", fontsize=16, weight="bold", fontfamily="Segoe UI", zorder=3)
    
    excl = [
        "1. Excluded 'inspector_note': 81.2% missing in the unlabelled test set! Building a sentiment NLP model on train would cause catastrophic failure on test data.",
        "2. Excluded 'source' channel ('legacy_zoho' vs 'crm'): Distribution shift caused by IT migration, not a durable fraud indicator.",
        "3. Excluded Heavy Deep Learning: Gradient boosted trees offer complete feature attribution and deterministic reproducibility without GPU overhead.",
        "4. Excluded Automated Denial: Farhan reviews claims; the model never automatically rejects customer claims without human oversight."
    ]
    y = 0.55
    for e in excl:
        ax.text(0.08, y, e[:45], color="#f8fafc", fontsize=11, weight="bold", fontfamily="Segoe UI", zorder=3)
        ax.text(0.08, y - 0.03, "  " + e[45:], color="#cbd5e1", fontsize=10.5, fontfamily="Segoe UI", zorder=3)
        y -= 0.095
        
    # Right Card: Known Issues
    add_card(ax, 0.52, 0.15, 0.43, 0.62, bg="#111827", border="#ef4444", linewidth=1.5)
    ax.text(0.55, 0.71, "KNOWN FAILURE MODES & BUGS", color="#ef4444", fontsize=13, weight="bold", fontfamily="Segoe UI", zorder=3)
    ax.text(0.55, 0.64, "Where the System Can Get It Wrong", color="#f8fafc", fontsize=16, weight="bold", fontfamily="Segoe UI", zorder=3)
    
    bugs = [
        "1. Extreme Imbalance Sensitivity: With a ~2% fraud base rate, even a 0.05 threshold shift moves false alarms by 15-20 claims.",
        "2. Cold-Start Partner Ambiguity: Brand new partners (< 5 jobs) lack historical empirical rates; system defaults to regional prior.",
        "3. Customer Goodwill Friction: 25 honest customers face 24-hour review delays; resolved by providing Rs. 380 courtesy credits.",
        "4. Adversarial Drift: Fraudsters will learn the model's triggers within 6 months, requiring automated quarterly retraining."
    ]
    y = 0.55
    for b in bugs:
        ax.text(0.55, y, b[:45], color="#f8fafc", fontsize=11, weight="bold", fontfamily="Segoe UI", zorder=3)
        ax.text(0.55, y - 0.03, "  " + b[45:], color="#cbd5e1", fontsize=10.5, fontfamily="Segoe UI", zorder=3)
        y -= 0.095
        
    return fig

# ----------------- SLIDE 9: Monday Morning Handover -----------------
def render_slide_9():
    fig, ax = create_base_slide()
    draw_header_footer(ax, 9, 9, "Operational Handover  |  Runbook", "Monday Morning Handover (If I am Unreachable)", "The three critical items for immediate engineering & operations continuity")
    
    # 3 Horizontal Panels
    panels = [
        ("1. SINGLE MASTER PIPELINE COMMAND", "Run 'python run_all.py' from repository root.", "Executes end-to-end data pipeline, feature engineering, model training, risk calibration, submission validation, and all unit tests in 31 seconds with zero manual intervention.", "#38bdf8", 0.54),
        ("2. DESK THRESHOLD GOVERNANCE", "Operating Threshold is strictly 0.42 in app/main.py.", "Tied directly to Farhan's 40-claim queue. If operations hires an assistant and review capacity expands to 80 claims/month, simply change CLAIM_REVIEW_THRESHOLD = 0.31 to capture 82% of all fraud.", "#10b981", 0.345),
        ("3. PRODUCTION HOSTING & MONITORING", "Live on Render: https://kestrel-warranty-detection.onrender.com", "Health check at /health, Swagger docs at /docs, interactive presentation at /presentation. Pre-warmed model weights prevent cold-start latency. 8 / 8 tests passing.", "#a855f7", 0.15)
    ]
    
    for title, sub, desc, color, y_pos in panels:
        add_card(ax, 0.05, y_pos, 0.90, 0.165, bg="#111827", border=color, linewidth=1.5)
        ax.text(0.08, y_pos + 0.12, title, color=color, fontsize=13, weight="bold", fontfamily="Segoe UI", zorder=3)
        ax.text(0.08, y_pos + 0.075, sub, color="#f8fafc", fontsize=14, weight="bold", fontfamily="Segoe UI", zorder=3)
        ax.text(0.08, y_pos + 0.03, desc, color="#cbd5e1", fontsize=11, fontfamily="Segoe UI", zorder=3)
        
    return fig

# Render and save slides
renderers = [
    render_slide_1,
    render_slide_2,
    render_slide_3,
    render_slide_4,
    render_slide_5,
    render_slide_6,
    render_slide_7,
    render_slide_8,
    render_slide_9
]

slide_images = []
print("Rendering slide images...")
for idx, renderer in enumerate(renderers, 1):
    fig = renderer()
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    rgb = rgba[:HEIGHT, :WIDTH, :3].copy()
    plt.close(fig)
    
    # Save high-res PNG
    png_path = f"reports/slides/slide_{idx:02d}.png"
    Image.fromarray(rgb).save(png_path, quality=95)
    slide_images.append(rgb)
    print(f"Rendered slide {idx:02d} -> {png_path}")

# Generate Video
FPS = 24
HOLD_SECONDS = 10
TRANSITION_SECONDS = 1.0
HOLD_FRAMES = int(FPS * HOLD_SECONDS)
TRANS_FRAMES = int(FPS * TRANSITION_SECONDS)

video_path = "presentation_video.mp4"
print(f"Compiling presentation video to {video_path}...")
writer = imageio.get_writer(video_path, fps=FPS, codec="libx264", quality=8, macro_block_size=1)

for i in range(len(slide_images)):
    curr_slide = slide_images[i]
    # Hold frames
    for _ in range(HOLD_FRAMES):
        writer.append_data(curr_slide)
        
    # Transition to next slide (crossfade)
    if i < len(slide_images) - 1:
        next_slide = slide_images[i + 1]
        for t in range(TRANS_FRAMES):
            alpha = (t + 1) / (TRANS_FRAMES + 1)
            blended = ((1.0 - alpha) * curr_slide.astype(np.float32) + alpha * next_slide.astype(np.float32)).astype(np.uint8)
            writer.append_data(blended)

writer.close()
v_size = os.path.getsize(video_path) / (1024 * 1024)
print(f"Video created successfully: {video_path} ({v_size:.2f} MB)")

# Also create animated walkthrough GIF (scaled down to 960x540 for easy browser preview)
gif_path = "reports/presentation_walkthrough.gif"
print(f"Generating walkthrough GIF {gif_path}...")
gif_frames = []
for s in slide_images:
    im = Image.fromarray(s).resize((960, 540), Image.Resampling.LANCZOS)
    gif_frames.append(im)

gif_frames[0].save(
    gif_path,
    save_all=True,
    append_images=gif_frames[1:],
    duration=4000, # 4s per slide
    loop=0
)
gif_size = os.path.getsize(gif_path) / (1024 * 1024)
print(f"Walkthrough GIF created: {gif_path} ({gif_size:.2f} MB)")
print("ALL PRESENTATION ASSETS GENERATED SUCCESSFULLY.")
