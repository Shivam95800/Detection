# MEMORANDUM

**TO:** Ritu Deshpande, Head of D2C Operations  
**FROM:** Applied Data Science Lead  
**DATE:** 23 October 2026  
**SUBJECT:** Warranty Fraud Review Model — Operational Strategy & Deployment Recommendations  

---

### 1. The Decision
We recommend deploying the **Kestrel Warranty Review Model** immediately as a real-time decision-support layer for Farhan Sheikh’s investigation desk. Instead of attempting to manually review 750 monthly claims, the model will route the **top 40 highest-risk claims each month** to your desk, allowing the remaining ~710 low-risk claims to be auto-approved and paid without customer friction.

---

### 2. The Number: What the Model Really Does, Honestly
At your current desk capacity of **40 reviews per month** (~5.3% of total claim volume):
- **Catches 68.2% of all fraudulent claims** before payout (15 out of 22 fraud cases intercepted in our out-of-time June benchmark).
- **Hits 37.5% Precision:** Out of 40 claims investigated, roughly 15 are confirmed frauds and 25 are genuine customer repairs held briefly. This is a **13.1x improvement** over investigating claims at random (where only 1 in 35 would be fraud).
- **Does not catch 100% of fraud:** Roughly 7 low-value or highly sophisticated fraud claims per month will slip through under current capacity limits.

---

### 3. The Rupees: Net Business Value & Cost Breakdown
Every genuine claim delayed costs Kestrel **₹380 in customer goodwill compensation** (`ops-policy.pdf` §4). Under the recommended 40-claim monthly review budget:
- **Gross Fraud Stopped:** **₹21,012 per month** (₹63,000+ per quarter) in fraudulent payouts prevented.
- **Goodwill Compensation (25 false alarms):** **-₹9,500 per month** (25 genuine repairs delayed $\times$ ₹380).
- **Investigation Desk Cost:** **₹0 additional marginal expense** (uses existing staff). If accounted at the ₹260 blended contact cost, expense is -₹10,400.
- **Net Rupee Impact:** **+₹11,512 net profit saved every month** (**+₹34,536 per quarter**).

---

### 4. Does Your "Newer Partners" Belief Hold?
**Partially true in the averages, but dangerously misleading as a management rule.**
- **The Surface Number:** Partners onboarded within the last 6 months do have a higher fraud rate (**3.83%** vs. **1.06%** for older partners).
- **The True Reality:** 86% of all service partners (315 of 366) have **never committed fraud**. Furthermore, our top three repeat fraud offenders are all **established franchise outlets operating for 2.5 to 4.0 years** (`SP3207` in Hyderabad with 14 frauds; `SP3095` in Bhopal with 11 frauds).
- **The Root Cause:** The spike in newer partner fraud was triggered by the **May 1 policy change**, where claims under ₹2,000 were auto-approved without inspection. A small cluster of 7 rogue partners onboarded in late 2025 exploited this rule. 
- **Action:** Do not penalize all new partners—most are honest and essential for Tier-2/3 expansion. Let the model flag specific suspicious claims regardless of partner age.

---

### 5. Why 97% Accuracy is the Wrong Target
A naive model that **approves every single claim and catches zero fraud** achieves **98.77% accuracy** simply because only 1.23% of claims are fraudulent. The Board's 97% accuracy KPI is mathematically achieved by doing nothing. In fraud operations, the right metrics are **Fraud Rupees Stopped** and **Recall at Desk Capacity (68.2%)**.

---

### 6. What You Should Do Next Week (4 Concrete Steps)
1. **Launch a 30-Day Shadow Pilot:** Feed incoming claims into the model. Route the top 40 flagged claims to Farhan’s team and compare findings against the model’s plain-English explanation codes.
2. **Close the May 1 Auto-Approval Loophole:** Mandate mandatory photographic evidence for sub-₹2,000 claims from outlets with less than 6 months of operating history.
3. **Track Investigation Outcomes in CRM:** Ensure investigators record definitive "Confirmed Fraud" vs "Genuine Fault" tags in CRM to eliminate undecided blanks.
4. **Schedule Monthly Model Retraining:** Implement monthly batch retraining to adapt to shifting fraud tactics across product seasons (e.g. room heaters in winter, purifiers in summer).

---

### 7. Limitations Stated Plainly
The model struggles with: (a) high-value first-time fraud claims on flagship vacuums with forged bills, (b) typo-prone manually typed serial numbers, and (c) newly onboarded partners committing fraud on Day 1 before establishing historical patterns.
