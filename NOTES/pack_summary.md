# Data Pack Summary — Kestrel Home (Variant C)

## 1. File Inventory and Schemas

| File | Rows | Columns | Date Range | Primary / Join Keys | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`train.csv`** | 12,029 | 14 | 2025-04-01 03:24 to 2026-06-30 23:55 (15 months) | `claim_id` (11,348 unique; 681 resubmissions = 1,362 duplicate rows) | Past claims with outcome (`is_fraud`). 11,669 non-fraud, 145 fraud, 215 undecided (`NaN`). |
| **`test_unlabelled.csv`** | 2,252 | 13 | 2026-07-01 00:41 to 2026-09-30 23:46 (3 months) | `claim_id` (2,252 unique; no duplicates) | Out-of-time test set to score. All rows sourced from `crm`. |
| **`sample_submission.csv`** | 2,252 | 2 | N/A | `claim_id`, `score` | Exact schema target for `predictions.csv`. Higher score = more likely fraudulent. |
| **`partners.csv`** | 380 | 4 | Onboarded: 2021-06-02 to 2026-08-27 | `partner_id` | Master partner table. Types: authorised_service_centre (174), franchise (134), freelance_technician (72). |
| **`products.csv`** | 21 | 4 | N/A | `sku` | Master product catalog across 7 families. List prices ₹1,199 to ₹28,999; warranties 12 or 24 months. |
| **`ops-policy.pdf`** | 1 page | Text | v4.1 | N/A | Kestrel operations policy covering costs, claim approval, review capacity, and systems migration. |
| **`email-thread.txt`** | 5 emails | Text | N/A | N/A | Internal messages between Ritu (Ops), Kabir (Account Lead), Tanmay (IT), Farhan (Finance), Meenal (Service Desk). |
| **`README.txt`** | Text | Text | N/A | N/A | File manifest, column descriptions, and submission target instructions. |

---

## 2. File Schemas & Column Definitions

### `train.csv` (12,029 rows, 14 columns) & `test_unlabelled.csv` (2,252 rows, 13 columns)
- **`claim_id`** (string): Unique claim identifier (e.g., `WC700000`). Note: In `train.csv`, 681 claim numbers appear twice (1,362 rows) due to partner resubmissions after initial rejection. In `test_unlabelled.csv`, all 2,252 are unique.
- **`submitted_at`** (string timestamp, `YYYY-MM-DD HH:MM`): Timestamp in Indian Standard Time (IST) as displayed in CRM.
- **`partner_id`** (string): Foreign key referencing `partners.csv` (`partner_id`). 0 unmatched keys in both train and test.
- **`sku`** (string): Foreign key referencing `products.csv` (`sku`). 0 unmatched keys in both train and test.
- **`product_serial`** (string): Serial number entered manually by the partner. 8,835 unique serials in train, 2,114 in test. 797 serials appear in both train and test.
- **`days_since_purchase`** (integer): Days elapsed between product purchase date and claim submission date (ranges 1 to 729 days). No negative values.
- **`claim_amount_inr`** (float): Claimed payout amount in Indian Rupees (₹350.0 to ₹8,420.0). No claims exceed 100% of the product list price (max ratio = 0.998).
- **`photo_attached`** (string, `Y`/`N`): Indicator of whether photographs were attached.
- **`partner_inspected`** (string, `Y`/`N`): Inspection sign-off by service partner. Sinks sharply after May 1, 2026 due to auto-approval policy.
- **`claim_description`** (string): Categorical fault description entered by partner (16 unique categories, e.g., 'power button not working', 'motor not running', 'water leaking').
- **`inspector_note`** (string): Free-text note from inspector. Missing in 3,237 train rows (26.9%) and 1,829 test rows (81.2%).
- **`customer_prior_claims`** (integer): Count of earlier warranty claims filed by this customer (ranges 0 to 6).
- **`source`** (string): System of origin: `legacy_zoho` (before 1 Oct 2025) or `crm` (from 1 Oct 2025 onwards). All test claims are `crm`.
- **`is_fraud`** (float, `train.csv` only): Ground truth label. `1.0` = fraud, `0.0` = genuine, `NaN` / blank = undecided at export.

### `partners.csv` (380 rows, 4 columns)
- **`partner_id`** (string, PK): Identifier `SP3000` to `SP3379`.
- **`city`** (string): 20 unique cities across India (e.g., Jaipur, Hubballi, Chennai, Kota, Pune, Bhopal, Lucknow).
- **`onboarded_date`** (date, `YYYY-MM-DD`): Partner onboarding date (2021-06-02 to 2026-08-27).
- **`partner_type`** (string): `authorised_service_centre` (174), `franchise` (134), `freelance_technician` (72).

### `products.csv` (21 rows, 4 columns)
- **`sku`** (string, PK): Product SKU code (e.g., `KH-AF-01` to `KH-WP-03`).
- **`family`** (string): 7 product lines: `Air Fryer` (3), `Ceiling Fan` (3), `Induction Cooktop` (3), `Mixer Grinder` (3), `Robot Vacuum` (3), `Room Heater` (3), `Water Purifier` (3).
- **`list_price_inr`** (integer): Retail price in INR (₹1,199 to ₹28,999).
- **`warranty_months`** (integer): Standard warranty duration in months (12 or 24 months).

---

## 3. Ground Truth Label & Base Rate

In `train.csv`:
- Total rows: **12,029**
- Resolved non-fraud (`0.0`): **11,669** (97.01% of all rows, 98.77% of resolved rows)
- Confirmed fraud (`1.0`): **145** (1.21% of all rows, 1.23% of resolved rows)
- Undecided / open (`NaN`): **215** (1.79% of all rows)
  - All 215 unresolved claims originate from the `crm` system between October 2025 and June 2026.
- **Base Rate on Resolved Cases:** **1.227%** (145 / 11,814).
- **The "Do-Nothing" Accuracy Trap:**
  - A trivial baseline model that predicts `0` (genuine) for every claim achieves an accuracy of **98.77%** on resolved cases.
  - The Board's stated KPI threshold of **>97% accuracy** is already surpassed by doing literally nothing. Accuracy is completely uninformative in this extreme class imbalance regime.

---

## 4. Business Numbers, Costs, Capacities, and Operational Rules

All exact figures and quotes extracted from `ops-policy.pdf` and `email-thread.txt`:

1. **Investigation Desk Capacity:**
   - **Quote:** *"The claims investigation desk can review at most 40 claims per month."* (`ops-policy.pdf` §5, reaffirmed by Farhan Sheikh in `email-thread.txt`: *"The investigation desk can look at forty claims a month, no more."*)
   - **Monthly Volume:** ~750 claims per month (`email-thread.txt` / brief).
   - **Operational Review Rate ($k$):** $\frac{40}{750} \approx 5.33\%$ of monthly claim volume. Over a 3-month test period (~2,252 claims, ~750/mo), the desk can review at most $40 \times 3 = 120$ claims ($k = 120$).

2. **Cost of Delaying Genuine Customers (False Positive Cost):**
   - **Quote:** *"A genuine claim held for review delays the customer's repair; Kestrel issues goodwill worth Rs 380 on average when that happens."* (`ops-policy.pdf` §4, reaffirmed by Farhan Sheikh: *"Every genuine customer we hold costs us goodwill."*)
   - **$C_{\text{FP}} = ₹380$** per genuine claim incorrectly held.

3. **Value of Fraud Prevented (True Positive Benefit):**
   - **Quote:** *"A fraudulent claim paid out costs the full claim amount."* (`ops-policy.pdf` §4).
   - Preventing a fraudulent claim saves the full claim amount: **Benefit = $\text{claim\_amount\_inr}$**.

4. **Service Contact Blended Cost:**
   - **Quote:** *"Blended cost of a service contact: Rs 260."* (`ops-policy.pdf` §4).

5. **Net Rupee Impact Formula:**
   $$\text{Net Rupee Impact} = \sum_{i \in \text{Flagged Fraud}} \text{claim\_amount\_inr}_i - (₹380 \times N_{\text{Flagged Genuine}}) - (\text{Investigation Cost} \times N_{\text{Flagged Total}})$$
   *(Note: Depending on whether investigation cost is accounted as desk fixed overhead or blended contact cost ₹260, sensitivity analysis will evaluate both ₹0 additional marginal cost and ₹260 marginal cost).*

6. **The May 1, 2026 Auto-Approval Policy Change:**
   - **Quote:** *"Until 30 Apr 2026 every claim required partner inspection sign-off. From 1 May 2026, claims under Rs 2,000 are auto-approved without inspection to cut turnaround time. Larger claims still need inspection."* (`ops-policy.pdf` §5).
   - **Empirical Effect:** In `train.csv`, claims under ₹2,000 had a fraud rate of 0.54% before May 2026. Following the auto-approval rule, fraud on claims under ₹2,000 spiked to **3.09%** (a 5.7x increase), confirming Meenal Joshi's observation: *"Since the May change the small claims have exploded - my team sees the same few outlets again and again."*
   - In `test_unlabelled.csv`, 77.0% (1,734 / 2,252) of claims are under ₹2,000 and have `partner_inspected == 'N'`.

7. **System Migration & Undecided Label Artifact:**
   - **Quote:** *"Until 30 Sep 2025 service and order operations ran on the legacy Zoho Desk instance. From 1 Oct 2025 everything runs on the Kestrel CRM. Legacy records were migrated in bulk; the source column says which system a row came from."* (`ops-policy.pdf` §9).
   - **Data Trap:** Tanmay Kulkarni notes: *"Cases still under investigation are blank in the CRM - but Zoho couldn't store blanks, so the old ones came across as 0."* In `train.csv`, `legacy_zoho` records have 0 `NaN`s because unresolved claims were coerced to 0 during migration. `crm` retains true `NaN` values for unresolved claims.

8. **Resubmitted Claims:**
   - **Quote:** *"Partners re-submit claims when they're bounced, so the same claim number can show up twice."* (`email-thread.txt`).
   - In `train.csv`, 681 claim IDs appear exactly twice (spaced 8 to 120 hours apart, mean 74.2 hours). Their labels are 100% consistent within each pair.

---

## 5. Submission Requirements & Rules (README.txt)

- **Target Output:** `predictions.csv` in the root repository.
- **Format:** Must match `sample_submission.csv` exactly:
  - Header: `claim_id,score`
  - Exactly 2,252 rows matching `test_unlabelled.csv` row-for-row.
  - Score interpretation: **Higher score = more likely fraudulent** ($P(\text{fraud})$ or risk score).
  - No missing values, no `NaN`, no `inf`, purely numeric scores.
