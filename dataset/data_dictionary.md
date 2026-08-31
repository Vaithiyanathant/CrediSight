# Data Dictionary — Loan Performance Intelligence Engine
> Intain Campus FinTech Challenge 2026 | AI Track
> Built from real HMDA public data (ffiec.cfpb.gov) — Version 1.0

---

## 1. loan_static_attributes.csv
Origination-level snapshot. One row per loan.

| Field | Type | Description | Example |
|---|---|---|---|
| loan_id | string | Unique loan ID | LN0000001 |
| origination_month | YYYY-MM | Month originated | 2019-06 |
| original_balance | float | Original UPB (USD) | 285000 |
| interest_rate | float | Annual note rate (%) | 5.875 |
| loan_term_months | int | Full loan term | 360 |
| credit_score_band | string | Credit score band | 700-739 |
| ltv_band | string | LTV ratio band | 81-90 |
| dti_band | string | DTI ratio band | 37-43 |
| state | string | Property state | CA |
| loan_purpose | string | Loan purpose | Purchase |
| occupancy_type | string | Occupancy | Primary |
| property_type | string | Property type | Single Family |
| servicer_name | string | Loan servicer | ServicerA |
| vintage_year | string | Origination year | 2019 |

---

## 2. loan_monthly_performance_train.csv
Monthly panel with targets. One row per loan per month (36 months).

| Field | Type | Description | Values |
|---|---|---|---|
| loan_id | string | FK to static table | |
| month_index | int | Sequential month (1-36) | 1-36 |
| reporting_month | YYYY-MM | Calendar month | 2021-01..2023-12 |
| origination_month | YYYY-MM | Loan origination month | |
| loan_age_months | int | Months since origination | 1-120+ |
| remaining_term_months | int | Remaining months | 0-360 |
| original_balance | float | Original UPB (USD) | |
| current_balance | float | Current UPB (USD) | |
| interest_rate | float | Annual note rate (%) | |
| credit_score_band | string | Credit band | <620..780+ |
| ltv_band | string | LTV band | <=60..>100 |
| dti_band | string | DTI band | <=28..>50 |
| state | string | Property state | |
| loan_purpose | string | Loan purpose | |
| occupancy_type | string | Occupancy | |
| property_type | string | Property type | |
| servicer_name | string | Servicer | |
| current_status | string | Loan status at month-end | Current/30DPD/60DPD/90DPD/Default/Prepaid/Closed |
| days_past_due | int | Days overdue | 0/30/60/90/120 |
| modification_flag | int | 1=modified | 0,1 |
| prepayment_flag | int | 1=prepaid this month | 0,1 |
| default_flag | int | 1=in default | 0,1 |
| loss_severity_band | string | Loss severity if defaulted | 0%/1-10%/11-25%/26-50%/>50%/N/A |
| last_updated_at | date | Last record update | YYYY-MM-DD |
| source_system | string | Source system | LOS/Servicer-Portal/Manual-Entry/Batch-Upload |
| document_status | string | Doc completeness | Complete/Missing-Income/Missing-Appraisal/Missing-ID/Stale |
| **next_3m_delinquency_flag** | int | TARGET: delinquent in 3m | 0,1 |
| **next_6m_delinquency_flag** | int | TARGET: delinquent in 6m | 0,1 |
| **next_12m_default_flag** | int | TARGET: default in 12m | 0,1 |
| **next_12m_prepayment_flag** | int | TARGET: prepay in 12m | 0,1 |
| **next_state** | string | TARGET: next month status | Current/30DPD/... |
| **exception_required** | int | TARGET: needs review | 0,1 |
| **exception_type** | string | TARGET: exception category | None/balance_anomaly/doc_gap/missing_modification |

---

## 3. loan_monthly_performance_test.csv
Same schema as train WITHOUT target columns. Participants predict these.

---

## 4. servicer_updates.csv
Second-source servicer data for conflict detection.

| Field | Type | Description |
|---|---|---|
| loan_id | string | FK to loan_static_attributes |
| update_date | date | Servicer update date |
| servicer_name | string | Servicer name |
| reported_balance | float | Balance per servicer (USD) |
| reported_status | string | Status per servicer |
| reported_rate | float | Rate per servicer (%) |
| source_system | string | Source system |
| conflict_type | string | balance_mismatch/status_conflict/stale_record/rate_discrepancy/none |
| stale_flag | int | 1=stale record (>65 days) |
| notes | string | Servicer notes |

---

## 5. macro_scenarios.csv
Scenario assumptions for simulation and stress testing.

| Field | Description |
|---|---|
| scenario_name | Base/Adverse-Credit/High-Prepayment/Stagflation |
| description | Plain-English description |
| gdp_growth_pct | Projected GDP growth (%) |
| unemployment_rate_pct | Projected unemployment rate (%) |
| hpi_change_pct | House Price Index change (%) |
| interest_rate_shock_bps | Rate shock in basis points |
| credit_spread_shock_bps | Credit spread shock in basis points |
| prepayment_cpr_assumption_pct | CPR assumption (%) |
| default_rate_multiplier | Multiplier on base default rate |
| delinquency_rate_multiplier | Multiplier on base delinquency rate |
| prepayment_rate_multiplier | Multiplier on base prepayment rate |

---

## 6. submission_template.csv
Required output format for final submission.

| Field | Description |
|---|---|
| loan_id | Loan identifier |
| reporting_month | Prediction month (YYYY-MM) |
| prob_next_3m_delinquency | P(delinquency in 3m) in [0,1] |
| prob_next_6m_delinquency | P(delinquency in 6m) in [0,1] |
| prob_next_12m_default | P(default in 12m) in [0,1] |
| prob_next_12m_prepayment | P(prepayment in 12m) in [0,1] |
| predicted_next_state | Next month predicted status |
| anomaly_score | Record anomaly score in [0,1] |
| exception_required | 1=exception review recommended |
| exception_type | Exception category |
| top_driver_1..3 | Top SHAP feature drivers |
| reviewer_action | No Action/Flag/Escalate |
| model_confidence | Model confidence in [0,1] |

---

## 7. HMDA Raw Data (hmda_raw/ folder)
Real public HMDA LAR data downloaded from ffiec.cfpb.gov
4.4M+ originated loan records, 99 fields each.
Coverage: States WY/ND/AK/VT/SD/MT/NH/ID/NE/NM/HI/KS/DE/WV/RI/MS | Years 2019-2023

Key HMDA fields mapped to problem statement fields:
- loan_amount (x1000) -> original_balance
- interest_rate -> interest_rate
- loan_term -> loan_term_months
- loan_to_value_ratio -> ltv_band
- debt_to_income_ratio -> dti_band
- state_code -> state
- loan_purpose (1=Purchase,2=Refinance,31/32=Refi types) -> loan_purpose
- occupancy_type (1=Primary,2=Secondary,3=Investment) -> occupancy_type
- property_value -> property value reference
- income -> borrower income reference

---

## Glossary

| Term | Definition |
|---|---|
| DPD | Days Past Due — days a payment is overdue |
| LTV | Loan-to-Value — loan balance / property value |
| DTI | Debt-to-Income — monthly debt payments / gross monthly income |
| CPR | Conditional Prepayment Rate — annualized prepayment speed |
| CDR | Conditional Default Rate — annualized default speed |
| HPI | House Price Index — residential property price changes |
| UPB | Unpaid Principal Balance |
| Modification | Formal change to loan terms by lender/borrower agreement |
| Vintage | Year a loan was originated; used for cohort analysis |
| Delinquency | Borrower missed one or more scheduled payments |
| Default | Severe delinquency (90+ DPD) triggering loss proceedings |
| Prepayment | Borrower pays off loan before maturity |
| Censoring | In survival analysis: loan exits before event occurs |
| Exception | Record violating a business rule requiring human review |
| SHAP | SHapley Additive exPlanations — ML model explanation method |
| ROC-AUC | Area Under the Receiver Operating Characteristic Curve |
| PR-AUC | Area Under the Precision-Recall Curve (better for imbalanced) |
| Brier Score | Mean squared error between predicted probabilities and actuals |
| RAG | Retrieval-Augmented Generation — LLM grounded in source docs |
