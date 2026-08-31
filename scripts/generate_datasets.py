"""
Intain Campus FinTech Challenge 2026 | AI Track
Dataset Generator from Real HMDA Public Data
"""
import numpy as np, pandas as pd, random, json, os, glob, warnings
from datetime import datetime, timedelta
warnings.filterwarnings("ignore")

SEED=42; np.random.seed(SEED); random.seed(SEED)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.environ.get("LPIE_DATASET_DIR", os.path.join(REPO_ROOT, "dataset"))
RAW=f"{OUT}/hmda_raw"
os.makedirs(OUT, exist_ok=True)

print("="*65)
print("  Intain Campus FinTech Challenge 2026 | AI Track")
print("  Dataset Generator from Real HMDA Public Data")
print("="*65)

# ══════════════════════════════════════════════════
# STEP 1: Load all downloaded HMDA data
# ══════════════════════════════════════════════════
print("\n[STEP 1/5] Loading real HMDA data...")
hmda_files = sorted(glob.glob(f"{RAW}/hmda_*.csv"))
parts = []
for fp in hmda_files:
    try:
        df = pd.read_csv(fp, dtype=str, low_memory=False)
        if "action_taken" in df.columns:
            df = df[df["action_taken"]=="1"]
        parts.append(df)
    except Exception as e:
        print(f"  skip {os.path.basename(fp)}: {e}")

hmda = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
print(f"  Loaded: {len(hmda):,} originated loans from {len(hmda_files)} files")

# Show states and year coverage
if not hmda.empty:
    st = hmda["state_code"].dropna().unique() if "state_code" in hmda else []
    yrs = hmda["activity_year"].dropna().unique() if "activity_year" in hmda else []
    print(f"  States: {sorted([str(s) for s in st])}")
    print(f"  Years:  {sorted([str(y) for y in yrs])}")

# ══════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════
CREDIT_BANDS=["<620","620-659","660-699","700-739","740-779","780+"]
LTV_BANDS=["<=60","61-70","71-80","81-90","91-95","96-100",">100"]
DTI_BANDS=["<=28","29-36","37-43","44-50",">50"]
STATUS_ALL=["Current","30DPD","60DPD","90DPD","Default","Prepaid","Closed"]
DOC_STATUS=["Complete","Missing-Income","Missing-Appraisal","Missing-ID","Stale"]
SOURCE_SYS=["LOS","Servicer-Portal","Manual-Entry","Batch-Upload"]
SERVICERS=["ServicerA","ServicerB","ServicerC","ServicerD","ServicerE"]
LOSS_SEV=["0%","1-10%","11-25%","26-50%",">50%","N/A"]
LOAN_PURPOSES=["Purchase","Refinance","Cash-Out Refinance","Rate-Term Refinance"]
OCCUPANCY=["Primary","Secondary","Investment"]
PROP_TYPES=["Single Family","Condo","PUD","Multi-Family","Co-op"]
STATES=["CA","TX","FL","NY","IL","WA","CO","GA","NC","AZ","NV","OH","PA","VA","MA",
        "TN","NJ","MN","OR","WY","ND","AK","VT","SD","MT","NH","ME","ID","NE","NM",
        "HI","KS","DE","WV","RI","MS"]

N_LOANS        = 10000
N_MONTHS_TRAIN = 36
N_MONTHS_TEST  = 6
BASE_DATE      = datetime(2021, 1, 1)

print(f"\n  Target: {N_LOANS:,} loans x {N_MONTHS_TRAIN} months = {N_LOANS*N_MONTHS_TRAIN:,} train rows")

# Helper: map HMDA purpose codes
PURPOSE_MAP = {"1":"Purchase","2":"Refinance","31":"Refinance","32":"Cash-Out Refinance",
               "3":"Refinance","4":"Other"}
OCC_MAP = {"1":"Primary","2":"Secondary","3":"Investment"}

def clean_amount(v):
    try:
        a = float(v) * 1000
        return round(a/1000)*1000 if 50000 <= a <= 3_000_000 else round(random.lognormvariate(12.0,0.55)/1000)*1000
    except:
        return round(random.lognormvariate(12.0,0.55)/1000)*1000

def clean_rate(v):
    try:
        r = float(v)
        return round(r,3) if 1.0 <= r <= 18.0 else round(random.gauss(5.8,1.2),3)
    except:
        return round(random.gauss(5.8,1.2),3)

def clean_term(v):
    try:
        t = int(float(v))
        return t if t in [120,180,240,360] else random.choice([180,240,360])
    except:
        return random.choice([180,240,360])

def clean_ltv(v):
    try:
        x=float(v)
        if x<=60: return "<=60"
        elif x<=70: return "61-70"
        elif x<=80: return "71-80"
        elif x<=90: return "81-90"
        elif x<=95: return "91-95"
        elif x<=100: return "96-100"
        else: return ">100"
    except:
        return random.choice(LTV_BANDS)

def clean_dti(v):
    try:
        x=float(v)
        if x<=28: return "<=28"
        elif x<=36: return "29-36"
        elif x<=43: return "37-43"
        elif x<=50: return "44-50"
        else: return ">50"
    except:
        return random.choice(DTI_BANDS)

def clean_credit(v):
    """Map HMDA applicant_credit_score_type to credit band (approximate)."""
    # HMDA does not give actual score, only type code — synthesize band
    p=[0.08,0.12,0.18,0.22,0.22,0.18]
    return np.random.choice(CREDIT_BANDS, p=p)

def clean_state(v):
    s=str(v).strip().upper()
    return s if s in STATES else random.choice(STATES)

def clean_purpose(v):
    return PURPOSE_MAP.get(str(v), random.choice(LOAN_PURPOSES))

def clean_occ(v):
    return OCC_MAP.get(str(v), random.choice(OCCUPANCY))

def rnd_orig_month():
    y=random.randint(2018,2022); m=random.randint(1,12)
    return f"{y}-{m:02d}"

# ══════════════════════════════════════════════════
# STEP 2: Build loan_static_attributes.csv
# ══════════════════════════════════════════════════
print("\n[STEP 2/5] Building loan_static_attributes.csv...")

loan_ids = [f"LN{str(i).zfill(7)}" for i in range(1, N_LOANS+1)]

# Sample from real HMDA if available
def hmda_col(col, n):
    if not hmda.empty and col in hmda.columns:
        vals = hmda[col].dropna().astype(str)
        vals = vals[(vals!="NA")&(vals!="Exempt")&(vals!="")]
        if len(vals) > 100:
            return vals.sample(n=n, replace=True, random_state=SEED).tolist()
    return [None]*n

raw_amt  = hmda_col("loan_amount", N_LOANS)
raw_rate = hmda_col("interest_rate", N_LOANS)
raw_term = hmda_col("loan_term", N_LOANS)
raw_ltv  = hmda_col("loan_to_value_ratio", N_LOANS)
raw_dti  = hmda_col("debt_to_income_ratio", N_LOANS)
raw_st   = hmda_col("state_code", N_LOANS)
raw_pur  = hmda_col("loan_purpose", N_LOANS)
raw_occ  = hmda_col("occupancy_type", N_LOANS)

orig_months   = [rnd_orig_month() for _ in range(N_LOANS)]
vintage_years = [m[:4] for m in orig_months]

rows_s = []
for i in range(N_LOANS):
    rows_s.append({
        "loan_id":           loan_ids[i],
        "origination_month": orig_months[i],
        "original_balance":  clean_amount(raw_amt[i]),
        "interest_rate":     clean_rate(raw_rate[i]),
        "loan_term_months":  clean_term(raw_term[i]),
        "credit_score_band": clean_credit(None),
        "ltv_band":          clean_ltv(raw_ltv[i]),
        "dti_band":          clean_dti(raw_dti[i]),
        "state":             clean_state(raw_st[i]) if raw_st[i] else random.choice(STATES),
        "loan_purpose":      clean_purpose(raw_pur[i]) if raw_pur[i] else random.choice(LOAN_PURPOSES),
        "occupancy_type":    clean_occ(raw_occ[i]) if raw_occ[i] else random.choice(OCCUPANCY),
        "property_type":     random.choice(PROP_TYPES),
        "servicer_name":     random.choice(SERVICERS),
        "vintage_year":      vintage_years[i],
    })

static_df = pd.DataFrame(rows_s)

# Inject 3% realistic missing
for col in ["credit_score_band","ltv_band","dti_band","state"]:
    mask = np.random.rand(N_LOANS) < 0.03
    static_df.loc[mask, col] = np.nan

static_df.to_csv(f"{OUT}/loan_static_attributes.csv", index=False)
print(f"  Saved: {len(static_df):,} loans -> loan_static_attributes.csv")
print(f"  Origination years: {sorted(static_df.vintage_year.unique())}")
print(f"  States sampled: {static_df.state.dropna().nunique()} unique")

# ══════════════════════════════════════════════════
# STEP 3: Build monthly panel (train + test)
# ══════════════════════════════════════════════════
print("\n[STEP 3/5] Building monthly panels...")

STATUS_TRANS = {
    "Current": {"Current":0.92,"30DPD":0.06,"Prepaid":0.02},
    "30DPD":   {"Current":0.55,"30DPD":0.20,"60DPD":0.20,"Prepaid":0.05},
    "60DPD":   {"Current":0.30,"30DPD":0.20,"60DPD":0.15,"90DPD":0.25,"Prepaid":0.10},
    "90DPD":   {"Current":0.15,"60DPD":0.15,"90DPD":0.25,"Default":0.40,"Prepaid":0.05},
    "Default": {"Default":0.70,"Closed":0.30},
    "Prepaid": {"Prepaid":1.00},
    "Closed":  {"Closed":1.00},
}
DPD_MAP={"Current":0,"30DPD":30,"60DPD":60,"90DPD":90,"Default":120,"Prepaid":0,"Closed":0}

def calc_risk(row):
    p=0.04
    if str(row.get("credit_score_band","")) in ["<620","620-659"]: p+=0.09
    elif str(row.get("credit_score_band","")) in ["660-699"]: p+=0.04
    if str(row.get("ltv_band","")) in [">100","96-100"]: p+=0.05
    if str(row.get("dti_band","")) in [">50","44-50"]: p+=0.04
    if str(row.get("loan_purpose",""))=="Cash-Out Refinance": p+=0.02
    return min(p, 0.35)

def transition(cur, risk):
    t=dict(STATUS_TRANS[cur])
    if risk>0.15:
        if "30DPD" in t: t["30DPD"]=min(t["30DPD"]*1.6,0.5)
        if "Default" in t: t["Default"]=min(t.get("Default",0)*1.4,0.65)
    keys=list(t.keys()); tot=sum(t.values())
    probs=[v/tot for v in t.values()]
    return np.random.choice(keys, p=probs)

lookup = static_df.set_index("loan_id").to_dict("index")
train_rows=[]; test_rows=[]

for idx, lid in enumerate(loan_ids):
    if idx % 2000 == 0:
        print(f"  Processing loan {idx:,}/{N_LOANS:,}...")
    s=lookup[lid]
    risk=calc_risk(s)
    ob=float(s["original_balance"]) if pd.notna(s.get("original_balance")) else 250000.0
    bal=ob
    term=int(s["loan_term_months"]) if pd.notna(s.get("loan_term_months")) else 360
    rate_pct=float(s["interest_rate"]) if pd.notna(s.get("interest_rate")) else 5.5
    try:
        orig_dt=datetime.strptime(str(s["origination_month"])+"-01","%Y-%m-%d")
    except:
        orig_dt=datetime(2019,6,1)
    cur_st="Current"
    for m in range(N_MONTHS_TRAIN):
        rep_dt=BASE_DATE+timedelta(days=m*30)
        age=max(1,(rep_dt-orig_dt).days//30)
        rem=max(0,term-age)
        dpd=DPD_MAP.get(cur_st,0)
        mod=1 if cur_st in ["60DPD","90DPD"] and random.random()<0.15 else 0
        pp=1 if cur_st=="Prepaid" else 0
        df_=1 if cur_st=="Default" else 0
        if cur_st not in ["Default","Prepaid","Closed"] and rem>0:
            mr=(rate_pct/100)/12
            if mr>0:
                pmt=bal*mr/(1-(1+mr)**(-rem))
                bal=max(0,bal-(pmt-bal*mr))
        elif cur_st in ["Prepaid","Closed","Default"]: bal=0.0
        lsev="N/A"
        if cur_st=="Default": lsev=random.choice(["1-10%","11-25%","26-50%",">50%"])
        disp_bal=bal*random.uniform(1.5,5.0) if random.random()<0.02 else bal
        row={
            "loan_id":lid,"month_index":m+1,
            "reporting_month":rep_dt.strftime("%Y-%m"),
            "origination_month":str(s["origination_month"]),
            "loan_age_months":age,"remaining_term_months":rem,
            "original_balance":ob,"current_balance":round(disp_bal,2),
            "interest_rate":rate_pct,
            "credit_score_band":s.get("credit_score_band"),
            "ltv_band":s.get("ltv_band"),"dti_band":s.get("dti_band"),
            "state":s.get("state"),"loan_purpose":s.get("loan_purpose"),
            "occupancy_type":s.get("occupancy_type"),
            "property_type":s.get("property_type"),
            "servicer_name":s.get("servicer_name"),
            "current_status":cur_st,"days_past_due":dpd,
            "modification_flag":mod,"prepayment_flag":pp,"default_flag":df_,
            "loss_severity_band":lsev,
            "last_updated_at":(rep_dt+timedelta(days=random.randint(0,5))).strftime("%Y-%m-%d"),
            "source_system":random.choices(SOURCE_SYS,[0.50,0.30,0.15,0.05])[0],
            "document_status":random.choices(DOC_STATUS,[0.80,0.07,0.05,0.04,0.04])[0],
        }
        train_rows.append(row)
        if cur_st not in ["Prepaid","Closed"]: cur_st=transition(cur_st,risk)
    last_st=cur_st; last_bal=bal
    for m in range(N_MONTHS_TEST):
        rep_dt=(BASE_DATE+timedelta(days=N_MONTHS_TRAIN*30))+timedelta(days=m*30)
        age=max(1,(rep_dt-orig_dt).days//30); rem=max(0,term-age)
        dpd=DPD_MAP.get(last_st,0)
        row={"loan_id":lid,"month_index":N_MONTHS_TRAIN+m+1,
            "reporting_month":rep_dt.strftime("%Y-%m"),
            "origination_month":str(s["origination_month"]),
            "loan_age_months":age,"remaining_term_months":rem,
            "original_balance":ob,"current_balance":round(last_bal,2),
            "interest_rate":rate_pct,
            "credit_score_band":s.get("credit_score_band"),
            "ltv_band":s.get("ltv_band"),"dti_band":s.get("dti_band"),
            "state":s.get("state"),"loan_purpose":s.get("loan_purpose"),
            "occupancy_type":s.get("occupancy_type"),
            "property_type":s.get("property_type"),
            "servicer_name":s.get("servicer_name"),
            "current_status":last_st,"days_past_due":dpd,
            "modification_flag":0,"prepayment_flag":int(last_st=="Prepaid"),
            "default_flag":int(last_st=="Default"),"loss_severity_band":"N/A",
            "last_updated_at":(rep_dt+timedelta(days=random.randint(0,5))).strftime("%Y-%m-%d"),
            "source_system":random.choices(SOURCE_SYS,[0.50,0.30,0.15,0.05])[0],
            "document_status":random.choices(DOC_STATUS,[0.80,0.07,0.05,0.04,0.04])[0],
        }
        test_rows.append(row)
        if last_st not in ["Prepaid","Closed"]: last_st=transition(last_st,risk)

train_df=pd.DataFrame(train_rows)
test_df=pd.DataFrame(test_rows)
print(f"  Train rows built: {len(train_df):,}")
print(f"  Test rows built:  {len(test_df):,}")

# Add target labels to train
print("  Adding target labels...")
train_df=train_df.sort_values(["loan_id","month_index"]).reset_index(drop=True)
BAD={"30DPD","60DPD","90DPD","Default"}
n3=[]; n6=[]; n12d=[]; n12p=[]; nst=[]; exr=[]; ext=[]
for lid, grp in train_df.groupby("loan_id"):
    sts=grp["current_status"].tolist()
    bals=grp["current_balance"].tolist()
    obs=grp["original_balance"].tolist()
    docs=grp["document_status"].tolist()
    mods=grp["modification_flag"].tolist()
    for i in range(len(sts)):
        n3.append(int(bool(BAD & set(sts[i+1:i+4]))))
        n6.append(int(bool(BAD & set(sts[i+1:i+7]))))
        n12d.append(int("Default" in sts[i+1:i+13]))
        n12p.append(int("Prepaid" in sts[i+1:i+13]))
        nst.append(sts[i+1] if i+1<len(sts) else sts[i])
        exc=[]
        if bals[i]>obs[i]*1.4: exc.append("balance_anomaly")
        if str(docs[i]) in ["Missing-Income","Missing-Appraisal","Missing-ID"]: exc.append("doc_gap")
        if sts[i] in ["90DPD","Default"] and mods[i]==0: exc.append("missing_modification")
        exr.append(1 if exc else 0)
        ext.append(exc[0] if exc else "None")
train_df["next_3m_delinquency_flag"]=n3
train_df["next_6m_delinquency_flag"]=n6
train_df["next_12m_default_flag"]=n12d
train_df["next_12m_prepayment_flag"]=n12p
train_df["next_state"]=nst
train_df["exception_required"]=exr
train_df["exception_type"]=ext
# Inject 5% realistic missing in noisy fields
for col in ["days_past_due","loss_severity_band","document_status","credit_score_band"]:
    mask=np.random.rand(len(train_df))<0.05
    train_df.loc[mask,col]=np.nan
train_df.to_csv(f"{OUT}/loan_monthly_performance_train.csv", index=False)
print(f"  Saved: {len(train_df):,} rows -> loan_monthly_performance_train.csv")
print(f"    3m delinquency rate : {train_df.next_3m_delinquency_flag.mean()*100:.1f}%")
print(f"    12m default rate    : {train_df.next_12m_default_flag.mean()*100:.1f}%")
print(f"    12m prepayment rate : {train_df.next_12m_prepayment_flag.mean()*100:.1f}%")
print(f"    Exception rate      : {train_df.exception_required.mean()*100:.1f}%")
test_df.to_csv(f"{OUT}/loan_monthly_performance_test.csv", index=False)
print(f"  Saved: {len(test_df):,} rows -> loan_monthly_performance_test.csv")

# ══════════════════════════════════════════════════
# STEP 4: servicer_updates.csv
# ══════════════════════════════════════════════════
print("\n[STEP 4/5] Building servicer_updates.csv...")
med_bal=train_df.groupby("loan_id")["current_balance"].median().to_dict()
svc_sample=np.random.choice(loan_ids, size=int(N_LOANS*0.5), replace=False)
svc_rows=[]
for lid in svc_sample:
    for _ in range(random.randint(1,6)):
        upd_dt=BASE_DATE+timedelta(days=random.randint(0,N_MONTHS_TRAIN*30))
        ct=random.choices(
            ["balance_mismatch","status_conflict","stale_record","rate_discrepancy","none"],
            [0.15,0.10,0.20,0.10,0.45])[0]
        bb=med_bal.get(lid,250000.0)
        rb=bb*random.uniform(0.5,2.0) if ct=="balance_mismatch" else bb
        svc_rows.append({
            "loan_id":lid,"update_date":upd_dt.strftime("%Y-%m-%d"),
            "servicer_name":random.choice(SERVICERS),
            "reported_balance":round(rb,2),
            "reported_status":random.choice(STATUS_ALL[:5]),
            "reported_rate":round(random.gauss(5.5,1.5),3),
            "source_system":random.choice(SOURCE_SYS),
            "conflict_type":ct,"stale_flag":int(ct=="stale_record"),
            "notes":f"Update from {random.choice(SERVICERS)}",
        })
pd.DataFrame(svc_rows).to_csv(f"{OUT}/servicer_updates.csv",index=False)
print(f"  Saved: {len(svc_rows):,} rows -> servicer_updates.csv")

# ══════════════════════════════════════════════════
# STEP 5: macro_scenarios.csv
# ══════════════════════════════════════════════════
print("\n[STEP 5/5] Building support files...")
scenarios=[
  {"scenario_name":"Base","description":"Baseline macro: moderate rates, stable credit",
   "gdp_growth_pct":2.3,"unemployment_rate_pct":4.1,"hpi_change_pct":3.5,
   "interest_rate_shock_bps":0,"credit_spread_shock_bps":0,
   "prepayment_cpr_assumption_pct":8.0,"default_rate_multiplier":1.0,
   "delinquency_rate_multiplier":1.0,"prepayment_rate_multiplier":1.0},
  {"scenario_name":"Adverse-Credit","description":"Adverse credit: rising unemployment, HPI decline",
   "gdp_growth_pct":-0.5,"unemployment_rate_pct":7.8,"hpi_change_pct":-8.0,
   "interest_rate_shock_bps":50,"credit_spread_shock_bps":200,
   "prepayment_cpr_assumption_pct":4.0,"default_rate_multiplier":2.8,
   "delinquency_rate_multiplier":2.5,"prepayment_rate_multiplier":0.5},
  {"scenario_name":"High-Prepayment","description":"Rate-drop: mass refinance wave",
   "gdp_growth_pct":3.1,"unemployment_rate_pct":3.5,"hpi_change_pct":5.2,
   "interest_rate_shock_bps":-150,"credit_spread_shock_bps":-30,
   "prepayment_cpr_assumption_pct":30.0,"default_rate_multiplier":0.7,
   "delinquency_rate_multiplier":0.8,"prepayment_rate_multiplier":3.5},
  {"scenario_name":"Stagflation","description":"High inflation, low growth, rising rates",
   "gdp_growth_pct":0.2,"unemployment_rate_pct":6.0,"hpi_change_pct":-3.0,
   "interest_rate_shock_bps":250,"credit_spread_shock_bps":100,
   "prepayment_cpr_assumption_pct":2.0,"default_rate_multiplier":1.9,
   "delinquency_rate_multiplier":1.7,"prepayment_rate_multiplier":0.3},
]
pd.DataFrame(scenarios).to_csv(f"{OUT}/macro_scenarios.csv",index=False)
print("  Saved: macro_scenarios.csv (4 scenarios)")

# submission_template.csv
sub=[]
for lid in loan_ids[:20]:
    sub.append({"loan_id":lid,"reporting_month":"2024-01",
        "prob_next_3m_delinquency":0.0,"prob_next_6m_delinquency":0.0,
        "prob_next_12m_default":0.0,"prob_next_12m_prepayment":0.0,
        "predicted_next_state":"Current","anomaly_score":0.0,
        "exception_required":0,"exception_type":"None",
        "top_driver_1":"","top_driver_2":"","top_driver_3":"",
        "reviewer_action":"No Action","model_confidence":0.0})
pd.DataFrame(sub).to_csv(f"{OUT}/submission_template.csv",index=False)
print("  Saved: submission_template.csv (20 example rows)")

# validation_rules.json
rules={"version":"1.0","generated":datetime.now().strftime("%Y-%m-%d"),
  "description":"Deterministic validation rules for Loan Performance Intelligence Engine",
  "rules":[
    {"rule_id":"VR-001","name":"balance_not_exceed_original",
     "description":"current_balance must not exceed original_balance*1.05 unless modification_flag=1",
     "field":"current_balance","severity":"ERROR","exception_type":"balance_anomaly",
     "condition":"current_balance <= original_balance*1.05 OR modification_flag==1"},
    {"rule_id":"VR-002","name":"dpd_status_consistency",
     "description":"days_past_due must match current_status: Current->0, 30DPD->30, 60DPD->60, 90DPD->90",
     "field":"days_past_due","severity":"ERROR","exception_type":"status_dpd_mismatch",
     "condition":"Current->0, 30DPD->[25-35], 60DPD->[55-65], 90DPD->[85-120]"},
    {"rule_id":"VR-003","name":"date_validity",
     "description":"reporting_month >= origination_month AND loan_age_months == months_diff",
     "field":"reporting_month","severity":"ERROR","exception_type":"date_inconsistency",
     "condition":"reporting_month >= origination_month"},
    {"rule_id":"VR-004","name":"prepaid_balance_zero",
     "description":"Prepaid loans must have current_balance=0",
     "field":"current_balance","severity":"ERROR","exception_type":"balance_anomaly",
     "condition":"IF prepayment_flag==1 THEN current_balance==0"},
    {"rule_id":"VR-005","name":"closed_balance_zero",
     "description":"Closed loans must have current_balance=0",
     "field":"current_balance","severity":"ERROR","exception_type":"balance_anomaly",
     "condition":"IF current_status==Closed THEN current_balance==0"},
    {"rule_id":"VR-006","name":"doc_gap_check",
     "description":"Missing critical documents trigger doc_gap exception",
     "field":"document_status","severity":"WARNING","exception_type":"doc_gap",
     "condition":"document_status NOT IN [Missing-Income, Missing-Appraisal, Missing-ID]"},
    {"rule_id":"VR-007","name":"servicer_conflict",
     "description":"Servicer reported balance deviating >10% from master flags conflict",
     "field":"current_balance","severity":"WARNING","exception_type":"servicer_conflict",
     "condition":"ABS(current_balance-servicer_balance)/current_balance<=0.10"},
    {"rule_id":"VR-008","name":"rate_range",
     "description":"interest_rate must be 1.0-20.0%",
     "field":"interest_rate","severity":"ERROR","exception_type":"rate_out_of_range",
     "condition":"1.0 <= interest_rate <= 20.0"},
    {"rule_id":"VR-009","name":"remaining_term_valid",
     "description":"remaining_term_months must be 0 to loan_term_months",
     "field":"remaining_term_months","severity":"ERROR","exception_type":"term_inconsistency",
     "condition":"0 <= remaining_term_months <= loan_term_months"},
    {"rule_id":"VR-010","name":"stale_record",
     "description":"last_updated_at must be within 65 days of reporting_month end",
     "field":"last_updated_at","severity":"WARNING","exception_type":"stale_record",
     "condition":"days_since_last_update <= 65"},
    {"rule_id":"VR-011","name":"loss_severity_default_only",
     "description":"loss_severity_band non-NA only when default_flag=1",
     "field":"loss_severity_band","severity":"WARNING","exception_type":"loss_sev_inconsistency",
     "condition":"IF default_flag==0 THEN loss_severity_band==N/A"},
    {"rule_id":"VR-012","name":"modification_high_dpd",
     "description":"Loans at 90DPD or Default without modification should be reviewed",
     "field":"modification_flag","severity":"WARNING","exception_type":"missing_modification",
     "condition":"IF current_status IN [90DPD,Default] AND modification_flag==0 THEN flag"},
  ]}
with open(f"{OUT}/validation_rules.json","w") as fv:
    json.dump(rules,fv,indent=2)
print("  Saved: validation_rules.json (12 rules)")

# data_dictionary.md
dd_text = open("/tmp/dd_content.txt").read()
with open(f"{OUT}/data_dictionary.md","w") as fdoc:
    fdoc.write(dd_text)
print("  Saved: data_dictionary.md")

# FINAL SUMMARY
print("\n" + "="*65)
print("  ALL DATASETS GENERATED SUCCESSFULLY")
print("="*65)

import os
all_f = sorted(os.listdir(OUT))
total_bytes=0
print(f"\n  {'File':<52} {'Size':>12}")
print(f"  {'-'*52} {'-'*12}")
for fn in all_f:
    fp = os.path.join(OUT, fn)
    if os.path.isfile(fp):
        sz = os.path.getsize(fp)
        total_bytes += sz
        lbl = f"{sz/1024:.1f} KB" if sz<1024*1024 else f"{sz/1024/1024:.2f} MB"
        print(f"  {fn:<52} {lbl:>12}")
    elif os.path.isdir(fp):
        dir_sz = sum(os.path.getsize(os.path.join(fp,ff)) for ff in os.listdir(fp) if os.path.isfile(os.path.join(fp,ff)))
        total_bytes += dir_sz
        print(f"  {(fn+'/'):<52} {dir_sz/1024/1024:.2f} MB  (HMDA raw dir)")
print(f"\n  Total size: {total_bytes/1024/1024:.1f} MB")
print(f"  Location  : {OUT}/")
print("="*65)
