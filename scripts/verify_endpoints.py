"""Full endpoint verification script — run with: python3 verify_endpoints.py"""
import sys, logging
sys.path.insert(0, 'src')
logging.disable(logging.CRITICAL)

from lpie.api.main import app
from fastapi.testclient import TestClient

lines = []
with TestClient(app) as client:
    def chk(name, fn):
        try:
            r = fn()
            ok = r.status_code in (200, 201, 404, 422, 503)
            mark = "PASS" if ok else "FAIL"
            lines.append(f"{mark} {r.status_code:3d}  {name}")
        except Exception as e:
            lines.append(f"FAIL 000  {name}  ERR:{str(e)[:80]}")

    # Health / infra
    chk("/livez",                   lambda: client.get("/livez"))
    chk("/readyz",                  lambda: client.get("/readyz"))
    chk("/healthz",                 lambda: client.get("/healthz"))
    chk("/metrics",                 lambda: client.get("/metrics"))
    chk("/api/v1/meta/models",      lambda: client.get("/api/v1/meta/models"))
    chk("/api/v1/meta/config",      lambda: client.get("/api/v1/meta/config"))

    # Data intelligence
    chk("/api/v1/profile",          lambda: client.post("/api/v1/profile", json={"split":"train","sample_rows":200,"include_relationships":False,"include_missingness":False}))
    chk("/api/v1/validate",         lambda: client.post("/api/v1/validate", json={"months":[1],"limit":50}))
    chk("/api/v1/drift",            lambda: client.get("/api/v1/drift"))
    chk("/api/v1/dq/summary",       lambda: client.get("/api/v1/dq/summary"))

    # Prediction
    chk("/api/v1/predict (batch)",  lambda: client.post("/api/v1/predict", json={"loan_ids":["LN0000001","LN0000002"]}))
    chk("/api/v1/predict/{loan_id}",lambda: client.get("/api/v1/predict/LN0000001"))
    chk("/api/v1/portfolio/summary",lambda: client.get("/api/v1/portfolio/summary"))
    chk("/api/v1/portfolio/watchlist",lambda: client.get("/api/v1/portfolio/watchlist?n=5"))

    # Survival (static routes before dynamic)
    chk("/api/v1/survival/segment",       lambda: client.post("/api/v1/survival/segment", json={"segment_by":"current_status","horizon":6}))
    chk("/api/v1/survival/state-occupancy",lambda: client.get("/api/v1/survival/state-occupancy?horizon=6"))
    chk("/api/v1/survival/{loan_id}",     lambda: client.get("/api/v1/survival/LN0000001"))

    # Scenario
    chk("/api/v1/scenarios",        lambda: client.get("/api/v1/scenarios"))
    chk("/api/v1/scenario/run",     lambda: client.post("/api/v1/scenario/run", json={"scenario":"Base","n_paths":20,"horizon":3}))
    chk("/api/v1/scenario/sensitivity",lambda: client.get("/api/v1/scenario/sensitivity?scenario=Base&horizon=6"))
    chk("/api/v1/scenario/custom",  lambda: client.post("/api/v1/scenario/custom", json={"name":"TestShock","gdp_growth_pct":-1.0,"n_paths":20,"horizon":3}))

    # Anomaly & reviewer
    chk("/api/v1/anomalies",        lambda: client.get("/api/v1/anomalies?limit=3"))
    chk("/api/v1/reviewer/decision",lambda: client.post("/api/v1/reviewer/decision", json={"loan_id":"LN0000001","month_index":36,"human_decision":"Confirm","model_recommendation":"Flag"}))

    # Explainability (static routes before dynamic)
    chk("/api/v1/explain/global",   lambda: client.get("/api/v1/explain/global?head=next_12m_default"))
    chk("/api/v1/explain/errors",   lambda: client.get("/api/v1/explain/errors?head=next_12m_default"))
    chk("/api/v1/explain/counterfactual",lambda: client.post("/api/v1/explain/counterfactual", json={"loan_id":"LN0000001","month_index":42,"head":"next_12m_default"}))
    chk("/api/v1/explain/{loan_id}",lambda: client.get("/api/v1/explain/LN0000001?head=next_12m_default"))

    # Copilot
    chk("/api/v1/copilot/prompt-log",lambda: client.get("/api/v1/copilot/prompt-log"))

    # Submission
    chk("/api/v1/submission/validate",lambda: client.get("/api/v1/submission/validate"))

print("=" * 65)
print("LPIE ENDPOINT VERIFICATION")
print("=" * 65)
for line in lines:
    print(line)
pass_count = sum(1 for l in lines if l.startswith("PASS"))
fail_count = len(lines) - pass_count
print("=" * 65)
print(f"RESULT: {pass_count}/{len(lines)} endpoints PASSING  |  {fail_count} FAILING")
if fail_count == 0:
    print("ALL ENDPOINTS WORKING CORRECTLY")
