import axios from "axios";
const BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
export const api = axios.create({ baseURL: BASE, timeout: 60_000, headers: { "Content-Type": "application/json" } });

// ── Portfolio ────────────────────────────────────────────────────────────────
// GET /api/v1/portfolio/summary
// GET /api/v1/portfolio/watchlist
export const portfolioApi = {
  summary:   ()                                                                                              => api.get("/api/v1/portfolio/summary"),
  // Backend query params are `n`, `min_default_prob`, `status_filter` — not
  // `limit`/`min_prob`. FastAPI silently drops unknown query params instead of
  // rejecting them, so the mismatch never surfaced as an error: it just made
  // the watchlist filter dropdown a no-op forever. See DEPLOY notes.
  watchlist: (p?: { n?: number; min_default_prob?: number; status_filter?: string }) => api.get("/api/v1/portfolio/watchlist", { params: p }),
};

// ── Prediction ───────────────────────────────────────────────────────────────
// POST /api/v1/predict
// GET  /api/v1/predict/{loan_id}
export const predictionApi = {
  byLoanId: (id: string) => api.get(`/api/v1/predict/${id}`),
  demo:     ()           => api.get("/api/v1/predict/demo"),
};

// ── Anomaly ──────────────────────────────────────────────────────────────────
// GET /api/v1/anomalies
// GET /api/v1/anomalies/{loan_id}/{month_index}
export const anomalyApi = {
  list: (p?: { limit?: number; offset?: number; type?: string; min_score?: number }) =>
    api.get("/api/v1/anomalies", { params: p }),
  card: (id: string, month_index = 0) =>
    api.get(`/api/v1/anomalies/${id}/${month_index}`),
};

// ── Reviewer ─────────────────────────────────────────────────────────────────
// POST /api/v1/reviewer/decision
export const reviewerApi = {
  decision: (d: ReviewerDecisionRequest) => api.post("/api/v1/reviewer/decision", d),
};

// ── Survival ─────────────────────────────────────────────────────────────────
// GET  /api/v1/survival/{loan_id}
// POST /api/v1/survival/segment
export const survivalApi = {
  byLoanId: (id: string)  => api.get(`/api/v1/survival/${id}`),
  segment:  (d: unknown)  => api.post("/api/v1/survival/segment", d),
};

// ── Scenario ─────────────────────────────────────────────────────────────────
// GET  /api/v1/scenarios
// POST /api/v1/scenario/run      body: { scenario, n_paths, horizon }
// POST /api/v1/scenario/custom   body: CustomScenarioRequest
// GET  /api/v1/scenario/sensitivity
export const scenarioApi = {
  list:        ()           => api.get("/api/v1/scenarios"),
  run:         (d: unknown) => api.post("/api/v1/scenario/run", d),
  custom:      (d: unknown) => api.post("/api/v1/scenario/custom", d),
  sensitivity: (p?: unknown) => api.get("/api/v1/scenario/sensitivity", { params: p }),
};

// ── Explainability ───────────────────────────────────────────────────────────
// GET  /api/v1/explain/global
// GET  /api/v1/explain/errors         (was "error-analysis" — fixed)
// POST /api/v1/explain/counterfactual
// GET  /api/v1/explain/{loan_id}      (was "local/{id}" — fixed)
export const explainApi = {
  global:        (p?: { head?: string; top_k?: number })       => api.get("/api/v1/explain/global", { params: p }),
  local:         (id: string, p?: { head?: string })           => api.get(`/api/v1/explain/${id}`, { params: p }),
  counterfactual:(d: unknown)                                  => api.post("/api/v1/explain/counterfactual", d),
  errorAnalysis: (p?: unknown)                                 => api.get("/api/v1/explain/errors", { params: p }),
};

// ── Copilot ──────────────────────────────────────────────────────────────────
// POST /api/v1/copilot/ask
// GET  /api/v1/copilot/prompt-log     (was "log" — fixed)
export const copilotApi = {
  // use_sql defaults to true: without it, the copilot can only answer from
  // retrieved documentation (RAG), never from live portfolio numbers. Every
  // "quick question" on the copilot page asks for a live figure (current
  // default rate, highest anomaly scores, expected loss under a scenario), so
  // leaving it unset made every one of them come back as the same class of
  // "the evidence packet does not contain that value" refusal.
  ask: (d: { question: string; loan_id?: string; use_sql?: boolean; top_k?: number; use_cache?: boolean }) =>
    api.post("/api/v1/copilot/ask", { use_sql: true, ...d }),
  log: (p?: { limit?: number; offset?: number })                      => api.get("/api/v1/copilot/prompt-log", { params: p }),
};

// ── Data Quality ─────────────────────────────────────────────────────────────
// GET /api/v1/dq/summary
export const dqApi = {
  summary: (p?: { months?: string; by_month?: boolean }) => api.get("/api/v1/dq/summary", { params: p }),
};

// ── Drift ────────────────────────────────────────────────────────────────────
// GET /api/v1/drift    query params are `ref` / `cur` (month_index windows,
// e.g. "31-36"), not `reference`/`current`. There is no `top_k` — the response
// always returns every profiled feature; slice client-side if needed.
export const driftApi = {
  detect: (p: { ref: string; cur: string; adversarial?: boolean }) => api.get("/api/v1/drift", { params: p }),
};

// ── Submission ───────────────────────────────────────────────────────────────
// POST /api/v1/submission/generate
// GET  /api/v1/submission/validate
export const submissionApi = {
  generate: (d: unknown) => api.post("/api/v1/submission/generate", d),
  validate: ()           => api.get("/api/v1/submission/validate"),
};

// ── Health ───────────────────────────────────────────────────────────────────
// GET /healthz  GET /readyz
export const healthApi = {
  healthz:  () => api.get("/healthz"),
  readiness:() => api.get("/readyz"),
};

// ── Shared types ─────────────────────────────────────────────────────────────
export interface ReviewerDecisionRequest {
  loan_id: string;
  month_index: number;
  human_decision: "Confirm" | "Reject" | "Escalate";
  model_recommendation?: string;
  rationale?: string;
  reviewer?: string;
  anomaly_score?: number;
  exception_type?: string;
}
export type RiskLevel = "low" | "medium" | "high";

