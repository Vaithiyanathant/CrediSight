export type RiskLevel = "low" | "medium" | "high" | "neutral";
export type DQGrade = "A" | "B" | "C" | "D" | "F";
export type LoanStatus = "Current" | "30DPD" | "60DPD" | "90DPD" | "Default" | "Prepaid" | "Closed";
export type VerifierVerdict = "PASS" | "REGENERATED" | "FALLBACK";
export type DriftVerdict = "KEEP" | "MONITOR" | "DROP_OR_ROBUSTIFY" | "UNKNOWN";

// Field names match GET /api/v1/portfolio/summary exactly (src/lpie/api/schemas.py
// PortfolioSummaryResponse). The previous version of this type guessed shorter
// names (n_total, as_of_month, default_rate_12m, expected_loss_rate as a 0-1
// fraction, …) that the backend never returns — every KPI on the Portfolio
// Risk page silently fell back to 0 because `summary?.n_total ?? 0` never saw
// the real field. `expected_loss_pct_of_balance` in particular is already a
// percentage number (0.29 means 0.29%), not a 0-1 fraction — do not run it
// through a `*100` formatter.
export interface PortfolioSummary {
  as_of_month_index: number; reporting_month: string | null; model_version: string;
  total_loans: number; total_balance: number; active_loans: number; terminal_loans: number;
  delinquency_rate: number; projected_default_rate: number; projected_prepayment_rate: number;
  expected_loss: number; expected_loss_pct_of_balance: number | null;
  risk_distribution: { low: number; medium: number; high: number };
  reviewer_action_distribution: Record<string, number>;
  dq_distribution: Record<string, number>;
  state_distribution: Record<string, number>;
  confidence_distribution: Record<string, unknown>;
  segments: Record<string, Array<Record<string, unknown>>>;
  computed_at: string; elapsed_ms: number;
}

export interface WatchlistEntry {
  rank: number; loan_id: string; month_index: number; current_status: string;
  current_balance: number; prob_next_12m_default: number; expected_loss: number;
  anomaly_score: number; exception_required: number; exception_type: string;
  reviewer_action: string; model_confidence: number; dq_grade: string | null;
  top_drivers: string[];
  prob_next_3m_delinquency?: number; anomaly_tier?: string;
  top_driver_1?: string; top_driver_2?: string; top_driver_3?: string;
}

export interface DriverBlock { feature: string; shap_value: number; direction: "positive"|"negative"; rank: number; }
export interface PredictionValue { value: number; lower?: number; upper?: number; ci?: [number,number]; }
export interface PredictionResponse {
  loan_id: string; as_of_month?: number; month_index?: number;
  predictions: {
    prob_next_3m_delinquency?: PredictionValue;
    prob_next_6m_delinquency?: PredictionValue;
    prob_next_12m_default?: PredictionValue;
    prob_next_12m_prepayment?: PredictionValue;
    // GET /api/v1/predict/{id} returns next_state as a STRUCTURED block
    // (NextStatePrediction in schemas.py) — the per-state probabilities live
    // under `.probs`. This was previously typed as a flat Record<string,number>,
    // so `Object.entries(next_state)` yielded ["predicted","Closed"] and
    // crashed the loan detail page with "prob.toFixed is not a function".
    next_state?: {
      predicted: string;
      probs: Record<string, number>;
      confidence?: number;
      legal_mask_applied?: boolean;
    };
    exception_required?: PredictionValue;
    next_3m_delinquency?: PredictionValue;
    next_6m_delinquency?: PredictionValue;
    next_12m_default?: PredictionValue;
    next_12m_prepayment?: PredictionValue;
  };
  top_drivers?: DriverBlock[];
  explanation?: { top_drivers?: Array<{ feature: string; shap?: number; value?: unknown; direction?: string }> };
  anomaly?: { score: number; tier: string; drivers?: string[]; rule_severity?: number };
  exception?: { required: number; type?: string; source?: string };
  data_quality?: { dq_score?: number; dq_grade?: string };
  confidence?: { model_confidence?: number };
  reviewer_action?: string;
  current_status: string; current_balance: number;
  is_terminal?: boolean;
  anomaly_score?: number; anomaly_tier?: string; dq_grade?: string;
  elapsed_ms?: number;
}

export interface AnomalyEntry {
  loan_id: string; month_index: number; anomaly_score: number; anomaly_tier: string;
  rule_violations?: string[]; rules_fired?: string[];
  unsupervised_score?: number; exception_prob?: number;
  exception_required?: number; exception_type?: string;
  current_status: string; current_balance: number; reviewer_action: string;
  dq_grade?: string; dq_score?: number; rule_severity?: number; worst_severity?: string;
}

export interface SurvivalPoint { t: number; survival: number; cif_default: number; cif_prepay: number; }
export interface SurvivalResponse {
  loan_id: string; month_index: number; current_status: string; model_version?: string;
  horizons_m?: number[]; survival?: number[]; cif_default?: number[]; cif_prepay?: number[];
  curve?: SurvivalPoint[]; prob_default_12m?: number; prob_prepay_12m?: number;
  median_survival_months?: number | null; elapsed_ms?: number;
}

export interface ScenarioInfo {
  scenario_name: string; description: string; gdp_growth_pct: number;
  unemployment_rate_pct: number; hpi_change_pct: number; interest_rate_shock_bps: number;
  default_rate_multiplier: number; delinquency_rate_multiplier: number; prepayment_rate_multiplier: number;
}
export interface MonteCarloPath { t: number; p5: number; p50: number; p95: number; }
export interface ScenarioFanChart { p5: number[]; p50: number[]; p95: number[]; mean: number[]; months: number[]; }
export interface ScenarioResponse {
  scenario: string; assumptions: Record<string, unknown>; paths: MonteCarloPath[];
  summary: {
    fan_charts?: { delinquency_rate?: ScenarioFanChart; cumulative_default_rate?: ScenarioFanChart; cumulative_prepayment_rate?: ScenarioFanChart; cumulative_loss?: ScenarioFanChart; outstanding_balance?: ScenarioFanChart; };
    terminal?: { default_rate?: { mean: number }; prepayment_rate?: { mean: number }; expected_loss?: { mean: number }; };
    risk_measures?: Record<string, number>;
  };
  elapsed_ms: number;
}

export interface CopilotCitation { chunk?: string; source?: string; text?: string; }
export interface CopilotResponseModel {
  task: string; answer: string; banner: string;
  verifier: { verdict: VerifierVerdict; passed: boolean; regenerated: boolean; used_fallback: boolean; n_failures: number; failures: string[]; numbers_checked: number; numbers_matched: number; };
  citations: (CopilotCitation | string)[]; evidence_hash: string; model: string; provider: string; latency_ms: number; prompt_log_id?: number;
}
export interface PromptLogEntry {
  id: number; ts: string; created_at?: string; question?: string; task?: string;
  verdict?: VerifierVerdict; verifier_verdict?: string; model: string; provider?: string; latency_ms: number;
  accepted?: number;
  // Full audit content — what was sent and what came back. `raw_output` is the
  // model's first attempt; `final_output` is what the client was actually
  // served. They differ whenever the verifier forced a regeneration or fell
  // back to the deterministic template.
  system_prompt?: string | null;
  user_prompt?: string | null;
  raw_output?: string | null;
  regenerated_output?: string | null;
  final_output?: string | null;
  verifier_failures?: Array<{ kind?: string; detail?: string; evidence?: string }>;
  retrieved_context?: Array<{ citation?: string; text?: string; score?: number; title?: string }>;
  evidence_packet?: Record<string, unknown> | null;
  input_tokens?: number | null;
  output_tokens?: number | null;
  request_id?: string | null;
}

export interface DQSummaryResponse {
  n_records?: number; mean_score?: number; mean_dq?: number;
  median_score?: number; median_dq?: number; grade?: string;
  grade_distribution: { A: number; B: number; C: number; D?: number; F?: number };
  dimension_means?: { completeness: number; validity: number; consistency: number; timeliness: number; uniqueness: number; cross_source: number; };
  mean_dimension_scores?: { completeness: number; validity: number; consistency: number; timeliness: number; uniqueness: number; cross_source: number; };
  top_violated_rules?: Array<{ rule_id: string; description?: string; name?: string; violation_count?: number; violation_rate?: number }>;
  // Real backend field is `by_month` (see GET /api/v1/dq/summary), not
  // `monthly_trend` — this type previously guessed a field name the API
  // never returns, so no page ever rendered a DQ trend over time.
  by_month?: Array<{ month_index: number; n: number; mean_dq: number; pct_grade_a: number }>;
  n_error_violations?: number;
  n_warning_violations?: number;
  computed_at?: string;
  elapsed_ms?: number;
}

export interface DriftFeature {
  feature: string; kind?: string;
  psi: number | null; ks_stat: number | null;
  js_divergence?: number | null; js_div?: number | null;
  missingness_delta?: number; missing_delta?: number;
  verdict: string; value_verdict?: string; missingness_verdict?: string;
  // True when the feature shifts deterministically with the calendar
  // (loan_age_months, remaining_term_months). Reported, never actionable.
  seasoning?: boolean;
}
// Matches GET /api/v1/drift exactly (DriftResponse in schemas.py). The
// previous shape guessed top-level `adversarial_auc`/`retrain_triggered`
// fields the backend never sends — the real numbers sit under `adversarial`
// and `retraining_trigger`, so both drift pages always read `undefined ?? 0`
// and silently rendered "model stable" even when a retrain was required.
export interface DriftResponse {
  ref_window: string; cur_window: string;
  n_reference_rows: number; n_current_rows: number; computed_at: string;
  features: DriftFeature[];
  verdict_counts?: Record<string, number>; max_psi?: number;
  max_psi_actionable?: number; seasoning_features?: string[];
  missingness_drift_leaders?: Array<Record<string, unknown>>;
  adversarial: {
    available: boolean; adversarial_auc?: number; n_reference?: number; n_current?: number;
    excluded_columns?: string[]; excluded_seasoning_columns?: string[]; interpretation?: string;
    top_drivers?: Array<{ feature: string; importance: number }>;
  };
  retraining_trigger: {
    retrain_required: boolean; reasons: string[]; notes?: string[];
    features_over_psi_threshold: string[]; n_features_over_threshold: number;
    seasoning_features_over_psi_threshold?: string[];
    adversarial_auc?: number; thresholds: Record<string, number>;
  };
  batch_verdict: "PASS" | "WARN" | "FAIL";
  elapsed_ms?: number;
}

export interface ShapEntry {
  feature: string; shap_value?: number; shap?: number; rank?: number;
  family?: string; value?: number; direction?: string; abs_shap?: number;
}
export interface LocalExplainResponse {
  loan_id: string; head: string;
  prediction?: number; probability?: number;
  base_value: number;
  shap_values?: ShapEntry[];
  top_contributions?: ShapEntry[];
  counterfactual?: Record<string, unknown>;
  elapsed_ms?: number;
}
export interface GlobalExplainResponse {
  head: string;
  model_version?: string;
  n_explained?: number;
  global_importance: Array<{ feature: string; mean_abs_shap?: number; shap_mean_abs?: number; rank: number; family?: string; direction?: string; }>;
  family_attribution: Array<{ family: string; share?: number; total_mean_abs_shap?: number; n_features?: number }> | Record<string, number>;
  importance?: Array<{ feature: string; shap_mean_abs?: number; mean_abs_shap?: number; family: string; rank: number }>;
  n_samples?: number; elapsed_ms?: number;
}

export interface ScenarioSummary { expected_loss: number; default_rate: number; prepay_rate: number; }
