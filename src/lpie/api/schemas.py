"""Pydantic v2 request/response contracts for every endpoint.

Typed everywhere: an untyped dict in a risk API is an undocumented API. Each
response carries enough provenance to answer, from the payload alone: which
loan, which observation time, which model, which feature version, which data
version, which rules fired, was it gated, how confident, and can it be
reproduced.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Probability = Annotated[float, Field(ge=0.0, le=1.0)]
MonthIndex = Annotated[int, Field(ge=1, le=600)]
LoanId = Annotated[str, Field(pattern=r"^LN\d{7}$")]
ReportingMonth = Annotated[str, Field(pattern=r"^\d{4}-(0[1-9]|1[0-2])$")]

LoanState = Literal["Current", "30DPD", "60DPD", "90DPD", "Default", "Prepaid", "Closed"]
ReviewerAction = Literal["No Action", "Flag", "Escalate"]
HumanDecision = Literal["Confirm", "Reject", "Escalate"]
HeadName = Literal[
    "next_3m_delinquency", "next_6m_delinquency", "next_12m_default",
    "next_12m_prepayment", "next_state", "exception_required",
]


class LPIEModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True, protected_namespaces=())


# --------------------------------------------------------------------------- #
# errors
# --------------------------------------------------------------------------- #
class ErrorDetail(LPIEModel):
    code: str = Field(description="Stable machine-readable error code")
    message: str = Field(description="Human-readable explanation, safe to display")
    request_id: str | None = Field(default=None, description="Correlates with the server log")
    details: dict[str, Any] | None = Field(default=None, description="Structured context")


class ErrorResponse(LPIEModel):
    error: ErrorDetail


# --------------------------------------------------------------------------- #
# health / meta
# --------------------------------------------------------------------------- #
class ComponentHealth(LPIEModel):
    status: str
    detail: dict[str, Any] = Field(default_factory=dict)


class HealthResponse(LPIEModel):
    status: Literal["ok", "degraded", "unhealthy"]
    ready: bool = Field(description="False when a mandatory artifact or the feature store is missing")
    service: str
    version: str
    model_version: str
    feature_version: str
    git_sha: str | None = None
    data_sha256: str | None = None
    started_at: str
    uptime_seconds: float
    database: dict[str, Any]
    app_store: dict[str, Any]
    artifacts: dict[str, Any]
    loaded_model_versions: dict[str, str]
    feature_store_months: int
    degraded_capabilities: list[str] = Field(default_factory=list)
    missing_artifacts: list[str] = Field(default_factory=list)
    startup_errors: list[dict[str, str]] = Field(default_factory=list)


class ModelRegistryEntry(LPIEModel):
    model_version: str
    head: str
    algo: str | None = None
    trained_at: str | None = None
    train_window: Any = None
    valid_window: Any = None
    embargo_months: int | None = None
    metrics: Any = None
    feature_hash: str | None = None
    config_hash: str | None = None
    code_git_sha: str | None = None
    data_sha256: str | None = None
    artifact_path: str | None = None
    artifact_sha256: str | None = None
    n_features: int | None = None
    status: Literal["candidate", "champion", "archived"]
    notes: str | None = None


class ModelsResponse(LPIEModel):
    model_version: str
    feature_version: str
    feature_hash: str
    config_hash: str
    git_sha: str | None
    data_sha256: str | None
    n_models: int
    champions: dict[str, str]
    models: list[ModelRegistryEntry]
    artifacts: list[dict[str, Any]]


# --------------------------------------------------------------------------- #
# profiling / validation / drift / dq
# --------------------------------------------------------------------------- #
class ProfileRequest(LPIEModel):
    split: Literal["train", "test", "panel"] = Field(
        default="train", description="Which slice of the panel to profile"
    )
    months: list[MonthIndex] | None = Field(
        default=None, description="Restrict to these month_index values"
    )
    columns: list[str] | None = Field(default=None, description="Restrict to these columns")
    sample_rows: int = Field(default=40000, ge=1000, le=200000)
    include_relationships: bool = True
    include_missingness: bool = True
    top_k: int = Field(default=10, ge=1, le=50)


class ColumnProfile(LPIEModel):
    model_config = ConfigDict(extra="allow")
    column: str
    dtype: str
    kind: str | None = None
    n: int
    n_null: int
    null_rate: float | None
    n_unique: int


class ProfileResponse(LPIEModel):
    name: str
    computed_at: str
    n_rows: int
    n_columns: int
    memory_mb: float
    sampled_for_expensive_stages: bool
    sample_rows: int
    schema_: dict[str, str] = Field(alias="schema")
    columns: list[ColumnProfile]
    degenerate_columns: list[str]
    constant_columns: list[str]
    high_null_columns: list[dict[str, Any]]
    missingness: dict[str, Any] | None = None
    relationships: dict[str, Any] | None = None
    temporal_integrity: dict[str, Any] | None = None
    state_machine: dict[str, Any] | None = None
    censoring: dict[str, Any] | None = None
    elapsed_ms: float


class ValidationRecord(LPIEModel):
    model_config = ConfigDict(extra="allow")
    loan_id: str | None = None
    month_index: int | None = None


class ValidationViolation(LPIEModel):
    loan_id: str | None
    month_index: float | None
    rule_id: str
    rule_name: str
    severity: Literal["ERROR", "WARNING"]
    exception_type: str
    dimension: str
    observed_value: str | None
    expected_condition: str | None
    description: str | None


class DQRecordScore(LPIEModel):
    loan_id: str | None = None
    month_index: int | None = None
    completeness: float
    validity: float
    consistency: float
    timeliness: float
    uniqueness: float
    cross_source: float
    dq_score: float
    dq_grade: Literal["A", "B", "C", "D", "F"]
    n_rules_violated: int


class ValidationRequest(LPIEModel):
    records: list[dict[str, Any]] | None = Field(
        default=None, description="Rows to validate. Omit to validate a stored slice."
    )
    split: Literal["train", "test", "panel"] = "train"
    months: list[MonthIndex] | None = None
    loan_ids: list[LoanId] | None = None
    limit: int = Field(default=5000, ge=1, le=100000)
    include_records: bool = Field(default=False, description="Return per-record rule results")
    max_violations: int = Field(default=500, ge=0, le=20000)
    batch_id: str = Field(default="adhoc", max_length=64)

    @model_validator(mode="after")
    def _one_source(self) -> ValidationRequest:
        if self.records is not None and (self.months or self.loan_ids):
            raise ValueError("Supply either `records` or a stored-slice selector, not both")
        return self


class ValidationResponse(LPIEModel):
    record_results: list[ValidationRecord]
    violations: list[ValidationViolation]
    dq_score: list[DQRecordScore]
    batch_score: dict[str, Any]
    summary: dict[str, Any]
    elapsed_ms: float


class DriftFeature(LPIEModel):
    feature: str
    kind: str
    psi: float | None
    ks_stat: float | None
    ks_pvalue: float | None
    js_div: float | None
    ref_null_rate: float
    cur_null_rate: float
    missing_delta: float
    value_verdict: str
    missingness_verdict: str
    verdict: str
    driver: str
    seasoning: bool = False


class DriftResponse(LPIEModel):
    ref_window: str
    cur_window: str
    n_reference_rows: int
    n_current_rows: int
    computed_at: str
    features: list[DriftFeature]
    verdict_counts: dict[str, int]
    max_psi: float | None
    # Max PSI over non-seasoning features only — the number the batch verdict
    # reads. `max_psi` remains the true max so the seasoning shift stays visible.
    max_psi_actionable: float | None = None
    seasoning_features: list[str] = []
    missingness_drift_leaders: list[dict[str, Any]]
    adversarial: dict[str, Any]
    retraining_trigger: dict[str, Any]
    batch_verdict: Literal["PASS", "WARN", "FAIL"]
    elapsed_ms: float


class DQSummaryResponse(LPIEModel):
    n_records: int
    mean_dq: float | None
    median_dq: float | None
    grade: str | None
    grade_distribution: dict[str, int]
    pct_grade_a: float | None
    pct_grade_f: float | None
    mean_dimension_scores: dict[str, float]
    top_violated_rules: list[dict[str, Any]]
    n_error_violations: int
    n_warning_violations: int
    by_month: list[dict[str, Any]] = Field(default_factory=list)
    computed_at: str


# --------------------------------------------------------------------------- #
# prediction
# --------------------------------------------------------------------------- #
class PredictionValue(LPIEModel):
    value: Probability
    ci: list[float] | None = Field(default=None, min_length=2, max_length=2)
    calibrated: bool
    raw: Probability | None = None
    ensemble_disagreement: float | None = None


class NextStatePrediction(LPIEModel):
    predicted: LoanState
    probs: dict[str, Probability]
    confidence: Probability
    legal_mask_applied: bool = True


class Predictions(LPIEModel):
    prob_next_3m_delinquency: PredictionValue
    prob_next_6m_delinquency: PredictionValue
    prob_next_12m_default: PredictionValue
    prob_next_12m_prepayment: PredictionValue
    next_state: NextStatePrediction


class RuleFired(LPIEModel):
    rule_id: str
    severity: str
    name: str | None = None
    exception_type: str | None = None
    observed: str | None = None
    expected: str | None = None
    message: str | None = None


class AnomalyBlock(LPIEModel):
    score: Probability
    tier: str
    rule_severity: float
    worst_severity: str
    detector_scores: dict[str, float | None] = Field(default_factory=dict)
    rules_fired: list[RuleFired] = Field(default_factory=list)
    drivers: list[str] = Field(default_factory=list)


class ExceptionBlock(LPIEModel):
    required: int = Field(ge=0, le=1)
    type: str
    source: str


class DriverBlock(LPIEModel):
    feature: str
    value: Any = None
    shap: float | None = None
    direction: str | None = None


class ExplanationBlock(LPIEModel):
    top_drivers: list[DriverBlock] = Field(default_factory=list)
    narrative: str | None = None


class ConfidenceBlock(LPIEModel):
    model_confidence: Probability
    conformal_width: float | None
    ensemble_disagreement: float | None
    segment_ece: float | None
    data_quality: float | None


class DataQualityBlock(LPIEModel):
    dq_score: float | None
    dq_grade: str | None
    violations: list[str] = Field(default_factory=list)


class SurvivalBlock(LPIEModel):
    horizons_m: list[int]
    survival: list[float]
    cif_default: list[float]
    cif_prepay: list[float]
    cif_closed: list[float]
    conservation_max_error: float


class PredictionBundle(LPIEModel):
    loan_id: str
    reporting_month: str | None
    month_index: int
    model_version: str
    feature_version: str
    scored_at: str
    current_status: LoanState
    current_balance: float
    is_terminal: bool
    gated_by_rule: str | None
    calibration_segment: str | None
    predictions: Predictions
    survival: SurvivalBlock | None = None
    anomaly: AnomalyBlock
    exception: ExceptionBlock
    explanation: ExplanationBlock
    confidence: ConfidenceBlock
    expected_loss: float
    reviewer_action: ReviewerAction
    data_quality: DataQualityBlock


class PredictionRequest(LPIEModel):
    records: list[dict[str, Any]] | None = Field(
        default=None,
        description="Raw monthly rows to score. Features are built online from stored history.",
    )
    loan_ids: list[LoanId] | None = Field(default=None, max_length=5000)
    months: list[MonthIndex] | None = Field(default=None, max_length=64)
    include_survival: bool = False
    survival_horizon: int = Field(default=24, ge=1, le=60)
    include_drivers: bool = True

    @model_validator(mode="after")
    def _need_a_selector(self) -> PredictionRequest:
        if self.records is None and not self.loan_ids:
            raise ValueError("Supply `records` to score, or `loan_ids` to score stored rows")
        if self.records is not None and self.loan_ids:
            raise ValueError("Supply either `records` or `loan_ids`, not both")
        return self


class PredictionResponse(LPIEModel):
    predictions: list[PredictionBundle]
    n_rows: int
    model_version: str
    feature_version: str
    scored_at: str
    degraded_components: list[str] = Field(default_factory=list)
    elapsed_ms: float


# --------------------------------------------------------------------------- #
# portfolio
# --------------------------------------------------------------------------- #
class PortfolioSummaryResponse(LPIEModel):
    as_of_month_index: int
    reporting_month: str | None
    model_version: str
    total_loans: int
    total_balance: float
    active_loans: int
    terminal_loans: int
    delinquency_rate: float
    projected_default_rate: float
    projected_prepayment_rate: float
    expected_loss: float
    expected_loss_pct_of_balance: float | None
    risk_distribution: dict[str, int]
    reviewer_action_distribution: dict[str, int]
    dq_distribution: dict[str, int]
    state_distribution: dict[str, int]
    confidence_distribution: dict[str, Any]
    segments: dict[str, list[dict[str, Any]]]
    computed_at: str
    elapsed_ms: float


class WatchlistEntry(LPIEModel):
    rank: int
    loan_id: str
    month_index: int
    current_status: LoanState
    current_balance: float
    prob_next_12m_default: Probability
    expected_loss: float
    anomaly_score: Probability
    exception_required: int
    exception_type: str
    reviewer_action: ReviewerAction
    model_confidence: Probability
    dq_grade: str | None
    top_drivers: list[str] = Field(default_factory=list)


class WatchlistResponse(LPIEModel):
    n: int
    ranked_by: str
    capacity: int
    total_expected_loss_in_watchlist: float
    share_of_portfolio_expected_loss: float | None
    filters: dict[str, Any]
    entries: list[WatchlistEntry]
    computed_at: str
    elapsed_ms: float


# --------------------------------------------------------------------------- #
# survival
# --------------------------------------------------------------------------- #
class SurvivalResponse(LPIEModel):
    loan_id: str
    month_index: int
    current_status: LoanState
    model_version: str
    horizons_m: list[int]
    survival: list[float]
    cif_default: list[float]
    cif_prepay: list[float]
    cif_closed: list[float]
    conservation_max_error: float
    conservation_holds: bool
    is_terminal: bool
    gated_by_rule: str | None
    baselines: dict[str, Any] = Field(default_factory=dict)
    elapsed_ms: float


class SegmentSurvivalRequest(LPIEModel):
    segment_by: Literal[
        "credit_score_band", "vintage_year_num", "state", "servicer_name", "ltv_band_ord", "current_status"
    ] = "credit_score_band"
    values: list[str] | None = Field(default=None, max_length=25)
    horizon: int = Field(default=24, ge=1, le=60)
    max_loans: int = Field(default=10000, ge=100, le=20000)
    include_kaplan_meier: bool = True


class SegmentSurvivalResponse(LPIEModel):
    segment_by: str
    horizon: int
    n_loans: int
    model_curves: list[dict[str, Any]]
    kaplan_meier: dict[str, Any] | None = None
    log_rank: dict[str, Any] | None = None
    elapsed_ms: float


class StateOccupancyResponse(LPIEModel):
    as_of_month_index: int
    horizon: int
    states: list[str]
    months: list[int]
    mean_share: list[list[float]]
    n_loans: int
    model_version: str
    elapsed_ms: float


# --------------------------------------------------------------------------- #
# scenario
# --------------------------------------------------------------------------- #
class ScenarioInfo(LPIEModel):
    scenario_name: str
    description: str
    gdp_growth_pct: float
    unemployment_rate_pct: float
    hpi_change_pct: float
    interest_rate_shock_bps: float
    credit_spread_shock_bps: float
    prepayment_cpr_assumption_pct: float
    default_rate_multiplier: float
    delinquency_rate_multiplier: float
    prepayment_rate_multiplier: float


class ScenariosResponse(LPIEModel):
    scenarios: list[ScenarioInfo]
    defaults: dict[str, Any]
    segment_options: list[str]


class ScenarioRequest(LPIEModel):
    scenario: str = Field(min_length=1, max_length=64)
    n_paths: int = Field(default=1000, ge=10, le=5000)
    horizon: int = Field(default=24, ge=1, le=60)
    segment_by: Literal["vintage", "credit_band", "state", "servicer", "ltv_band"] | None = None
    seed: int | None = Field(default=None, ge=0)
    reanchor: bool = True
    max_loans: int | None = Field(default=None, ge=100, le=20000)


class CustomScenarioRequest(LPIEModel):
    name: str = Field(default="Custom", max_length=64)
    description: str = Field(default="User-defined macro shock", max_length=512)
    gdp_growth_pct: float = Field(default=0.0, ge=-20, le=20)
    unemployment_rate_pct: float = Field(default=4.1, ge=0, le=50)
    hpi_change_pct: float = Field(default=0.0, ge=-60, le=60)
    interest_rate_shock_bps: float = Field(default=0.0, ge=-1000, le=1000)
    credit_spread_shock_bps: float = Field(default=0.0, ge=-1000, le=2000)
    prepayment_cpr_assumption_pct: float = Field(default=8.0, ge=0, le=100)
    default_rate_multiplier: float = Field(default=1.0, ge=0.01, le=20)
    delinquency_rate_multiplier: float = Field(default=1.0, ge=0.01, le=20)
    prepayment_rate_multiplier: float = Field(default=1.0, ge=0.01, le=20)
    n_paths: int = Field(default=500, ge=10, le=5000)
    horizon: int = Field(default=24, ge=1, le=60)
    segment_by: Literal["vintage", "credit_band", "state", "servicer", "ltv_band"] | None = None
    seed: int | None = Field(default=None, ge=0)
    reanchor: bool = False


class ScenarioResponse(LPIEModel):
    scenario: str
    assumptions: dict[str, Any]
    summary: dict[str, Any]
    segments: list[dict[str, Any]]
    reanchoring: dict[str, Any]
    invariants_passed: bool
    cache_key: str
    elapsed_ms: float


class SensitivityResponse(LPIEModel):
    scenario: str
    horizon_months: int
    n_loans: int
    metric: str
    tornado: dict[str, Any]
    shapley: dict[str, Any] | None = None
    tornado_vs_shapley: dict[str, Any] | None = None
    elapsed_ms: float


# --------------------------------------------------------------------------- #
# anomaly / reviewer
# --------------------------------------------------------------------------- #
class AnomalyEntry(LPIEModel):
    loan_id: str
    month_index: int
    current_status: LoanState
    anomaly_score: Probability
    anomaly_tier: str
    exception_required: int
    exception_type: str
    rule_severity: float
    worst_severity: str
    current_balance: float
    dq_score: float | None
    dq_grade: str | None
    reviewer_action: ReviewerAction
    rules_fired: list[str] = Field(default_factory=list)


class AnomalyListResponse(LPIEModel):
    n: int
    total_matching: int
    filters: dict[str, Any]
    entries: list[AnomalyEntry]
    detectors_available: dict[str, bool]
    elapsed_ms: float


class AnomalyCardResponse(LPIEModel):
    model_config = ConfigDict(extra="allow")
    loan_id: str
    month_index: int
    anomaly: dict[str, Any]
    exception: dict[str, Any]
    rules_fired: list[dict[str, Any]]
    shap_drivers: list[dict[str, Any]]
    nearest_normal: list[dict[str, Any]]
    history: dict[str, Any]
    data_quality: dict[str, Any]
    governance: dict[str, Any]
    elapsed_ms: float


class ReviewerDecisionRequest(LPIEModel):
    loan_id: LoanId
    month_index: MonthIndex
    human_decision: HumanDecision
    model_recommendation: ReviewerAction | None = None
    rationale: str | None = Field(default=None, max_length=2000)
    reviewer: str | None = Field(default=None, max_length=128)
    anomaly_score: Probability | None = None
    exception_type: str | None = Field(default=None, max_length=64)

    @field_validator("rationale", "reviewer", "exception_type")
    @classmethod
    def _strip(cls, value: str | None) -> str | None:
        return value.strip() if isinstance(value, str) else value


class ReviewerDecisionResponse(LPIEModel):
    id: int
    loan_id: str
    month_index: int
    human_decision: HumanDecision
    model_recommendation: str | None
    agreed_with_model: bool | None
    decided_at: str
    agreement_stats: dict[str, Any]


# --------------------------------------------------------------------------- #
# explainability
# --------------------------------------------------------------------------- #
class GlobalExplainResponse(LPIEModel):
    head: str
    model_version: str
    n_explained: int
    global_importance: list[dict[str, Any]]
    family_attribution: list[dict[str, Any]]
    permutation_importance: list[dict[str, Any]] = Field(default_factory=list)
    monotonicity_audit: list[dict[str, Any]] = Field(default_factory=list)
    interpretation: str
    elapsed_ms: float


class LocalExplainResponse(LPIEModel):
    loan_id: str
    month_index: int
    head: str
    model_version: str
    probability: Probability
    base_value: float
    top_contributions: list[dict[str, Any]]
    narrative: str
    peer_comparison: list[dict[str, Any]]
    history_strip: dict[str, Any]
    conformal_interval: list[float] | None
    confidence: dict[str, Any]
    semantics: dict[str, str]
    elapsed_ms: float


class CounterfactualRequest(LPIEModel):
    loan_id: LoanId
    month_index: MonthIndex
    head: HeadName = "next_12m_default"
    target_probability: Probability | None = None
    max_changes: int = Field(default=3, ge=1, le=5)
    features: list[str] | None = Field(default=None, max_length=12)


class CounterfactualResponse(LPIEModel):
    loan_id: str
    month_index: int
    head: str
    found: bool
    original_probability: Probability | None = None
    target_probability: float | None = None
    counterfactual: dict[str, Any] | None = None
    alternatives: list[dict[str, Any]] = Field(default_factory=list)
    actionable_features: list[str] = Field(default_factory=list)
    forbidden_features: list[str] = Field(default_factory=list)
    narrative: str | None = None
    governance: str | None = None
    reason: str | None = None
    n_evaluated: int = 0
    elapsed_ms: float


class ErrorAnalysisResponse(LPIEModel):
    head: str
    threshold: float
    confusion_profile: dict[str, Any]
    error_slices: dict[str, Any]
    segment_performance: dict[str, Any]
    calibration_by_segment: list[dict[str, Any]]
    fairness: dict[str, Any]
    elapsed_ms: float


# --------------------------------------------------------------------------- #
# copilot
# --------------------------------------------------------------------------- #
class VerifierResult(LPIEModel):
    verdict: Literal["PASS", "REGENERATED", "FALLBACK", "FAIL"]
    passed: bool
    regenerated: bool
    used_fallback: bool
    n_failures: int
    failures: list[dict[str, Any]] = Field(default_factory=list)
    numbers_checked: int = 0
    numbers_matched: int = 0
    fields_checked: int = 0
    rules_checked: int = 0


class CopilotRequest(LPIEModel):
    question: str = Field(min_length=3, max_length=2000)
    loan_id: LoanId | None = None
    month_index: MonthIndex | None = None
    use_sql: bool = Field(default=False, description="Allow generated read-only SQL over DuckDB")
    top_k: int = Field(default=6, ge=1, le=20)
    use_cache: bool = True


class ReviewerNoteRequest(LPIEModel):
    loan_id: LoanId
    month_index: MonthIndex
    use_cache: bool = True


class ScenarioSummaryRequest(LPIEModel):
    scenario: str = Field(min_length=1, max_length=64)
    n_paths: int = Field(default=500, ge=10, le=5000)
    horizon: int = Field(default=24, ge=1, le=60)
    use_cache: bool = True


class CopilotResponseModel(LPIEModel):
    task: str
    answer: str
    banner: str
    verifier: VerifierResult
    citations: list[dict[str, Any]] = Field(default_factory=list)
    evidence_hash: str
    model: str
    provider: str
    latency_ms: int
    prompt_log_id: int | None
    sql: str | None = None
    query_preview: list[dict[str, Any]] = Field(default_factory=list)
    llm_error_code: str | None = Field(
        default=None,
        description=(
            "Provider failure that forced the deterministic fallback, when one "
            "occurred: GROQ_API_KEY_EXHAUSTED, GROQ_AUTH_ERROR, "
            "GROQ_MODEL_NOT_FOUND, GROQ_TIMEOUT, LLM_UNAVAILABLE. Null on a "
            "normal generation. Never contains the key itself."
        ),
    )
    llm_error: str | None = Field(
        default=None, description="Human-readable form of `llm_error_code`, key-redacted."
    )
    governance: dict[str, Any] = Field(default_factory=dict)


class PromptLogEntry(LPIEModel):
    """One audited LLM call.

    The whole point of the prompt log is that a reviewer can see *exactly* what
    was sent and what came back, so the content fields below are part of the
    contract, not debug extras. `raw_output` is what the model first returned;
    `final_output` is what the client was actually served — they differ
    whenever the verifier forced a regeneration or fell back to the
    deterministic template, and seeing both side by side is what makes a
    rejection auditable.
    """

    model_config = ConfigDict(extra="allow")
    id: int
    ts: str
    task: str | None
    model: str | None
    provider: str | None
    verifier_verdict: str | None
    accepted: int | None
    latency_ms: int | None
    # --- content: what went in, what came back ---
    system_prompt: str | None = None
    user_prompt: str | None = None
    raw_output: str | None = None
    regenerated_output: str | None = None
    final_output: str | None = None
    verifier_failures: list[dict[str, Any]] = Field(default_factory=list)
    retrieved_context: list[dict[str, Any]] = Field(default_factory=list)
    evidence_packet: dict[str, Any] | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    request_id: str | None = None


class PromptLogResponse(LPIEModel):
    entries: list[PromptLogEntry]
    stats: dict[str, Any]
    rejection_gallery: list[dict[str, Any]] = Field(default_factory=list)
    n: int


# --------------------------------------------------------------------------- #
# submission
# --------------------------------------------------------------------------- #
class SubmissionGenerateRequest(LPIEModel):
    as_of_month: MonthIndex | None = Field(
        default=None, description="Panel month to score. Defaults to the first test month."
    )
    reporting_month: ReportingMonth | None = None
    write_file: bool = True
    strict_row_set: bool = Field(
        default=False, description="Restrict output to the template's loan IDs"
    )


class SubmissionResponse(LPIEModel):
    valid: bool
    path: str | None
    n_rows: int
    n_loans: int
    reporting_month: str
    validation: dict[str, Any]
    manifest: dict[str, Any] | None = None
    preview: list[dict[str, Any]] = Field(default_factory=list)
    elapsed_ms: float


class SubmissionValidateResponse(LPIEModel):
    valid: bool
    path: str | None
    n_rows: int
    n_loans: int
    n_errors: int
    n_warnings: int
    errors: list[dict[str, Any]]
    warnings: list[dict[str, Any]]
    summary: dict[str, Any]
    template: str
