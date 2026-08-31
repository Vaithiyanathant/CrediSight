"""DuckDB analytical store.

One embedded OLAP database holds the raw zone, the quality zone, the serving
zone and a view over the Parquet feature store. Connections are pooled per
thread — a DuckDB connection is not thread-safe, and opening one per request
would dominate latency on a small container.
"""

from __future__ import annotations

import threading
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from lpie.core.config import Settings, get_settings
from lpie.core.exceptions import InvalidRequestError
from lpie.core.logging import get_logger

log = get_logger(__name__)

SCHEMA_SQL = """
-- ============ RAW ZONE (immutable, as-ingested) ============
CREATE TABLE IF NOT EXISTS raw_monthly (
  loan_id VARCHAR, month_index INTEGER, reporting_month VARCHAR,
  origination_month VARCHAR, loan_age_months INTEGER, remaining_term_months INTEGER,
  original_balance DOUBLE, current_balance DOUBLE, interest_rate DOUBLE,
  credit_score_band VARCHAR, ltv_band VARCHAR, dti_band VARCHAR, state VARCHAR,
  loan_purpose VARCHAR, occupancy_type VARCHAR, property_type VARCHAR,
  servicer_name VARCHAR, current_status VARCHAR, days_past_due DOUBLE,
  modification_flag INTEGER, prepayment_flag INTEGER, default_flag INTEGER,
  loss_severity_band VARCHAR, last_updated_at DATE, source_system VARCHAR,
  document_status VARCHAR,
  next_3m_delinquency_flag INTEGER, next_6m_delinquency_flag INTEGER,
  next_12m_default_flag INTEGER, next_12m_prepayment_flag INTEGER,
  next_state VARCHAR, exception_required INTEGER, exception_type VARCHAR,
  _split VARCHAR
);
CREATE TABLE IF NOT EXISTS raw_static (
  loan_id VARCHAR, origination_month VARCHAR, original_balance DOUBLE,
  interest_rate DOUBLE, loan_term_months INTEGER, credit_score_band VARCHAR,
  ltv_band VARCHAR, dti_band VARCHAR, state VARCHAR, loan_purpose VARCHAR,
  occupancy_type VARCHAR, property_type VARCHAR, servicer_name VARCHAR,
  vintage_year VARCHAR
);
CREATE TABLE IF NOT EXISTS raw_servicer (
  loan_id VARCHAR, update_date DATE, servicer_name VARCHAR, reported_balance DOUBLE,
  reported_status VARCHAR, reported_rate DOUBLE, source_system VARCHAR,
  conflict_type VARCHAR, stale_flag INTEGER, notes VARCHAR
);
CREATE TABLE IF NOT EXISTS raw_macro (
  scenario_name VARCHAR, description VARCHAR, gdp_growth_pct DOUBLE,
  unemployment_rate_pct DOUBLE, hpi_change_pct DOUBLE, interest_rate_shock_bps DOUBLE,
  credit_spread_shock_bps DOUBLE, prepayment_cpr_assumption_pct DOUBLE,
  default_rate_multiplier DOUBLE, delinquency_rate_multiplier DOUBLE,
  prepayment_rate_multiplier DOUBLE
);

-- ============ QUALITY ZONE ============
CREATE TABLE IF NOT EXISTS dq_rule_results (
  loan_id VARCHAR, month_index INTEGER, rule_id VARCHAR, severity VARCHAR,
  exception_type VARCHAR, dimension VARCHAR,
  observed_value VARCHAR, expected_condition VARCHAR
);
CREATE TABLE IF NOT EXISTS dq_record_scores (
  loan_id VARCHAR, month_index INTEGER,
  completeness DOUBLE, validity DOUBLE, consistency DOUBLE,
  timeliness DOUBLE, uniqueness DOUBLE, cross_source DOUBLE,
  dq_score DOUBLE, dq_grade VARCHAR, n_rules_violated INTEGER
);
CREATE TABLE IF NOT EXISTS dq_batch_scores (
  batch_id VARCHAR, month_index INTEGER, n_records INTEGER, mean_dq DOUBLE,
  pct_grade_a DOUBLE, pct_grade_f DOUBLE, n_error_violations INTEGER,
  n_warning_violations INTEGER, drift_verdict VARCHAR, computed_at TIMESTAMP
);
CREATE TABLE IF NOT EXISTS drift_metrics (
  feature VARCHAR, ref_window VARCHAR, cur_window VARCHAR, psi DOUBLE,
  ks_stat DOUBLE, ks_pvalue DOUBLE, js_div DOUBLE, missing_delta DOUBLE,
  verdict VARCHAR, kind VARCHAR, computed_at TIMESTAMP
);

-- ============ SERVING ZONE ============
CREATE TABLE IF NOT EXISTS predictions (
  loan_id VARCHAR, month_index INTEGER, reporting_month VARCHAR, model_version VARCHAR,
  feature_version VARCHAR,
  prob_next_3m_delinquency DOUBLE, prob_next_6m_delinquency DOUBLE,
  prob_next_12m_default DOUBLE, prob_next_12m_prepayment DOUBLE,
  predicted_next_state VARCHAR, next_state_probs VARCHAR,
  anomaly_score DOUBLE, exception_required INTEGER, exception_type VARCHAR,
  top_driver_1 VARCHAR, top_driver_2 VARCHAR, top_driver_3 VARCHAR,
  reviewer_action VARCHAR, model_confidence DOUBLE,
  expected_loss DOUBLE, current_balance DOUBLE,
  is_terminal BOOLEAN, gated_by_rule VARCHAR, scored_at TIMESTAMP
);
CREATE TABLE IF NOT EXISTS survival_curves (
  loan_id VARCHAR, month_index INTEGER, model_version VARCHAR, horizon INTEGER,
  survival DOUBLE, cif_default DOUBLE, cif_prepay DOUBLE, cif_closed DOUBLE
);
CREATE TABLE IF NOT EXISTS anomaly_scores (
  loan_id VARCHAR, month_index INTEGER, model_version VARCHAR,
  anomaly_score DOUBLE, tier VARCHAR, iforest DOUBLE, ecod DOUBLE,
  autoencoder DOUBLE, self_z DOUBLE, unsupervised_rank DOUBLE,
  rule_severity DOUBLE, exception_required INTEGER, exception_type VARCHAR
);
"""

_INDEX_SQL = [
    "CREATE UNIQUE INDEX IF NOT EXISTS ix_raw_monthly_pk ON raw_monthly (loan_id, month_index)",
    "CREATE UNIQUE INDEX IF NOT EXISTS ix_raw_static_pk ON raw_static (loan_id)",
    "CREATE INDEX IF NOT EXISTS ix_dq_rule_results ON dq_rule_results (loan_id, month_index)",
    "CREATE UNIQUE INDEX IF NOT EXISTS ix_dq_record_scores ON dq_record_scores (loan_id, month_index)",
    "CREATE INDEX IF NOT EXISTS ix_predictions ON predictions (loan_id, month_index, model_version)",
    "CREATE INDEX IF NOT EXISTS ix_anomaly ON anomaly_scores (loan_id, month_index)",
]

# Only these may be referenced by a copilot-generated query. Anything else is a
# hard reject before the SQL ever reaches DuckDB.
READ_ONLY_ALLOWLIST: frozenset[str] = frozenset({
    "raw_monthly", "raw_static", "raw_servicer", "raw_macro",
    "dq_rule_results", "dq_record_scores", "dq_batch_scores", "drift_metrics",
    "predictions", "survival_curves", "anomaly_scores", "features",
})


class DuckDBStore:
    """Thread-local connection pool over one DuckDB file."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.path: Path = self.settings.path("duckdb_path")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._write_lock = threading.Lock()
        self._memory_limit = str(self.settings.get("runtime.duckdb_memory_limit", "1GB"))
        self._threads = int(self.settings.get("runtime.duckdb_threads", 4))

    # ------------------------------------------------------------------ #
    def _configure(self, con: duckdb.DuckDBPyConnection) -> None:
        con.execute(f"SET memory_limit='{self._memory_limit}'")
        con.execute(f"SET threads TO {self._threads}")
        con.execute("SET enable_progress_bar=false")

    def connect(self) -> duckdb.DuckDBPyConnection:
        con = getattr(self._local, "con", None)
        if con is None:
            try:
                con = duckdb.connect(str(self.path))
            except duckdb.IOException:
                # Another process (e.g. uvicorn) holds an exclusive lock.
                # Fall back to read-only mode; if that also fails, re-raise as
                # a service-unavailable error so the API layer can return 503.
                try:
                    con = duckdb.connect(str(self.path), read_only=True)
                except duckdb.IOException as exc:
                    from lpie.core.exceptions import ModelNotLoadedError
                    raise ModelNotLoadedError(
                        f"DuckDB file is locked by another process and cannot be opened "
                        f"in read-only mode either. Path: {self.path}",
                        details={"path": str(self.path), "error": str(exc)},
                    ) from exc
            self._configure(con)
            self._local.con = con
        return con

    def close(self) -> None:
        con = getattr(self._local, "con", None)
        if con is not None:
            con.close()
            self._local.con = None

    # ------------------------------------------------------------------ #
    def initialise(self) -> None:
        """Create every table and index. Idempotent; safe on every startup."""
        con = self.connect()
        with self._write_lock:
            con.execute(SCHEMA_SQL)
            for stmt in _INDEX_SQL:
                try:
                    con.execute(stmt)
                except duckdb.Error:
                    # A unique index cannot be created over data that already
                    # violates it (raw_monthly legitimately may, before dedup).
                    log.debug("duckdb.index_skipped", stmt=stmt)
        self.refresh_feature_view()

    def refresh_feature_view(self) -> None:
        """(Re)point the `features` view at the Parquet store, if it exists."""
        store = self.settings.path("feature_store_dir")
        glob = str(store / "month_index=*" / "*.parquet")
        con = self.connect()
        with self._write_lock:
            if any(store.glob("month_index=*/*.parquet")):
                con.execute(
                    "CREATE OR REPLACE VIEW features AS "
                    f"SELECT * FROM read_parquet('{glob}', hive_partitioning=true)"
                )
            else:
                con.execute("DROP VIEW IF EXISTS features")

    def has_feature_store(self) -> bool:
        store = self.settings.path("feature_store_dir")
        return any(store.glob("month_index=*/*.parquet"))

    # ------------------------------------------------------------------ #
    def query(self, sql: str, params: Sequence[Any] | None = None) -> pd.DataFrame:
        con = self.connect()
        return con.execute(sql, list(params) if params else None).fetch_df()

    def execute(self, sql: str, params: Sequence[Any] | None = None) -> None:
        con = self.connect()
        with self._write_lock:
            con.execute(sql, list(params) if params else None)

    def scalar(self, sql: str, params: Sequence[Any] | None = None) -> Any:
        con = self.connect()
        row = con.execute(sql, list(params) if params else None).fetchone()
        return None if row is None else row[0]

    def table_exists(self, name: str) -> bool:
        return bool(
            self.scalar(
                "SELECT count(*) FROM duckdb_tables() WHERE table_name = ?", [name]
            )
        ) or bool(
            self.scalar(
                "SELECT count(*) FROM duckdb_views() WHERE view_name = ?", [name]
            )
        )

    def row_count(self, name: str) -> int:
        if not self.table_exists(name):
            return 0
        try:
            return int(self.scalar(f"SELECT count(*) FROM {name}") or 0)
        except duckdb.Error:
            return 0

    def replace_table(self, name: str, df: pd.DataFrame) -> int:
        """Atomically swap a table's contents. Used by offline pipeline stages only."""
        con = self.connect()
        with self._write_lock:
            con.register("_incoming", df)
            con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM _incoming")
            con.unregister("_incoming")
        return int(len(df))

    def append_table(self, name: str, df: pd.DataFrame) -> int:
        if df.empty:
            return 0
        con = self.connect()
        with self._write_lock:
            con.register("_incoming", df)
            cols = ", ".join(f'"{c}"' for c in df.columns)
            con.execute(f"INSERT INTO {name} ({cols}) SELECT {cols} FROM _incoming")
            con.unregister("_incoming")
        return int(len(df))

    def delete_where(self, name: str, where: str, params: Sequence[Any] | None = None) -> None:
        self.execute(f"DELETE FROM {name} WHERE {where}", params)

    # ------------------------------------------------------------------ #
    def read_features(
        self,
        months: Iterable[int] | None = None,
        loan_ids: Sequence[str] | None = None,
        columns: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """Partition-pruned read from the Parquet feature store."""
        if not self.has_feature_store():
            return pd.DataFrame()
        self.refresh_feature_view()
        cols = ", ".join(f'"{c}"' for c in columns) if columns else "*"
        clauses: list[str] = []
        params: list[Any] = []
        if months is not None:
            months = [int(m) for m in months]
            if not months:
                return pd.DataFrame()
            clauses.append(f"month_index IN ({', '.join(['?'] * len(months))})")
            params.extend(months)
        if loan_ids is not None:
            loan_ids = list(loan_ids)
            if not loan_ids:
                return pd.DataFrame()
            clauses.append(f"loan_id IN ({', '.join(['?'] * len(loan_ids))})")
            params.extend(loan_ids)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        return self.query(f"SELECT {cols} FROM features {where}", params)

    def health(self) -> dict[str, Any]:
        try:
            self.scalar("SELECT 1")
            tables = {
                name: self.row_count(name)
                for name in ("raw_monthly", "raw_static", "dq_record_scores", "predictions")
            }
            return {
                "status": "ok",
                "path": str(self.path),
                "tables": tables,
                "feature_store": self.has_feature_store(),
            }
        except Exception as exc:  # pragma: no cover - defensive
            return {"status": "error", "path": str(self.path), "error": str(exc)}


# --------------------------------------------------------------------------- #
_STORE: DuckDBStore | None = None
_STORE_LOCK = threading.Lock()


def get_store(settings: Settings | None = None) -> DuckDBStore:
    global _STORE
    if _STORE is None:
        with _STORE_LOCK:
            if _STORE is None:
                _STORE = DuckDBStore(settings)
    return _STORE


def reset_store() -> None:
    global _STORE
    with _STORE_LOCK:
        if _STORE is not None:
            _STORE.close()
        _STORE = None


def validate_read_only_sql(sql: str) -> str:
    """Gate for copilot-generated SQL.

    Read-only, single-statement, allowlisted tables only. This is the only place
    in the system where model-authored SQL can reach the database, so the check
    is deliberately conservative: reject anything not provably safe.
    """
    import re

    text = sql.strip().rstrip(";").strip()
    if not text:
        raise InvalidRequestError("Empty SQL statement")
    if ";" in text:
        raise InvalidRequestError("Only a single SQL statement is permitted")

    lowered = re.sub(r"--[^\n]*", " ", text.lower())
    lowered = re.sub(r"/\*.*?\*/", " ", lowered, flags=re.S)

    if not re.match(r"^\s*(with|select)\b", lowered):
        raise InvalidRequestError("Only SELECT / WITH queries are permitted")

    forbidden = (
        "insert", "update", "delete", "drop", "create", "alter", "attach", "detach",
        "copy", "install", "load", "pragma", "export", "import", "call", "set ",
        "read_csv", "read_parquet", "read_json", "glob", "system", "shell",
    )
    for word in forbidden:
        if re.search(rf"\b{re.escape(word.strip())}\b", lowered):
            raise InvalidRequestError(f"Forbidden SQL keyword: {word.strip()}")

    referenced = set(re.findall(r"\b(?:from|join)\s+([a-zA-Z_][a-zA-Z0-9_]*)", lowered))
    # CTE names defined in this query are legitimate references.
    cte_names = set(re.findall(r"(?:with|,)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+as\s*\(", lowered))
    unknown = referenced - READ_ONLY_ALLOWLIST - cte_names
    if unknown:
        raise InvalidRequestError(
            f"Query references non-allowlisted table(s): {sorted(unknown)}",
        )
    return text
