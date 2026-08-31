"""SQLite application store (WAL).

Transactional app state that must survive restarts and be trivially inspectable:
reviewer decisions (the human-in-the-loop record), the copilot prompt log (the
governance audit trail), and the model registry (champion/challenger lineage).

DuckDB holds analytics; this holds *decisions*. Keeping them apart means an
analytical rebuild can never destroy an audit record.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from lpie.core.config import Settings, get_settings
from lpie.core.logging import get_logger
from lpie.core.timing import utcnow_iso

log = get_logger(__name__)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS reviewer_decisions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  loan_id TEXT NOT NULL,
  month_index INTEGER NOT NULL,
  model_recommendation TEXT,
  model_version TEXT,
  human_decision TEXT NOT NULL,
  rationale TEXT,
  reviewer TEXT,
  decided_at TEXT NOT NULL,
  agreed_with_model INTEGER,
  anomaly_score REAL,
  exception_type TEXT
);
CREATE INDEX IF NOT EXISTS ix_rd_loan ON reviewer_decisions (loan_id, month_index);
CREATE INDEX IF NOT EXISTS ix_rd_time ON reviewer_decisions (decided_at);

CREATE TABLE IF NOT EXISTS prompt_log (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts TEXT NOT NULL,
  request_id TEXT,
  model TEXT,
  task TEXT,
  system_prompt TEXT,
  user_prompt TEXT,
  retrieved_context TEXT,
  evidence_packet TEXT,
  raw_output TEXT,
  verifier_verdict TEXT,
  verifier_failures TEXT,
  regenerated_output TEXT,
  final_output TEXT,
  accepted INTEGER,
  latency_ms INTEGER,
  input_tokens INTEGER,
  output_tokens INTEGER,
  provider TEXT
);
CREATE INDEX IF NOT EXISTS ix_pl_ts ON prompt_log (ts);
CREATE INDEX IF NOT EXISTS ix_pl_verdict ON prompt_log (verifier_verdict);

CREATE TABLE IF NOT EXISTS model_registry (
  model_version TEXT NOT NULL,
  head TEXT NOT NULL,
  algo TEXT,
  trained_at TEXT,
  train_window TEXT,
  valid_window TEXT,
  embargo_months INTEGER,
  metrics TEXT,
  feature_hash TEXT,
  config_hash TEXT,
  code_git_sha TEXT,
  data_sha256 TEXT,
  artifact_path TEXT,
  artifact_sha256 TEXT,
  n_features INTEGER,
  status TEXT NOT NULL CHECK(status IN ('candidate','champion','archived')),
  notes TEXT,
  PRIMARY KEY (model_version, head)
);
CREATE INDEX IF NOT EXISTS ix_mr_head_status ON model_registry (head, status);

CREATE TABLE IF NOT EXISTS pipeline_runs (
  run_id TEXT PRIMARY KEY,
  stage TEXT NOT NULL,
  started_at TEXT,
  finished_at TEXT,
  status TEXT,
  inputs TEXT,
  outputs TEXT,
  row_counts TEXT,
  duration_ms INTEGER,
  git_sha TEXT,
  data_sha256 TEXT,
  error TEXT
);
"""


class AppStore:
    """Thread-local SQLite connections in WAL mode."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.path: Path = self.settings.path("sqlite_path")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()

    def connect(self) -> sqlite3.Connection:
        con = getattr(self._local, "con", None)
        if con is None:
            con = sqlite3.connect(str(self.path), timeout=30.0, check_same_thread=False)
            con.row_factory = sqlite3.Row
            con.execute("PRAGMA journal_mode=WAL")
            con.execute("PRAGMA synchronous=NORMAL")
            con.execute("PRAGMA busy_timeout=30000")
            con.execute("PRAGMA foreign_keys=ON")
            self._local.con = con
        return con

    def close(self) -> None:
        con = getattr(self._local, "con", None)
        if con is not None:
            con.close()
            self._local.con = None

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        con = self.connect()
        try:
            yield con
            con.commit()
        except Exception:
            con.rollback()
            raise

    def initialise(self) -> None:
        with self.tx() as con:
            con.executescript(SCHEMA_SQL)

    # ------------------------------------------------------------------ #
    # reviewer decisions
    # ------------------------------------------------------------------ #
    def record_reviewer_decision(self, payload: dict[str, Any]) -> int:
        row = {
            "loan_id": payload["loan_id"],
            "month_index": int(payload["month_index"]),
            "model_recommendation": payload.get("model_recommendation"),
            "model_version": payload.get("model_version"),
            "human_decision": payload["human_decision"],
            "rationale": payload.get("rationale"),
            "reviewer": payload.get("reviewer"),
            "decided_at": payload.get("decided_at") or utcnow_iso(),
            "agreed_with_model": (
                None
                if payload.get("agreed_with_model") is None
                else int(bool(payload["agreed_with_model"]))
            ),
            "anomaly_score": payload.get("anomaly_score"),
            "exception_type": payload.get("exception_type"),
        }
        cols = ", ".join(row)
        marks = ", ".join("?" * len(row))
        with self.tx() as con:
            cur = con.execute(
                f"INSERT INTO reviewer_decisions ({cols}) VALUES ({marks})",
                list(row.values()),
            )
            return int(cur.lastrowid)

    def list_reviewer_decisions(
        self, loan_id: str | None = None, limit: int = 100, offset: int = 0
    ) -> list[dict[str, Any]]:
        sql = "SELECT * FROM reviewer_decisions"
        params: list[Any] = []
        if loan_id:
            sql += " WHERE loan_id = ?"
            params.append(loan_id)
        sql += " ORDER BY id DESC LIMIT ? OFFSET ?"
        params.extend([int(limit), int(offset)])
        return [dict(r) for r in self.connect().execute(sql, params).fetchall()]

    def reviewer_agreement_stats(self) -> dict[str, Any]:
        row = self.connect().execute(
            "SELECT count(*) AS n, "
            "       sum(CASE WHEN agreed_with_model = 1 THEN 1 ELSE 0 END) AS n_agreed "
            "FROM reviewer_decisions WHERE agreed_with_model IS NOT NULL"
        ).fetchone()
        n = int(row["n"] or 0)
        agreed = int(row["n_agreed"] or 0)
        by_decision = {
            r["human_decision"]: int(r["n"])
            for r in self.connect().execute(
                "SELECT human_decision, count(*) AS n FROM reviewer_decisions "
                "GROUP BY human_decision"
            ).fetchall()
        }
        return {
            "n_decisions_with_comparison": n,
            "n_agreed": agreed,
            "agreement_rate": (agreed / n) if n else None,
            "by_decision": by_decision,
            "n_total": int(
                self.connect().execute("SELECT count(*) AS n FROM reviewer_decisions").fetchone()["n"]
            ),
        }

    # ------------------------------------------------------------------ #
    # prompt log
    # ------------------------------------------------------------------ #
    def log_prompt(self, record: dict[str, Any]) -> int:
        row = {
            "ts": record.get("ts") or utcnow_iso(),
            "request_id": record.get("request_id"),
            "model": record.get("model"),
            "task": record.get("task"),
            "system_prompt": record.get("system_prompt"),
            "user_prompt": record.get("user_prompt"),
            "retrieved_context": _as_text(record.get("retrieved_context")),
            "evidence_packet": _as_text(record.get("evidence_packet")),
            "raw_output": record.get("raw_output"),
            "verifier_verdict": record.get("verifier_verdict"),
            "verifier_failures": _as_text(record.get("verifier_failures")),
            "regenerated_output": record.get("regenerated_output"),
            "final_output": record.get("final_output"),
            "accepted": int(bool(record.get("accepted"))),
            "latency_ms": record.get("latency_ms"),
            "input_tokens": record.get("input_tokens"),
            "output_tokens": record.get("output_tokens"),
            "provider": record.get("provider"),
        }
        cols = ", ".join(row)
        marks = ", ".join("?" * len(row))
        with self.tx() as con:
            cur = con.execute(f"INSERT INTO prompt_log ({cols}) VALUES ({marks})", list(row.values()))
            return int(cur.lastrowid)

    def list_prompt_log(
        self, limit: int = 50, offset: int = 0, verdict: str | None = None, task: str | None = None
    ) -> list[dict[str, Any]]:
        sql = "SELECT * FROM prompt_log"
        clauses, params = [], []
        if verdict:
            clauses.append("verifier_verdict = ?")
            params.append(verdict)
        if task:
            clauses.append("task = ?")
            params.append(task)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY id DESC LIMIT ? OFFSET ?"
        params.extend([int(limit), int(offset)])
        rows = [dict(r) for r in self.connect().execute(sql, params).fetchall()]
        for r in rows:
            for k in ("retrieved_context", "evidence_packet", "verifier_failures"):
                r[k] = _from_text(r.get(k))
        return rows

    def prompt_log_stats(self) -> dict[str, Any]:
        rows = self.connect().execute(
            "SELECT verifier_verdict AS v, count(*) AS n FROM prompt_log GROUP BY verifier_verdict"
        ).fetchall()
        by_verdict = {str(r["v"]): int(r["n"]) for r in rows}
        total = sum(by_verdict.values())
        return {
            "total": total,
            "by_verdict": by_verdict,
            "verification_failure_rate": (
                (total - by_verdict.get("PASS", 0)) / total if total else None
            ),
        }

    # ------------------------------------------------------------------ #
    # model registry
    # ------------------------------------------------------------------ #
    def register_model(self, entry: dict[str, Any], *, promote: bool = False) -> None:
        row = {
            "model_version": entry["model_version"],
            "head": entry["head"],
            "algo": entry.get("algo"),
            "trained_at": entry.get("trained_at") or utcnow_iso(),
            "train_window": _as_text(entry.get("train_window")),
            "valid_window": _as_text(entry.get("valid_window")),
            "embargo_months": entry.get("embargo_months"),
            "metrics": _as_text(entry.get("metrics")),
            "feature_hash": entry.get("feature_hash"),
            "config_hash": entry.get("config_hash"),
            "code_git_sha": entry.get("code_git_sha"),
            "data_sha256": entry.get("data_sha256"),
            "artifact_path": entry.get("artifact_path"),
            "artifact_sha256": entry.get("artifact_sha256"),
            "n_features": entry.get("n_features"),
            "status": entry.get("status", "candidate"),
            "notes": entry.get("notes"),
        }
        cols = ", ".join(row)
        marks = ", ".join("?" * len(row))
        with self.tx() as con:
            con.execute(
                f"INSERT OR REPLACE INTO model_registry ({cols}) VALUES ({marks})",
                list(row.values()),
            )
            if promote or row["status"] == "champion":
                con.execute(
                    "UPDATE model_registry SET status='archived' "
                    "WHERE head=? AND model_version<>? AND status='champion'",
                    (row["head"], row["model_version"]),
                )
                con.execute(
                    "UPDATE model_registry SET status='champion' WHERE head=? AND model_version=?",
                    (row["head"], row["model_version"]),
                )

    def get_champion(self, head: str) -> dict[str, Any] | None:
        row = self.connect().execute(
            "SELECT * FROM model_registry WHERE head=? AND status='champion' "
            "ORDER BY trained_at DESC LIMIT 1",
            (head,),
        ).fetchone()
        return _decode_registry_row(row)

    def list_models(self, head: str | None = None, status: str | None = None) -> list[dict[str, Any]]:
        sql = "SELECT * FROM model_registry"
        clauses, params = [], []
        if head:
            clauses.append("head = ?")
            params.append(head)
        if status:
            clauses.append("status = ?")
            params.append(status)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY head, trained_at DESC"
        return [_decode_registry_row(r) for r in self.connect().execute(sql, params).fetchall()]

    def set_status(self, model_version: str, head: str, status: str) -> None:
        if status not in {"candidate", "champion", "archived"}:
            raise ValueError(f"invalid status {status!r}")
        with self.tx() as con:
            if status == "champion":
                con.execute(
                    "UPDATE model_registry SET status='archived' WHERE head=? AND status='champion'",
                    (head,),
                )
            con.execute(
                "UPDATE model_registry SET status=? WHERE model_version=? AND head=?",
                (status, model_version, head),
            )

    # ------------------------------------------------------------------ #
    def record_pipeline_run(self, record: dict[str, Any]) -> None:
        row = {
            "run_id": record["run_id"],
            "stage": record["stage"],
            "started_at": record.get("started_at"),
            "finished_at": record.get("finished_at"),
            "status": record.get("status"),
            "inputs": _as_text(record.get("inputs")),
            "outputs": _as_text(record.get("outputs")),
            "row_counts": _as_text(record.get("row_counts")),
            "duration_ms": record.get("duration_ms"),
            "git_sha": record.get("git_sha"),
            "data_sha256": record.get("data_sha256"),
            "error": record.get("error"),
        }
        cols = ", ".join(row)
        marks = ", ".join("?" * len(row))
        with self.tx() as con:
            con.execute(
                f"INSERT OR REPLACE INTO pipeline_runs ({cols}) VALUES ({marks})",
                list(row.values()),
            )

    def health(self) -> dict[str, Any]:
        try:
            con = self.connect()
            counts = {
                t: int(con.execute(f"SELECT count(*) FROM {t}").fetchone()[0])
                for t in ("reviewer_decisions", "prompt_log", "model_registry")
            }
            return {"status": "ok", "path": str(self.path), "tables": counts}
        except Exception as exc:  # pragma: no cover - defensive
            return {"status": "error", "path": str(self.path), "error": str(exc)}


def _as_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return json.dumps(value, default=str)


def _from_text(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return value


def _decode_registry_row(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    d = dict(row)
    for k in ("metrics", "train_window", "valid_window"):
        d[k] = _from_text(d.get(k))
    return d


_APP_STORE: AppStore | None = None
_LOCK = threading.Lock()


def get_app_store(settings: Settings | None = None) -> AppStore:
    global _APP_STORE
    if _APP_STORE is None:
        with _LOCK:
            if _APP_STORE is None:
                _APP_STORE = AppStore(settings)
    return _APP_STORE


def reset_app_store() -> None:
    global _APP_STORE
    with _LOCK:
        if _APP_STORE is not None:
            _APP_STORE.close()
        _APP_STORE = None
