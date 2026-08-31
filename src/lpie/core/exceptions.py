"""Typed error hierarchy.

Every failure surfaced to a client is one of these. The API exception handler
maps `.http_status` / `.code` into the standard envelope:

    {"error": {"code": ..., "message": ..., "request_id": ..., "details": {...}}}

Stack traces are logged server-side, never returned.
"""

from __future__ import annotations

from typing import Any


class LPIEError(Exception):
    """Base class for every deliberate LPIE failure."""

    code: str = "LPIE_ERROR"
    http_status: int = 500
    public_message: str = "Internal error"

    def __init__(
        self,
        message: str | None = None,
        *,
        details: dict[str, Any] | None = None,
        http_status: int | None = None,
        code: str | None = None,
    ) -> None:
        self.message = message or self.public_message
        self.details = details or {}
        if http_status is not None:
            self.http_status = http_status
        if code is not None:
            self.code = code
        super().__init__(self.message)

    def to_payload(self, request_id: str | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.details:
            payload["details"] = self.details
        if request_id:
            payload["request_id"] = request_id
        return {"error": payload}


# --------------------------------------------------------------------------- #
# Data / contract layer
# --------------------------------------------------------------------------- #
class DataContractError(LPIEError):
    code = "DATA_CONTRACT_VIOLATION"
    http_status = 400
    public_message = "Input data violates the declared schema contract"


class DataNotFoundError(LPIEError):
    code = "DATA_NOT_FOUND"
    http_status = 404
    public_message = "Requested data was not found"


class ValidationRuleError(LPIEError):
    code = "VALIDATION_RULE_ERROR"
    http_status = 500
    public_message = "A validation rule could not be evaluated"


class InvalidRequestError(LPIEError):
    code = "INVALID_REQUEST"
    http_status = 400
    public_message = "The request is malformed for this resource"


# --------------------------------------------------------------------------- #
# Artifact / model layer
# --------------------------------------------------------------------------- #
class ArtifactNotFoundError(LPIEError):
    code = "ARTIFACT_NOT_FOUND"
    http_status = 503
    public_message = "A required model artifact is not available"


class ModelNotLoadedError(LPIEError):
    code = "MODEL_NOT_LOADED"
    http_status = 503
    public_message = "Champion model is unavailable"


class FeatureContractError(LPIEError):
    code = "FEATURE_CONTRACT_VIOLATION"
    http_status = 500
    public_message = "Feature matrix does not satisfy the declared feature contract"


class FeatureStoreEmptyError(LPIEError):
    code = "FEATURE_STORE_EMPTY"
    http_status = 503
    public_message = "Feature store has not been built — run `make features`"


class PredictionError(LPIEError):
    code = "PREDICTION_FAILED"
    http_status = 500
    public_message = "Prediction could not be produced"


class LeakageError(LPIEError):
    code = "LEAKAGE_DETECTED"
    http_status = 500
    public_message = "A point-in-time leakage guard failed"


# --------------------------------------------------------------------------- #
# Downstream engines
# --------------------------------------------------------------------------- #
class ScenarioInvariantError(LPIEError):
    code = "SCENARIO_INVARIANT_VIOLATION"
    http_status = 500
    public_message = "Scenario simulation violated a probability invariant"


class ScenarioNotFoundError(LPIEError):
    code = "SCENARIO_NOT_FOUND"
    http_status = 404
    public_message = "Unknown scenario"


class ScenarioRunError(LPIEError):
    """A named scenario exists but its simulation failed.

    Distinct from ScenarioNotFoundError on purpose: reporting an engine fault as
    404 "unknown scenario" sends the caller to check their spelling instead of
    the server logs.
    """

    code = "SCENARIO_RUN_FAILED"
    http_status = 500
    public_message = "Scenario simulation failed"


class CopilotVerificationError(LPIEError):
    code = "COPILOT_VERIFICATION_FAILED"
    http_status = 502
    public_message = "Generated text failed numeric verification and no fallback was available"


class CopilotUnavailableError(LPIEError):
    code = "COPILOT_UNAVAILABLE"
    http_status = 503
    public_message = "LLM provider is not configured or unreachable"


class SubmissionError(LPIEError):
    code = "SUBMISSION_INVALID"
    http_status = 400
    public_message = "Submission failed schema validation"


class UnsupportedOperationError(LPIEError):
    code = "UNSUPPORTED_OPERATION"
    http_status = 501
    public_message = "Operation is not supported by this deployment"


class RateLimitError(LPIEError):
    code = "RATE_LIMITED"
    http_status = 429
    public_message = "Too many requests"


class PayloadTooLargeError(LPIEError):
    code = "PAYLOAD_TOO_LARGE"
    http_status = 413
    public_message = "Request body exceeds the configured limit"


class AuthError(LPIEError):
    code = "UNAUTHORIZED"
    http_status = 401
    public_message = "Missing or invalid API key"


class ConflictError(LPIEError):
    code = "STATE_CONFLICT"
    http_status = 409
    public_message = "Request conflicts with current resource state"
