"""Centralised exception handling.

Every error leaves through one envelope and one code path:

    {"error": {"code": ..., "message": ..., "request_id": ..., "details": {...}}}

Stack traces are logged server-side with the request ID and never returned. An
unexpected exception becomes a generic 500 with the request ID attached, so a
support conversation starts with "what was your request ID" rather than with a
guess.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError as PydanticValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from lpie.core.exceptions import LPIEError
from lpie.core.logging import get_logger

log = get_logger(__name__)

HTTP_CODE_NAMES = {
    400: "BAD_REQUEST",
    401: "UNAUTHORIZED",
    403: "FORBIDDEN",
    404: "NOT_FOUND",
    405: "METHOD_NOT_ALLOWED",
    409: "STATE_CONFLICT",
    413: "PAYLOAD_TOO_LARGE",
    422: "REQUEST_VALIDATION_FAILED",
    429: "RATE_LIMITED",
    500: "INTERNAL_ERROR",
    501: "NOT_IMPLEMENTED",
    503: "SERVICE_UNAVAILABLE",
}


def _request_id(request: Request) -> str | None:
    return getattr(request.state, "request_id", None)


def _envelope(
    code: str, message: str, request_id: str | None, details: dict[str, Any] | None = None
) -> dict[str, Any]:
    payload: dict[str, Any] = {"code": code, "message": message}
    if request_id:
        payload["request_id"] = request_id
    if details:
        payload["details"] = details
    return {"error": payload}


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(LPIEError)
    async def _lpie_error(request: Request, exc: LPIEError) -> JSONResponse:
        request_id = _request_id(request)
        log.warning(
            "request.lpie_error",
            request_id=request_id, code=exc.code, status=exc.http_status,
            path=request.url.path, message=exc.message, details=exc.details,
        )
        return JSONResponse(
            status_code=exc.http_status,
            content=exc.to_payload(request_id),
            headers={"X-Error-Code": exc.code},
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        request_id = _request_id(request)
        errors = [
            {
                "location": list(e.get("loc", [])),
                "message": e.get("msg"),
                "type": e.get("type"),
            }
            for e in exc.errors()[:25]
        ]
        log.info("request.validation_failed", request_id=request_id,
                 path=request.url.path, n_errors=len(errors))
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=_envelope(
                "REQUEST_VALIDATION_FAILED",
                "The request body or query does not satisfy the endpoint contract",
                request_id,
                {"errors": errors},
            ),
        )

    @app.exception_handler(PydanticValidationError)
    async def _pydantic_error(request: Request, exc: PydanticValidationError) -> JSONResponse:
        request_id = _request_id(request)
        log.error("response.validation_failed", request_id=request_id,
                  path=request.url.path, error=str(exc)[:500])
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=_envelope(
                "RESPONSE_CONTRACT_VIOLATION",
                "The server produced a response that does not satisfy its own contract",
                request_id,
            ),
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        request_id = _request_id(request)
        code = HTTP_CODE_NAMES.get(exc.status_code, "HTTP_ERROR")
        detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
        return JSONResponse(
            status_code=exc.status_code,
            content=_envelope(code, detail, request_id),
            headers=getattr(exc, "headers", None),
        )

    @app.exception_handler(HTTPException)
    async def _fastapi_http_error(request: Request, exc: HTTPException) -> JSONResponse:
        return await _http_error(request, exc)

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        request_id = _request_id(request)
        # Full traceback server-side; nothing about internals leaves the process.
        log.exception(
            "request.unhandled_exception",
            request_id=request_id, path=request.url.path,
            method=request.method, error_type=type(exc).__name__,
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=_envelope(
                "INTERNAL_ERROR",
                "An unexpected internal error occurred. The request ID identifies it in the "
                "server log.",
                request_id,
            ),
        )
