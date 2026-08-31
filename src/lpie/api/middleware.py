"""Request middleware: IDs, timing, size limits, rate limiting, API key.

Ordering is deliberate — the outermost layer assigns the request ID so every
later layer, including the error handlers, can attach it.
"""

from __future__ import annotations

import time
import uuid
from collections import deque
from typing import Any

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from lpie.core.config import Settings
from lpie.core.logging import bind_contextvars, clear_contextvars, get_logger

log = get_logger("lpie.api.access")


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Assign a request ID, time the call, and emit one structured access line."""

    def __init__(self, app: Any, settings: Settings) -> None:
        super().__init__(app)
        self.settings = settings
        self.header = str(settings.get("api.request_id_header", "X-Request-ID"))
        self.slow_ms = float(settings.get("logging.slow_request_ms", 2000))

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request_id = request.headers.get(self.header) or uuid.uuid4().hex
        request.state.request_id = request_id
        bind_contextvars(request_id=request_id, path=request.url.path, method=request.method)

        started = time.perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        finally:
            latency_ms = round((time.perf_counter() - started) * 1000, 3)
            try:
                response.headers[self.header] = request_id  # type: ignore[name-defined]
                response.headers["X-Response-Time-Ms"] = str(latency_ms)  # type: ignore[name-defined]
                response.headers["X-Model-Version"] = self.settings.model_version  # type: ignore[name-defined]
            except (NameError, UnboundLocalError, AttributeError):
                pass

            from lpie.api.metrics import METRICS

            METRICS.observe_request(request.url.path, request.method, status_code, latency_ms)
            log.info(
                "request",
                request_id=request_id, method=request.method, path=request.url.path,
                status_code=status_code, latency_ms=latency_ms,
                model_version=self.settings.model_version,
                slow=latency_ms > self.slow_ms,
            )
            clear_contextvars()


class BodySizeLimitMiddleware(BaseHTTPMiddleware):
    """Reject oversized bodies before they are parsed.

    The Content-Length check is cheap and catches the common case; the streaming
    check catches a chunked upload that lies about its size.
    """

    def __init__(self, app: Any, max_bytes: int) -> None:
        super().__init__(app)
        self.max_bytes = int(max_bytes)

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        declared = request.headers.get("content-length")
        if declared is not None:
            try:
                if int(declared) > self.max_bytes:
                    return self._too_large(request, int(declared))
            except ValueError:
                pass
        return await call_next(request)

    def _too_large(self, request: Request, size: int) -> JSONResponse:
        request_id = getattr(request.state, "request_id", None)
        return JSONResponse(
            status_code=413,
            content={
                "error": {
                    "code": "PAYLOAD_TOO_LARGE",
                    "message": f"Request body of {size} bytes exceeds the {self.max_bytes}-byte limit",
                    "request_id": request_id,
                }
            },
        )


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Fixed-window-per-client token counter.

    In-process and deliberately simple. This deployment is a single container by
    design, so a shared counter would need infrastructure the architecture
    explicitly excludes. Behind more than one replica, put the limit at the proxy.
    """

    def __init__(self, app: Any, requests_per_minute: int, burst: int, exempt_paths: set[str]) -> None:
        super().__init__(app)
        self.limit = int(requests_per_minute)
        self.burst = int(burst)
        self.exempt = exempt_paths
        self._windows: dict[str, deque[float]] = {}

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        if request.url.path in self.exempt:
            return await call_next(request)

        client = request.headers.get("X-Forwarded-For", "").split(",")[0].strip() or (
            request.client.host if request.client else "unknown"
        )
        now = time.monotonic()
        window = self._windows.setdefault(client, deque())
        while window and now - window[0] > 60.0:
            window.popleft()

        if len(window) >= self.limit + self.burst:
            retry_after = max(1, int(60 - (now - window[0])))
            from lpie.api.metrics import METRICS

            METRICS.increment("lpie_rate_limited_total")
            return JSONResponse(
                status_code=429,
                content={
                    "error": {
                        "code": "RATE_LIMITED",
                        "message": f"Rate limit of {self.limit} requests/minute exceeded",
                        "request_id": getattr(request.state, "request_id", None),
                    }
                },
                headers={"Retry-After": str(retry_after)},
            )
        window.append(now)

        # Bound the client table so a spray of unique source IPs cannot grow it.
        if len(self._windows) > 4096:
            for key in [k for k, v in self._windows.items() if not v][:1024]:
                self._windows.pop(key, None)

        return await call_next(request)


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Optional shared-secret gate. Off unless LPIE_API_KEY is set."""

    def __init__(self, app: Any, api_key: str, header: str, public_paths: set[str]) -> None:
        super().__init__(app)
        self.api_key = api_key
        self.header = header
        self.public_paths = public_paths

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        if request.url.path in self.public_paths or request.method == "OPTIONS":
            return await call_next(request)

        supplied = request.headers.get(self.header)
        if not supplied or not _constant_time_equals(supplied, self.api_key):
            # The key itself is never logged, on any path.
            log.warning("request.unauthorized", path=request.url.path,
                        request_id=getattr(request.state, "request_id", None))
            return JSONResponse(
                status_code=401,
                content={
                    "error": {
                        "code": "UNAUTHORIZED",
                        "message": f"A valid {self.header} header is required",
                        "request_id": getattr(request.state, "request_id", None),
                    }
                },
            )
        return await call_next(request)


def _constant_time_equals(a: str, b: str) -> bool:
    import hmac

    return hmac.compare_digest(a.encode("utf-8"), b.encode("utf-8"))
