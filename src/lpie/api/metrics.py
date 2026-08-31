"""Prometheus-style metrics, in-process and dependency-free.

A counter registry and a latency histogram are all this deployment needs; adding
`prometheus_client` would pull a dependency for a `/metrics` endpoint that is
forty lines of arithmetic. Bucket boundaries are chosen for the workload: batch
scoring is tens of milliseconds, a Monte-Carlo run is seconds.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from typing import Any

LATENCY_BUCKETS = (5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 30000)


def _normalise_path(path: str) -> str:
    """Collapse identifiers so a per-loan endpoint is one label, not ten thousand."""
    parts = []
    for segment in path.strip("/").split("/"):
        if segment.startswith("LN") and segment[2:].isdigit():
            parts.append("{loan_id}")
        elif segment.isdigit():
            parts.append("{n}")
        else:
            parts.append(segment)
    return "/" + "/".join(parts)


class MetricsRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.started_at = time.time()
        self._counters: dict[str, float] = defaultdict(float)
        self._labelled: dict[tuple[str, tuple[tuple[str, str], ...]], float] = defaultdict(float)
        self._histograms: dict[str, list[float]] = defaultdict(lambda: [0.0] * (len(LATENCY_BUCKETS) + 1))
        self._histogram_sums: dict[str, float] = defaultdict(float)
        self._histogram_counts: dict[str, float] = defaultdict(float)
        self._gauges: dict[str, float] = {}

    # ------------------------------------------------------------------ #
    def increment(self, name: str, value: float = 1.0, **labels: str) -> None:
        with self._lock:
            if labels:
                key = (name, tuple(sorted(labels.items())))
                self._labelled[key] += value
            else:
                self._counters[name] += value

    def set_gauge(self, name: str, value: float) -> None:
        with self._lock:
            self._gauges[name] = float(value)

    def observe(self, name: str, value_ms: float) -> None:
        with self._lock:
            buckets = self._histograms[name]
            placed = False
            for i, bound in enumerate(LATENCY_BUCKETS):
                if value_ms <= bound:
                    buckets[i] += 1
                    placed = True
                    break
            if not placed:
                buckets[-1] += 1
            self._histogram_sums[name] += value_ms
            self._histogram_counts[name] += 1

    def observe_request(self, path: str, method: str, status_code: int, latency_ms: float) -> None:
        route = _normalise_path(path)
        self.increment("lpie_http_requests_total", route=route, method=method,
                       status=str(status_code))
        if status_code >= 500:
            self.increment("lpie_http_errors_total", route=route, status=str(status_code))
        elif status_code >= 400:
            self.increment("lpie_http_client_errors_total", route=route, status=str(status_code))
        self.observe("lpie_http_request_duration_ms", latency_ms)

    # ------------------------------------------------------------------ #
    def render(self, extra_gauges: dict[str, Any] | None = None) -> str:
        with self._lock:
            lines: list[str] = []

            lines.append("# HELP lpie_uptime_seconds Seconds since process start")
            lines.append("# TYPE lpie_uptime_seconds gauge")
            lines.append(f"lpie_uptime_seconds {time.time() - self.started_at:.3f}")

            grouped: dict[str, list[tuple[tuple[tuple[str, str], ...], float]]] = defaultdict(list)
            for (name, labels), value in self._labelled.items():
                grouped[name].append((labels, value))

            for name, entries in sorted(grouped.items()):
                lines.append(f"# TYPE {name} counter")
                for labels, value in sorted(entries):
                    label_text = ",".join(f'{k}="{_escape(v)}"' for k, v in labels)
                    lines.append(f"{name}{{{label_text}}} {value:g}")

            for name, value in sorted(self._counters.items()):
                lines.append(f"# TYPE {name} counter")
                lines.append(f"{name} {value:g}")

            for name, buckets in sorted(self._histograms.items()):
                lines.append(f"# TYPE {name} histogram")
                cumulative = 0.0
                for i, bound in enumerate(LATENCY_BUCKETS):
                    cumulative += buckets[i]
                    lines.append(f'{name}_bucket{{le="{bound}"}} {cumulative:g}')
                cumulative += buckets[-1]
                lines.append(f'{name}_bucket{{le="+Inf"}} {cumulative:g}')
                lines.append(f"{name}_sum {self._histogram_sums[name]:g}")
                lines.append(f"{name}_count {self._histogram_counts[name]:g}")

            gauges = {**self._gauges, **(extra_gauges or {})}
            for name, value in sorted(gauges.items()):
                try:
                    numeric = float(value)
                except (TypeError, ValueError):
                    continue
                lines.append(f"# TYPE {name} gauge")
                lines.append(f"{name} {numeric:g}")

        return "\n".join(lines) + "\n"

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "counters": dict(self._counters),
                "labelled_counters": {
                    f"{name}{{{','.join(f'{k}={v}' for k, v in labels)}}}": value
                    for (name, labels), value in self._labelled.items()
                },
                "histogram_counts": dict(self._histogram_counts),
                "histogram_means_ms": {
                    name: round(self._histogram_sums[name] / count, 3)
                    for name, count in self._histogram_counts.items()
                    if count
                },
                "gauges": dict(self._gauges),
                "uptime_seconds": round(time.time() - self.started_at, 3),
            }


def _escape(value: str) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")


METRICS = MetricsRegistry()
