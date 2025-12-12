"""Metrics collection and monitoring utilities."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LatencyMetrics:
    """Track latency statistics."""

    window_size: int = 1000
    _latencies: deque = field(default_factory=lambda: deque(maxlen=1000))

    def record(self, latency_ms: float) -> None:
        """Record a latency measurement."""
        self._latencies.append(latency_ms)

    @property
    def count(self) -> int:
        """Number of recorded latencies."""
        return len(self._latencies)

    @property
    def mean(self) -> float:
        """Mean latency."""
        if not self._latencies:
            return 0.0
        return sum(self._latencies) / len(self._latencies)

    @property
    def p50(self) -> float:
        """50th percentile (median) latency."""
        return self._percentile(50)

    @property
    def p95(self) -> float:
        """95th percentile latency."""
        return self._percentile(95)

    @property
    def p99(self) -> float:
        """99th percentile latency."""
        return self._percentile(99)

    def _percentile(self, p: int) -> float:
        """Calculate percentile."""
        if not self._latencies:
            return 0.0
        sorted_latencies = sorted(self._latencies)
        idx = int(len(sorted_latencies) * p / 100)
        idx = min(idx, len(sorted_latencies) - 1)
        return sorted_latencies[idx]

    def to_dict(self) -> dict[str, float]:
        """Export metrics as dictionary."""
        return {
            "count": self.count,
            "mean_ms": self.mean,
            "p50_ms": self.p50,
            "p95_ms": self.p95,
            "p99_ms": self.p99,
        }


@dataclass
class ThroughputMetrics:
    """Track throughput statistics."""

    _count: int = 0
    _start_time: float | None = None
    _window_counts: deque = field(default_factory=lambda: deque(maxlen=60))
    _window_times: deque = field(default_factory=lambda: deque(maxlen=60))

    def record(self, count: int = 1) -> None:
        """Record processed items."""
        if self._start_time is None:
            self._start_time = time.time()

        self._count += count
        now = time.time()
        self._window_counts.append(count)
        self._window_times.append(now)

    @property
    def total_count(self) -> int:
        """Total items processed."""
        return self._count

    @property
    def elapsed_seconds(self) -> float:
        """Elapsed time since start."""
        if self._start_time is None:
            return 0.0
        return time.time() - self._start_time

    @property
    def overall_rate(self) -> float:
        """Overall throughput rate (items/sec)."""
        elapsed = self.elapsed_seconds
        if elapsed <= 0:
            return 0.0
        return self._count / elapsed

    @property
    def recent_rate(self) -> float:
        """Recent throughput rate from sliding window."""
        if len(self._window_times) < 2:
            return self.overall_rate

        time_span = self._window_times[-1] - self._window_times[0]
        if time_span <= 0:
            return 0.0

        return sum(self._window_counts) / time_span

    def to_dict(self) -> dict[str, Any]:
        """Export metrics as dictionary."""
        return {
            "total_count": self.total_count,
            "elapsed_seconds": self.elapsed_seconds,
            "overall_rate": self.overall_rate,
            "recent_rate": self.recent_rate,
        }


class ServiceMetrics:
    """Aggregate metrics for a service."""

    def __init__(self, service_name: str):
        self.service_name = service_name
        self.latency = LatencyMetrics()
        self.throughput = ThroughputMetrics()
        self.errors = 0
        self.start_time = time.time()

    def record_success(self, latency_ms: float) -> None:
        """Record a successful operation."""
        self.latency.record(latency_ms)
        self.throughput.record()

    def record_error(self) -> None:
        """Record an error."""
        self.errors += 1

    def to_dict(self) -> dict[str, Any]:
        """Export all metrics as dictionary."""
        return {
            "service": self.service_name,
            "uptime_seconds": time.time() - self.start_time,
            "errors": self.errors,
            "latency": self.latency.to_dict(),
            "throughput": self.throughput.to_dict(),
        }

    def __repr__(self) -> str:
        metrics = self.to_dict()
        return (
            f"ServiceMetrics({self.service_name}: "
            f"count={metrics['throughput']['total_count']}, "
            f"rate={metrics['throughput']['overall_rate']:.1f}/s, "
            f"p50={metrics['latency']['p50_ms']:.2f}ms, "
            f"p99={metrics['latency']['p99_ms']:.2f}ms)"
        )

