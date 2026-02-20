"""
Lightweight Enhanced Casey implementation used for automated regression tests.

The real production agent offers richer capabilities, but this module provides
enough structure for the test harness to exercise monitoring, analytics, and
messaging flows without requiring external services.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class AnalysisRecord:
    analysis_type: str
    priority: str
    confidence: float
    findings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    timestamp: _dt.datetime = field(default_factory=_dt.datetime.utcnow)


class EnhancedCaseyAgent:
    """
    Minimal enhanced orchestrator with monitoring, analytics, and messaging hooks.
    """

    def __init__(self) -> None:
        self.monitored_agents: Dict[str, Dict[str, Any]] = {}
        self.message_history: List[Dict[str, Any]] = []
        self.system_metrics: List[Any] = []
        self.analysis_history: List[AnalysisRecord] = []
        self._optimization_suggestions: List[str] = []

        self.config: Dict[str, Any] = {
            "ai": {"enable_predictive_analysis": True},
            "monitoring": {"system_check_interval": 5, "metric_window": 60},
            "thresholds": {"cpu_warning": 75, "cpu_critical": 90},
        }

    # ------------------------------------------------------------------
    # Monitoring
    # ------------------------------------------------------------------
    def monitor_process(self, name: str, pid: int) -> None:
        self.monitored_agents[name] = {
            "pid": pid,
            "start_time": _dt.datetime.utcnow(),
        }

    def get_system_status(self) -> Dict[str, Any]:
        recent_metrics = [m for m in self.system_metrics[-10:] if m is not None]
        return {
            "timestamp": _dt.datetime.utcnow().isoformat(),
            "monitored_agents": list(self.monitored_agents.keys()),
            "system_metrics": recent_metrics,
            "message_history_count": len(self.message_history),
        }

    # ------------------------------------------------------------------
    # Messaging
    # ------------------------------------------------------------------
    def receive_message(self, message: Dict[str, Any]) -> None:
        self.message_history.append(message)
        if message.get("type") == "error_report":
            self._optimization_suggestions.append(
                f"Investigate {message.get('agent_name', 'unknown')} error."
            )

    # ------------------------------------------------------------------
    # Analytics
    # ------------------------------------------------------------------
    def _summaries_from_metrics(self) -> Dict[str, float]:
        cpu_values: List[float] = []
        mem_values: List[float] = []
        for metric in self.system_metrics:
            if metric is None:
                continue
            cpu = getattr(metric, "cpu_percent", None)
            mem = getattr(metric, "memory_percent", None)
            if isinstance(cpu, (int, float)):
                cpu_values.append(float(cpu))
            if isinstance(mem, (int, float)):
                mem_values.append(float(mem))
        return {
            "cpu_avg": sum(cpu_values) / len(cpu_values) if cpu_values else 0.0,
            "mem_avg": sum(mem_values) / len(mem_values) if mem_values else 0.0,
        }

    def _record_analysis(self, record: AnalysisRecord) -> None:
        self.analysis_history.append(record)

    def _predictive_analysis(self) -> Optional[Dict[str, Any]]:
        if len(self.system_metrics) < 5:
            return None
        summary = self._summaries_from_metrics()
        cpu_trend = summary["cpu_avg"] * 0.01
        mem_trend = summary["mem_avg"] * 0.01
        return {
            "system": {
                "cpu_trend": cpu_trend,
                "memory_trend": mem_trend,
                "predicted_cpu_1h": summary["cpu_avg"] + cpu_trend * 60,
                "predicted_memory_1h": summary["mem_avg"] + mem_trend * 60,
            }
        }

    def _calculate_overall_system_health(self) -> float:
        summary = self._summaries_from_metrics()
        cpu_score = max(0.0, 100.0 - summary["cpu_avg"])
        mem_score = max(0.0, 100.0 - summary["mem_avg"])
        return (cpu_score + mem_score) / 2 if (cpu_score or mem_score) else 100.0

    def _analyze_system_performance(self) -> Optional[AnalysisRecord]:
        if not self.system_metrics:
            return None
        summary = self._summaries_from_metrics()
        findings = [
            f"Average CPU: {summary['cpu_avg']:.1f}%",
            f"Average Memory: {summary['mem_avg']:.1f}%",
        ]
        recommendations = ["Review resource allocation", "Balance workload across agents"]
        record = AnalysisRecord(
            analysis_type="system_performance",
            priority="medium",
            confidence=0.85,
            findings=findings,
            recommendations=recommendations,
        )
        self._record_analysis(record)
        return record

    def get_analysis_summary(self) -> Dict[str, Any]:
        health_score = self._calculate_overall_system_health()
        return {
            "total_analyses": len(self.analysis_history),
            "system_health_score": health_score,
            "optimization_suggestions": list(self._optimization_suggestions),
        }

    # ------------------------------------------------------------------
    # Convenience helpers for tests
    # ------------------------------------------------------------------
    def get_system_status_snapshot(self) -> Dict[str, Any]:  # pragma: no cover
        return self.get_system_status()


