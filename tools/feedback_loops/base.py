"""Base classes for feedback loops."""

from __future__ import annotations

import datetime
import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class FeedbackResult:
    """Result produced by a feedback loop execution."""

    loop_name: str
    timestamp: str
    observations: Dict[str, Any] = field(default_factory=dict)
    insights: Dict[str, Any] = field(default_factory=dict)
    actions: Dict[str, Any] = field(default_factory=dict)
    status: str = "ok"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "loop": self.loop_name,
            "timestamp": self.timestamp,
            "observations": self.observations,
            "insights": self.insights,
            "actions": self.actions,
            "status": self.status,
        }


class FeedbackLoop(ABC):
    """Abstract feedback loop with collect -> analyze -> act phases."""

    def __init__(
        self,
        name: str,
        agent: Any,
        history_limit: int = 200,
        log_directory: str = "logs/feedback",
    ):
        self.name = name
        self.agent = agent
        self.history_limit = history_limit
        self.history: list[FeedbackResult] = []
        self.log_directory = log_directory
        os.makedirs(self.log_directory, exist_ok=True)
        self.log_path = os.path.join(self.log_directory, f"{self.name}.jsonl")

    def run(self, context: Dict[str, Any]) -> Optional[FeedbackResult]:
        """Execute the loop phases sequentially."""
        observations = self.collect(context)
        if observations is None:
            return None

        insights = self.analyze(observations, context)
        actions = self.act(insights, context)
        result = FeedbackResult(
            loop_name=self.name,
            timestamp=datetime.datetime.now().isoformat(),
            observations=observations,
            insights=insights or {},
            actions=actions or {},
        )
        self._record(result)
        return result

    def _record(self, result: FeedbackResult) -> None:
        """Persist history in memory and append to log file."""
        self.history.append(result)
        if len(self.history) > self.history_limit:
            self.history.pop(0)

        try:
            with open(self.log_path, "a") as log_file:
                log_file.write(json.dumps(result.to_dict()) + "\n")
        except Exception as exc:
            if hasattr(self.agent, "log_action"):
                self.agent.log_action(
                    f"[FeedbackLoop:{self.name}] Failed to persist log: {exc}"
                )

    @abstractmethod
    def collect(self, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Gather observations/raw data used by the loop."""

    @abstractmethod
    def analyze(
        self, observations: Dict[str, Any], context: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Generate insights from observations."""

    @abstractmethod
    def act(
        self, insights: Dict[str, Any], context: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Apply actions or recommendations based on insights."""



