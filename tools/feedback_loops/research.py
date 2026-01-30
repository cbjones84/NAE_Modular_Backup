"""Research automation feedback loop for Casey."""

from __future__ import annotations

import collections
from typing import Any, Dict, Optional

from .base import FeedbackLoop


class ResearchFeedbackLoop(FeedbackLoop):
    """Track research discoveries and prioritise integration efforts."""

    def __init__(self, agent: Any):
        super().__init__(name="research", agent=agent, history_limit=300)
        self.source_counter: collections.Counter[str] = collections.Counter()
        self.priority_counter: collections.Counter[str] = collections.Counter()

    def collect(self, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        finding = context.get("finding")
        plan = context.get("plan")
        if not finding:
            return None

        source = finding.get("source", "unknown")
        priority = (plan or {}).get("priority", finding.get("priority", "medium"))

        self.source_counter[source] += 1
        self.priority_counter[priority] += 1

        return {
            "algorithm": finding.get("name"),
            "source": source,
            "priority": priority,
            "plan_status": (plan or {}).get("status"),
        }

    def analyze(
        self, observations: Dict[str, Any], context: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        return {
            "top_sources": self.source_counter.most_common(3),
            "priority_distribution": self.priority_counter.most_common(),
            "latest_algorithm": observations.get("algorithm"),
        }

    def act(
        self, insights: Dict[str, Any], context: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        agent = self.agent
        finding = context.get("finding")
        plan = context.get("plan") or {}
        priority = plan.get("priority", "medium")

        if not hasattr(agent, "research_dashboard"):
            agent.research_dashboard = {}
        agent.research_dashboard.update(
            {
                "top_sources": insights.get("top_sources"),
                "priority_distribution": insights.get("priority_distribution"),
                "last_update": insights.get("latest_algorithm"),
            }
        )

        actions: Dict[str, Any] = {"dashboard_updated": True}

        # Encourage timely escalation of high-priority discoveries
        if priority in {"critical", "high"}:
            if hasattr(agent, "improvement_suggestions"):
                suggestion = {
                    "type": "research_follow_up",
                    "algorithm": finding.get("name"),
                    "priority": priority,
                    "source": finding.get("source"),
                    "plan": plan,
                }
                agent.improvement_suggestions.append(suggestion)
                actions["suggestion_added"] = suggestion
            if hasattr(agent, "log_action"):
                agent.log_action(
                    f"[FeedbackLoop:research] Highlighted {finding.get('name')} ({priority}) for rapid review."
                )

        return actions



