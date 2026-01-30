"""Risk management feedback loop."""

from __future__ import annotations

from typing import Any, Dict, Optional

from .base import FeedbackLoop


class RiskFeedbackLoop(FeedbackLoop):
    """Assess drawdowns and trigger protective actions."""

    def __init__(
        self,
        agent: Any,
        daily_loss_threshold: float = 0.02,
        consecutive_loss_limit: int = 4,
    ):
        super().__init__(name="risk", agent=agent, history_limit=500)
        self.daily_loss_threshold = daily_loss_threshold
        self.consecutive_loss_limit = consecutive_loss_limit

    def collect(self, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        metrics = context.get("risk_metrics") or getattr(self.agent, "risk_metrics", None)
        if metrics is None:
            return None

        nav = getattr(self.agent, "nav", None)
        daily_loss = metrics.get("daily_loss")
        realized_pnl = metrics.get("realized_pnl")
        unrealized_pnl = metrics.get("unrealized_pnl")
        consecutive_losses = getattr(self.agent, "consecutive_losses", 0)

        observation = {
            "nav": nav,
            "daily_loss": daily_loss,
            "realized_pnl": realized_pnl,
            "unrealized_pnl": unrealized_pnl,
            "consecutive_losses": consecutive_losses,
            "trade_status": context.get("trade_result", {}).get("status"),
        }
        return observation

    def analyze(
        self, observations: Dict[str, Any], context: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        nav = observations.get("nav") or 0
        daily_loss = observations.get("daily_loss") or 0
        realized = observations.get("realized_pnl") or 0
        unrealized = observations.get("unrealized_pnl") or 0
        total_pnl = realized + unrealized
        consecutive_losses = observations.get("consecutive_losses", 0)

        risk_flags = []
        severity_level = 0  # 0=normal,1=high,2=critical

        if nav and daily_loss < -(nav * self.daily_loss_threshold):
            risk_flags.append("daily_loss_limit")
            severity_level = max(severity_level, 1)
        if consecutive_losses >= self.consecutive_loss_limit:
            risk_flags.append("consecutive_losses")
            severity_level = max(severity_level, 2)
        if total_pnl < -(nav * 0.05 if nav else 0):
            risk_flags.append("drawdown_warning")
            severity_level = max(severity_level, 1)

        severity_map = {0: "normal", 1: "high", 2: "critical"}
        severity = severity_map[severity_level]

        return {
            "nav": nav,
            "daily_loss": daily_loss,
            "total_pnl": total_pnl,
            "consecutive_losses": consecutive_losses,
            "risk_flags": risk_flags,
            "severity": severity,
        }

    def act(
        self, insights: Dict[str, Any], context: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        if insights is None:
            return None

        agent = self.agent
        severity = insights.get("severity", "normal")
        risk_flags = insights.get("risk_flags", [])

        if not hasattr(agent, "risk_state"):
            agent.risk_state = {}
        agent.risk_state.update(insights)

        actions: Dict[str, Any] = {"severity": severity}

        if severity == "critical":
            if hasattr(agent, "activate_kill_switch"):
                agent.activate_kill_switch("Risk feedback loop detected critical condition")
                actions["kill_switch_triggered"] = True
            if hasattr(agent, "log_action"):
                agent.log_action("[FeedbackLoop:risk] Critical condition detected; kill switch engaged.")
        elif severity == "high" and hasattr(agent, "log_action"):
            agent.log_action(
                f"[FeedbackLoop:risk] Elevated risk detected ({', '.join(risk_flags) or 'none'})"
            )

        return actions


