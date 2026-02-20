"""Trading performance feedback loop."""

from __future__ import annotations

import statistics
from collections import deque
from typing import Any, Deque, Dict, Optional

from .base import FeedbackLoop


class PerformanceFeedbackLoop(FeedbackLoop):
    """Monitor trade outcomes and adapt risk controls for Optimus."""

    def __init__(
        self,
        agent: Any,
        max_trades_tracked: int = 100,
        slippage_threshold: float = 0.4,
    ):
        super().__init__(name="performance", agent=agent, history_limit=500)
        self.trade_history: Deque[Dict[str, Any]] = deque(maxlen=max_trades_tracked)
        self.slippage_threshold = slippage_threshold

    def collect(self, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        result = context.get("result", {})
        execution_details = context.get("execution_details", {})

        status = result.get("status")
        if status not in {"filled", "submitted", "cancelled"}:
            return None

        # Capture numeric observations
        pnl = result.get("pnl")
        fill_price = result.get("execution_price") or result.get("fill_price")
        requested_price = execution_details.get("price")
        slippage = None
        if fill_price is not None and requested_price not in (None, 0):
            slippage = float(fill_price) - float(requested_price)

        observation = {
            "status": status,
            "symbol": execution_details.get("symbol"),
            "side": execution_details.get("side"),
            "pnl": pnl,
            "slippage": slippage,
            "confidence": execution_details.get("entry_confidence"),
            "timing_score": execution_details.get("entry_timing_score"),
        }
        self.trade_history.append(observation)
        return observation

    def analyze(
        self, observations: Dict[str, Any], context: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        if not self.trade_history:
            return None

        pnls = [entry["pnl"] for entry in self.trade_history if entry.get("pnl") is not None]
        wins = [p for p in pnls if p is not None and p > 0]
        losses = [p for p in pnls if p is not None and p <= 0]
        positive_trades = [
            entry for entry in self.trade_history if entry.get("pnl") is not None and entry["pnl"] > 0
        ]
        negative_trades = [
            entry for entry in self.trade_history if entry.get("pnl") is not None and entry["pnl"] <= 0
        ]

        slippages = [
            abs(entry["slippage"])
            for entry in self.trade_history
            if entry.get("slippage") is not None
        ]

        trade_count = len(self.trade_history)
        win_rate = len(positive_trades) / trade_count if trade_count else 0.0
        avg_pnl = statistics.mean(pnls) if pnls else 0.0
        avg_win = statistics.mean(wins) if wins else 0.0
        avg_loss = statistics.mean(losses) if losses else 0.0
        avg_slippage = statistics.mean(slippages) if slippages else 0.0

        timing_scores = [
            entry["timing_score"]
            for entry in self.trade_history
            if entry.get("timing_score") is not None
        ]
        avg_timing_score = statistics.mean(timing_scores) if timing_scores else None

        return {
            "trade_count": trade_count,
            "win_rate": round(win_rate, 4),
            "average_pnl": round(avg_pnl, 4),
            "average_win": round(avg_win, 4),
            "average_loss": round(avg_loss, 4),
            "average_slippage": round(avg_slippage, 6),
            "average_timing_score": round(avg_timing_score, 2) if avg_timing_score else None,
        }

    def act(
        self, insights: Dict[str, Any], context: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        if insights is None:
            return None

        actions: Dict[str, Any] = {}
        agent = self.agent

        # Ensure agent has somewhere to store the performance snapshot
        if not hasattr(agent, "performance_snapshot"):
            agent.performance_snapshot = {}
        agent.performance_snapshot.update(insights)
        actions["snapshot_updated"] = True

        trade_count = insights.get("trade_count", 0)
        win_rate = insights.get("win_rate", 0.0)
        avg_slippage = insights.get("average_slippage", 0.0)

        # Initialize adaptive scalars if needed
        if not hasattr(agent, "dynamic_risk_scalar"):
            agent.dynamic_risk_scalar = 1.0
        if not hasattr(agent, "dynamic_slippage_penalty"):
            agent.dynamic_slippage_penalty = 1.0

        # Risk down adjustments when performance degrades
        if trade_count >= 8 and win_rate < 0.45:
            agent.dynamic_risk_scalar = max(0.5, agent.dynamic_risk_scalar * 0.9)
            actions["risk_scalar"] = agent.dynamic_risk_scalar
            if hasattr(agent, "log_action"):
                agent.log_action(
                    "[FeedbackLoop:performance] Win rate below 45%; tightening risk exposure."
                )

        # Reward improvements with gradual scaling
        elif trade_count >= 10 and win_rate > 0.62 and insights.get("average_pnl", 0) > 0:
            agent.dynamic_risk_scalar = min(1.5, agent.dynamic_risk_scalar * 1.05)
            actions["risk_scalar"] = agent.dynamic_risk_scalar
            if hasattr(agent, "log_action"):
                agent.log_action(
                    "[FeedbackLoop:performance] Strong performance; loosening risk exposure slightly."
                )

        # Monitor slippage
        if avg_slippage and avg_slippage > self.slippage_threshold:
            agent.dynamic_slippage_penalty = min(3.0, agent.dynamic_slippage_penalty * 1.1)
            actions["slippage_penalty"] = agent.dynamic_slippage_penalty
            if hasattr(agent, "log_action"):
                agent.log_action(
                    "[FeedbackLoop:performance] Elevated slippage detected; increasing routing penalty."
                )
        elif avg_slippage and avg_slippage < self.slippage_threshold / 2:
            agent.dynamic_slippage_penalty = max(1.0, agent.dynamic_slippage_penalty * 0.95)
            actions["slippage_penalty"] = agent.dynamic_slippage_penalty

        return actions


