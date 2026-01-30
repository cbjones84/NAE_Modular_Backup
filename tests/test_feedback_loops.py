from tools.feedback_loops import (
    PerformanceFeedbackLoop,
    ResearchFeedbackLoop,
    RiskFeedbackLoop,
)


class _StubAgent:
    def __init__(self):
        self.nav = 10000
        self.consecutive_losses = 0
        self.log_messages = []
        self.kill_switch_calls = 0
        self.dynamic_risk_scalar = 1.0
        self.dynamic_slippage_penalty = 1.0
        self.performance_snapshot = {}
        self.risk_state = {}
        self.improvement_suggestions = []

    def log_action(self, message):
        self.log_messages.append(message)

    def activate_kill_switch(self, reason: str):
        self.kill_switch_calls += 1
        self.log_messages.append(f"KILL_SWITCH:{reason}")


def test_performance_feedback_loop_scales_risk_down():
    agent = _StubAgent()
    loop = PerformanceFeedbackLoop(agent=agent, max_trades_tracked=20)

    # Inject a sequence of losses to trigger risk tightening
    context = {
        "execution_details": {"symbol": "AAPL", "side": "sell"},
        "result": {"status": "filled", "pnl": -50, "execution_price": 100},
    }
    for _ in range(8):
        loop.run(context)

    assert agent.dynamic_risk_scalar < 1.0
    assert "snapshot_updated" in loop.history[-1].actions


def test_research_feedback_loop_adds_high_priority_suggestion():
    agent = _StubAgent()
    loop = ResearchFeedbackLoop(agent=agent)

    finding = {"name": "RL_Volatility", "source": "arXiv"}
    plan = {"priority": "high", "status": "pending_review"}

    loop.run({"finding": finding, "plan": plan})

    assert any(
        suggestion.get("algorithm") == "RL_Volatility"
        for suggestion in agent.improvement_suggestions
    )


def test_risk_feedback_loop_triggers_kill_switch_on_consecutive_losses():
    agent = _StubAgent()
    agent.consecutive_losses = 5
    loop = RiskFeedbackLoop(agent=agent, daily_loss_threshold=0.01, consecutive_loss_limit=4)

    risk_metrics = {"daily_loss": -200, "realized_pnl": -500, "unrealized_pnl": -300}
    loop.run({"risk_metrics": risk_metrics, "trade_result": {"status": "filled"}})

    assert agent.kill_switch_calls == 1
    assert agent.risk_state["severity"] == "critical"



