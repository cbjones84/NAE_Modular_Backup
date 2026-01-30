# NAE Feedback Loops

This document summarises the new feedback-loop architecture introduced for the Neural Agency Engine (NAE).  
The implementation draws on industry-standard continuous-improvement frameworks such as PDCA (Plan → Do → Check → Act), OODA (Observe → Orient → Decide → Act), and reinforcement learning feedback cycles common in quantitative trading desks.

## 1. Performance Feedback (OptimusAgent)
- **Purpose**: Monitor trade outcomes, win-rate trends, and slippage in near real time.
- **Loop**: Collect → Analyse → Act.
  - *Collect*: Captures P&L, realised slippage, and timing confidence for each trade.
  - *Analyse*: Computes rolling win rate, average gains/losses, and slippage statistics.
  - *Act*: Dynamically adjusts `dynamic_risk_scalar` and `dynamic_slippage_penalty`, updates `performance_snapshot`, and logs advisory messages for Splinter/Supervision.
- **Benefits**: Implements a PDCA-style loop around every execution cycle allowing Optimus to self-tune within bounded limits before human review.

## 2. Risk Feedback (OptimusAgent / Bebop / Splinter)
- **Purpose**: Enforce tighter controls when drawdowns or consecutive losses reach thresholds aligned with FINRA/SEC best practices.
- **Loop**:
  - *Collect*: Uses Optimus `risk_metrics`, NAV, and consecutive loss counters.
  - *Analyse*: Calculates severity levels (normal/high/critical) based on daily loss limits and cumulative drawdown.
  - *Act*: Updates `risk_state`; at critical levels it calls `activate_kill_switch`, mirroring OODA-style rapid response used on trading floors.
- **Benefits**: Ensures that risk escalation pathways are consistent with documented kill-switch policy and surfaces structured alerts to Bebop/Splinter.

## 3. Research Feedback (CaseyAgent)
- **Purpose**: Track the quality and source effectiveness of research discoveries, and automatically queue high-priority items.
- **Loop**:
  - *Collect*: Records metadata for each discovery and integration plan produced by Casey.
  - *Analyse*: Maintains source frequency and priority distribution to highlight the most productive research channels.
  - *Act*: Updates `research_dashboard` and pushes high-priority suggestions into Casey’s `improvement_suggestions` backlog for review.
- **Benefits**: Provides a continuous-improvement loop for research automation, enabling prioritisation similar to modern model governance pipelines.

## 4. Feedback Manager
- `FeedbackManager` centralises registration and execution of loops.
- Agents attach loops in their constructors and run them within existing workflows without cross-coupled dependencies.
- Logs are written to `logs/feedback/*.jsonl` for post-trade analytics or external monitoring dashboards.

## Operational Notes
- Feedback loops are intentionally conservative: adjustments stay within pre-defined bounds and do not bypass manual approvals.
- Logs and snapshots allow Splinter/Bebop to audit loop actions and broadcast improvement recommendations through the existing agent messaging system.
- The instrumentation is covered by unit tests in `tests/test_feedback_loops.py` and regression tests for Alpaca paper trading.

For further integration ideas, see:
- `tools/feedback_loops/performance.py`
- `tools/feedback_loops/risk.py`
- `tools/feedback_loops/research.py`



