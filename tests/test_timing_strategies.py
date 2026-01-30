import datetime

from tools.profit_algorithms.timing_strategies import (
    ExitReason,
    TimingStrategyEngine,
)


def test_trailing_stop_triggers_after_peak_drawdown():
    engine = TimingStrategyEngine(pdt_prevention=False)

    price_data = [
        {"high": 100.0, "close": 100.0},
        {"high": 110.0, "close": 109.0},
        {"high": 109.5, "close": 108.0},
        {"high": 108.0, "close": 105.0},
    ]

    entry_price = 100.0
    current_price = 103.0  # Below trailing stop derived from 110 high
    current_pnl = current_price - entry_price
    current_pnl_pct = current_pnl / entry_price

    entry_time = datetime.datetime.now() - datetime.timedelta(days=2)

    analysis = engine.analyze_exit_timing(
        symbol="TEST",
        entry_price=entry_price,
        entry_time=entry_time,
        current_price=current_price,
        quantity=1,
        price_data=price_data,
        current_pnl=current_pnl,
        current_pnl_pct=current_pnl_pct,
    )

    assert analysis.should_exit is True
    assert analysis.exit_reason == ExitReason.TRAILING_STOP
    assert any("Trailing stop triggered" in reason for reason in analysis.exit_reasons)

