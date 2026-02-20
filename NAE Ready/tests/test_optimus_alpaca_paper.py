"""
Regression coverage for Optimus when operating with the Alpaca paper adapter.

These tests avoid external network calls by stubbing Alpaca and vault access.
"""

import datetime
from typing import Any, Dict, List

import pytest

from agents import optimus


class _StubAdapter:
    """Minimal stand-in for AlpacaAdapter used during tests."""

    def __init__(self):
        self.placed_orders: List[Dict[str, Any]] = []
        self.cancelled = False

    def auth(self) -> bool:  # pragma: no cover - trivial
        return True

    def get_positions(self):  # pragma: no cover - empty by default
        return []

    def place_order(self, order: Dict[str, Any]) -> Dict[str, Any]:
        self.placed_orders.append(order)
        return {
            "order_id": "alpaca_test_order",
            "status": "filled",
            "timestamp": datetime.datetime.utcnow().isoformat(),
        }

    def cancel_all_orders(self):  # pragma: no cover - simple flag set
        self.cancelled = True
        return {"status": "cancelled", "orders_cancelled": len(self.placed_orders)}


class _StubAlpacaClient:
    """Drop-in replacement for agents.optimus.AlpacaClient."""

    def __init__(self, *_, **kwargs):
        self.adapter = _StubAdapter()
        self.paper_trading = kwargs.get("paper_trading", True)

    def submit_order(self, order_data: Dict[str, Any]) -> Dict[str, Any]:
        return self.adapter.place_order(order_data)


class _StubVaultClient:
    """No-op vault client so tests don't touch real secrets."""

    def __init__(self):
        pass

    def get_secret(self, *_args, **_kwargs):
        return None


@pytest.fixture()
def optimus_paper(monkeypatch):
    """Provision an OptimusAgent instance with stubbed Alpaca + vault."""

    created_clients: List[_StubAlpacaClient] = []

    def _factory(*args, **kwargs):
        client = _StubAlpacaClient(*args, **kwargs)
        created_clients.append(client)
        return client

    monkeypatch.setattr(optimus, "AlpacaClient", _factory)
    monkeypatch.setattr(optimus, "VaultClient", _StubVaultClient)

    agent = optimus.OptimusAgent(sandbox=False)  # PAPER mode
    yield agent, created_clients


def test_execute_trade_routes_through_alpaca(optimus_paper):
    agent, clients = optimus_paper

    order = {
        "symbol": "AAPL",
        "side": "buy",
        "quantity": 1,
        "price": 10.0,
        "order_type": "market",
        "time_in_force": "day",
        "override_timing": True,
    }

    result = agent.execute_trade(order)

    assert result["broker"] == "alpaca"
    assert result["mode"] == "paper"
    assert result["status"] == "filled"
    assert clients[0].adapter.placed_orders, "Adapter should record placed orders"


def test_trading_status_reports_alpaca_configured(optimus_paper):
    agent, _ = optimus_paper

    status = agent.get_trading_status()
    brokers = status["broker_clients"]

    assert brokers["alpaca_configured"] is True
    assert brokers["ibkr_configured"] is False


def test_cancel_all_orders_uses_adapter(optimus_paper):
    agent, clients = optimus_paper

    agent._cancel_all_orders()

    assert clients[0].adapter.cancelled is True

