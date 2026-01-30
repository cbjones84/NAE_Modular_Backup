"""
Transaction cost and execution modelling utilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class ExecutionInputs:
    mid_price: float
    bid_price: float
    ask_price: float
    order_size: int
    venue_liquidity_score: float
    volatility: float


@dataclass
class ExecutionCost:
    expected_slippage: float
    spread_cost: float
    market_impact_cost: float
    total_cost: float


class ExecutionCostModel:
    """
    Simple execution cost estimator combining spread, slippage and impact.
    """

    def __init__(self, impact_coefficient: float = 0.0005):
        self.impact_coefficient = impact_coefficient

    def estimate(self, inputs: ExecutionInputs) -> ExecutionCost:
        spread = max(inputs.ask_price - inputs.bid_price, 0.0)
        spread_cost = 0.5 * spread * inputs.order_size

        liquidity_factor = max(inputs.venue_liquidity_score, 1e-6)
        expected_slippage = (spread / liquidity_factor) * 0.25 * inputs.order_size

        market_impact = (
            self.impact_coefficient
            * inputs.order_size
            * max(inputs.volatility, 1e-4)
        )

        total_cost = spread_cost + expected_slippage + market_impact

        return ExecutionCost(
            expected_slippage=float(expected_slippage),
            spread_cost=float(spread_cost),
            market_impact_cost=float(market_impact),
            total_cost=float(total_cost),
        )


