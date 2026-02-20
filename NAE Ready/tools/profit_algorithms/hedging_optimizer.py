"""
Dynamic Greeks hedging optimiser with cost-aware rebalance rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class GreekExposure:
    delta: float
    gamma: float
    theta: float
    vega: float


@dataclass
class HedgingDecision:
    should_rebalance: bool
    expected_benefit: float
    transaction_cost: float
    hedging_frequency_hours: float
    rationale: str


class HedgingOptimizer:
    """
    Optimises hedging cadence based on expected gamma P&L vs transaction costs.
    """

    def __init__(self, min_frequency_hours: float = 1.0):
        self.min_frequency_hours = min_frequency_hours

    def expected_gamma_pnl(
        self,
        gamma: float,
        expected_vol: float,
        time_step_hours: float,
        underlying_price: float,
    ) -> float:
        """
        Approximate gamma-related P&L over the hedging horizon.
        """
        dt = time_step_hours / (24 * 365)
        return 0.5 * gamma * (expected_vol**2) * (underlying_price**2) * dt

    def decide(
        self,
        exposures: GreekExposure,
        expected_vol: float,
        transaction_cost_per_unit: float,
        hedge_size: float,
        underlying_price: float,
        current_frequency_hours: float,
    ) -> HedgingDecision:
        expected_benefit = self.expected_gamma_pnl(
            exposures.gamma, expected_vol, current_frequency_hours, underlying_price
        )
        transaction_cost = transaction_cost_per_unit * hedge_size

        should_rebalance = expected_benefit > transaction_cost
        rationale = (
            "Benefit exceeds cost"
            if should_rebalance
            else "Cost outweighs expected benefit"
        )

        new_frequency = max(self.min_frequency_hours, current_frequency_hours)

        return HedgingDecision(
            should_rebalance=should_rebalance,
            expected_benefit=float(expected_benefit),
            transaction_cost=float(transaction_cost),
            hedging_frequency_hours=new_frequency,
            rationale=rationale,
        )

    def optimise_frequency_grid(
        self,
        exposures: GreekExposure,
        expected_vol: float,
        transaction_cost_per_unit: float,
        hedge_size: float,
        underlying_price: float,
        candidate_frequencies: Dict[str, float],
    ) -> HedgingDecision:
        """
        Evaluate multiple frequencies and select the optimal hedging cadence.
        """
        best_decision: HedgingDecision | None = None
        for label, freq in candidate_frequencies.items():
            decision = self.decide(
                exposures=exposures,
                expected_vol=expected_vol,
                transaction_cost_per_unit=transaction_cost_per_unit,
                hedge_size=hedge_size,
                underlying_price=underlying_price,
                current_frequency_hours=freq,
            )
            if (
                best_decision is None
                or decision.expected_benefit - decision.transaction_cost
                > best_decision.expected_benefit - best_decision.transaction_cost
            ):
                best_decision = decision
                best_decision.rationale += f" (candidate: {label})"

        if best_decision is None:
            raise ValueError("No candidate frequencies provided.")
        return best_decision


