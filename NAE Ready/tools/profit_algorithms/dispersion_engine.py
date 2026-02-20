"""
Dispersion and correlation arbitrage utilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd


def implied_correlation(
    index_variance: float,
    constituent_vols: Iterable[float],
    weights: Iterable[float],
) -> float:
    """
    Compute implied correlation given index variance and constituent vols.
    """
    vols = np.asarray(list(constituent_vols), dtype=float)
    weights_arr = np.asarray(list(weights), dtype=float)

    if vols.shape != weights_arr.shape:
        raise ValueError("vols and weights must be same length.")

    weights_arr = weights_arr / weights_arr.sum()
    weighted_var = np.sum((weights_arr**2) * (vols**2))
    numerator = index_variance - weighted_var
    denom = 2 * np.sum(
        weights_arr[:, None]
        * weights_arr[None, :]
        * vols[:, None]
        * vols[None, :]
    )
    # Only consider off-diagonal elements
    denom = denom.sum() - np.sum((weights_arr**2) * (vols**2))
    if denom <= 0:
        return np.nan
    return float(numerator / denom)


@dataclass
class DispersionSignal:
    """
    Captures the outputs of the dispersion engine.
    """

    implied_corr: float
    expected_corr: float
    spread: float
    direction: str
    recommended_trade: str
    metadata: Dict[str, float]


class DispersionEngine:
    """
    Generates dispersion trade signals using implied and realised correlation.
    """

    def __init__(self, threshold_bps: float = 50.0):
        self.threshold_bps = threshold_bps

    def estimate_realised_correlation(
        self, constituent_returns: pd.DataFrame, weights: pd.Series
    ) -> float:
        """
        Estimate realised correlation using weighted constituents.
        """
        aligned_returns = constituent_returns.dropna()
        weights = weights.reindex(aligned_returns.columns).fillna(0.0)
        weights = weights / weights.sum()

        weighted_cov = aligned_returns.cov().values
        corr_matrix = aligned_returns.corr().values
        # Weighted average correlation, ignoring diagonal
        mask = ~np.eye(len(weights), dtype=bool)
        weighted_corr = (corr_matrix * mask).sum() / mask.sum()
        return float(weighted_corr)

    def generate_signal(
        self,
        index_iv: float,
        constituent_vols: pd.Series,
        weights: pd.Series,
        realised_corr: float,
    ) -> DispersionSignal:
        index_variance = index_iv**2
        implied_corr_value = implied_correlation(
            index_variance, constituent_vols.values, weights.values
        )
        spread = (implied_corr_value - realised_corr) * 10_000  # basis points

        metadata = {
            "implied_corr": implied_corr_value,
            "realised_corr": realised_corr,
            "spread_bps": spread,
        }

        if np.isnan(implied_corr_value):
            return DispersionSignal(
                implied_corr=np.nan,
                expected_corr=realised_corr,
                spread=np.nan,
                direction="neutral",
                recommended_trade="insufficient_data",
                metadata=metadata,
            )

        if spread > self.threshold_bps:
            direction = "long_dispersion"
            trade = "Buy weighted constituents IV, sell index IV"
        elif spread < -self.threshold_bps:
            direction = "short_dispersion"
            trade = "Sell constituent IV, buy index IV"
        else:
            direction = "neutral"
            trade = "No action"

        return DispersionSignal(
            implied_corr=implied_corr_value,
            expected_corr=realised_corr,
            spread=spread,
            direction=direction,
            recommended_trade=trade,
            metadata=metadata,
        )


