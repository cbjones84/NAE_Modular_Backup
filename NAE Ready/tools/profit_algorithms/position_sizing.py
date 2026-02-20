"""
Hybrid Kelly position sizing with volatility regime scaling.

VERY_AGGRESSIVE MODE: Optimized for $6.2M target / $15.7M stretch goal

Growth Milestones:
Year 1: $9,411 | Year 5: $982,500
Year 2: $44,110 | Year 6: $2,477,897
Year 3: $152,834 | Year 7: $6,243,561 (TARGET)
Year 4: $388,657 | Year 8: $15,726,144 (STRETCH)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class KellyInput:
    expected_return: float  # per-trade expectation
    variance: float
    max_fraction: float = 0.25  # VERY_AGGRESSIVE: 25% max (was 5%)
    vix_baseline: float = 20.0
    current_vix: float = 20.0


@dataclass
class KellyResult:
    optimal_fraction: float
    capped_fraction: float
    adjusted_fraction: float
    position_size: float


class HybridKellySizer:
    """
    VERY_AGGRESSIVE Kelly sizing with milestone acceleration.
    
    Key changes from conservative mode:
    - Higher max_fraction cap (25% vs 5%)
    - Volatility opportunity: Higher VIX = opportunity, not just risk
    - Milestone-aware acceleration when behind schedule
    """

    def __init__(self, account_equity: float, aggressiveness: str = "VERY_AGGRESSIVE"):
        self.account_equity = account_equity
        self.aggressiveness = aggressiveness
        
        # Aggressiveness multipliers
        self.aggression_multipliers = {
            "CONSERVATIVE": 0.5,
            "MODERATE": 0.75,
            "MODERATE_PLUS": 1.0,
            "AGGRESSIVE": 1.25,
            "VERY_AGGRESSIVE": 1.50,
        }
        self.multiplier = self.aggression_multipliers.get(aggressiveness, 1.50)

    def compute_fraction(self, kelly_input: KellyInput) -> KellyResult:
        if kelly_input.variance <= 0:
            raise ValueError("Variance must be positive for Kelly sizing.")

        raw_fraction = kelly_input.expected_return / kelly_input.variance
        
        # VERY_AGGRESSIVE: Higher cap
        aggressive_max = kelly_input.max_fraction * self.multiplier
        capped_fraction = min(raw_fraction, aggressive_max)

        # VOLATILITY OPPORTUNITY: In aggressive mode, higher VIX = opportunity
        if self.aggressiveness in ["AGGRESSIVE", "VERY_AGGRESSIVE"]:
            # Inverse relationship: Higher VIX slightly increases position (up to 20% boost)
            vix_opportunity = min(kelly_input.current_vix / kelly_input.vix_baseline, 1.2)
            regime_scaler = vix_opportunity
        else:
            # Conservative: Higher VIX = smaller position
            regime_scaler = kelly_input.vix_baseline / max(kelly_input.current_vix, 1e-6)
        
        adjusted_fraction = max(min(capped_fraction * regime_scaler, aggressive_max), 0.0)

        position_size = adjusted_fraction * self.account_equity

        return KellyResult(
            optimal_fraction=raw_fraction,
            capped_fraction=capped_fraction,
            adjusted_fraction=adjusted_fraction,
            position_size=position_size,
        )
    
    def compute_with_acceleration(
        self, 
        kelly_input: KellyInput, 
        milestone_progress_pct: float = 100.0
    ) -> KellyResult:
        """
        Compute position size with milestone acceleration
        
        Args:
            kelly_input: Kelly input parameters
            milestone_progress_pct: Progress toward milestone (100 = on track)
            
        Returns:
            KellyResult with accelerated position sizing
        """
        base_result = self.compute_fraction(kelly_input)
        
        # Calculate acceleration based on milestone progress
        if milestone_progress_pct < 50:
            acceleration = 1.50  # TURBO: Far behind
        elif milestone_progress_pct < 80:
            acceleration = 1.30  # AGGRESSIVE: Behind
        elif milestone_progress_pct < 100:
            acceleration = 1.15  # MODERATE: Slightly behind
        else:
            acceleration = 1.0   # MAINTAIN: On track
        
        # Apply acceleration
        accelerated_position = base_result.position_size * acceleration
        accelerated_fraction = base_result.adjusted_fraction * acceleration
        
        # Cap at 50% of account equity for safety
        max_position = self.account_equity * 0.50
        final_position = min(accelerated_position, max_position)
        
        return KellyResult(
            optimal_fraction=base_result.optimal_fraction * acceleration,
            capped_fraction=base_result.capped_fraction * acceleration,
            adjusted_fraction=accelerated_fraction,
            position_size=final_position,
        )


