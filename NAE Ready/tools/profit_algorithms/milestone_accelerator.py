#!/usr/bin/env python3
"""
MILESTONE ACCELERATION ENGINE - DYNAMIC GROWTH OPTIMIZER
=========================================================
Intelligently adjusts trading parameters based on progress toward milestones.
Accelerates growth when behind, maintains momentum when ahead.

Growth Milestones (Target: $6,243,561 | Stretch: $15,726,144):
Year 0: $100 (Start)
Year 1: $9,411 (94x growth)
Year 2: $44,110 (4.7x from Y1)
Year 3: $152,834 (3.5x from Y2)
Year 4: $388,657 (2.5x from Y3)
Year 5: $982,500 (2.5x from Y4)
Year 6: $2,477,897 (2.5x from Y5)
Year 7: $6,243,561 (2.5x from Y6) - TARGET GOAL
Year 8: $15,726,144 (2.5x from Y7) - STRETCH GOAL

This module provides:
1. Dynamic risk scaling based on milestone proximity
2. Compound acceleration when behind schedule
3. Momentum maintenance when ahead of schedule
4. Adaptive position sizing based on account phase
5. Strategy prioritization based on growth requirements
"""

import datetime
import math
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum


# =============================================================================
# GROWTH MILESTONES
# =============================================================================

GROWTH_MILESTONES = {
    0: 100.0,            # Starting capital
    1: 9_411.0,          # Year 1
    2: 44_110.0,         # Year 2
    3: 152_834.0,        # Year 3
    4: 388_657.0,        # Year 4
    5: 982_500.0,        # Year 5
    6: 2_477_897.0,      # Year 6
    7: 6_243_561.0,      # Year 7 - TARGET
    8: 15_726_144.0,     # Year 8 - STRETCH
}

# Monthly milestones (interpolated)
MONTHLY_MILESTONES = {}
for year in range(8):
    start = GROWTH_MILESTONES[year]
    end = GROWTH_MILESTONES[year + 1]
    for month in range(12):
        # Exponential interpolation
        progress = month / 12
        MONTHLY_MILESTONES[year * 12 + month] = start * ((end / start) ** progress)


class AccountPhase(Enum):
    """Account growth phases with dynamic risk parameters"""
    MICRO = "micro"      # $100 - $1,000
    SMALL = "small"      # $1,000 - $10,000
    GROWTH = "growth"    # $10,000 - $100,000
    SCALE = "scale"      # $100,000 - $1,000,000
    WEALTHY = "wealthy"  # $1,000,000+


@dataclass
class AccelerationProfile:
    """Trading acceleration parameters"""
    risk_per_trade: float           # % of NAV to risk per trade
    kelly_fraction: float           # Kelly multiplier (0.5 = half Kelly)
    max_position_pct: float         # Max % of NAV per position
    daily_trades_target: int        # Target trades per day
    min_confidence: float           # Minimum confidence to trade
    momentum_boost: float           # Extra boost for momentum trades
    compound_multiplier: float      # Compound growth accelerator
    strategy_priority: List[str]    # Prioritized strategy types


@dataclass
class MilestoneStatus:
    """Current milestone progress status"""
    current_nav: float
    current_month: int
    current_year: float
    target_at_month: float
    progress_pct: float             # 100% = exactly on track
    months_ahead: float             # Positive = ahead, Negative = behind
    acceleration_level: str         # TURBO, AGGRESSIVE, MODERATE, MAINTAIN
    next_milestone: float
    days_to_next_milestone: int
    required_daily_return: float    # To get back on track


class MilestoneAccelerator:
    """
    Dynamic growth optimization engine that adapts trading parameters
    based on progress toward $15.7M stretch goal
    """
    
    def __init__(self, start_date: Optional[datetime.date] = None):
        """
        Initialize the accelerator
        
        Args:
            start_date: Date when NAE started trading (defaults to Jan 6, 2026)
        """
        self.start_date = start_date or datetime.date(2026, 1, 6)
        self.target_goal = 6_243_561.0
        self.stretch_goal = 15_726_144.0
        
    def get_current_month(self) -> int:
        """Calculate current month number from start date"""
        today = datetime.date.today()
        months = (today.year - self.start_date.year) * 12 + \
                 (today.month - self.start_date.month)
        return max(0, months)
    
    def get_milestone_for_month(self, month: int) -> float:
        """Get expected NAV for a given month"""
        if month in MONTHLY_MILESTONES:
            return MONTHLY_MILESTONES[month]
        elif month <= 0:
            return GROWTH_MILESTONES[0]
        else:
            return GROWTH_MILESTONES[8]  # Cap at stretch goal
    
    def analyze_progress(self, current_nav: float) -> MilestoneStatus:
        """
        Analyze current progress against milestones
        
        Args:
            current_nav: Current Net Asset Value
            
        Returns:
            MilestoneStatus with detailed progress analysis
        """
        current_month = self.get_current_month()
        current_year = current_month / 12
        target_at_month = self.get_milestone_for_month(current_month)
        
        # Calculate progress percentage (100% = on track)
        progress_pct = (current_nav / target_at_month) * 100 if target_at_month > 0 else 100
        
        # Calculate months ahead/behind
        # Find which month's milestone matches our current NAV
        months_ahead = 0
        for m in range(96):  # 8 years
            if self.get_milestone_for_month(m) >= current_nav:
                months_ahead = current_month - m
                break
        
        # Determine acceleration level
        if progress_pct < 50:
            acceleration_level = "TURBO"      # Far behind - maximum aggression
        elif progress_pct < 80:
            acceleration_level = "AGGRESSIVE"  # Behind - high aggression
        elif progress_pct < 100:
            acceleration_level = "MODERATE"    # Slightly behind - moderate boost
        else:
            acceleration_level = "MAINTAIN"    # On track or ahead - maintain momentum
        
        # Calculate next milestone
        next_year = min(int(current_year) + 1, 8)
        next_milestone = GROWTH_MILESTONES.get(next_year, self.stretch_goal)
        
        # Days to next milestone (end of year)
        today = datetime.date.today()
        next_year_date = datetime.date(self.start_date.year + next_year, 
                                        self.start_date.month, 
                                        self.start_date.day)
        days_to_next = (next_year_date - today).days
        
        # Required daily return to get back on track
        if current_nav >= target_at_month:
            required_daily_return = 0.0
        else:
            gap = target_at_month / current_nav
            days_in_month = 30
            required_daily_return = (gap ** (1 / days_in_month) - 1) * 100
        
        return MilestoneStatus(
            current_nav=current_nav,
            current_month=current_month,
            current_year=current_year,
            target_at_month=target_at_month,
            progress_pct=progress_pct,
            months_ahead=months_ahead,
            acceleration_level=acceleration_level,
            next_milestone=next_milestone,
            days_to_next_milestone=days_to_next,
            required_daily_return=required_daily_return
        )
    
    def get_acceleration_profile(self, current_nav: float) -> AccelerationProfile:
        """
        Get optimal trading parameters based on current progress
        
        Args:
            current_nav: Current Net Asset Value
            
        Returns:
            AccelerationProfile with tuned parameters
        """
        status = self.analyze_progress(current_nav)
        phase = self._determine_phase(current_nav)
        
        # Base parameters by phase
        base_params = self._get_base_params(phase)
        
        # Apply acceleration multipliers based on progress
        multiplier = self._get_acceleration_multiplier(status)
        
        return AccelerationProfile(
            risk_per_trade=min(base_params['risk'] * multiplier, 0.50),
            kelly_fraction=min(base_params['kelly'] * multiplier, 0.75),
            max_position_pct=min(base_params['max_pos'] * multiplier, 0.50),
            daily_trades_target=int(base_params['trades'] * multiplier),
            min_confidence=max(base_params['min_conf'] / multiplier, 0.30),
            momentum_boost=base_params['momentum'] * multiplier,
            compound_multiplier=multiplier,
            strategy_priority=self._get_priority_strategies(status, phase)
        )
    
    def _determine_phase(self, nav: float) -> AccountPhase:
        """Determine current account phase"""
        if nav < 1_000:
            return AccountPhase.MICRO
        elif nav < 10_000:
            return AccountPhase.SMALL
        elif nav < 100_000:
            return AccountPhase.GROWTH
        elif nav < 1_000_000:
            return AccountPhase.SCALE
        else:
            return AccountPhase.WEALTHY
    
    def _get_base_params(self, phase: AccountPhase) -> Dict[str, float]:
        """Get base trading parameters by phase"""
        params = {
            AccountPhase.MICRO: {
                'risk': 0.20,      # 20% risk (very aggressive for small account)
                'kelly': 0.60,     # 60% Kelly
                'max_pos': 0.40,   # 40% max position
                'trades': 5,       # 5 trades/day
                'min_conf': 0.50,  # Lower confidence threshold
                'momentum': 1.5,   # High momentum boost
            },
            AccountPhase.SMALL: {
                'risk': 0.15,      # 15% risk
                'kelly': 0.55,     # 55% Kelly
                'max_pos': 0.35,   # 35% max position
                'trades': 6,       # 6 trades/day
                'min_conf': 0.55,
                'momentum': 1.4,
            },
            AccountPhase.GROWTH: {
                'risk': 0.10,      # 10% risk
                'kelly': 0.50,     # 50% Kelly
                'max_pos': 0.30,   # 30% max position
                'trades': 8,       # 8 trades/day
                'min_conf': 0.55,
                'momentum': 1.3,
            },
            AccountPhase.SCALE: {
                'risk': 0.07,      # 7% risk
                'kelly': 0.45,     # 45% Kelly
                'max_pos': 0.25,   # 25% max position
                'trades': 10,      # 10 trades/day
                'min_conf': 0.60,
                'momentum': 1.2,
            },
            AccountPhase.WEALTHY: {
                'risk': 0.05,      # 5% risk
                'kelly': 0.40,     # 40% Kelly
                'max_pos': 0.20,   # 20% max position
                'trades': 12,      # 12 trades/day
                'min_conf': 0.60,
                'momentum': 1.1,
            },
        }
        return params.get(phase, params[AccountPhase.GROWTH])
    
    def _get_acceleration_multiplier(self, status: MilestoneStatus) -> float:
        """Calculate acceleration multiplier based on progress"""
        if status.acceleration_level == "TURBO":
            # Far behind - maximum boost
            return 1.5
        elif status.acceleration_level == "AGGRESSIVE":
            # Behind - high boost
            return 1.3
        elif status.acceleration_level == "MODERATE":
            # Slightly behind - moderate boost
            return 1.15
        else:  # MAINTAIN
            # On track - sustain
            return 1.0
    
    def _get_priority_strategies(self, status: MilestoneStatus, phase: AccountPhase) -> List[str]:
        """Get prioritized strategy list based on current status"""
        # High-growth strategies for catching up
        high_growth = ["0DTE_options", "momentum_breakout", "earnings_volatility", "leveraged_etf"]
        
        # Steady growth strategies for maintaining
        steady = ["swing_trading", "momentum_fractional", "trend_following", "covered_calls"]
        
        if status.acceleration_level in ["TURBO", "AGGRESSIVE"]:
            # Prioritize high-growth strategies when behind
            if phase in [AccountPhase.MICRO, AccountPhase.SMALL]:
                return ["momentum_fractional", "high_growth_stocks", "momentum_breakout"]
            else:
                return high_growth + steady[:2]
        else:
            # Mix of strategies when on track
            return steady + high_growth[:2]
    
    def calculate_optimal_position_size(
        self, 
        current_nav: float, 
        entry_price: float,
        confidence: float,
        volatility: float = 0.20
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate optimal position size with acceleration factors
        
        Args:
            current_nav: Current account value
            entry_price: Entry price per share
            confidence: Strategy confidence (0.0 to 1.0)
            volatility: Expected volatility
            
        Returns:
            Tuple of (position_size_dollars, details_dict)
        """
        profile = self.get_acceleration_profile(current_nav)
        
        # Base position size
        base_size = current_nav * profile.risk_per_trade
        
        # Adjust for confidence
        confidence_factor = 0.5 + (confidence * 0.5)  # Maps 0-1 to 0.5-1.0
        
        # Adjust for volatility (higher vol = slightly smaller position)
        vol_factor = 1.0 / (1.0 + volatility)
        
        # Apply compound multiplier for catch-up
        accelerated_size = base_size * confidence_factor * vol_factor * profile.compound_multiplier
        
        # Cap at maximum position percentage
        max_size = current_nav * profile.max_position_pct
        final_size = min(accelerated_size, max_size)
        
        # Calculate shares
        shares = int(final_size / entry_price) if entry_price > 0 else 0
        
        details = {
            "base_size": base_size,
            "confidence_factor": confidence_factor,
            "volatility_factor": vol_factor,
            "compound_multiplier": profile.compound_multiplier,
            "final_size": final_size,
            "shares": shares,
            "position_pct": (final_size / current_nav) * 100 if current_nav > 0 else 0,
            "acceleration_level": self.analyze_progress(current_nav).acceleration_level
        }
        
        return final_size, details
    
    def should_trade(self, current_nav: float, opportunity_confidence: float) -> Tuple[bool, str]:
        """
        Determine if a trade should be taken based on acceleration needs
        
        Args:
            current_nav: Current account value
            opportunity_confidence: Confidence in the opportunity (0.0 to 1.0)
            
        Returns:
            Tuple of (should_trade, reason)
        """
        profile = self.get_acceleration_profile(current_nav)
        status = self.analyze_progress(current_nav)
        
        # Lower confidence threshold when behind schedule
        effective_min_conf = profile.min_confidence
        
        if opportunity_confidence >= effective_min_conf:
            return True, f"Confidence {opportunity_confidence:.2%} >= {effective_min_conf:.2%} threshold ({status.acceleration_level})"
        
        # Even lower bar for TURBO mode
        if status.acceleration_level == "TURBO" and opportunity_confidence >= 0.40:
            return True, f"TURBO mode override: confidence {opportunity_confidence:.2%} acceptable when far behind"
        
        return False, f"Confidence {opportunity_confidence:.2%} < {effective_min_conf:.2%} minimum"
    
    def get_growth_report(self, current_nav: float) -> str:
        """Generate a detailed growth report"""
        status = self.analyze_progress(current_nav)
        profile = self.get_acceleration_profile(current_nav)
        
        report = f"""
╔═══════════════════════════════════════════════════════════════════════════╗
║                    NAE MILESTONE ACCELERATION REPORT                       ║
╠═══════════════════════════════════════════════════════════════════════════╣
║ Current NAV:          ${status.current_nav:>14,.2f}                        ║
║ Target at Month {status.current_month:2}:   ${status.target_at_month:>14,.2f}                        ║
║ Progress:             {status.progress_pct:>14.1f}%                        ║
║ Months Ahead/Behind:  {status.months_ahead:>+14.1f}                        ║
╠═══════════════════════════════════════════════════════════════════════════╣
║ ACCELERATION LEVEL:   {status.acceleration_level:>14}                        ║
║ Required Daily Return: {status.required_daily_return:>13.2f}%                        ║
╠═══════════════════════════════════════════════════════════════════════════╣
║ TRADING PARAMETERS                                                         ║
║ Risk per Trade:       {profile.risk_per_trade:>14.1%}                        ║
║ Kelly Fraction:       {profile.kelly_fraction:>14.1%}                        ║
║ Max Position:         {profile.max_position_pct:>14.1%}                        ║
║ Min Confidence:       {profile.min_confidence:>14.1%}                        ║
║ Daily Trades Target:  {profile.daily_trades_target:>14}                        ║
╠═══════════════════════════════════════════════════════════════════════════╣
║ NEXT MILESTONE                                                             ║
║ Target:               ${status.next_milestone:>14,.2f}                        ║
║ Days Remaining:       {status.days_to_next_milestone:>14}                        ║
╠═══════════════════════════════════════════════════════════════════════════╣
║ GROWTH TRAJECTORY                                                          ║
║ Year 1: $9,411      │ Year 5: $982,500                                     ║
║ Year 2: $44,110     │ Year 6: $2,477,897                                   ║
║ Year 3: $152,834    │ Year 7: $6,243,561 ← TARGET                          ║
║ Year 4: $388,657    │ Year 8: $15,726,144 ← STRETCH                        ║
╚═══════════════════════════════════════════════════════════════════════════╝
"""
        return report


# Global accelerator instance
_accelerator: Optional[MilestoneAccelerator] = None

def get_accelerator() -> MilestoneAccelerator:
    """Get global accelerator instance"""
    global _accelerator
    if _accelerator is None:
        _accelerator = MilestoneAccelerator()
    return _accelerator


def calculate_accelerated_position(
    nav: float, 
    entry_price: float, 
    confidence: float
) -> Tuple[float, Dict[str, Any]]:
    """
    Convenience function for accelerated position sizing
    
    Args:
        nav: Current Net Asset Value
        entry_price: Entry price per share
        confidence: Strategy confidence (0.0 to 1.0)
        
    Returns:
        Tuple of (position_size_dollars, details)
    """
    return get_accelerator().calculate_optimal_position_size(
        nav, entry_price, confidence
    )


if __name__ == "__main__":
    # Demo the accelerator
    accelerator = MilestoneAccelerator()
    
    # Test with different NAV levels
    test_navs = [100, 500, 5000, 50000, 500000, 2000000]
    
    for nav in test_navs:
        print(accelerator.get_growth_report(nav))
        print("\n" + "="*80 + "\n")

