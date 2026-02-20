#!/usr/bin/env python3
"""
OPTIMUS ACCELERATOR MODULE - AGGRESSIVE $5M PATH
=================================================
Maximum growth strategies designed for small accounts ($100 start)
with bi-weekly $100 deposits, targeting $5M in 8 years.

AGGRESSIVE RETURN TARGETS:
- MICRO Phase ($100-$1,000):  30% monthly = 2,230% annual
- SMALL Phase ($1,000-$10,000): 20% monthly = 791% annual  
- GROWTH Phase ($10,000-$100,000): 12% monthly = 290% annual
- SCALE Phase ($100,000+): 8% monthly = 152% annual

Path to $5M:
- Starting: $100
- Bi-weekly deposits: $100 ($2,600/year)
- Total deposits over 8 years: $20,900
- Target: $5,000,000+ through aggressive compounding

Strategy Phases:
- Phase 1 (Micro): $100 - $1,000 → Fractional momentum, high-growth stocks
- Phase 2 (Small): $1,000 - $10,000 → 0DTE options, leveraged ETFs, swing trading
- Phase 3 (Growth): $10,000 - $100,000 → Full strategy suite, earnings plays
- Phase 4 (Scale): $100,000+ → Diversified aggressive growth

Key Accelerator Strategies (6 Total):
1. Momentum Fractional Trading (All phases) - Quick momentum plays
2. Breakout Swing Trading ($500+) - Multi-day breakouts
3. Leveraged ETF Trading ($1,000+) - TQQQ, SQQQ, SOXL
4. High Growth Stocks (All phases) - NVDA, TSLA, COIN, etc.
5. 0DTE Options ($2,500+) - Same-day expiration trades [NEW]
6. Earnings Volatility Plays ($1,000+) - IV crush opportunities [NEW]

WARNING: This is an AGGRESSIVE growth strategy with HIGH RISK.
Only use capital you can afford to lose.
"""

import os
import sys
import json
import math
import datetime
import requests
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

# Setup paths
SCRIPT_DIR = Path(__file__).parent
NAE_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(NAE_DIR))

# Environment
os.environ.setdefault("TRADIER_SANDBOX", "false")
os.environ.setdefault("TRADIER_API_KEY", "27Ymk28vtbgqY1LFYxhzaEmIuwJb")
os.environ.setdefault("TRADIER_ACCOUNT_ID", "6YB66744")

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class AcceleratorConfig:
    """
    VERY_AGGRESSIVE configuration for $15.7M stretch goal in 8 years
    
    Growth Milestones:
    Year 1: $9,411 | Year 5: $982,500
    Year 2: $44,110 | Year 6: $2,477,897
    Year 3: $152,834 | Year 7: $6,243,561 (TARGET)
    Year 4: $388,657 | Year 8: $15,726,144 (STRETCH)
    """
    starting_capital: float = 100.0
    biweekly_deposit: float = 100.0
    target_goal: float = 6_243_561.0      # Year 7 target (was $5M)
    stretch_goal: float = 15_726_144.0    # Year 8 stretch goal
    target_years: int = 8
    
    # Risk parameters by phase - VERY AGGRESSIVE
    micro_risk_pct: float = 0.20      # 20% risk per trade (ULTRA aggressive for micro)
    small_risk_pct: float = 0.15      # 15% risk per trade
    growth_risk_pct: float = 0.10     # 10% risk per trade
    scale_risk_pct: float = 0.07      # 7% risk per trade
    wealthy_risk_pct: float = 0.05    # 5% risk for $1M+
    
    # Phase thresholds (aligned with MilestoneAccelerator)
    micro_max: float = 1_000.0
    small_max: float = 10_000.0
    growth_max: float = 100_000.0
    scale_max: float = 1_000_000.0
    
    # Target returns by phase (monthly) - ULTRA AGGRESSIVE for $15.7M
    micro_target_monthly: float = 0.40   # 40% monthly (was 30%)
    small_target_monthly: float = 0.30   # 30% monthly (was 20%)
    growth_target_monthly: float = 0.20  # 20% monthly (was 12%)
    scale_target_monthly: float = 0.12   # 12% monthly (was 8%)

class AccountPhase(Enum):
    """Account growth phases"""
    MICRO = "micro"      # $100 - $1,000
    SMALL = "small"      # $1,000 - $10,000
    GROWTH = "growth"    # $10,000 - $100,000
    SCALE = "scale"      # $100,000+

# =============================================================================
# GROWTH PROJECTION
# =============================================================================

@dataclass
class GrowthProjection:
    """Projected growth milestones"""
    month: int
    year: float
    balance: float
    phase: str
    total_deposited: float
    total_returns: float
    monthly_return_pct: float
    on_track: bool
    target_at_month: float

def calculate_growth_projection(config: AcceleratorConfig) -> List[GrowthProjection]:
    """
    Calculate projected growth path with bi-weekly deposits and phase-adjusted returns
    """
    projections = []
    balance = config.starting_capital
    total_deposited = config.starting_capital
    total_months = config.target_years * 12
    
    # Calculate required trajectory
    # We need exponential growth that reaches $5M
    # With deposits, we use a modified growth model
    
    for month in range(1, total_months + 1):
        year = month / 12
        
        # Add bi-weekly deposits (2 per month on average)
        monthly_deposits = config.biweekly_deposit * 2
        
        # Determine phase
        if balance < config.micro_max:
            phase = AccountPhase.MICRO
            target_return = config.micro_target_monthly
            risk_pct = config.micro_risk_pct
        elif balance < config.small_max:
            phase = AccountPhase.SMALL
            target_return = config.small_target_monthly
            risk_pct = config.small_risk_pct
        elif balance < config.growth_max:
            phase = AccountPhase.GROWTH
            target_return = config.growth_target_monthly
            risk_pct = config.growth_risk_pct
        else:
            phase = AccountPhase.SCALE
            target_return = config.scale_target_monthly
            risk_pct = config.scale_risk_pct
        
        # Calculate returns for the month
        monthly_return = balance * target_return
        
        # Update balance
        old_balance = balance
        balance = balance + monthly_return + monthly_deposits
        total_deposited += monthly_deposits
        
        # Calculate target trajectory (linear interpolation for comparison)
        target_at_month = config.starting_capital + (
            (config.target_goal - config.starting_capital) * (month / total_months)
        )
        
        projections.append(GrowthProjection(
            month=month,
            year=round(year, 2),
            balance=round(balance, 2),
            phase=phase.value,
            total_deposited=round(total_deposited, 2),
            total_returns=round(balance - total_deposited, 2),
            monthly_return_pct=round(target_return * 100, 1),
            on_track=balance >= target_at_month * 0.8,  # Within 80% of target
            target_at_month=round(target_at_month, 2)
        ))
    
    return projections

def print_growth_milestones(projections: List[GrowthProjection]):
    """Print key growth milestones"""
    print("\n" + "=" * 70)
    print("  GROWTH PROJECTION: $100 -> $5,000,000 (8 Years)")
    print("  Bi-weekly deposits: $100")
    print("=" * 70)
    
    milestones = [1, 3, 6, 12, 24, 36, 48, 60, 72, 84, 96]
    
    print(f"\n{'Month':<8}{'Year':<8}{'Balance':<15}{'Phase':<10}{'Deposited':<12}{'Returns':<12}{'Target':<12}")
    print("-" * 77)
    
    for p in projections:
        if p.month in milestones:
            status = "[OK]" if p.on_track else "[!!]"
            print(f"{p.month:<8}{p.year:<8}${p.balance:>12,.2f}  {p.phase:<10}${p.total_deposited:>9,.0f}  ${p.total_returns:>9,.0f}  {status}")
    
    final = projections[-1]
    print("-" * 77)
    print(f"\nFINAL: ${final.balance:,.2f}")
    print(f"Total Deposited: ${final.total_deposited:,.2f}")
    print(f"Total Returns: ${final.total_returns:,.2f}")
    print(f"Return Multiple: {final.balance / final.total_deposited:.1f}x")
    
    if final.balance >= 5_000_000:
        print("\n[TARGET] TARGET ACHIEVED!")
    else:
        shortfall = 5_000_000 - final.balance
        print(f"\n[!!] Shortfall: ${shortfall:,.2f}")
        print(f"   Need {(5_000_000/final.balance - 1)*100:.1f}% more aggressive returns")

# =============================================================================
# ACCELERATOR STRATEGIES
# =============================================================================

class AcceleratorStrategy:
    """Base class for accelerator strategies"""
    
    def __init__(self, name: str, min_capital: float, risk_level: str):
        self.name = name
        self.min_capital = min_capital
        self.risk_level = risk_level
        self.enabled = True
        
    def analyze(self, symbol: str, data: Dict) -> Dict:
        """Analyze opportunity - override in subclass"""
        raise NotImplementedError
        
    def get_position_size(self, capital: float, risk_pct: float, price: float) -> int:
        """Calculate position size"""
        risk_amount = capital * risk_pct
        return max(1, int(risk_amount / price))


class MomentumFractionalStrategy(AcceleratorStrategy):
    """
    Momentum trading with fractional shares
    Best for: Micro accounts ($100-$1000)
    Target: 5-20% gains per trade
    """
    
    def __init__(self):
        super().__init__(
            name="Momentum Fractional",
            min_capital=50.0,
            risk_level="HIGH"
        )
        self.lookback_days = 5
        self.min_momentum = 0.03  # 3% move
        
    def analyze(self, symbol: str, quote: Dict, history: List[Dict] = None) -> Dict:
        """Analyze momentum opportunity"""
        result = {
            "strategy": self.name,
            "symbol": symbol,
            "signal": "NONE",
            "score": 0,
            "reasons": []
        }
        
        if not quote:
            return result
            
        price = float(quote.get('last', 0))
        change_pct = float(quote.get('change_percentage', 0)) / 100
        volume = int(quote.get('volume', 0))
        avg_volume = int(quote.get('average_volume', 1))
        
        # Momentum score
        score = 0
        
        # Strong daily move
        if change_pct > 0.05:  # Up 5%+
            score += 30
            result["reasons"].append(f"Strong momentum: +{change_pct:.1%}")
        elif change_pct > 0.03:  # Up 3%+
            score += 20
            result["reasons"].append(f"Good momentum: +{change_pct:.1%}")
        elif change_pct < -0.05:  # Down 5%+ (potential bounce)
            score += 15
            result["reasons"].append(f"Oversold bounce: {change_pct:.1%}")
        
        # Volume confirmation
        vol_ratio = volume / avg_volume if avg_volume > 0 else 0
        if vol_ratio > 2.0:
            score += 25
            result["reasons"].append(f"High volume: {vol_ratio:.1f}x avg")
        elif vol_ratio > 1.5:
            score += 15
            result["reasons"].append(f"Above avg volume: {vol_ratio:.1f}x")
        
        # Price accessibility (for small accounts)
        if price < 20:
            score += 10
            result["reasons"].append("Accessible price point")
        elif price < 50:
            score += 5
        
        result["score"] = score
        result["price"] = price
        result["change_pct"] = change_pct
        result["volume_ratio"] = vol_ratio
        
        if score >= 50:
            result["signal"] = "STRONG_BUY"
        elif score >= 35:
            result["signal"] = "BUY"
        elif score >= 20:
            result["signal"] = "WATCH"
        
        return result


class BreakoutSwingStrategy(AcceleratorStrategy):
    """
    Breakout swing trading
    Best for: Small-Growth accounts ($1000+)
    Target: 10-30% gains per trade
    Hold time: 2-10 days
    """
    
    def __init__(self):
        super().__init__(
            name="Breakout Swing",
            min_capital=500.0,
            risk_level="MEDIUM-HIGH"
        )
        
    def analyze(self, symbol: str, quote: Dict, history: List[Dict] = None) -> Dict:
        """Analyze breakout opportunity"""
        result = {
            "strategy": self.name,
            "symbol": symbol,
            "signal": "NONE",
            "score": 0,
            "reasons": []
        }
        
        if not quote:
            return result
            
        price = float(quote.get('last', 0))
        high = float(quote.get('high', 0))
        low = float(quote.get('low', 0))
        prev_close = float(quote.get('prevclose', price))
        change_pct = (price - prev_close) / prev_close if prev_close > 0 else 0
        volume = int(quote.get('volume', 0))
        avg_volume = int(quote.get('average_volume', 1))
        
        score = 0
        
        # Gap up (potential breakout)
        if low > prev_close * 1.02:  # Gapped up 2%+
            score += 25
            result["reasons"].append(f"Gap up: {((low/prev_close)-1)*100:.1f}%")
        
        # Breaking to new highs (intraday)
        if price == high and change_pct > 0.02:
            score += 20
            result["reasons"].append("At day high with momentum")
        
        # Volume surge
        vol_ratio = volume / avg_volume if avg_volume > 0 else 0
        if vol_ratio > 3.0:
            score += 30
            result["reasons"].append(f"Volume surge: {vol_ratio:.1f}x")
        elif vol_ratio > 2.0:
            score += 20
            result["reasons"].append(f"High volume: {vol_ratio:.1f}x")
        
        # Strong move
        if change_pct > 0.07:
            score += 20
            result["reasons"].append(f"Strong move: +{change_pct:.1%}")
        elif change_pct > 0.04:
            score += 10
            result["reasons"].append(f"Good move: +{change_pct:.1%}")
        
        result["score"] = score
        result["price"] = price
        result["change_pct"] = change_pct
        
        if score >= 60:
            result["signal"] = "STRONG_BUY"
        elif score >= 40:
            result["signal"] = "BUY"
        elif score >= 25:
            result["signal"] = "WATCH"
        
        return result


class LeveragedETFStrategy(AcceleratorStrategy):
    """
    Leveraged ETF trading (TQQQ, SQQQ, UPRO, SPXU, etc.)
    Best for: Small+ accounts ($1000+)
    Target: Quick 5-15% moves
    WARNING: High risk, for aggressive growth only
    """
    
    LEVERAGED_ETFS = {
        "TQQQ": {"underlying": "QQQ", "leverage": 3, "direction": "BULL"},
        "SQQQ": {"underlying": "QQQ", "leverage": 3, "direction": "BEAR"},
        "UPRO": {"underlying": "SPY", "leverage": 3, "direction": "BULL"},
        "SPXU": {"underlying": "SPY", "leverage": 3, "direction": "BEAR"},
        "LABU": {"underlying": "XBI", "leverage": 3, "direction": "BULL"},
        "LABD": {"underlying": "XBI", "leverage": 3, "direction": "BEAR"},
        "SOXL": {"underlying": "SOXX", "leverage": 3, "direction": "BULL"},
        "SOXS": {"underlying": "SOXX", "leverage": 3, "direction": "BEAR"},
        "FNGU": {"underlying": "FANG+", "leverage": 3, "direction": "BULL"},
    }
    
    def __init__(self):
        super().__init__(
            name="Leveraged ETF",
            min_capital=1000.0,
            risk_level="VERY HIGH"
        )
        
    def analyze(self, symbol: str, quote: Dict, market_direction: str = "NEUTRAL") -> Dict:
        """Analyze leveraged ETF opportunity"""
        result = {
            "strategy": self.name,
            "symbol": symbol,
            "signal": "NONE",
            "score": 0,
            "reasons": []
        }
        
        if symbol not in self.LEVERAGED_ETFS:
            return result
            
        etf_info = self.LEVERAGED_ETFS[symbol]
        
        if not quote:
            return result
            
        price = float(quote.get('last', 0))
        change_pct = float(quote.get('change_percentage', 0)) / 100
        
        score = 0
        
        # Direction alignment
        if market_direction == "BULLISH" and etf_info["direction"] == "BULL":
            score += 30
            result["reasons"].append(f"Bullish market + 3x Bull ETF")
        elif market_direction == "BEARISH" and etf_info["direction"] == "BEAR":
            score += 30
            result["reasons"].append(f"Bearish market + 3x Bear ETF")
        
        # Momentum in right direction
        if etf_info["direction"] == "BULL" and change_pct > 0.03:
            score += 20
            result["reasons"].append(f"Bull momentum: +{change_pct:.1%}")
        elif etf_info["direction"] == "BEAR" and change_pct > 0.03:
            score += 20
            result["reasons"].append(f"Bear momentum: +{change_pct:.1%}")
        
        # Oversold bounce potential (contrarian)
        if etf_info["direction"] == "BULL" and change_pct < -0.08:
            score += 25
            result["reasons"].append(f"Oversold bounce potential: {change_pct:.1%}")
        
        result["score"] = score
        result["price"] = price
        result["leverage"] = etf_info["leverage"]
        
        if score >= 45:
            result["signal"] = "BUY"
        elif score >= 30:
            result["signal"] = "WATCH"
        
        return result


class ZeroDTEOptionsStrategy(AcceleratorStrategy):
    """
    0DTE (Zero Days to Expiration) Options Strategy
    EXTREMELY HIGH RISK - HIGH REWARD
    Best for: Small+ accounts ($2,500+)
    Target: 50-500% gains per trade
    
    WARNING: Can lose 100% of position. Only use small allocation.
    """
    
    # Symbols with good 0DTE liquidity
    ZERODTDE_SYMBOLS = ["SPY", "QQQ", "IWM", "AAPL", "TSLA", "NVDA", "AMD"]
    
    def __init__(self):
        super().__init__(
            name="0DTE Options",
            min_capital=2500.0,  # Need $2,500+ for options
            risk_level="EXTREME"
        )
        self.max_allocation_pct = 0.05  # Max 5% of account per 0DTE trade
        
    def analyze(self, symbol: str, quote: Dict, market_sentiment: str = "NEUTRAL") -> Dict:
        """
        Analyze 0DTE opportunity
        
        Strategy Logic:
        1. Look for strong directional moves (>1% pre-market or early)
        2. Buy slightly OTM calls/puts in direction of move
        3. Target 50-100% gain, stop at 50% loss
        4. Close by 2PM ET (before decay accelerates)
        """
        result = {
            "strategy": self.name,
            "symbol": symbol,
            "signal": "NONE",
            "score": 0,
            "option_type": None,
            "reasons": []
        }
        
        if symbol not in self.ZERODTDE_SYMBOLS:
            return result
            
        if not quote:
            return result
            
        price = float(quote.get('last', 0))
        change_pct = float(quote.get('change_percentage', 0)) / 100
        volume = int(quote.get('volume', 0))
        avg_volume = int(quote.get('average_volume', 1))
        high = float(quote.get('high', 0))
        low = float(quote.get('low', 0))
        
        score = 0
        option_type = None
        
        # Strong directional move (required for 0DTE)
        if change_pct > 0.015:  # Up 1.5%+
            score += 35
            option_type = "CALL"
            result["reasons"].append(f"Strong bullish move: +{change_pct:.2%}")
        elif change_pct < -0.015:  # Down 1.5%+
            score += 35
            option_type = "PUT"
            result["reasons"].append(f"Strong bearish move: {change_pct:.2%}")
        elif abs(change_pct) > 0.008:  # Moderate move 0.8%+
            score += 20
            option_type = "CALL" if change_pct > 0 else "PUT"
            result["reasons"].append(f"Moderate directional move: {change_pct:+.2%}")
        
        # Volume confirmation (critical for 0DTE)
        vol_ratio = volume / avg_volume if avg_volume > 0 else 0
        if vol_ratio > 2.0:
            score += 25
            result["reasons"].append(f"High volume confirmation: {vol_ratio:.1f}x")
        elif vol_ratio > 1.5:
            score += 15
            result["reasons"].append(f"Above avg volume: {vol_ratio:.1f}x")
        
        # Price near day's extreme (momentum)
        if high > low:
            price_position = (price - low) / (high - low)
            if option_type == "CALL" and price_position > 0.8:
                score += 15
                result["reasons"].append("Price near day high (bullish momentum)")
            elif option_type == "PUT" and price_position < 0.2:
                score += 15
                result["reasons"].append("Price near day low (bearish momentum)")
        
        # Time of day consideration (0DTE works best early in day)
        import datetime
        now = datetime.datetime.now()
        hour = now.hour
        if 9 <= hour <= 11:  # Morning session
            score += 10
            result["reasons"].append("Optimal time window (morning)")
        elif 11 < hour <= 14:  # Midday
            score += 5
        else:  # Late day - avoid
            score -= 20
            result["reasons"].append("Late day - higher risk")
        
        result["score"] = max(0, score)
        result["price"] = price
        result["option_type"] = option_type
        result["change_pct"] = change_pct
        
        # Higher threshold for 0DTE due to extreme risk
        if score >= 60 and option_type:
            result["signal"] = "STRONG_BUY"
            result["strike_suggestion"] = self._suggest_strike(price, option_type)
        elif score >= 45 and option_type:
            result["signal"] = "BUY"
            result["strike_suggestion"] = self._suggest_strike(price, option_type)
        elif score >= 30:
            result["signal"] = "WATCH"
        
        return result
    
    def _suggest_strike(self, price: float, option_type: str) -> Dict:
        """Suggest strike price for 0DTE trade"""
        if option_type == "CALL":
            # Slightly OTM call (1-2% above current price)
            strike = round(price * 1.01, 0)
            return {
                "type": "CALL",
                "strike": strike,
                "delta_approx": 0.40,
                "target_pct": 0.75,  # 75% profit target
                "stop_pct": 0.50     # 50% stop loss
            }
        else:
            # Slightly OTM put (1-2% below current price)
            strike = round(price * 0.99, 0)
            return {
                "type": "PUT",
                "strike": strike,
                "delta_approx": -0.40,
                "target_pct": 0.75,
                "stop_pct": 0.50
            }


class EarningsPlayStrategy(AcceleratorStrategy):
    """
    Earnings volatility plays
    Trade IV crush or directional moves around earnings
    Best for: Small+ accounts ($1,000+)
    Target: 30-100% gains
    """
    
    def __init__(self):
        super().__init__(
            name="Earnings Play",
            min_capital=1000.0,
            risk_level="HIGH"
        )
        
    def analyze(self, symbol: str, quote: Dict, days_to_earnings: int = None) -> Dict:
        """Analyze earnings play opportunity"""
        result = {
            "strategy": self.name,
            "symbol": symbol,
            "signal": "NONE",
            "score": 0,
            "play_type": None,
            "reasons": []
        }
        
        if not quote:
            return result
            
        price = float(quote.get('last', 0))
        
        # If earnings within 5 days, look for IV crush play
        if days_to_earnings is not None:
            if days_to_earnings <= 2:
                result["score"] += 30
                result["play_type"] = "IV_CRUSH"
                result["reasons"].append(f"Earnings in {days_to_earnings} days - IV crush opportunity")
            elif days_to_earnings <= 5:
                result["score"] += 20
                result["play_type"] = "PRE_EARNINGS"
                result["reasons"].append(f"Earnings in {days_to_earnings} days - pre-earnings run potential")
        
        result["price"] = price
        
        if result["score"] >= 30:
            result["signal"] = "WATCH"
        
        return result


class HighGrowthStockStrategy(AcceleratorStrategy):
    """
    High-growth stock picking
    Focus on stocks with 100%+ potential
    Best for: All phases
    """
    
    # High-conviction growth targets
    GROWTH_WATCHLIST = [
        # AI/Tech
        "NVDA", "AMD", "SMCI", "ARM", "PLTR", "SNOW", "NET", "CRWD",
        # EV/Energy  
        "TSLA", "RIVN", "LCID", "ENPH", "FSLR",
        # Biotech
        "MRNA", "BNTX", "CRSP",
        # Fintech
        "SQ", "COIN", "SOFI", "AFRM",
        # High Beta
        "ROKU", "SHOP", "SE", "DKNG", "RBLX"
    ]
    
    def __init__(self):
        super().__init__(
            name="High Growth Stocks",
            min_capital=100.0,
            risk_level="HIGH"
        )
        
    def analyze(self, symbol: str, quote: Dict) -> Dict:
        """Analyze high-growth opportunity"""
        result = {
            "strategy": self.name,
            "symbol": symbol,
            "signal": "NONE",
            "score": 0,
            "reasons": []
        }
        
        # Bonus for watchlist stocks
        if symbol in self.GROWTH_WATCHLIST:
            result["score"] += 15
            result["reasons"].append("High-conviction growth stock")
        
        if not quote:
            return result
            
        price = float(quote.get('last', 0))
        change_pct = float(quote.get('change_percentage', 0)) / 100
        week_52_high = float(quote.get('week_52_high', price))
        week_52_low = float(quote.get('week_52_low', price))
        
        # Distance from 52-week high (value opportunity)
        if week_52_high > 0:
            from_high = (price - week_52_high) / week_52_high
            if from_high < -0.30:  # 30%+ off highs
                result["score"] += 25
                result["reasons"].append(f"Down {abs(from_high):.0%} from 52w high (value)")
            elif from_high < -0.15:
                result["score"] += 15
                result["reasons"].append(f"Down {abs(from_high):.0%} from 52w high")
        
        # Near 52-week low (contrarian buy)
        if week_52_low > 0:
            from_low = (price - week_52_low) / week_52_low
            if from_low < 0.10:  # Within 10% of low
                result["score"] += 20
                result["reasons"].append("Near 52-week low (contrarian)")
        
        # Momentum
        if change_pct > 0.05:
            result["score"] += 15
            result["reasons"].append(f"Strong momentum: +{change_pct:.1%}")
        
        result["price"] = price
        
        if result["score"] >= 50:
            result["signal"] = "STRONG_BUY"
        elif result["score"] >= 35:
            result["signal"] = "BUY"
        elif result["score"] >= 20:
            result["signal"] = "WATCH"
        
        return result


# =============================================================================
# ACCELERATOR ENGINE
# =============================================================================

class OptimusAccelerator:
    """
    Main accelerator engine for aggressive growth
    """
    
    def __init__(self, config: AcceleratorConfig = None):
        self.config = config or AcceleratorConfig()
        self.strategies = [
            MomentumFractionalStrategy(),
            BreakoutSwingStrategy(),
            LeveragedETFStrategy(),
            HighGrowthStockStrategy(),
            ZeroDTEOptionsStrategy(),    # NEW: 0DTE for $2,500+ accounts
            EarningsPlayStrategy(),       # NEW: Earnings volatility plays
        ]
        self.current_balance = self.config.starting_capital
        self.phase = AccountPhase.MICRO
        self.trade_history = []
        
    def get_phase(self, balance: float) -> AccountPhase:
        """Determine account phase based on balance"""
        if balance < self.config.micro_max:
            return AccountPhase.MICRO
        elif balance < self.config.small_max:
            return AccountPhase.SMALL
        elif balance < self.config.growth_max:
            return AccountPhase.GROWTH
        else:
            return AccountPhase.SCALE
    
    def get_risk_pct(self, phase: AccountPhase) -> float:
        """Get risk percentage for current phase"""
        risk_map = {
            AccountPhase.MICRO: self.config.micro_risk_pct,
            AccountPhase.SMALL: self.config.small_risk_pct,
            AccountPhase.GROWTH: self.config.growth_risk_pct,
            AccountPhase.SCALE: self.config.scale_risk_pct,
        }
        return risk_map.get(phase, 0.02)
    
    def get_available_strategies(self, balance: float) -> List[AcceleratorStrategy]:
        """Get strategies available for current balance"""
        return [s for s in self.strategies if s.min_capital <= balance and s.enabled]
    
    def fetch_quote(self, symbol: str) -> Optional[Dict]:
        """Fetch quote from Tradier"""
        try:
            api_key = os.environ.get("TRADIER_API_KEY")
            url = "https://api.tradier.com/v1/markets/quotes"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Accept": "application/json"
            }
            response = requests.get(url, headers=headers, params={"symbols": symbol}, timeout=10)
            if response.status_code == 200:
                data = response.json()
                quote = data.get("quotes", {}).get("quote", {})
                if isinstance(quote, list):
                    quote = quote[0] if quote else {}
                return quote
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
        return None
    
    def scan_opportunities(self, symbols: List[str] = None) -> List[Dict]:
        """Scan for trading opportunities"""
        if symbols is None:
            # Default scan list based on phase
            symbols = [
                # High-liquidity for micro accounts
                "SPY", "QQQ", "AAPL", "MSFT", "NVDA", "AMD", "TSLA",
                # Leveraged ETFs
                "TQQQ", "SQQQ", "UPRO", "SOXL",
                # Growth stocks
                "PLTR", "SOFI", "COIN", "ROKU", "DKNG"
            ]
        
        opportunities = []
        available_strategies = self.get_available_strategies(self.current_balance)
        
        print(f"\n[SCAN] Scanning {len(symbols)} symbols with {len(available_strategies)} strategies...")
        print(f"   Account Phase: {self.phase.value.upper()}")
        print(f"   Available Capital: ${self.current_balance:,.2f}")
        print(f"   Risk Per Trade: {self.get_risk_pct(self.phase)*100:.0f}%\n")
        
        for symbol in symbols:
            quote = self.fetch_quote(symbol)
            if not quote:
                continue
            
            for strategy in available_strategies:
                try:
                    if isinstance(strategy, LeveragedETFStrategy):
                        result = strategy.analyze(symbol, quote, "NEUTRAL")
                    else:
                        result = strategy.analyze(symbol, quote)
                    
                    if result["signal"] in ["BUY", "STRONG_BUY"]:
                        opportunities.append(result)
                        
                except Exception as e:
                    print(f"Error analyzing {symbol} with {strategy.name}: {e}")
        
        # Sort by score
        opportunities.sort(key=lambda x: x["score"], reverse=True)
        
        return opportunities
    
    def generate_trade_plan(self, opportunities: List[Dict], max_trades: int = 3) -> List[Dict]:
        """Generate trade plan from opportunities"""
        if not opportunities:
            return []
        
        trade_plan = []
        remaining_capital = self.current_balance
        risk_pct = self.get_risk_pct(self.phase)
        
        for opp in opportunities[:max_trades]:
            if remaining_capital < 10:  # Minimum $10 per trade
                break
                
            price = opp.get("price", 0)
            if price <= 0:
                continue
            
            # Calculate position size
            risk_amount = remaining_capital * risk_pct
            
            # For micro accounts, can use fractional shares
            if self.phase == AccountPhase.MICRO:
                # Use up to 50% of remaining capital per trade
                position_value = min(risk_amount * 2, remaining_capital * 0.5)
                shares = position_value / price
            else:
                shares = max(1, int(risk_amount / price))
                position_value = shares * price
            
            if position_value > remaining_capital:
                continue
            
            trade = {
                "symbol": opp["symbol"],
                "strategy": opp["strategy"],
                "signal": opp["signal"],
                "score": opp["score"],
                "price": price,
                "shares": round(shares, 4),
                "position_value": round(position_value, 2),
                "risk_amount": round(risk_amount, 2),
                "reasons": opp.get("reasons", []),
                "stop_loss": round(price * 0.95, 2),  # 5% stop
                "target_1": round(price * 1.10, 2),   # 10% target
                "target_2": round(price * 1.20, 2),   # 20% target
            }
            
            trade_plan.append(trade)
            remaining_capital -= position_value
        
        return trade_plan
    
    def execute_trade_plan(self, trade_plan: List[Dict], dry_run: bool = True) -> List[Dict]:
        """Execute the trade plan"""
        results = []
        
        for trade in trade_plan:
            print(f"\n{'='*50}")
            print(f"TRADE: {trade['signal']} {trade['symbol']}")
            print(f"{'='*50}")
            print(f"Strategy: {trade['strategy']}")
            print(f"Score: {trade['score']}/100")
            print(f"Price: ${trade['price']:.2f}")
            print(f"Shares: {trade['shares']}")
            print(f"Position: ${trade['position_value']:.2f}")
            print(f"Stop Loss: ${trade['stop_loss']:.2f}")
            print(f"Target 1: ${trade['target_1']:.2f} (+10%)")
            print(f"Target 2: ${trade['target_2']:.2f} (+20%)")
            print(f"Reasons: {', '.join(trade['reasons'])}")
            
            if dry_run:
                print(f"\n[DRY RUN] Trade not executed")
                trade["status"] = "dry_run"
            else:
                # Execute via Tradier
                trade["status"] = "pending"
                # TODO: Add actual execution
            
            results.append(trade)
        
        return results
    
    def run_accelerator(self, execute: bool = False):
        """Run the full accelerator pipeline"""
        print("\n" + "=" * 70)
        print("  [ROCKET] OPTIMUS ACCELERATOR - AGGRESSIVE GROWTH MODE [ROCKET]")
        print(f"  Target: $100 -> $5,000,000 in 8 years")
        print("=" * 70)
        
        # Update phase
        self.phase = self.get_phase(self.current_balance)
        
        # Show projection
        projections = calculate_growth_projection(self.config)
        print_growth_milestones(projections)
        
        # Scan for opportunities
        print("\n" + "=" * 70)
        print("  OPPORTUNITY SCAN")
        print("=" * 70)
        
        opportunities = self.scan_opportunities()
        
        if not opportunities:
            print("\n[X] No opportunities found meeting criteria")
            return
        
        print(f"\n[OK] Found {len(opportunities)} opportunities:")
        for i, opp in enumerate(opportunities[:5], 1):
            print(f"   {i}. {opp['symbol']} - {opp['strategy']} - Score: {opp['score']}")
        
        # Generate trade plan
        print("\n" + "=" * 70)
        print("  TRADE PLAN")
        print("=" * 70)
        
        trade_plan = self.generate_trade_plan(opportunities)
        
        if not trade_plan:
            print("\n[X] Could not generate trade plan")
            return
        
        # Execute
        results = self.execute_trade_plan(trade_plan, dry_run=not execute)
        
        # Summary
        print("\n" + "=" * 70)
        print("  ACCELERATOR SUMMARY")
        print("=" * 70)
        print(f"  Account Balance: ${self.current_balance:,.2f}")
        print(f"  Phase: {self.phase.value.upper()}")
        print(f"  Opportunities Found: {len(opportunities)}")
        print(f"  Trades Planned: {len(trade_plan)}")
        total_deployed = sum(t['position_value'] for t in trade_plan)
        print(f"  Capital to Deploy: ${total_deployed:,.2f}")
        print(f"  Remaining Cash: ${self.current_balance - total_deployed:,.2f}")
        print("=" * 70)


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point"""
    # Configuration for $100 start with bi-weekly $100 deposits
    config = AcceleratorConfig(
        starting_capital=100.0,
        biweekly_deposit=100.0,
        target_goal=5_000_000.0,
        target_years=8
    )
    
    # Create accelerator
    accelerator = OptimusAccelerator(config)
    
    # For testing, set a realistic starting balance
    # In production, this would come from actual account balance
    accelerator.current_balance = 100.0  # Start with $100
    
    # Run accelerator (dry run by default)
    accelerator.run_accelerator(execute=False)


if __name__ == "__main__":
    main()

