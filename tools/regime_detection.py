# NAE/tools/regime_detection.py
"""
Regime Detection & Adaptive Strategies

Detects market regimes:
- Low volatility
- Trending
- Mean-reverting
- Crisis/High volatility

Routes strategy selection based on regime
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime


class MarketRegime(Enum):
    LOW_VOLATILITY = "low_volatility"
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    MEAN_REVERTING = "mean_reverting"
    HIGH_VOLATILITY = "high_volatility"
    CRISIS = "crisis"


@dataclass
class RegimeFeatures:
    """Features for regime detection"""
    realized_volatility: float
    trend_strength: float
    mean_reversion_score: float
    volume_profile: float
    iv_surface_shape: float
    order_flow_imbalance: float = 0.0


class RegimeDetector:
    """
    Market regime detection system
    """
    
    def __init__(self):
        self.regime_history: List[Tuple[datetime, MarketRegime]] = []
        self.feature_history: List[RegimeFeatures] = []
    
    def detect_regime(
        self,
        price_data: pd.DataFrame,
        volume_data: Optional[pd.Series] = None,
        iv_data: Optional[pd.Series] = None
    ) -> Tuple[MarketRegime, float, Dict[str, Any]]:
        """
        Detect current market regime
        
        Args:
            price_data: Price data (OHLCV)
            volume_data: Volume data (optional)
            iv_data: Implied volatility data (optional)
        
        Returns:
            (regime, confidence, details)
        """
        # Calculate features
        features = self._calculate_features(price_data, volume_data, iv_data)
        self.feature_history.append(features)
        
        # Classify regime
        regime, confidence, details = self._classify_regime(features)
        
        # Record regime
        self.regime_history.append((datetime.now(), regime))
        
        return regime, confidence, details
    
    def _calculate_features(
        self,
        price_data: pd.DataFrame,
        volume_data: Optional[pd.Series],
        iv_data: Optional[pd.Series]
    ) -> RegimeFeatures:
        """Calculate regime detection features"""
        prices = price_data['close'] if 'close' in price_data.columns else price_data.iloc[:, 0]
        
        # Realized volatility (20-day rolling)
        returns = prices.pct_change().dropna()
        if len(returns) >= 20:
            realized_vol = returns.tail(20).std() * np.sqrt(252)
        else:
            realized_vol = returns.std() * np.sqrt(252) if len(returns) > 1 else 0.0
        
        # Trend strength (ADX-like)
        if len(prices) >= 14:
            high_low = (prices.max() - prices.min()) / prices.mean() if prices.mean() > 0 else 0.0
            trend_strength = min(1.0, high_low * 10)  # Normalize
        else:
            trend_strength = 0.0
        
        # Mean reversion score (Hurst exponent approximation)
        if len(returns) >= 20:
            # Simplified mean reversion score
            autocorr = returns.autocorr(lag=1)
            mean_reversion_score = max(0.0, -autocorr) if not np.isnan(autocorr) else 0.0
        else:
            mean_reversion_score = 0.0
        
        # Volume profile
        if volume_data is not None and len(volume_data) >= 20:
            volume_ma = volume_data.tail(20).mean()
            current_volume = volume_data.iloc[-1]
            volume_profile = min(2.0, current_volume / volume_ma) if volume_ma > 0 else 1.0
        else:
            volume_profile = 1.0
        
        # IV surface shape (skew approximation)
        if iv_data is not None and len(iv_data) >= 5:
            iv_surface_shape = iv_data.tail(5).std() / iv_data.tail(5).mean() if iv_data.tail(5).mean() > 0 else 0.0
        else:
            iv_surface_shape = 0.0
        
        return RegimeFeatures(
            realized_volatility=realized_vol,
            trend_strength=trend_strength,
            mean_reversion_score=mean_reversion_score,
            volume_profile=volume_profile,
            iv_surface_shape=iv_surface_shape
        )
    
    def _classify_regime(
        self,
        features: RegimeFeatures
    ) -> Tuple[MarketRegime, float, Dict[str, Any]]:
        """Classify regime based on features"""
        details = {}
        
        # High volatility / Crisis detection
        if features.realized_volatility > 0.40:  # >40% annual vol
            regime = MarketRegime.CRISIS
            confidence = min(1.0, features.realized_volatility / 0.50)
            details["reason"] = f"Very high volatility: {features.realized_volatility:.2%}"
        
        # High volatility
        elif features.realized_volatility > 0.25:  # >25% annual vol
            regime = MarketRegime.HIGH_VOLATILITY
            confidence = min(1.0, (features.realized_volatility - 0.25) / 0.15)
            details["reason"] = f"High volatility: {features.realized_volatility:.2%}"
        
        # Low volatility
        elif features.realized_volatility < 0.10:  # <10% annual vol
            regime = MarketRegime.LOW_VOLATILITY
            confidence = min(1.0, (0.10 - features.realized_volatility) / 0.10)
            details["reason"] = f"Low volatility: {features.realized_volatility:.2%}"
        
        # Mean reverting
        elif features.mean_reversion_score > 0.3 and features.trend_strength < 0.3:
            regime = MarketRegime.MEAN_REVERTING
            confidence = min(1.0, features.mean_reversion_score)
            details["reason"] = f"Mean reverting pattern detected"
        
        # Trending up
        elif features.trend_strength > 0.5:
            # Check if trend is up (simplified)
            regime = MarketRegime.TRENDING_UP
            confidence = min(1.0, features.trend_strength)
            details["reason"] = f"Strong uptrend: {features.trend_strength:.2f}"
        
        # Trending down
        elif features.trend_strength > 0.3:
            regime = MarketRegime.TRENDING_DOWN
            confidence = min(1.0, features.trend_strength)
            details["reason"] = f"Downtrend: {features.trend_strength:.2f}"
        
        # Default: mean reverting
        else:
            regime = MarketRegime.MEAN_REVERTING
            confidence = 0.5
            details["reason"] = "Default: mean reverting"
        
        details.update({
            "realized_volatility": features.realized_volatility,
            "trend_strength": features.trend_strength,
            "mean_reversion_score": features.mean_reversion_score
        })
        
        return regime, confidence, details
    
    def get_recommended_strategy(self, regime: MarketRegime) -> Dict[str, Any]:
        """
        Get recommended strategy for regime
        
        Returns strategy recommendations based on regime
        """
        recommendations = {
            MarketRegime.LOW_VOLATILITY: {
                "strategies": ["credit_spreads", "iron_condors", "calendar_spreads"],
                "avoid": ["naked_calls", "high_gamma"],
                "position_sizing": "moderate",
                "risk_level": "low"
            },
            MarketRegime.TRENDING_UP: {
                "strategies": ["call_debit_spreads", "covered_calls", "bull_put_spreads"],
                "avoid": ["put_buying", "bear_spreads"],
                "position_sizing": "aggressive",
                "risk_level": "moderate"
            },
            MarketRegime.TRENDING_DOWN: {
                "strategies": ["put_debit_spreads", "protective_puts", "bear_call_spreads"],
                "avoid": ["call_buying", "bull_spreads"],
                "position_sizing": "defensive",
                "risk_level": "moderate"
            },
            MarketRegime.MEAN_REVERTING: {
                "strategies": ["straddles", "strangles", "iron_condors"],
                "avoid": ["directional_spreads"],
                "position_sizing": "moderate",
                "risk_level": "moderate"
            },
            MarketRegime.HIGH_VOLATILITY: {
                "strategies": ["credit_spreads", "iron_condors", "volatility_harvesting"],
                "avoid": ["naked_options", "high_leverage"],
                "position_sizing": "conservative",
                "risk_level": "high"
            },
            MarketRegime.CRISIS: {
                "strategies": ["protective_puts", "cash", "hedging"],
                "avoid": ["naked_positions", "high_exposure"],
                "position_sizing": "minimal",
                "risk_level": "critical"
            }
        }
        
        return recommendations.get(regime, {
            "strategies": [],
            "avoid": [],
            "position_sizing": "moderate",
            "risk_level": "moderate"
        })
    
    def get_regime_history(self, days: int = 30) -> List[Tuple[datetime, MarketRegime]]:
        """Get recent regime history"""
        cutoff = datetime.now() - pd.Timedelta(days=days)
        return [(dt, reg) for dt, reg in self.regime_history if dt >= cutoff]

