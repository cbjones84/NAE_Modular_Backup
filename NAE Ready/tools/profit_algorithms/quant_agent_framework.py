# NAE/tools/profit_algorithms/quant_agent_framework.py
"""
Multi-Agent LLM Framework for Trading (QuantAgent-inspired)
Based on arXiv:2509.09995 - Multi-Agent LLM for High-Frequency Trading

This framework uses specialized LLM agents for different aspects of trading:
- Indicator Agent: Technical indicator analysis
- Pattern Agent: Pattern recognition
- Trend Agent: Trend identification
- Risk Agent: Risk assessment

All agents work together to make rapid, risk-aware trading decisions.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class MarketSignal:
    """Structured market signal from agent analysis"""
    indicators: Dict[str, float]
    patterns: List[str]
    trend_direction: str  # "bullish", "bearish", "neutral"
    trend_strength: float  # 0.0 to 1.0
    risk_score: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    recommendation: str  # "buy", "sell", "hold", "avoid"


class IndicatorAgent:
    """Specialized agent for technical indicator analysis"""
    
    def __init__(self):
        self.indicators = {}
    
    def analyze(self, market_data: pd.DataFrame) -> Dict[str, float]:
        """
        Analyze market data using technical indicators.
        
        Args:
            market_data: DataFrame with OHLCV data
            
        Returns:
            Dict of indicator values
        """
        if market_data.empty or len(market_data) < 20:
            return {}
        
        close = market_data['close'] if 'close' in market_data.columns else market_data.iloc[:, -1]
        
        indicators = {}
        
        # Moving averages
        indicators['sma_20'] = close.rolling(20).mean().iloc[-1]
        indicators['sma_50'] = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else None
        indicators['ema_12'] = close.ewm(span=12).mean().iloc[-1]
        indicators['ema_26'] = close.ewm(span=26).mean().iloc[-1] if len(close) >= 26 else None
        
        # RSI
        if len(close) >= 14:
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            indicators['rsi'] = 100 - (100 / (1 + rs.iloc[-1]))
        
        # MACD
        if indicators.get('ema_12') and indicators.get('ema_26'):
            indicators['macd'] = indicators['ema_12'] - indicators['ema_26']
        
        # Bollinger Bands
        if len(close) >= 20:
            sma_20 = close.rolling(20).mean()
            std_20 = close.rolling(20).std()
            indicators['bb_upper'] = (sma_20 + 2 * std_20).iloc[-1]
            indicators['bb_lower'] = (sma_20 - 2 * std_20).iloc[-1]
            indicators['bb_position'] = (close.iloc[-1] - indicators['bb_lower']) / (indicators['bb_upper'] - indicators['bb_lower'])
        
        # Volume indicators
        if 'volume' in market_data.columns:
            volume = market_data['volume']
            indicators['volume_sma'] = volume.rolling(20).mean().iloc[-1]
            indicators['volume_ratio'] = volume.iloc[-1] / indicators['volume_sma'] if indicators['volume_sma'] > 0 else 1.0
        
        return indicators


class PatternAgent:
    """Specialized agent for pattern recognition"""
    
    def detect(self, market_data: pd.DataFrame) -> List[str]:
        """
        Detect chart patterns in market data.
        
        Args:
            market_data: DataFrame with OHLCV data
            
        Returns:
            List of detected patterns
        """
        patterns = []
        
        if market_data.empty or len(market_data) < 10:
            return patterns
        
        close = market_data['close'] if 'close' in market_data.columns else market_data.iloc[:, -1]
        high = market_data['high'] if 'high' in market_data.columns else close
        low = market_data['low'] if 'low' in market_data.columns else close
        
        # Simple pattern detection
        recent_prices = close.tail(10).values
        
        # Double top/bottom
        if len(recent_prices) >= 5:
            peaks = []
            troughs = []
            for i in range(1, len(recent_prices) - 1):
                if recent_prices[i] > recent_prices[i-1] and recent_prices[i] > recent_prices[i+1]:
                    peaks.append(recent_prices[i])
                elif recent_prices[i] < recent_prices[i-1] and recent_prices[i] < recent_prices[i+1]:
                    troughs.append(recent_prices[i])
            
            if len(peaks) >= 2 and abs(peaks[-1] - peaks[-2]) / peaks[-1] < 0.02:
                patterns.append("double_top")
            if len(troughs) >= 2 and abs(troughs[-1] - troughs[-2]) / troughs[-1] < 0.02:
                patterns.append("double_bottom")
        
        # Trend patterns
        if len(recent_prices) >= 5:
            trend = np.polyfit(range(len(recent_prices)), recent_prices, 1)[0]
            if trend > 0:
                patterns.append("uptrend")
            elif trend < 0:
                patterns.append("downtrend")
            else:
                patterns.append("sideways")
        
        return patterns


class TrendAgent:
    """Specialized agent for trend identification"""
    
    def identify(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Identify market trends.
        
        Args:
            market_data: DataFrame with OHLCV data
            
        Returns:
            Dict with trend direction and strength
        """
        if market_data.empty or len(market_data) < 20:
            return {"direction": "neutral", "strength": 0.0}
        
        close = market_data['close'] if 'close' in market_data.columns else market_data.iloc[:, -1]
        
        # Short-term trend (5 periods)
        short_trend = np.polyfit(range(min(5, len(close))), close.tail(5).values, 1)[0]
        
        # Medium-term trend (20 periods)
        medium_trend = np.polyfit(range(min(20, len(close))), close.tail(20).values, 1)[0]
        
        # Long-term trend (50 periods if available)
        if len(close) >= 50:
            long_trend = np.polyfit(range(50), close.tail(50).values, 1)[0]
        else:
            long_trend = medium_trend
        
        # Determine direction
        avg_trend = (short_trend + medium_trend + long_trend) / 3
        
        if avg_trend > 0.01:
            direction = "bullish"
        elif avg_trend < -0.01:
            direction = "bearish"
        else:
            direction = "neutral"
        
        # Calculate strength (0.0 to 1.0)
        strength = min(abs(avg_trend) * 100, 1.0)
        
        return {
            "direction": direction,
            "strength": strength,
            "short_trend": short_trend,
            "medium_trend": medium_trend,
            "long_trend": long_trend
        }


class RiskAgent:
    """Specialized agent for risk assessment"""
    
    def assess(self, market_data: pd.DataFrame, position_size: float = 0.0) -> Dict[str, float]:
        """
        Assess market risk.
        
        Args:
            market_data: DataFrame with OHLCV data
            position_size: Current position size (optional)
            
        Returns:
            Dict with risk metrics
        """
        if market_data.empty:
            return {"risk_score": 0.5, "volatility": 0.0, "drawdown": 0.0}
        
        close = market_data['close'] if 'close' in market_data.columns else market_data.iloc[:, -1]
        
        # Volatility (standard deviation of returns)
        returns = close.pct_change().dropna()
        volatility = returns.std() * np.sqrt(252) if len(returns) > 1 else 0.0
        
        # Drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = ((cumulative - running_max) / running_max).min()
        
        # Risk score (0.0 = low risk, 1.0 = high risk)
        risk_score = min((volatility * 2 + abs(drawdown)) / 2, 1.0)
        
        return {
            "risk_score": risk_score,
            "volatility": volatility,
            "drawdown": abs(drawdown),
            "max_drawdown": abs(drawdown)
        }


class QuantAgentFramework:
    """
    Multi-Agent LLM Framework for Trading
    Coordinates specialized agents to make trading decisions
    """
    
    def __init__(self):
        self.indicator_agent = IndicatorAgent()
        self.pattern_agent = PatternAgent()
        self.trend_agent = TrendAgent()
        self.risk_agent = RiskAgent()
    
    def analyze_market(self, market_data: pd.DataFrame, position_size: float = 0.0) -> MarketSignal:
        """
        Analyze market using all specialized agents.
        
        Args:
            market_data: DataFrame with OHLCV data
            position_size: Current position size (optional)
            
        Returns:
            MarketSignal with synthesized analysis
        """
        # Get analysis from each agent
        indicators = self.indicator_agent.analyze(market_data)
        patterns = self.pattern_agent.detect(market_data)
        trend = self.trend_agent.identify(market_data)
        risk = self.risk_agent.assess(market_data, position_size)
        
        # Synthesize recommendations
        recommendation, confidence = self._synthesize(indicators, patterns, trend, risk)
        
        return MarketSignal(
            indicators=indicators,
            patterns=patterns,
            trend_direction=trend["direction"],
            trend_strength=trend["strength"],
            risk_score=risk["risk_score"],
            confidence=confidence,
            recommendation=recommendation
        )
    
    def _synthesize(self, indicators: Dict[str, float], patterns: List[str],
                   trend: Dict[str, Any], risk: Dict[str, float]) -> tuple:
        """
        Synthesize agent analyses into trading recommendation.
        
        Returns:
            Tuple of (recommendation, confidence)
        """
        # Base recommendation on trend
        if trend["direction"] == "bullish" and trend["strength"] > 0.5:
            base_rec = "buy"
            base_confidence = trend["strength"]
        elif trend["direction"] == "bearish" and trend["strength"] > 0.5:
            base_rec = "sell"
            base_confidence = trend["strength"]
        else:
            base_rec = "hold"
            base_confidence = 0.5
        
        # Adjust for risk
        if risk["risk_score"] > 0.7:
            base_rec = "avoid"
            base_confidence = 0.3
        elif risk["risk_score"] > 0.5:
            base_confidence *= 0.7  # Reduce confidence with higher risk
        
        # Adjust for patterns
        if "double_top" in patterns and base_rec == "buy":
            base_rec = "sell"
            base_confidence = 0.6
        elif "double_bottom" in patterns and base_rec == "sell":
            base_rec = "buy"
            base_confidence = 0.6
        
        # Adjust for indicators
        if indicators.get("rsi"):
            rsi = indicators["rsi"]
            if rsi > 70 and base_rec == "buy":
                base_rec = "hold"
                base_confidence *= 0.8
            elif rsi < 30 and base_rec == "sell":
                base_rec = "hold"
                base_confidence *= 0.8
        
        return base_rec, min(base_confidence, 1.0)

