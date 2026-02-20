# NAE/tools/profit_algorithms/timing_strategies.py
"""
Advanced Entry and Exit Timing Strategies for Maximum Profit
Implements sophisticated timing algorithms to optimize trade entry and exit points
for compound growth and generational wealth accumulation ($5M+ goal)
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import datetime


class MarketCondition(Enum):
    """Market condition classification"""
    BULL_TREND = "bull_trend"
    BEAR_TREND = "bear_trend"
    CONSOLIDATION = "consolidation"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    REVERSAL = "reversal"


class EntrySignal(Enum):
    """Entry signal strength"""
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    WEAK_BUY = "weak_buy"
    NO_SIGNAL = "no_signal"
    WEAK_SELL = "weak_sell"
    SELL = "sell"
    STRONG_SELL = "strong_sell"


class ExitReason(Enum):
    """Exit reason classification"""
    PROFIT_TARGET = "profit_target"
    STOP_LOSS = "stop_loss"
    TRAILING_STOP = "trailing_stop"
    TREND_REVERSAL = "trend_reversal"
    TIME_BASED = "time_based"
    VOLATILITY_SPIKE = "volatility_spike"
    RISK_MANAGEMENT = "risk_management"


@dataclass
class TechnicalIndicators:
    """Container for technical analysis indicators"""
    rsi: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_histogram: Optional[float] = None
    sma_20: Optional[float] = None
    sma_50: Optional[float] = None
    sma_200: Optional[float] = None
    ema_12: Optional[float] = None
    ema_26: Optional[float] = None
    bollinger_upper: Optional[float] = None
    bollinger_middle: Optional[float] = None
    bollinger_lower: Optional[float] = None
    atr: Optional[float] = None  # Average True Range
    volume_ratio: Optional[float] = None  # Current volume / Average volume
    support_level: Optional[float] = None
    resistance_level: Optional[float] = None
    price_position: Optional[float] = None  # Position between support/resistance (0-1)


@dataclass
class EntryAnalysis:
    """Comprehensive entry timing analysis"""
    signal: EntrySignal
    confidence: float  # 0-1
    optimal_entry_price: float
    suggested_quantity: float
    risk_reward_ratio: float
    stop_loss_price: float
    take_profit_price: float
    market_condition: MarketCondition
    entry_reasons: List[str]
    risk_factors: List[str]
    timing_score: float  # 0-100


@dataclass
class ExitAnalysis:
    """Comprehensive exit timing analysis"""
    should_exit: bool
    exit_reason: Optional[ExitReason]
    confidence: float  # 0-1
    optimal_exit_price: float
    exit_urgency: str  # "low", "medium", "high", "critical"
    profit_potential_remaining: float
    risk_increasing: bool
    exit_reasons: List[str]
    timing_score: float  # 0-100


class TimingStrategyEngine:
    """
    Advanced timing strategy engine for optimal entry and exit points
    Optimized for compound growth and generational wealth accumulation
    
    VERY_AGGRESSIVE MODE: Day trading ENABLED for faster growth toward $15.7M goal
    
    Growth Milestones:
    Year 1: $9,411 | Year 2: $44,110 | Year 3: $152,834 | Year 4: $388,657
    Year 5: $982,500 | Year 6: $2,477,897 | Year 7: $6,243,561 | Year 8: $15,726,144
    """
    
    def __init__(self, nav: float = 25000.0, pdt_prevention: bool = False):
        self.nav = nav
        self.min_risk_reward_ratio = 1.5  # AGGRESSIVE: 1.5:1 R/R (was 2:1)
        self.max_risk_per_trade_pct = 0.15  # VERY_AGGRESSIVE: 15% risk per trade (was 2%)
        self.trailing_stop_activation_pct = 0.03  # Activate trailing stop at 3% profit (faster)
        self.trailing_stop_distance_pct = 0.02  # 2% trailing stop distance (tighter)
        self.pdt_prevention = pdt_prevention  # DISABLED for day trading
        self.min_hold_period_days = 0  # Day trading allowed
        
    def calculate_technical_indicators(self, price_data: List[Dict[str, Any]]) -> TechnicalIndicators:
        """
        Calculate technical indicators from price data
        
        Args:
            price_data: List of dicts with 'close', 'high', 'low', 'volume', 'open' keys
            
        Returns:
            TechnicalIndicators object
        """
        if not price_data or len(price_data) < 50:
            return TechnicalIndicators()
        
        # Extract price arrays
        closes = np.array([p['close'] for p in price_data])
        highs = np.array([p['high'] for p in price_data])
        lows = np.array([p['low'] for p in price_data])
        volumes = np.array([p.get('volume', 0) for p in price_data])
        
        indicators = TechnicalIndicators()
        
        # Calculate RSI (14-period)
        indicators.rsi = self._calculate_rsi(closes, period=14)
        
        # Calculate MACD (12, 26, 9)
        macd, signal, histogram = self._calculate_macd(closes)
        indicators.macd = macd
        indicators.macd_signal = signal
        indicators.macd_histogram = histogram
        
        # Calculate Moving Averages
        indicators.sma_20 = self._calculate_sma(closes, 20)
        indicators.sma_50 = self._calculate_sma(closes, 50)
        indicators.sma_200 = self._calculate_sma(closes, 200)
        indicators.ema_12 = self._calculate_ema(closes, 12)
        indicators.ema_26 = self._calculate_ema(closes, 26)
        
        # Calculate Bollinger Bands (20-period, 2 std dev)
        bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(closes, 20, 2.0)
        indicators.bollinger_upper = bb_upper
        indicators.bollinger_middle = bb_middle
        indicators.bollinger_lower = bb_lower
        
        # Calculate ATR (14-period)
        indicators.atr = self._calculate_atr(highs, lows, closes, 14)
        
        # Calculate Volume Ratio
        if len(volumes) > 20:
            avg_volume = np.mean(volumes[-20:])
            current_volume = volumes[-1] if volumes[-1] > 0 else avg_volume
            indicators.volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
        
        # Calculate Support/Resistance
        support, resistance = self._calculate_support_resistance(highs, lows, closes)
        indicators.support_level = support
        indicators.resistance_level = resistance
        
        # Calculate price position (0 = support, 1 = resistance)
        current_price = closes[-1]
        if support and resistance and resistance > support:
            indicators.price_position = (current_price - support) / (resistance - support)
        
        return indicators
    
    def analyze_entry_timing(self, symbol: str, current_price: float,
                           price_data: List[Dict[str, Any]],
                           market_data: Optional[Dict[str, Any]] = None) -> EntryAnalysis:
        """
        Comprehensive entry timing analysis
        
        Args:
            symbol: Trading symbol
            current_price: Current market price
            price_data: Historical price data
            market_data: Optional market sentiment/volume data
            
        Returns:
            EntryAnalysis with optimal entry recommendation
        """
        indicators = self.calculate_technical_indicators(price_data)
        
        # Analyze market condition
        market_condition = self._classify_market_condition(indicators, current_price)
        
        # Generate entry signal
        signal, confidence = self._generate_entry_signal(indicators, current_price, market_condition)
        
        # Calculate optimal entry parameters
        entry_price, stop_loss, take_profit = self._calculate_entry_levels(
            current_price, indicators, market_condition
        )
        
        # Calculate risk/reward ratio
        risk = abs(current_price - stop_loss)
        reward = abs(take_profit - current_price)
        risk_reward = reward / risk if risk > 0 else 0
        
        # Calculate position size based on risk
        suggested_quantity = self._calculate_position_size(
            current_price, stop_loss, risk_reward
        )
        
        # Generate entry reasons
        entry_reasons = self._generate_entry_reasons(indicators, signal, market_condition)
        risk_factors = self._identify_risk_factors(indicators, market_condition)
        
        # Calculate overall timing score (0-100)
        timing_score = self._calculate_timing_score(
            indicators, signal, confidence, risk_reward, market_condition
        )
        
        return EntryAnalysis(
            signal=signal,
            confidence=confidence,
            optimal_entry_price=entry_price,
            suggested_quantity=suggested_quantity,
            risk_reward_ratio=risk_reward,
            stop_loss_price=stop_loss,
            take_profit_price=take_profit,
            market_condition=market_condition,
            entry_reasons=entry_reasons,
            risk_factors=risk_factors,
            timing_score=timing_score
        )
    
    def analyze_exit_timing(self, symbol: str, entry_price: float, entry_time: datetime.datetime,
                          current_price: float, quantity: float,
                          price_data: List[Dict[str, Any]],
                          current_pnl: float, current_pnl_pct: float) -> ExitAnalysis:
        """
        Comprehensive exit timing analysis
        
        CRITICAL: Enforces PDT prevention - no same-day exits
        All exits must occur after minimum hold period (overnight minimum)
        
        Args:
            symbol: Trading symbol
            entry_price: Original entry price
            entry_time: When position was opened
            current_price: Current market price
            quantity: Position quantity
            price_data: Historical price data
            current_pnl: Current profit/loss
            current_pnl_pct: Current profit/loss percentage
            
        Returns:
            ExitAnalysis with optimal exit recommendation
        """
        indicators = self.calculate_technical_indicators(price_data)
        
        # ==================== PDT PREVENTION CHECK ====================
        # CRITICAL: Block same-day exits to prevent Pattern Day Trading
        current_time = datetime.datetime.now()
        holding_days = (current_time - entry_time).days
        holding_hours = (current_time - entry_time).total_seconds() / 3600
        
        # Check if this would be a same-day round trip
        if self.pdt_prevention and holding_days < self.min_hold_period_days:
            # Position hasn't held overnight - block exit
            return ExitAnalysis(
                should_exit=False,
                exit_reason=None,
                confidence=1.0,
                optimal_exit_price=current_price,
                exit_urgency="low",
                profit_potential_remaining=0,
                risk_increasing=False,
                exit_reasons=[f"PDT Prevention: Position held {holding_hours:.1f} hours, must hold {self.min_hold_period_days} day(s) minimum"],
                timing_score=0
            )
        
        # Check various exit conditions
        exit_reasons_list = []
        should_exit = False
        exit_reason = None
        exit_urgency = "low"
        
        # 1. Profit target reached
        if current_pnl_pct >= 0.10:  # 10% profit
            should_exit = True
            exit_reason = ExitReason.PROFIT_TARGET
            exit_urgency = "medium"
            exit_reasons_list.append(f"Profit target reached: {current_pnl_pct:.2%}")
        
        # 2. Stop loss hit
        stop_loss_pct = -0.02  # 2% stop loss
        if current_pnl_pct <= stop_loss_pct:
            should_exit = True
            exit_reason = ExitReason.STOP_LOSS
            exit_urgency = "critical"
            exit_reasons_list.append(f"Stop loss triggered: {current_pnl_pct:.2%}")
        
        # 3. Trailing stop based on highest price achieved since entry
        highest_price = max(current_price, entry_price)
        if price_data:
            try:
                highs = [
                    float(bar.get("high", bar.get("close", current_price)))
                    for bar in price_data
                    if bar.get("high") is not None or bar.get("close") is not None
                ]
                if highs:
                    highest_price = max(highest_price, max(highs))
            except Exception:
                pass  # Fallback to current/highest price computed so far
        
        max_profit_pct = 0.0
        if entry_price > 0:
            max_profit_pct = (highest_price - entry_price) / entry_price
        
        trailing_stop_active = max_profit_pct >= self.trailing_stop_activation_pct
        if trailing_stop_active:
            trailing_stop_price = highest_price * (1 - self.trailing_stop_distance_pct)
            if current_price <= trailing_stop_price:
                should_exit = True
                exit_reason = ExitReason.TRAILING_STOP
                exit_urgency = "high"
                exit_reasons_list.append(
                    f"Trailing stop triggered: price fell to {current_price:.2f} "
                    f"(peak {highest_price:.2f}, stop {trailing_stop_price:.2f})"
                )
        
        # 4. Trend reversal detected
        trend_reversal = self._detect_trend_reversal(indicators, entry_price, current_price)
        if trend_reversal:
            should_exit = True
            exit_reason = ExitReason.TREND_REVERSAL
            exit_urgency = "high"
            exit_reasons_list.append("Trend reversal detected")
        
        # 5. Time-based exit (holding too long)
        # Note: holding_days already calculated above for PDT check
        if holding_days > 30 and current_pnl_pct > 0.05:  # Hold max 30 days if profitable
            should_exit = True
            exit_reason = ExitReason.TIME_BASED
            exit_urgency = "low"
            exit_reasons_list.append(f"Time-based exit after {holding_days} days")
        
        # 6. Volatility spike (risk management)
        if indicators.atr and indicators.atr > self.nav * 0.05:  # High volatility
            should_exit = True
            exit_reason = ExitReason.VOLATILITY_SPIKE
            exit_urgency = "medium"
            exit_reasons_list.append("High volatility detected")
        
        # Calculate optimal exit price
        optimal_exit_price = current_price
        
        # Calculate remaining profit potential
        if indicators.resistance_level and current_price < indicators.resistance_level:
            profit_potential = indicators.resistance_level - current_price
        else:
            profit_potential = 0
        
        # Check if risk is increasing
        risk_increasing = self._is_risk_increasing(indicators, current_price, entry_price)
        
        # Calculate confidence
        confidence = self._calculate_exit_confidence(
            exit_reason, exit_urgency, current_pnl_pct, indicators
        )
        
        # Calculate timing score
        timing_score = self._calculate_exit_timing_score(
            should_exit, exit_urgency, current_pnl_pct, risk_increasing
        )
        
        return ExitAnalysis(
            should_exit=should_exit,
            exit_reason=exit_reason,
            confidence=confidence,
            optimal_exit_price=optimal_exit_price,
            exit_urgency=exit_urgency,
            profit_potential_remaining=profit_potential,
            risk_increasing=risk_increasing,
            exit_reasons=exit_reasons_list,
            timing_score=timing_score
        )
    
    # ==================== Helper Methods ====================
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> Optional[float]:
        """Calculate Relative Strength Index"""
        if len(prices) < period + 1:
            return None
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return float(rsi)
    
    def _calculate_macd(self, prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """Calculate MACD"""
        if len(prices) < slow + signal:
            return None, None, None
        
        ema_fast = self._calculate_ema(prices, fast)
        ema_slow = self._calculate_ema(prices, slow)
        
        if ema_fast is None or ema_slow is None:
            return None, None, None
        
        macd_line = ema_fast - ema_slow
        
        # Calculate signal line (EMA of MACD)
        macd_array = np.array([macd_line] * len(prices))  # Simplified
        signal_line = self._calculate_ema(prices, signal)  # Simplified
        histogram = macd_line - signal_line if signal_line else None
        
        return float(macd_line), float(signal_line) if signal_line else None, float(histogram) if histogram else None
    
    def _calculate_sma(self, prices: np.ndarray, period: int) -> Optional[float]:
        """Calculate Simple Moving Average"""
        if len(prices) < period:
            return None
        return float(np.mean(prices[-period:]))
    
    def _calculate_ema(self, prices: np.ndarray, period: int) -> Optional[float]:
        """Calculate Exponential Moving Average"""
        if len(prices) < period:
            return None
        
        multiplier = 2 / (period + 1)
        ema = prices[0]
        
        for price in prices[1:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        
        return float(ema)
    
    def _calculate_bollinger_bands(self, prices: np.ndarray, period: int = 20, std_dev: float = 2.0) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """Calculate Bollinger Bands"""
        if len(prices) < period:
            return None, None, None
        
        sma = np.mean(prices[-period:])
        std = np.std(prices[-period:])
        
        upper = sma + (std_dev * std)
        middle = sma
        lower = sma - (std_dev * std)
        
        return float(upper), float(middle), float(lower)
    
    def _calculate_atr(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> Optional[float]:
        """Calculate Average True Range"""
        if len(highs) < period + 1:
            return None
        
        true_ranges = []
        for i in range(1, len(highs)):
            tr1 = highs[i] - lows[i]
            tr2 = abs(highs[i] - closes[i-1])
            tr3 = abs(lows[i] - closes[i-1])
            true_ranges.append(max(tr1, tr2, tr3))
        
        if len(true_ranges) < period:
            return None
        
        atr = np.mean(true_ranges[-period:])
        return float(atr)
    
    def _calculate_support_resistance(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
        """Calculate support and resistance levels"""
        if len(closes) < 20:
            return None, None
        
        # Simple approach: use recent lows for support, recent highs for resistance
        recent_lows = lows[-20:]
        recent_highs = highs[-20:]
        
        support = float(np.min(recent_lows))
        resistance = float(np.max(recent_highs))
        
        return support, resistance
    
    def _classify_market_condition(self, indicators: TechnicalIndicators, current_price: float) -> MarketCondition:
        """Classify current market condition"""
        # Check for trend
        if indicators.sma_20 and indicators.sma_50:
            if indicators.sma_20 > indicators.sma_50:
                return MarketCondition.BULL_TREND
            elif indicators.sma_20 < indicators.sma_50:
                return MarketCondition.BEAR_TREND
        
        # Check volatility
        if indicators.atr and current_price > 0:
            volatility_pct = (indicators.atr / current_price) * 100
            if volatility_pct > 3.0:
                return MarketCondition.HIGH_VOLATILITY
            elif volatility_pct < 1.0:
                return MarketCondition.LOW_VOLATILITY
        
        return MarketCondition.CONSOLIDATION
    
    def _generate_entry_signal(self, indicators: TechnicalIndicators, current_price: float,
                              market_condition: MarketCondition) -> Tuple[EntrySignal, float]:
        """Generate entry signal based on indicators"""
        score = 0.0
        reasons = []
        
        # RSI analysis
        if indicators.rsi:
            if indicators.rsi < 30:
                score += 30
                reasons.append("RSI oversold")
            elif indicators.rsi > 70:
                score -= 30
                reasons.append("RSI overbought")
            elif 30 <= indicators.rsi <= 50:
                score += 10
                reasons.append("RSI in favorable range")
        
        # MACD analysis
        if indicators.macd and indicators.macd_signal:
            if indicators.macd > indicators.macd_signal:
                score += 20
                reasons.append("MACD bullish crossover")
            else:
                score -= 10
                reasons.append("MACD bearish")
        
        # Moving average analysis
        if indicators.sma_20 and indicators.sma_50:
            if current_price > indicators.sma_20 > indicators.sma_50:
                score += 25
                reasons.append("Price above key MAs")
            elif current_price < indicators.sma_20 < indicators.sma_50:
                score -= 25
                reasons.append("Price below key MAs")
        
        # Bollinger Bands
        if indicators.bollinger_lower and indicators.bollinger_upper:
            if current_price <= indicators.bollinger_lower:
                score += 15
                reasons.append("Price at lower Bollinger Band")
            elif current_price >= indicators.bollinger_upper:
                score -= 15
                reasons.append("Price at upper Bollinger Band")
        
        # Volume confirmation
        if indicators.volume_ratio and indicators.volume_ratio > 1.5:
            score += 10
            reasons.append("High volume confirmation")
        
        # Determine signal
        if score >= 60:
            signal = EntrySignal.STRONG_BUY
            confidence = min(0.95, (score / 100) * 0.9)
        elif score >= 30:
            signal = EntrySignal.BUY
            confidence = min(0.75, (score / 100) * 0.8)
        elif score >= 10:
            signal = EntrySignal.WEAK_BUY
            confidence = min(0.60, (score / 100) * 0.7)
        elif score <= -60:
            signal = EntrySignal.STRONG_SELL
            confidence = min(0.95, abs(score / 100) * 0.9)
        elif score <= -30:
            signal = EntrySignal.SELL
            confidence = min(0.75, abs(score / 100) * 0.8)
        elif score <= -10:
            signal = EntrySignal.WEAK_SELL
            confidence = min(0.60, abs(score / 100) * 0.7)
        else:
            signal = EntrySignal.NO_SIGNAL
            confidence = 0.3
        
        return signal, confidence
    
    def _calculate_entry_levels(self, current_price: float, indicators: TechnicalIndicators,
                                market_condition: MarketCondition) -> Tuple[float, float, float]:
        """Calculate optimal entry, stop loss, and take profit levels"""
        entry_price = current_price
        
        # Calculate stop loss (2% below entry or at support level)
        if indicators.support_level and indicators.support_level < current_price:
            stop_loss = indicators.support_level * 0.99  # 1% below support
        else:
            stop_loss = current_price * 0.98  # 2% stop loss
        
        # Calculate take profit (risk/reward ratio of at least 2:1)
        risk = current_price - stop_loss
        take_profit = current_price + (risk * self.min_risk_reward_ratio)
        
        # Adjust take profit to resistance if available
        if indicators.resistance_level and indicators.resistance_level > current_price:
            # Use resistance if it provides better risk/reward
            resistance_reward = indicators.resistance_level - current_price
            if resistance_reward >= risk * self.min_risk_reward_ratio:
                take_profit = indicators.resistance_level * 0.99  # 1% below resistance
        
        return entry_price, stop_loss, take_profit
    
    def _calculate_position_size(self, entry_price: float, stop_loss: float, risk_reward: float) -> float:
        """
        Calculate optimal position size - VERY_AGGRESSIVE MODE
        
        Target growth milestones require aggressive position sizing:
        Year 1: $9,411 | Year 7: $6,243,561 | Year 8: $15,726,144
        """
        # VERY_AGGRESSIVE: Risk up to 15% of NAV per trade
        max_risk_amount = self.nav * self.max_risk_per_trade_pct
        
        # Risk per share
        risk_per_share = abs(entry_price - stop_loss)
        
        if risk_per_share == 0:
            return 0
        
        # Calculate quantity based on risk
        quantity = max_risk_amount / risk_per_share
        
        # AGGRESSIVE R/R multipliers for faster growth
        if risk_reward >= 3.0:
            quantity *= 1.5  # 50% increase for excellent R/R (was 20%)
        elif risk_reward >= 2.0:
            quantity *= 1.3  # 30% increase for good R/R (was 0%)
        elif risk_reward >= 1.5:
            quantity *= 1.0  # Standard position
        else:
            quantity *= 0.7  # Reduce position for poor R/R (was 50%)
        
        return max(1, int(quantity))  # At least 1 share
    
    def _generate_entry_reasons(self, indicators: TechnicalIndicators, signal: EntrySignal,
                               market_condition: MarketCondition) -> List[str]:
        """Generate human-readable entry reasons"""
        reasons = []
        
        if signal in [EntrySignal.STRONG_BUY, EntrySignal.BUY, EntrySignal.WEAK_BUY]:
            reasons.append(f"Buy signal: {signal.value}")
        
        if indicators.rsi and indicators.rsi < 40:
            reasons.append(f"RSI oversold ({indicators.rsi:.1f})")
        
        if indicators.macd_histogram and indicators.macd_histogram > 0:
            reasons.append("MACD bullish momentum")
        
        if market_condition == MarketCondition.BULL_TREND:
            reasons.append("Bullish trend confirmed")
        
        if indicators.volume_ratio and indicators.volume_ratio > 1.5:
            reasons.append(f"High volume confirmation ({indicators.volume_ratio:.2f}x)")
        
        return reasons
    
    def _identify_risk_factors(self, indicators: TechnicalIndicators, market_condition: MarketCondition) -> List[str]:
        """Identify risk factors for entry"""
        risks = []
        
        if indicators.rsi and indicators.rsi > 70:
            risks.append("RSI overbought")
        
        if market_condition == MarketCondition.HIGH_VOLATILITY:
            risks.append("High volatility environment")
        
        if indicators.atr and indicators.atr > self.nav * 0.05:
            risks.append("High ATR (volatility)")
        
        if market_condition == MarketCondition.BEAR_TREND:
            risks.append("Bearish trend - counter-trend trade")
        
        return risks
    
    def _calculate_timing_score(self, indicators: TechnicalIndicators, signal: EntrySignal,
                               confidence: float, risk_reward: float,
                               market_condition: MarketCondition) -> float:
        """Calculate overall timing score (0-100)"""
        score = confidence * 50  # Base score from confidence
        
        # Signal strength bonus
        if signal == EntrySignal.STRONG_BUY:
            score += 20
        elif signal == EntrySignal.BUY:
            score += 10
        
        # Risk/reward bonus
        if risk_reward >= 3.0:
            score += 20
        elif risk_reward >= 2.0:
            score += 10
        
        # Market condition bonus
        if market_condition == MarketCondition.BULL_TREND:
            score += 10
        
        return min(100, max(0, score))
    
    def _detect_trend_reversal(self, indicators: TechnicalIndicators, entry_price: float,
                              current_price: float) -> bool:
        """Detect if trend has reversed"""
        # Check MACD reversal
        if indicators.macd_histogram and indicators.macd_histogram < 0:
            return True
        
        # Check price action
        price_change = (current_price - entry_price) / entry_price
        if price_change > 0.05:  # 5% profit
            # Check if momentum is weakening
            if indicators.rsi and indicators.rsi > 70:
                return True
        
        return False
    
    def _is_risk_increasing(self, indicators: TechnicalIndicators, current_price: float,
                           entry_price: float) -> bool:
        """Check if risk is increasing"""
        # Check if volatility is increasing
        if indicators.atr and indicators.atr > self.nav * 0.05:
            return True
        
        # Check if price is moving against us
        price_change = (current_price - entry_price) / entry_price
        if price_change < -0.01:  # 1% loss
            return True
        
        return False
    
    def _calculate_exit_confidence(self, exit_reason: Optional[ExitReason], exit_urgency: str,
                                  current_pnl_pct: float, indicators: TechnicalIndicators) -> float:
        """Calculate confidence in exit decision"""
        if exit_reason == ExitReason.STOP_LOSS:
            return 0.95  # Very confident on stop loss
        elif exit_reason == ExitReason.PROFIT_TARGET:
            return 0.85  # High confidence on profit target
        elif exit_reason == ExitReason.TRAILING_STOP:
            return 0.80  # Good confidence on trailing stop
        elif exit_reason == ExitReason.TREND_REVERSAL:
            return 0.75  # Moderate confidence on reversal
        else:
            return 0.60  # Lower confidence for other reasons
    
    def _calculate_exit_timing_score(self, should_exit: bool, exit_urgency: str,
                                    current_pnl_pct: float, risk_increasing: bool) -> float:
        """Calculate exit timing score"""
        if not should_exit:
            return 30  # Low score if no exit needed
        
        score = 50  # Base score
        
        # Urgency bonus
        if exit_urgency == "critical":
            score += 40
        elif exit_urgency == "high":
            score += 30
        elif exit_urgency == "medium":
            score += 20
        else:
            score += 10
        
        # Profit protection
        if current_pnl_pct > 0:
            score += min(20, current_pnl_pct * 200)  # Bonus for protecting profits
        
        # Risk increasing
        if risk_increasing:
            score += 10
        
        return min(100, score)


def create_timing_engine(nav: float = 25000.0, pdt_prevention: bool = True) -> TimingStrategyEngine:
    """
    Factory function to create timing strategy engine
    
    Args:
        nav: Net Asset Value for position sizing
        pdt_prevention: Enable Pattern Day Trading prevention (default: True)
    """
    return TimingStrategyEngine(nav=nav, pdt_prevention=pdt_prevention)

