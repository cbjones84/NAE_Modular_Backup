# NAE/tools/kalshi_learning_system.py
"""
Kalshi Learning System - Continuous Learning for Prediction Market Trading
==========================================================================

Implements active learning capabilities for the KalshiTrader agent:

1. PREDICTION FEEDBACK LOOP
   - Track predictions vs actual market outcomes
   - Calculate calibration scores (Brier score)
   - Identify category-specific accuracy patterns

2. STRATEGY OPTIMIZER (RL-based)
   - Optimize strategy selection based on market conditions
   - Adaptive position sizing using Kelly-like criterion
   - Multi-armed bandit for strategy exploration vs exploitation

3. META-LABELING
   - Secondary ML layer for trade signal filtering
   - Learn which signals are more reliable
   - Category-specific confidence adjustments

4. ONLINE LEARNING
   - Incremental model updates without forgetting
   - Replay buffer for experience retention
   - Drift detection for market regime changes

ALIGNED WITH NAE GOALS:
1. Achieve generational wealth
2. Generate $6,243,561+ within 8 years
3. Optimize prediction market returns through continuous learning
"""

import os
import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum
import math


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class PredictionRecord:
    """Record of a prediction for feedback tracking"""
    prediction_id: str
    market_ticker: str
    market_title: str
    category: str
    predicted_probability: float
    market_price: float  # Price at prediction time
    side: str  # "yes" or "no"
    strategy: str
    confidence: float
    timestamp: str
    resolved: bool = False
    actual_outcome: Optional[bool] = None  # True if YES won, False if NO won
    resolution_timestamp: Optional[str] = None
    pnl_cents: int = 0
    
    def brier_score(self) -> Optional[float]:
        """Calculate Brier score for this prediction"""
        if not self.resolved or self.actual_outcome is None:
            return None
        outcome = 1.0 if self.actual_outcome else 0.0
        return (self.predicted_probability - outcome) ** 2


@dataclass
class StrategyPerformance:
    """Performance metrics for a strategy"""
    strategy_name: str
    total_predictions: int = 0
    correct_predictions: int = 0
    total_pnl_cents: int = 0
    avg_confidence: float = 0.0
    avg_brier_score: float = 0.5
    win_rate: float = 0.0
    last_updated: str = ""
    
    def update_win_rate(self):
        if self.total_predictions > 0:
            self.win_rate = self.correct_predictions / self.total_predictions


@dataclass
class CategoryPerformance:
    """Performance metrics by Kalshi category"""
    category: str
    total_predictions: int = 0
    correct_predictions: int = 0
    avg_brier_score: float = 0.5
    calibration_score: float = 0.5
    recommended_confidence_adjustment: float = 1.0


@dataclass
class LearningState:
    """Current state of the learning system"""
    total_predictions: int = 0
    resolved_predictions: int = 0
    overall_brier_score: float = 0.5
    overall_calibration: float = 0.5
    strategy_performance: Dict[str, StrategyPerformance] = field(default_factory=dict)
    category_performance: Dict[str, CategoryPerformance] = field(default_factory=dict)
    last_training_time: Optional[str] = None
    model_version: int = 0


# =============================================================================
# PREDICTION FEEDBACK LOOP
# =============================================================================

class PredictionFeedbackLoop:
    """
    Tracks predictions and actual outcomes to improve forecasting
    
    Key metrics:
    - Brier Score: Mean squared error of probability predictions
    - Calibration: How well predicted probabilities match actual frequencies
    - Category accuracy: Performance breakdown by market category
    """
    
    def __init__(self, data_dir: str = "data/kalshi_learning"):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        
        self.predictions: List[PredictionRecord] = []
        self.pending_predictions: Dict[str, PredictionRecord] = {}  # ticker -> prediction
        
        # Performance tracking
        self.strategy_performance: Dict[str, StrategyPerformance] = {}
        self.category_performance: Dict[str, CategoryPerformance] = {}
        
        # Calibration bins (for reliability diagram)
        self.calibration_bins = defaultdict(lambda: {"count": 0, "correct": 0})
        
        # Load historical data
        self._load_data()
    
    def record_prediction(
        self,
        market_ticker: str,
        market_title: str,
        category: str,
        predicted_probability: float,
        market_price: float,
        side: str,
        strategy: str,
        confidence: float
    ) -> str:
        """
        Record a new prediction for tracking
        
        Returns:
            prediction_id for later resolution
        """
        prediction_id = f"pred_{market_ticker}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        record = PredictionRecord(
            prediction_id=prediction_id,
            market_ticker=market_ticker,
            market_title=market_title,
            category=category.lower(),
            predicted_probability=predicted_probability,
            market_price=market_price,
            side=side,
            strategy=strategy,
            confidence=confidence,
            timestamp=datetime.now().isoformat()
        )
        
        self.predictions.append(record)
        self.pending_predictions[market_ticker] = record
        
        self._save_data()
        
        return prediction_id
    
    def resolve_prediction(
        self,
        market_ticker: str,
        actual_outcome: bool,
        pnl_cents: int = 0
    ) -> Optional[Dict[str, Any]]:
        """
        Resolve a prediction with actual outcome
        
        Args:
            market_ticker: The market that resolved
            actual_outcome: True if YES won, False if NO won
            pnl_cents: Actual P&L in cents
            
        Returns:
            Resolution summary with learning insights
        """
        if market_ticker not in self.pending_predictions:
            # Try to find by ticker in recent predictions
            for pred in reversed(self.predictions[-100:]):
                if pred.market_ticker == market_ticker and not pred.resolved:
                    self.pending_predictions[market_ticker] = pred
                    break
        
        if market_ticker not in self.pending_predictions:
            return None
        
        record = self.pending_predictions.pop(market_ticker)
        record.resolved = True
        record.actual_outcome = actual_outcome
        record.resolution_timestamp = datetime.now().isoformat()
        record.pnl_cents = pnl_cents
        
        # Calculate if prediction was correct
        if record.side == "yes":
            was_correct = actual_outcome
        else:
            was_correct = not actual_outcome
        
        # Update strategy performance
        self._update_strategy_performance(record, was_correct)
        
        # Update category performance
        self._update_category_performance(record, was_correct)
        
        # Update calibration bins
        self._update_calibration(record)
        
        self._save_data()
        
        return {
            "prediction_id": record.prediction_id,
            "was_correct": was_correct,
            "predicted_probability": record.predicted_probability,
            "actual_outcome": actual_outcome,
            "brier_score": record.brier_score(),
            "pnl_cents": pnl_cents,
            "strategy": record.strategy,
            "category": record.category
        }
    
    def _update_strategy_performance(self, record: PredictionRecord, was_correct: bool):
        """Update performance metrics for strategy"""
        strategy = record.strategy
        
        if strategy not in self.strategy_performance:
            self.strategy_performance[strategy] = StrategyPerformance(strategy_name=strategy)
        
        perf = self.strategy_performance[strategy]
        perf.total_predictions += 1
        if was_correct:
            perf.correct_predictions += 1
        perf.total_pnl_cents += record.pnl_cents
        
        # Update running averages
        n = perf.total_predictions
        perf.avg_confidence = ((n - 1) * perf.avg_confidence + record.confidence) / n
        
        brier = record.brier_score()
        if brier is not None:
            perf.avg_brier_score = ((n - 1) * perf.avg_brier_score + brier) / n
        
        perf.update_win_rate()
        perf.last_updated = datetime.now().isoformat()
    
    def _update_category_performance(self, record: PredictionRecord, was_correct: bool):
        """Update performance metrics for category"""
        category = record.category
        
        if category not in self.category_performance:
            self.category_performance[category] = CategoryPerformance(category=category)
        
        perf = self.category_performance[category]
        perf.total_predictions += 1
        if was_correct:
            perf.correct_predictions += 1
        
        # Update Brier score
        n = perf.total_predictions
        brier = record.brier_score()
        if brier is not None:
            perf.avg_brier_score = ((n - 1) * perf.avg_brier_score + brier) / n
        
        # Calculate recommended confidence adjustment
        if perf.total_predictions >= 10:
            actual_win_rate = perf.correct_predictions / perf.total_predictions
            # If we're overconfident, reduce. If underconfident, increase
            if actual_win_rate > 0.05:  # Avoid division issues
                perf.recommended_confidence_adjustment = min(1.5, max(0.5, actual_win_rate / 0.5))
    
    def _update_calibration(self, record: PredictionRecord):
        """Update calibration bins for reliability diagram"""
        # Bin predictions by probability (0.1 intervals)
        prob = record.predicted_probability
        bin_idx = min(9, int(prob * 10))
        
        self.calibration_bins[bin_idx]["count"] += 1
        if record.actual_outcome == (record.side == "yes"):
            self.calibration_bins[bin_idx]["correct"] += 1
    
    def get_calibration_report(self) -> Dict[str, Any]:
        """
        Generate calibration report
        
        Good calibration: predicted probabilities match actual frequencies
        """
        report = {
            "bins": [],
            "overall_calibration_error": 0.0,
            "is_overconfident": False,
            "is_underconfident": False
        }
        
        total_error = 0.0
        total_count = 0
        
        for bin_idx in range(10):
            bin_data = self.calibration_bins[bin_idx]
            count = bin_data["count"]
            
            if count > 0:
                expected_prob = (bin_idx + 0.5) / 10
                actual_prob = bin_data["correct"] / count
                error = abs(expected_prob - actual_prob)
                
                report["bins"].append({
                    "range": f"{bin_idx/10:.1f}-{(bin_idx+1)/10:.1f}",
                    "count": count,
                    "expected": expected_prob,
                    "actual": actual_prob,
                    "error": error
                })
                
                total_error += error * count
                total_count += count
        
        if total_count > 0:
            report["overall_calibration_error"] = total_error / total_count
            
            # Check for systematic bias
            high_conf_bins = [b for b in report["bins"] if float(b["range"].split("-")[0]) >= 0.7]
            if high_conf_bins:
                avg_expected = np.mean([b["expected"] for b in high_conf_bins])
                avg_actual = np.mean([b["actual"] for b in high_conf_bins])
                report["is_overconfident"] = avg_actual < avg_expected - 0.1
                report["is_underconfident"] = avg_actual > avg_expected + 0.1
        
        return report
    
    def get_learning_recommendations(self) -> List[Dict[str, Any]]:
        """Generate recommendations based on learning data"""
        recommendations = []
        
        # Check calibration
        calibration = self.get_calibration_report()
        if calibration["is_overconfident"]:
            recommendations.append({
                "type": "calibration",
                "priority": "high",
                "message": "System is overconfident - reduce confidence on high-probability predictions",
                "adjustment": "Multiply confidence by 0.85 for predictions >70%"
            })
        elif calibration["is_underconfident"]:
            recommendations.append({
                "type": "calibration",
                "priority": "medium",
                "message": "System is underconfident - can increase position sizes",
                "adjustment": "Multiply confidence by 1.15 for high-confidence predictions"
            })
        
        # Check category performance
        for category, perf in self.category_performance.items():
            if perf.total_predictions >= 20:
                if perf.avg_brier_score > 0.3:
                    recommendations.append({
                        "type": "category",
                        "priority": "high",
                        "category": category,
                        "message": f"Poor performance in {category} markets (Brier: {perf.avg_brier_score:.3f})",
                        "adjustment": f"Reduce position size in {category} by 50%"
                    })
                elif perf.avg_brier_score < 0.15:
                    recommendations.append({
                        "type": "category",
                        "priority": "low",
                        "category": category,
                        "message": f"Excellent performance in {category} markets (Brier: {perf.avg_brier_score:.3f})",
                        "adjustment": f"Can increase position size in {category} by 25%"
                    })
        
        # Check strategy performance
        for strategy, perf in self.strategy_performance.items():
            if perf.total_predictions >= 20 and perf.win_rate < 0.4:
                recommendations.append({
                    "type": "strategy",
                    "priority": "high",
                    "strategy": strategy,
                    "message": f"Strategy {strategy} underperforming (win rate: {perf.win_rate:.1%})",
                    "adjustment": f"Review and potentially disable {strategy} strategy"
                })
        
        return recommendations
    
    def _load_data(self):
        """Load historical prediction data"""
        data_file = os.path.join(self.data_dir, "prediction_history.json")
        if os.path.exists(data_file):
            try:
                with open(data_file, 'r') as f:
                    data = json.load(f)
                    self.predictions = [PredictionRecord(**p) for p in data.get("predictions", [])]
                    
                    # Rebuild pending predictions
                    for pred in self.predictions:
                        if not pred.resolved:
                            self.pending_predictions[pred.market_ticker] = pred
                    
                    # Load performance data
                    for s, perf_data in data.get("strategy_performance", {}).items():
                        self.strategy_performance[s] = StrategyPerformance(**perf_data)
                    
                    for c, perf_data in data.get("category_performance", {}).items():
                        self.category_performance[c] = CategoryPerformance(**perf_data)
                    
            except Exception as e:
                print(f"Error loading prediction data: {e}")
    
    def _save_data(self):
        """Save prediction data"""
        data_file = os.path.join(self.data_dir, "prediction_history.json")
        try:
            data = {
                "predictions": [asdict(p) for p in self.predictions[-1000:]],  # Keep last 1000
                "strategy_performance": {k: asdict(v) for k, v in self.strategy_performance.items()},
                "category_performance": {k: asdict(v) for k, v in self.category_performance.items()},
                "last_updated": datetime.now().isoformat()
            }
            with open(data_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving prediction data: {e}")


# =============================================================================
# STRATEGY OPTIMIZER (Multi-Armed Bandit)
# =============================================================================

class StrategyOptimizer:
    """
    Multi-armed bandit for strategy selection
    
    Uses Thompson Sampling for exploration vs exploitation
    Learns which strategies work best for different market conditions
    """
    
    def __init__(self):
        # Strategy arms with Beta distributions (successes, failures)
        self.strategy_arms: Dict[str, Dict[str, float]] = {
            "bonding": {"alpha": 1.0, "beta": 1.0},  # Prior: uniform
            "arbitrage": {"alpha": 1.0, "beta": 1.0},
            "superforecast": {"alpha": 1.0, "beta": 1.0},
            "semantic": {"alpha": 1.0, "beta": 1.0},
            "event_driven": {"alpha": 1.0, "beta": 1.0}
        }
        
        # Category-specific arms
        self.category_arms: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(
            lambda: {s: {"alpha": 1.0, "beta": 1.0} for s in self.strategy_arms.keys()}
        )
        
        # Position sizing parameters
        self.position_sizing_params = {
            "base_kelly_fraction": 0.25,
            "max_position_pct": 0.10,
            "min_confidence_threshold": 0.55
        }
        
        self.data_file = "data/kalshi_learning/strategy_optimizer.json"
        self._load_state()
    
    def select_strategy(
        self,
        available_strategies: List[str],
        category: Optional[str] = None,
        explore: bool = True
    ) -> Tuple[str, float]:
        """
        Select strategy using Thompson Sampling
        
        Args:
            available_strategies: List of available strategy names
            category: Market category (for category-specific selection)
            explore: Whether to explore or exploit
            
        Returns:
            (strategy_name, expected_value)
        """
        if not available_strategies:
            return "bonding", 0.5
        
        # Get relevant arms
        if category:
            arms = self.category_arms[category.lower()]
        else:
            arms = self.strategy_arms
        
        if explore:
            # Thompson Sampling: sample from Beta distribution
            samples = {}
            for strategy in available_strategies:
                if strategy in arms:
                    alpha = arms[strategy]["alpha"]
                    beta = arms[strategy]["beta"]
                    samples[strategy] = np.random.beta(alpha, beta)
                else:
                    samples[strategy] = 0.5  # Unknown strategy
            
            best_strategy = max(samples.keys(), key=lambda s: samples[s])
            return best_strategy, samples[best_strategy]
        else:
            # Exploitation: use expected value
            expected_values = {}
            for strategy in available_strategies:
                if strategy in arms:
                    alpha = arms[strategy]["alpha"]
                    beta = arms[strategy]["beta"]
                    expected_values[strategy] = alpha / (alpha + beta)
                else:
                    expected_values[strategy] = 0.5
            
            best_strategy = max(expected_values.keys(), key=lambda s: expected_values[s])
            return best_strategy, expected_values[best_strategy]
    
    def update_strategy(
        self,
        strategy: str,
        was_successful: bool,
        category: Optional[str] = None
    ):
        """
        Update strategy arm based on outcome
        
        Args:
            strategy: Strategy that was used
            was_successful: Whether the trade was profitable
            category: Market category
        """
        # Update global arm
        if strategy in self.strategy_arms:
            if was_successful:
                self.strategy_arms[strategy]["alpha"] += 1
            else:
                self.strategy_arms[strategy]["beta"] += 1
        
        # Update category-specific arm
        if category:
            cat = category.lower()
            if strategy in self.category_arms[cat]:
                if was_successful:
                    self.category_arms[cat][strategy]["alpha"] += 1
                else:
                    self.category_arms[cat][strategy]["beta"] += 1
        
        self._save_state()
    
    def get_optimal_position_size(
        self,
        edge: float,
        confidence: float,
        win_probability: float,
        bankroll: float
    ) -> float:
        """
        Calculate optimal position size using modified Kelly criterion
        
        Args:
            edge: Expected edge (probability - price)
            confidence: Model confidence in the prediction
            win_probability: Estimated probability of winning
            bankroll: Current bankroll
            
        Returns:
            Optimal position size in dollars
        """
        if confidence < self.position_sizing_params["min_confidence_threshold"]:
            return 0.0
        
        if edge <= 0:
            return 0.0
        
        # Kelly formula: f = (bp - q) / b where b is odds, p is win prob, q is loss prob
        # For prediction markets: simplified as edge * confidence
        kelly_fraction = edge * confidence * win_probability
        
        # Apply Kelly fraction limit
        kelly_fraction = min(kelly_fraction, self.position_sizing_params["base_kelly_fraction"])
        
        # Apply max position limit
        max_position = bankroll * self.position_sizing_params["max_position_pct"]
        position_size = min(bankroll * kelly_fraction, max_position)
        
        return max(0, position_size)
    
    def get_strategy_rankings(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get strategy rankings by expected value"""
        if category:
            arms = self.category_arms[category.lower()]
        else:
            arms = self.strategy_arms
        
        rankings = []
        for strategy, params in arms.items():
            alpha = params["alpha"]
            beta = params["beta"]
            expected = alpha / (alpha + beta)
            total_trials = alpha + beta - 2  # Subtract prior
            
            rankings.append({
                "strategy": strategy,
                "expected_value": expected,
                "confidence_interval": self._beta_confidence_interval(alpha, beta),
                "total_trials": total_trials,
                "successes": alpha - 1,
                "failures": beta - 1
            })
        
        rankings.sort(key=lambda x: x["expected_value"], reverse=True)
        return rankings
    
    def _beta_confidence_interval(self, alpha: float, beta: float, confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval for Beta distribution"""
        from scipy import stats
        low = stats.beta.ppf((1 - confidence) / 2, alpha, beta)
        high = stats.beta.ppf(1 - (1 - confidence) / 2, alpha, beta)
        return (low, high)
    
    def _load_state(self):
        """Load optimizer state"""
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                    self.strategy_arms = data.get("strategy_arms", self.strategy_arms)
                    self.category_arms = defaultdict(
                        lambda: {s: {"alpha": 1.0, "beta": 1.0} for s in self.strategy_arms.keys()},
                        data.get("category_arms", {})
                    )
                    self.position_sizing_params = data.get("position_sizing_params", self.position_sizing_params)
            except Exception as e:
                print(f"Error loading optimizer state: {e}")
    
    def _save_state(self):
        """Save optimizer state"""
        os.makedirs(os.path.dirname(self.data_file), exist_ok=True)
        try:
            data = {
                "strategy_arms": self.strategy_arms,
                "category_arms": dict(self.category_arms),
                "position_sizing_params": self.position_sizing_params,
                "last_updated": datetime.now().isoformat()
            }
            with open(self.data_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving optimizer state: {e}")


# =============================================================================
# KALSHI META-LABELING
# =============================================================================

class KalshiMetaLabeler:
    """
    Meta-labeling specifically for Kalshi prediction markets
    
    Learns which trade signals are more reliable based on:
    - Market characteristics (volume, spread, time to close)
    - Category
    - Strategy
    - Historical performance
    """
    
    def __init__(self):
        self.training_data: List[Dict[str, Any]] = []
        self.is_trained = False
        self.feature_weights: Dict[str, float] = {
            "volume_score": 0.15,
            "spread_score": 0.10,
            "days_to_close_score": 0.10,
            "category_score": 0.20,
            "strategy_score": 0.20,
            "confidence_score": 0.15,
            "edge_score": 0.10
        }
        
        # Category historical performance
        self.category_scores: Dict[str, float] = {
            "economics": 0.7,
            "politics": 0.6,
            "weather": 0.5,
            "finance": 0.65,
            "science": 0.4,
            "entertainment": 0.5
        }
        
        # Strategy historical performance
        self.strategy_scores: Dict[str, float] = {
            "bonding": 0.8,
            "arbitrage": 0.75,
            "superforecast": 0.6,
            "semantic": 0.55,
            "event_driven": 0.65
        }
        
        self.data_file = "data/kalshi_learning/meta_labeler.json"
        self._load_state()
    
    def score_signal(
        self,
        market_ticker: str,
        category: str,
        strategy: str,
        edge: float,
        confidence: float,
        volume: int,
        spread: float,
        days_to_close: int
    ) -> Tuple[float, Dict[str, float]]:
        """
        Score a trade signal using meta-labeling
        
        Returns:
            (final_score, component_scores)
        """
        component_scores = {}
        
        # Volume score (higher is better, normalized)
        component_scores["volume_score"] = min(1.0, volume / 10000)
        
        # Spread score (lower spread is better)
        component_scores["spread_score"] = max(0.0, 1.0 - spread * 10)
        
        # Days to close score (prefer 3-14 days)
        if 3 <= days_to_close <= 14:
            component_scores["days_to_close_score"] = 1.0
        elif days_to_close < 3:
            component_scores["days_to_close_score"] = 0.5
        else:
            component_scores["days_to_close_score"] = max(0.3, 1.0 - (days_to_close - 14) / 30)
        
        # Category score
        component_scores["category_score"] = self.category_scores.get(category.lower(), 0.5)
        
        # Strategy score
        component_scores["strategy_score"] = self.strategy_scores.get(strategy.lower(), 0.5)
        
        # Confidence score
        component_scores["confidence_score"] = confidence
        
        # Edge score (higher edge is better, but extreme edges might be wrong)
        if 0.05 <= edge <= 0.30:
            component_scores["edge_score"] = 1.0
        elif edge > 0.30:
            component_scores["edge_score"] = 0.7  # Suspiciously high edge
        else:
            component_scores["edge_score"] = edge / 0.05
        
        # Calculate weighted final score
        final_score = sum(
            self.feature_weights[k] * component_scores[k]
            for k in self.feature_weights.keys()
        )
        
        return final_score, component_scores
    
    def should_trade(
        self,
        signal_score: float,
        threshold: float = 0.55
    ) -> Tuple[bool, str]:
        """
        Determine if a signal should be traded
        
        Returns:
            (should_trade, reason)
        """
        if signal_score >= 0.75:
            return True, "High confidence signal"
        elif signal_score >= threshold:
            return True, "Signal meets threshold"
        elif signal_score >= 0.45:
            return False, "Signal below threshold but close - consider smaller position"
        else:
            return False, "Signal too weak - skip trade"
    
    def record_outcome(
        self,
        category: str,
        strategy: str,
        was_profitable: bool,
        signal_score: float
    ):
        """Record outcome for learning"""
        self.training_data.append({
            "category": category,
            "strategy": strategy,
            "was_profitable": was_profitable,
            "signal_score": signal_score,
            "timestamp": datetime.now().isoformat()
        })
        
        # Update category and strategy scores with exponential moving average
        alpha = 0.1  # Learning rate
        
        outcome_value = 1.0 if was_profitable else 0.0
        
        if category.lower() in self.category_scores:
            current = self.category_scores[category.lower()]
            self.category_scores[category.lower()] = (1 - alpha) * current + alpha * outcome_value
        
        if strategy.lower() in self.strategy_scores:
            current = self.strategy_scores[strategy.lower()]
            self.strategy_scores[strategy.lower()] = (1 - alpha) * current + alpha * outcome_value
        
        self._save_state()
    
    def _load_state(self):
        """Load meta-labeler state"""
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                    self.category_scores = data.get("category_scores", self.category_scores)
                    self.strategy_scores = data.get("strategy_scores", self.strategy_scores)
                    self.training_data = data.get("training_data", [])[-500:]  # Keep last 500
            except Exception as e:
                print(f"Error loading meta-labeler state: {e}")
    
    def _save_state(self):
        """Save meta-labeler state"""
        os.makedirs(os.path.dirname(self.data_file), exist_ok=True)
        try:
            data = {
                "category_scores": self.category_scores,
                "strategy_scores": self.strategy_scores,
                "training_data": self.training_data[-500:],
                "last_updated": datetime.now().isoformat()
            }
            with open(self.data_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving meta-labeler state: {e}")


# =============================================================================
# MAIN LEARNING SYSTEM
# =============================================================================

class KalshiLearningSystem:
    """
    Unified learning system for KalshiTrader
    
    Integrates:
    - Prediction feedback loop
    - Strategy optimizer
    - Meta-labeling
    """
    
    def __init__(self, data_dir: str = "data/kalshi_learning"):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Initialize components
        self.feedback_loop = PredictionFeedbackLoop(data_dir)
        self.strategy_optimizer = StrategyOptimizer()
        self.meta_labeler = KalshiMetaLabeler()
        
        # Learning state
        self.is_learning_enabled = True
        self.learning_mode = "active"  # "active", "passive", "evaluation"
        
        # Metrics
        self.total_predictions = 0
        self.total_resolved = 0
        self.learning_iterations = 0
        
        print(f"[KalshiLearning] Learning system initialized")
        print(f"   Data directory: {data_dir}")
        print(f"   Mode: {self.learning_mode}")
    
    def record_trade_signal(
        self,
        market_ticker: str,
        market_title: str,
        category: str,
        predicted_probability: float,
        market_price: float,
        side: str,
        strategy: str,
        confidence: float,
        volume: int = 0,
        spread: float = 0.0,
        days_to_close: int = 7
    ) -> Dict[str, Any]:
        """
        Record a trade signal and get meta-labeling assessment
        
        Returns:
            Dict with prediction_id, meta_score, should_trade, recommendations
        """
        # Record prediction for feedback
        prediction_id = self.feedback_loop.record_prediction(
            market_ticker=market_ticker,
            market_title=market_title,
            category=category,
            predicted_probability=predicted_probability,
            market_price=market_price,
            side=side,
            strategy=strategy,
            confidence=confidence
        )
        
        # Get meta-labeling score
        edge = abs(predicted_probability - market_price)
        meta_score, component_scores = self.meta_labeler.score_signal(
            market_ticker=market_ticker,
            category=category,
            strategy=strategy,
            edge=edge,
            confidence=confidence,
            volume=volume,
            spread=spread,
            days_to_close=days_to_close
        )
        
        # Get trade recommendation
        should_trade, reason = self.meta_labeler.should_trade(meta_score)
        
        # Get optimal position size if trading
        position_size = 0.0
        if should_trade:
            position_size = self.strategy_optimizer.get_optimal_position_size(
                edge=edge,
                confidence=confidence,
                win_probability=predicted_probability if side == "yes" else (1 - predicted_probability),
                bankroll=10000  # This should come from actual bankroll
            )
        
        self.total_predictions += 1
        
        return {
            "prediction_id": prediction_id,
            "meta_score": meta_score,
            "component_scores": component_scores,
            "should_trade": should_trade,
            "trade_reason": reason,
            "recommended_position_size": position_size,
            "confidence_adjusted": confidence * self.meta_labeler.strategy_scores.get(strategy.lower(), 1.0)
        }
    
    def record_trade_outcome(
        self,
        market_ticker: str,
        actual_outcome: bool,
        pnl_cents: int,
        strategy: str,
        category: str
    ) -> Dict[str, Any]:
        """
        Record trade outcome and update all learning components
        
        Returns:
            Dict with resolution details and learning updates
        """
        # Resolve prediction
        resolution = self.feedback_loop.resolve_prediction(
            market_ticker=market_ticker,
            actual_outcome=actual_outcome,
            pnl_cents=pnl_cents
        )
        
        if resolution:
            was_successful = pnl_cents > 0
            
            # Update strategy optimizer
            self.strategy_optimizer.update_strategy(
                strategy=strategy,
                was_successful=was_successful,
                category=category
            )
            
            # Update meta-labeler
            self.meta_labeler.record_outcome(
                category=category,
                strategy=strategy,
                was_profitable=was_successful,
                signal_score=0.5  # Would be tracked from signal time
            )
            
            self.total_resolved += 1
            self.learning_iterations += 1
        
        return resolution or {"status": "not_found"}
    
    def select_best_strategy(
        self,
        available_strategies: List[str],
        category: Optional[str] = None,
        explore: bool = True
    ) -> Tuple[str, float]:
        """
        Select the best strategy using learned data
        
        Returns:
            (strategy_name, expected_value)
        """
        return self.strategy_optimizer.select_strategy(
            available_strategies=available_strategies,
            category=category,
            explore=explore
        )
    
    def get_learning_report(self) -> Dict[str, Any]:
        """Generate comprehensive learning report"""
        calibration = self.feedback_loop.get_calibration_report()
        recommendations = self.feedback_loop.get_learning_recommendations()
        strategy_rankings = self.strategy_optimizer.get_strategy_rankings()
        
        return {
            "summary": {
                "total_predictions": self.total_predictions,
                "total_resolved": self.total_resolved,
                "learning_iterations": self.learning_iterations,
                "learning_mode": self.learning_mode
            },
            "calibration": calibration,
            "recommendations": recommendations,
            "strategy_rankings": strategy_rankings,
            "category_scores": self.meta_labeler.category_scores,
            "strategy_scores": self.meta_labeler.strategy_scores,
            "generated_at": datetime.now().isoformat()
        }
    
    def get_confidence_adjustment(self, category: str, strategy: str) -> float:
        """
        Get confidence adjustment factor based on learning
        
        Returns multiplier to apply to raw confidence
        """
        category_adj = self.meta_labeler.category_scores.get(category.lower(), 0.5) / 0.5
        strategy_adj = self.meta_labeler.strategy_scores.get(strategy.lower(), 0.5) / 0.5
        
        # Blend adjustments
        return (category_adj * 0.6 + strategy_adj * 0.4)


# =============================================================================
# SINGLETON ACCESSOR
# =============================================================================

_learning_system_instance: Optional[KalshiLearningSystem] = None

def get_kalshi_learning_system() -> KalshiLearningSystem:
    """Get global Kalshi learning system instance"""
    global _learning_system_instance
    if _learning_system_instance is None:
        _learning_system_instance = KalshiLearningSystem()
    return _learning_system_instance


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("KALSHI LEARNING SYSTEM TEST")
    print("="*70)
    
    # Initialize
    learning = KalshiLearningSystem()
    
    # Simulate some predictions
    print("\n=== Recording Predictions ===")
    
    result1 = learning.record_trade_signal(
        market_ticker="KXFED-2024-RATE",
        market_title="Fed Rate Decision",
        category="economics",
        predicted_probability=0.85,
        market_price=0.80,
        side="yes",
        strategy="bonding",
        confidence=0.75,
        volume=5000,
        spread=0.02,
        days_to_close=7
    )
    print(f"Signal 1: {result1}")
    
    result2 = learning.record_trade_signal(
        market_ticker="KXBTC-50K",
        market_title="Bitcoin above 50K",
        category="finance",
        predicted_probability=0.65,
        market_price=0.55,
        side="yes",
        strategy="superforecast",
        confidence=0.60,
        volume=2000,
        spread=0.05,
        days_to_close=14
    )
    print(f"Signal 2: {result2}")
    
    # Simulate outcomes
    print("\n=== Recording Outcomes ===")
    
    outcome1 = learning.record_trade_outcome(
        market_ticker="KXFED-2024-RATE",
        actual_outcome=True,
        pnl_cents=500,
        strategy="bonding",
        category="economics"
    )
    print(f"Outcome 1: {outcome1}")
    
    # Get learning report
    print("\n=== Learning Report ===")
    report = learning.get_learning_report()
    print(json.dumps(report, indent=2, default=str))
    
    # Strategy selection
    print("\n=== Strategy Selection ===")
    best_strategy, expected = learning.select_best_strategy(
        available_strategies=["bonding", "arbitrage", "superforecast"],
        category="economics"
    )
    print(f"Best strategy for economics: {best_strategy} (expected: {expected:.2f})")
    
    print("\n✅ Learning system test complete!")

