#!/usr/bin/env python3
"""
Lightweight ML Suite for NAE — Replaces TensorFlow/PyTorch

Total RAM footprint: ~60 MB (vs 4+ GB for TensorFlow + PyTorch)
Trains in seconds on 2 CPU cores.

Components:
  1. LightGBMPredictor  — Price direction prediction (replaces LSTM)
  2. ContextualBandit    — Online strategy selection (replaces RL agent)
  3. OnlineMetaLabeler   — Trade quality filter (replaces RandomForest meta-labeling)
  4. WeekendRetrainer    — Batch retrain on full history (Saturday cron)
"""

import os
import json
import math
import time
import pickle
import logging
import datetime
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Check for LightGBM (pip install lightgbm — ~2 MB)
# ---------------------------------------------------------------------------
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except Exception:
    lgb = None
    LIGHTGBM_AVAILABLE = False
    logger.warning("LightGBM not available. Run: pip install lightgbm && brew install libomp")


# ===================================================================
# 1. LightGBM Price Direction Predictor (replaces LSTM)
# ===================================================================
class LightGBMPredictor:
    """
    Predicts next-day price direction using gradient-boosted decision trees.

    Why LightGBM > LSTM for this task:
      - Trains in <1 second on 60 days of data (LSTM: 30+ seconds)
      - ~50 MB RAM (LSTM/TensorFlow: 2+ GB)
      - Handles tabular features natively (LSTM needs sequence reshaping)
      - Equal or better accuracy on financial time series

    Features extracted per day:
      - Returns (1d, 3d, 5d, 10d, 20d)
      - Volatility (5d, 10d, 20d rolling std)
      - RSI (14-period)
      - Volume ratio (vs 20d average)
      - Distance from rolling highs/lows
      - Day of week
    """

    def __init__(self, model_dir: str = "logs/ml_models"):
        self.model = None
        self.is_trained = False
        self.model_dir = model_dir
        self.model_file = os.path.join(model_dir, "lgbm_price_direction.pkl")
        self.feature_names: List[str] = []
        self.train_history: Dict[str, Any] = {}

        os.makedirs(model_dir, exist_ok=True)
        self._load_model()

    def _extract_features(self, prices: List[float], volumes: Optional[List[float]] = None) -> np.ndarray:
        """
        Extract technical features from price series.
        Returns a 2D array: (num_samples, num_features)
        """
        prices = np.array(prices, dtype=np.float64)
        n = len(prices)
        if n < 25:
            return np.array([])

        # Default volumes if not provided
        if volumes is None:
            volumes = np.ones(n)
        else:
            volumes = np.array(volumes, dtype=np.float64)

        features_list = []
        self.feature_names = [
            "ret_1d", "ret_3d", "ret_5d", "ret_10d", "ret_20d",
            "vol_5d", "vol_10d", "vol_20d",
            "rsi_14",
            "vol_ratio",
            "dist_high_10d", "dist_low_10d",
            "dist_high_20d", "dist_low_20d",
            "day_of_week",
        ]

        for i in range(20, n):  # Need 20 lookback minimum
            row = []

            # Returns
            row.append((prices[i] - prices[i-1]) / prices[i-1] if prices[i-1] > 0 else 0)
            row.append((prices[i] - prices[i-3]) / prices[i-3] if prices[i-3] > 0 else 0)
            row.append((prices[i] - prices[i-5]) / prices[i-5] if prices[i-5] > 0 else 0)
            row.append((prices[i] - prices[i-10]) / prices[i-10] if i >= 10 and prices[i-10] > 0 else 0)
            row.append((prices[i] - prices[i-20]) / prices[i-20] if prices[i-20] > 0 else 0)

            # Rolling volatility (std of returns)
            ret_window = np.diff(prices[max(0, i-5):i+1]) / prices[max(0, i-5):i][:-1] if i >= 5 else [0]
            row.append(np.std(ret_window) if len(ret_window) > 1 else 0)
            ret_window = np.diff(prices[max(0, i-10):i+1]) / prices[max(0, i-10):i][:-1] if i >= 10 else [0]
            row.append(np.std(ret_window) if len(ret_window) > 1 else 0)
            ret_window = np.diff(prices[max(0, i-20):i+1]) / prices[max(0, i-20):i][:-1] if i >= 20 else [0]
            row.append(np.std(ret_window) if len(ret_window) > 1 else 0)

            # RSI (14-period)
            if i >= 14:
                gains, losses = [], []
                for j in range(i-13, i+1):
                    change = prices[j] - prices[j-1]
                    if change > 0:
                        gains.append(change)
                        losses.append(0)
                    else:
                        gains.append(0)
                        losses.append(abs(change))
                avg_gain = np.mean(gains) if gains else 0
                avg_loss = np.mean(losses) if losses else 0
                rs = avg_gain / avg_loss if avg_loss > 0 else 100
                rsi = 100 - (100 / (1 + rs))
                row.append(rsi / 100.0)  # Normalize to 0-1
            else:
                row.append(0.5)

            # Volume ratio
            avg_vol = np.mean(volumes[max(0, i-20):i]) if i >= 20 else np.mean(volumes[:i]) if i > 0 else 1
            row.append(volumes[i] / avg_vol if avg_vol > 0 else 1.0)

            # Distance from rolling highs/lows
            high_10 = np.max(prices[max(0, i-10):i+1])
            low_10 = np.min(prices[max(0, i-10):i+1])
            high_20 = np.max(prices[max(0, i-20):i+1])
            low_20 = np.min(prices[max(0, i-20):i+1])
            row.append((prices[i] - high_10) / high_10 if high_10 > 0 else 0)
            row.append((prices[i] - low_10) / low_10 if low_10 > 0 else 0)
            row.append((prices[i] - high_20) / high_20 if high_20 > 0 else 0)
            row.append((prices[i] - low_20) / low_20 if low_20 > 0 else 0)

            # Day of week (0=Mon, 4=Fri) — normalized
            row.append(datetime.date.today().weekday() / 4.0)

            features_list.append(row)

        return np.array(features_list) if features_list else np.array([])

    def train(self, prices: List[float], volumes: Optional[List[float]] = None,
              look_forward: int = 1) -> Tuple[bool, Dict[str, Any]]:
        """
        Train the LightGBM model on historical price data.

        Args:
            prices: List of daily closing prices (at least 60)
            volumes: Optional list of daily volumes
            look_forward: Days ahead to predict (default 1 = next day)

        Returns:
            (success, result_dict)
        """
        if not LIGHTGBM_AVAILABLE:
            return False, {"error": "LightGBM not installed. Run: pip install lightgbm"}

        if len(prices) < 30:
            return False, {"error": f"Need at least 30 prices, got {len(prices)}"}

        try:
            X = self._extract_features(prices, volumes)
            if len(X) < 10:
                return False, {"error": "Not enough feature samples after extraction"}

            # Target: 1 if price goes up in look_forward days, 0 otherwise
            prices_arr = np.array(prices)
            targets = []
            offset = 20  # Feature extraction starts at index 20
            for i in range(offset, len(prices_arr)):
                if i + look_forward < len(prices_arr):
                    target = 1 if prices_arr[i + look_forward] > prices_arr[i] else 0
                    targets.append(target)

            # Align X and y lengths
            min_len = min(len(X), len(targets))
            X = X[:min_len]
            y = np.array(targets[:min_len])

            if len(X) < 10:
                return False, {"error": "Not enough aligned samples"}

            # Train/validation split (80/20)
            split = int(len(X) * 0.8)
            X_train, X_val = X[:split], X[split:]
            y_train, y_val = y[:split], y[split:]

            train_data = lgb.Dataset(X_train, label=y_train, feature_name=self.feature_names)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

            params = {
                "objective": "binary",
                "metric": "binary_logloss",
                "num_leaves": 15,        # Small tree for small data
                "learning_rate": 0.05,
                "min_data_in_leaf": 3,
                "feature_fraction": 0.8,
                "bagging_fraction": 0.8,
                "bagging_freq": 5,
                "verbose": -1,
            }

            callbacks = [lgb.log_evaluation(period=0)]  # Suppress output
            self.model = lgb.train(
                params,
                train_data,
                num_boost_round=100,
                valid_sets=[val_data],
                callbacks=callbacks,
            )

            # Evaluate
            val_pred = (self.model.predict(X_val) > 0.5).astype(int)
            accuracy = np.mean(val_pred == y_val) if len(y_val) > 0 else 0
            up_ratio = np.mean(y) if len(y) > 0 else 0.5

            self.is_trained = True
            self.train_history = {
                "timestamp": datetime.datetime.now().isoformat(),
                "samples": len(X),
                "accuracy": float(accuracy),
                "up_ratio": float(up_ratio),
                "features": len(self.feature_names),
            }

            self._save_model()

            logger.info(f"LightGBM trained: {len(X)} samples, accuracy={accuracy:.1%}, "
                       f"up_ratio={up_ratio:.1%}")
            return True, self.train_history

        except Exception as e:
            logger.error(f"LightGBM training error: {e}")
            return False, {"error": str(e)}

    def predict(self, prices: List[float], volumes: Optional[List[float]] = None) -> Optional[Dict[str, Any]]:
        """
        Predict next-day price direction.

        Returns:
            Dict with 'direction' (1=up, 0=down), 'confidence' (0-1),
            'predicted_price' (estimated), or None if unable
        """
        if not self.is_trained or self.model is None:
            return self._fallback_predict(prices)

        try:
            X = self._extract_features(prices, volumes)
            if len(X) == 0:
                return self._fallback_predict(prices)

            # Use last row (most recent data point)
            latest = X[-1:, :]
            prob_up = float(self.model.predict(latest)[0])

            direction = 1 if prob_up > 0.5 else 0
            confidence = abs(prob_up - 0.5) * 2  # Scale 0.5-1.0 to 0-1

            # Estimate price move based on recent volatility
            recent_returns = np.diff(prices[-10:]) / np.array(prices[-10:])[:-1]
            avg_move = np.mean(np.abs(recent_returns)) if len(recent_returns) > 0 else 0.01
            predicted_change = avg_move if direction == 1 else -avg_move
            predicted_price = prices[-1] * (1 + predicted_change)

            return {
                "direction": direction,
                "direction_label": "UP" if direction == 1 else "DOWN",
                "confidence": float(confidence),
                "probability_up": float(prob_up),
                "predicted_price": float(predicted_price),
                "predicted_change_pct": float(predicted_change * 100),
                "model": "lightgbm",
            }

        except Exception as e:
            logger.debug(f"LightGBM prediction error: {e}")
            return self._fallback_predict(prices)

    @staticmethod
    def _fallback_predict(prices: List[float]) -> Optional[Dict[str, Any]]:
        """Simple momentum-based fallback when model isn't trained."""
        if len(prices) < 5:
            return None
        ret_5d = (prices[-1] - prices[-5]) / prices[-5] if prices[-5] > 0 else 0
        direction = 1 if ret_5d > 0 else 0
        return {
            "direction": direction,
            "direction_label": "UP" if direction == 1 else "DOWN",
            "confidence": min(abs(ret_5d) * 10, 0.6),  # Low confidence for fallback
            "probability_up": 0.5 + (ret_5d * 5),
            "predicted_price": prices[-1] * (1 + ret_5d * 0.2),
            "predicted_change_pct": ret_5d * 20,
            "model": "momentum_fallback",
        }

    def _save_model(self):
        try:
            with open(self.model_file, "wb") as f:
                pickle.dump({
                    "model": self.model,
                    "feature_names": self.feature_names,
                    "train_history": self.train_history,
                }, f)
            logger.info(f"LightGBM model saved to {self.model_file}")
        except Exception as e:
            logger.warning(f"Could not save LightGBM model: {e}")

    def _load_model(self):
        try:
            if os.path.exists(self.model_file):
                with open(self.model_file, "rb") as f:
                    data = pickle.load(f)
                self.model = data.get("model")
                self.feature_names = data.get("feature_names", [])
                self.train_history = data.get("train_history", {})
                self.is_trained = self.model is not None
                if self.is_trained:
                    logger.info("LightGBM model loaded from disk")
        except Exception as e:
            logger.debug(f"Could not load LightGBM model: {e}")

    def load_model(self):
        """Public alias for compatibility with LSTM interface."""
        self._load_model()


# ===================================================================
# 2. Contextual Bandit for Strategy Selection (replaces RL Agent)
# ===================================================================
class ContextualBandit:
    """
    Online contextual bandit for strategy selection.

    Instead of a neural-net RL agent (~1.5 GB PyTorch), this uses
    Thompson Sampling with Bayesian updates — ~5 MB RAM, learns per trade.

    How it works:
      - Each strategy has a Beta distribution (alpha, beta) tracking wins/losses
      - Context features (market regime, volatility, trend) shift the priors
      - Thompson Sampling: sample from each Beta, pick highest
      - Updates immediately after each trade result (online learning)

    Decay: Older observations fade via a decay factor, so the bandit
    adapts to regime changes automatically.
    """

    def __init__(self, strategies: Optional[List[str]] = None, decay: float = 0.995,
                 state_file: str = "logs/ml_models/bandit_state.json"):
        self.decay = decay
        self.state_file = state_file

        # Default strategies NAE uses
        self.strategies = strategies or [
            "momentum_scalp", "volatility_breakout", "mean_reversion",
            "gap_trading", "news_trading", "day_trading",
            "options_call", "options_put",
        ]

        # Beta distribution parameters per strategy per context
        # Key: strategy_name -> {"alpha": float, "beta": float}
        self.arms: Dict[str, Dict[str, float]] = {}

        # Context-conditioned arms: (strategy, context_bucket) -> (alpha, beta)
        self.context_arms: Dict[str, Dict[str, float]] = {}

        # Track stats
        self.total_selections = 0
        self.total_rewards = 0.0

        self._init_arms()
        self._load_state()

    def _init_arms(self):
        """Initialize Beta priors for all strategies."""
        for s in self.strategies:
            if s not in self.arms:
                self.arms[s] = {"alpha": 2.0, "beta": 2.0}  # Weak prior (50/50)

    def _get_context_key(self, context: Dict[str, Any]) -> str:
        """Discretize context into a bucket string."""
        trend = context.get("trend", "neutral")  # bullish/bearish/neutral
        vol = context.get("volatility", "medium")  # low/medium/high
        return f"{trend}_{vol}"

    def select_strategy(self, context: Optional[Dict[str, Any]] = None) -> Tuple[str, float]:
        """
        Select the best strategy given current market context.

        Args:
            context: {"trend": "bullish"|"bearish"|"neutral",
                      "volatility": "low"|"medium"|"high"}

        Returns:
            (strategy_name, expected_value)
        """
        samples = {}
        ctx_key = self._get_context_key(context or {})

        for strategy in self.strategies:
            # Blend global arm with context-specific arm
            global_arm = self.arms.get(strategy, {"alpha": 2.0, "beta": 2.0})
            ctx_arm_key = f"{strategy}_{ctx_key}"
            ctx_arm = self.context_arms.get(ctx_arm_key, {"alpha": 1.0, "beta": 1.0})

            # Weighted blend: 60% context, 40% global
            alpha = 0.4 * global_arm["alpha"] + 0.6 * ctx_arm["alpha"]
            beta = 0.4 * global_arm["beta"] + 0.6 * ctx_arm["beta"]

            # Thompson sample
            sample = np.random.beta(max(alpha, 0.1), max(beta, 0.1))
            samples[strategy] = sample

        # Pick highest sample
        best = max(samples, key=samples.get)
        self.total_selections += 1

        logger.debug(f"Bandit selected: {best} (score={samples[best]:.3f})")
        return best, float(samples[best])

    def update(self, strategy: str, reward: float, context: Optional[Dict[str, Any]] = None):
        """
        Update the bandit after observing a trade result.

        Args:
            strategy: The strategy that was used
            reward: 1.0 for profit, 0.0 for loss (or continuous 0-1)
            context: Market context when trade was made
        """
        # Apply decay to all arms (forget old data gradually)
        for s in self.arms:
            self.arms[s]["alpha"] *= self.decay
            self.arms[s]["beta"] *= self.decay
            # Floor to prevent collapse
            self.arms[s]["alpha"] = max(self.arms[s]["alpha"], 0.5)
            self.arms[s]["beta"] = max(self.arms[s]["beta"], 0.5)

        # Update global arm
        if strategy not in self.arms:
            self.arms[strategy] = {"alpha": 2.0, "beta": 2.0}
        self.arms[strategy]["alpha"] += reward
        self.arms[strategy]["beta"] += (1.0 - reward)

        # Update context arm
        if context:
            ctx_key = f"{strategy}_{self._get_context_key(context)}"
            if ctx_key not in self.context_arms:
                self.context_arms[ctx_key] = {"alpha": 1.0, "beta": 1.0}
            self.context_arms[ctx_key]["alpha"] += reward
            self.context_arms[ctx_key]["beta"] += (1.0 - reward)

        self.total_rewards += reward
        self._save_state()

    def get_strategy_rankings(self) -> List[Dict[str, Any]]:
        """Get all strategies ranked by expected reward."""
        rankings = []
        for s, arm in self.arms.items():
            expected = arm["alpha"] / (arm["alpha"] + arm["beta"])
            rankings.append({
                "strategy": s,
                "expected_reward": float(expected),
                "alpha": float(arm["alpha"]),
                "beta": float(arm["beta"]),
                "observations": float(arm["alpha"] + arm["beta"] - 4),  # Subtract prior
            })
        rankings.sort(key=lambda x: x["expected_reward"], reverse=True)
        return rankings

    def _save_state(self):
        try:
            os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
            with open(self.state_file, "w") as f:
                json.dump({
                    "arms": self.arms,
                    "context_arms": self.context_arms,
                    "total_selections": self.total_selections,
                    "total_rewards": self.total_rewards,
                    "timestamp": datetime.datetime.now().isoformat(),
                }, f, indent=2)
        except Exception as e:
            logger.debug(f"Could not save bandit state: {e}")

    def _load_state(self):
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, "r") as f:
                    data = json.load(f)
                self.arms = data.get("arms", self.arms)
                self.context_arms = data.get("context_arms", {})
                self.total_selections = data.get("total_selections", 0)
                self.total_rewards = data.get("total_rewards", 0)
                logger.info(f"Bandit state loaded: {self.total_selections} selections, "
                           f"avg reward={self.total_rewards/max(1, self.total_selections):.2%}")
        except Exception as e:
            logger.debug(f"Could not load bandit state: {e}")


# ===================================================================
# 3. Online Meta-Labeler (replaces sklearn RandomForest meta-labeling)
# ===================================================================
class OnlineMetaLabeler:
    """
    Online logistic regression for trade quality filtering.

    Replaces the sklearn RandomForestClassifier meta-labeling which:
      - Required batch retraining (min 10 trades to even start)
      - Used ~100 MB for sklearn import + model
      - Couldn't adapt between retraining sessions

    This version:
      - Updates after EVERY trade (true online learning)
      - Uses ~1 MB RAM (pure numpy logistic regression)
      - Adapts instantly to regime changes
      - Falls back to heuristic scoring until enough data accumulates
    """

    def __init__(self, n_features: int = 12, learning_rate: float = 0.01,
                 state_file: str = "logs/ml_models/meta_labeler_state.json"):
        self.n_features = n_features
        self.lr = learning_rate
        self.state_file = state_file

        # Logistic regression weights (initialized near zero with small random)
        self.weights = np.random.randn(n_features) * 0.01
        self.bias = 0.0

        # Running stats for feature normalization
        self.feature_means = np.zeros(n_features)
        self.feature_vars = np.ones(n_features)
        self.n_updates = 0

        # Track performance
        self.correct_predictions = 0
        self.total_predictions = 0
        self.is_trained = False  # True after first update

        self._load_state()

    def extract_features(self, signal: Dict[str, Any]) -> np.ndarray:
        """
        Extract features from a trade signal/strategy.
        Compatible with the old MetaLabelingModel interface.
        """
        features = np.zeros(self.n_features)

        # Feature 0: Trust/confidence score (0-1)
        features[0] = min(1.0, signal.get("trust_score", 50) / 100.0)

        # Feature 1: Backtest score (0-1)
        features[1] = min(1.0, signal.get("backtest_score", 50) / 100.0)

        # Feature 2: Consensus count (normalized)
        features[2] = min(1.0, signal.get("consensus_count", 1) / 10.0)

        # Feature 3: Sharpe ratio (normalized)
        features[3] = min(1.0, max(-1.0, signal.get("sharpe_ratio", 0) / 5.0))

        # Feature 4: Max drawdown
        features[4] = signal.get("max_drawdown", 0.3)

        # Feature 5: Market volatility
        features[5] = signal.get("market_volatility", 0.2)

        # Feature 6: Market trend (-1 to 1)
        features[6] = max(-1.0, min(1.0, signal.get("market_trend", 0)))

        # Feature 7: Is options trade
        features[7] = 1.0 if signal.get("trade_type") == "option" else 0.0

        # Feature 8: Is momentum strategy
        features[8] = 1.0 if "momentum" in str(signal.get("strategy_name", "")).lower() else 0.0

        # Feature 9: Position size ratio (vs account)
        features[9] = min(1.0, signal.get("position_size_pct", 0.05) / 0.25)

        # Feature 10: Entry score (from timing engine)
        features[10] = min(1.0, signal.get("timing_score", 50) / 100.0)

        # Feature 11: Recent win rate
        features[11] = signal.get("recent_win_rate", 0.5)

        return features

    def predict_confidence(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict confidence score for a trade signal.

        Returns:
            Dict with 'confidence' (0-1), 'should_trade' (bool),
            'model_type' (str)
        """
        features = self.extract_features(signal)

        # Normalize features using running stats
        if self.n_updates > 5:
            safe_std = np.sqrt(self.feature_vars + 1e-8)
            features_norm = (features - self.feature_means) / safe_std
        else:
            features_norm = features

        # Logistic regression: sigmoid(w . x + b)
        logit = np.dot(self.weights, features_norm) + self.bias
        confidence = 1.0 / (1.0 + np.exp(-np.clip(logit, -10, 10)))

        # Blend with heuristic until we have enough data
        heuristic_conf = self._heuristic_confidence(signal)
        if self.n_updates < 10:
            # Mostly heuristic early on
            blend_weight = self.n_updates / 10.0
            final_confidence = (1 - blend_weight) * heuristic_conf + blend_weight * confidence
            model_type = "heuristic_blend"
        else:
            final_confidence = float(confidence)
            model_type = "online_logistic"

        should_trade = final_confidence > 0.45  # Slightly below 0.5 for aggressive mode

        self.total_predictions += 1

        return {
            "confidence": float(final_confidence),
            "should_trade": should_trade,
            "raw_model_confidence": float(confidence),
            "heuristic_confidence": float(heuristic_conf),
            "model_type": model_type,
            "total_updates": self.n_updates,
        }

    def update(self, signal: Dict[str, Any], was_profitable: bool):
        """
        Update model after observing trade outcome.
        Called after every trade — this is the online learning step.

        Args:
            signal: The trade signal that was used
            was_profitable: True if the trade made money
        """
        features = self.extract_features(signal)
        label = 1.0 if was_profitable else 0.0

        # Update running feature statistics (Welford's online algorithm)
        self.n_updates += 1
        delta = features - self.feature_means
        self.feature_means += delta / self.n_updates
        delta2 = features - self.feature_means
        self.feature_vars += (delta * delta2 - self.feature_vars) / self.n_updates

        # Normalize
        if self.n_updates > 5:
            safe_std = np.sqrt(self.feature_vars + 1e-8)
            features_norm = (features - self.feature_means) / safe_std
        else:
            features_norm = features

        # Forward pass
        logit = np.dot(self.weights, features_norm) + self.bias
        prediction = 1.0 / (1.0 + np.exp(-np.clip(logit, -10, 10)))

        # Gradient descent step
        error = label - prediction
        self.weights += self.lr * error * features_norm
        self.bias += self.lr * error

        # Track accuracy
        predicted_label = 1 if prediction > 0.5 else 0
        if predicted_label == int(label):
            self.correct_predictions += 1

        self.is_trained = True
        self._save_state()

        accuracy = self.correct_predictions / max(1, self.total_predictions)
        logger.debug(f"MetaLabeler update #{self.n_updates}: "
                    f"label={label}, pred={prediction:.2f}, accuracy={accuracy:.1%}")

    @staticmethod
    def _heuristic_confidence(signal: Dict[str, Any]) -> float:
        """Heuristic confidence scoring (used before model is trained)."""
        conf = 0.5  # Base

        # Trust score boost
        trust = signal.get("trust_score", 50)
        conf += (trust - 50) / 200.0  # +/- 0.25 for trust

        # Consensus boost
        consensus = signal.get("consensus_count", 1)
        if consensus >= 3:
            conf += 0.1
        elif consensus >= 2:
            conf += 0.05

        # Timing score
        timing = signal.get("timing_score", 50)
        conf += (timing - 50) / 200.0

        return max(0.0, min(1.0, conf))

    def get_accuracy(self) -> float:
        """Get current prediction accuracy."""
        return self.correct_predictions / max(1, self.total_predictions)

    def _save_state(self):
        try:
            os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
            with open(self.state_file, "w") as f:
                json.dump({
                    "weights": self.weights.tolist(),
                    "bias": float(self.bias),
                    "feature_means": self.feature_means.tolist(),
                    "feature_vars": self.feature_vars.tolist(),
                    "n_updates": self.n_updates,
                    "correct_predictions": self.correct_predictions,
                    "total_predictions": self.total_predictions,
                    "timestamp": datetime.datetime.now().isoformat(),
                }, f)
        except Exception as e:
            logger.debug(f"Could not save meta-labeler state: {e}")

    def _load_state(self):
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, "r") as f:
                    data = json.load(f)
                self.weights = np.array(data["weights"])
                self.bias = float(data["bias"])
                self.feature_means = np.array(data["feature_means"])
                self.feature_vars = np.array(data["feature_vars"])
                self.n_updates = data.get("n_updates", 0)
                self.correct_predictions = data.get("correct_predictions", 0)
                self.total_predictions = data.get("total_predictions", 0)
                self.is_trained = self.n_updates > 0
                logger.info(f"MetaLabeler loaded: {self.n_updates} updates, "
                           f"accuracy={self.get_accuracy():.1%}")
        except Exception as e:
            logger.debug(f"Could not load meta-labeler state: {e}")


# ===================================================================
# 4. Weekend Retrainer
# ===================================================================
class WeekendRetrainer:
    """
    Manages batch retraining of the LightGBM model on weekends.

    Called from the Optimus monitoring loop — checks if it's Saturday
    and the model hasn't been retrained this week, then retrains on
    full available price history.
    """

    def __init__(self, predictor: LightGBMPredictor,
                 state_file: str = "logs/ml_models/retrain_state.json"):
        self.predictor = predictor
        self.state_file = state_file
        self.last_retrain_date: Optional[str] = None
        self._load_state()

    def should_retrain(self) -> bool:
        """Check if we should retrain (Saturday, not yet done this week)."""
        today = datetime.date.today()

        # Only retrain on Saturday (weekday=5) or Sunday (weekday=6)
        if today.weekday() not in (5, 6):
            return False

        # Check if already retrained this week
        if self.last_retrain_date:
            try:
                last = datetime.date.fromisoformat(self.last_retrain_date)
                days_since = (today - last).days
                if days_since < 6:  # Already retrained within the week
                    return False
            except (ValueError, TypeError):
                pass

        return True

    def retrain(self, prices: List[float], volumes: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        Execute batch retraining on full price history.

        Args:
            prices: Full available price history
            volumes: Corresponding volume data

        Returns:
            Result dict with training metrics
        """
        logger.info(f"Weekend batch retrain starting ({len(prices)} prices)...")

        success, result = self.predictor.train(prices, volumes)

        if success:
            self.last_retrain_date = datetime.date.today().isoformat()
            self._save_state()
            logger.info(f"Weekend retrain complete: accuracy={result.get('accuracy', 0):.1%}")
        else:
            logger.warning(f"Weekend retrain failed: {result.get('error')}")

        return {
            "success": success,
            "date": datetime.date.today().isoformat(),
            **result,
        }

    def _save_state(self):
        try:
            os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
            with open(self.state_file, "w") as f:
                json.dump({"last_retrain_date": self.last_retrain_date}, f)
        except Exception:
            pass

    def _load_state(self):
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, "r") as f:
                    data = json.load(f)
                self.last_retrain_date = data.get("last_retrain_date")
        except Exception:
            pass
