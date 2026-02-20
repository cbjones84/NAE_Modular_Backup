# NAE/tools/profit_algorithms/meta_labeling.py
"""
Meta-Labeling: Secondary ML layer for trade confidence scoring
Improves precision without sacrificing recall by filtering less reliable signals

VERY_AGGRESSIVE MODE: Optimized for $5M→$15.7M growth trajectory
- Higher base confidence for faster trade execution
- Aggressive confidence boosts for momentum signals
- Integrated with MilestoneAccelerator for dynamic adjustments
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle
import os
import datetime


class MetaLabelingModel:
    """
    Meta-labeling model that evaluates confidence of primary trading signals
    Adds a secondary decision layer to improve trade quality
    """
    
    def __init__(self):
        self.model = None
        self.is_trained = False
        self.training_data = []
        self.model_file = "logs/meta_labeling_model.pkl"
        
    def _extract_features(self, strategy: Dict[str, Any]) -> np.ndarray:
        """Extract features from strategy for meta-labeling"""
        features = []
        
        # Strategy quality metrics
        features.append(strategy.get("trust_score", 0) / 100.0)  # Normalized
        features.append(strategy.get("backtest_score", 0) / 100.0)  # Normalized
        features.append(strategy.get("consensus_count", 0) / 10.0)  # Normalized
        features.append(strategy.get("max_drawdown", 0.6))
        features.append(strategy.get("sharpe_ratio", 0) / 5.0)  # Normalized
        
        # Source reputation
        sources = strategy.get("sources", [])
        if sources:
            # Average source reputation (normalized)
            source_scores = []
            source_reputations = {
                "Grok": 85, "DeepSeek": 80, "Claude": 82,
                "toptrader.com": 75, "optionsforum.com": 60,
                "reddit": 50, "twitter": 45
            }
            for source in sources:
                source_lower = str(source).lower()
                for key, val in source_reputations.items():
                    if key.lower() in source_lower:
                        source_scores.append(val / 100.0)
            features.append(np.mean(source_scores) if source_scores else 0.5)
        else:
            features.append(0.5)
        
        # Market conditions (if available)
        features.append(strategy.get("market_volatility", 0.2))
        features.append(strategy.get("market_trend", 0))  # -1 to 1
        
        # Strategy type indicators
        features.append(1.0 if "options" in str(strategy.get("aggregated_details", "")).lower() else 0.0)
        features.append(1.0 if "momentum" in str(strategy.get("name", "")).lower() else 0.0)
        features.append(1.0 if "mean" in str(strategy.get("name", "")).lower() else 0.0)
        
        return np.array(features)
    
    def train(self, strategies: List[Dict[str, Any]], labels: List[float]):
        """
        Train meta-labeling model
        labels: 1.0 for profitable trades, 0.0 for unprofitable trades
        """
        try:
            X = np.array([self._extract_features(s) for s in strategies])
            y = np.array(labels)
            
            if len(X) < 10:
                # Not enough data, use simple heuristic
                self.is_trained = False
                return False, {"error": "insufficient_data", "samples": len(X)}
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train Random Forest classifier
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                min_samples_split=2
            )
            self.model.fit(X_train, y_train)
            
            # Evaluate
            train_score = self.model.score(X_train, y_train)
            test_score = self.model.score(X_test, y_test)
            
            self.is_trained = True
            
            # Save model
            os.makedirs(os.path.dirname(self.model_file), exist_ok=True)
            with open(self.model_file, 'wb') as f:
                pickle.dump(self.model, f)
            
            return True, {"train_score": train_score, "test_score": test_score}
            
        except Exception as e:
            print(f"Meta-labeling training error: {e}")
            self.is_trained = False
            return False, {"error": str(e)}
    
    def predict_confidence(self, strategy: Dict[str, Any]) -> float:
        """
        Predict confidence score (0.0 to 1.0) for a strategy
        Higher score = higher confidence = better trade quality
        """
        if not self.is_trained or self.model is None:
            # Fallback to heuristic-based confidence
            return self._heuristic_confidence(strategy)
        
        try:
            features = self._extract_features(strategy).reshape(1, -1)
            proba = self.model.predict_proba(features)[0]
            
            # Return probability of positive class (profitable trade)
            # Assuming binary classification: [unprofitable, profitable]
            if len(proba) == 2:
                return float(proba[1])  # Probability of profitable trade
            else:
                return float(proba[0])
                
        except Exception as e:
            print(f"Meta-labeling prediction error: {e}")
            return self._heuristic_confidence(strategy)
    
    def _heuristic_confidence(self, strategy: Dict[str, Any]) -> float:
        """
        VERY_AGGRESSIVE heuristic confidence for faster growth
        Optimized for $5M → $15.7M growth trajectory
        """
        # AGGRESSIVE: Higher base confidence (was 0.5)
        confidence = 0.60  # Start higher for more trading opportunities
        
        # Trust score boost (AGGRESSIVE scaling)
        trust_score = strategy.get("trust_score", 0)
        confidence += (trust_score - 50) / 150.0  # Scale from 50-100 to 0-0.33 (was 55-100 to 0-0.225)
        
        # Backtest score boost (AGGRESSIVE scaling)
        backtest_score = strategy.get("backtest_score", 0)
        confidence += (backtest_score - 25) / 250.0  # Scale from 25-100 to 0-0.30 (was 30-100 to 0-0.2)
        
        # Consensus boost (INCREASED)
        consensus = strategy.get("consensus_count", 0)
        confidence += min(consensus / 15.0, 0.20)  # Up to 0.20 boost (was 0.15)
        
        # Drawdown penalty (REDUCED penalty for aggressive mode)
        drawdown = strategy.get("max_drawdown", 0.6)
        confidence -= (drawdown - 0.3) / 3.0  # Less penalty (was /2.0)
        
        # MOMENTUM BOOST: Extra confidence for momentum strategies
        strategy_name = str(strategy.get("name", "")).lower()
        strategy_details = str(strategy.get("aggregated_details", "")).lower()
        if any(term in strategy_name or term in strategy_details 
               for term in ["momentum", "breakout", "trend", "0dte", "earnings"]):
            confidence += 0.10  # 10% boost for high-growth strategies
        
        # VOLATILITY OPPORTUNITY: Higher vol = more opportunity in aggressive mode
        volatility = strategy.get("market_volatility", 0.2)
        if 0.2 <= volatility <= 0.5:  # Sweet spot for aggressive trading
            confidence += 0.05
        
        return max(0.0, min(1.0, confidence))  # Clamp to [0, 1]
    
    def calculate_position_size(self, base_size: float, confidence: float, 
                               max_position_pct: float = 0.50) -> float:
        """
        VERY_AGGRESSIVE position sizing for accelerated growth
        Higher confidence = larger position (up to 50% max)
        
        Target: $6,243,561 by Year 7, $15,726,144 by Year 8
        """
        # AGGRESSIVE: Higher minimum (20% vs 10%), maximum 50% (was 25%)
        # Minimum 20% of base even for low confidence
        scaled_size = base_size * (0.2 + 0.8 * confidence)
        
        # AGGRESSIVE: Allow larger positions
        max_size = max_position_pct * base_size * 10
        return min(scaled_size, max_size)
    
    def calculate_accelerated_position(
        self, 
        base_size: float, 
        confidence: float, 
        current_nav: float,
        milestone_progress_pct: float = 100.0
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate position with milestone acceleration
        
        Args:
            base_size: Base position size
            confidence: Strategy confidence (0.0 to 1.0)
            current_nav: Current Net Asset Value
            milestone_progress_pct: Progress toward milestone (100 = on track)
            
        Returns:
            Tuple of (position_size, details)
        """
        # Calculate acceleration factor based on milestone progress
        if milestone_progress_pct < 50:
            acceleration = 1.50  # TURBO: 50% boost when far behind
        elif milestone_progress_pct < 80:
            acceleration = 1.30  # AGGRESSIVE: 30% boost when behind
        elif milestone_progress_pct < 100:
            acceleration = 1.15  # MODERATE: 15% boost when slightly behind
        else:
            acceleration = 1.0   # MAINTAIN: On track
        
        # Base calculation
        base_position = self.calculate_position_size(base_size, confidence)
        
        # Apply acceleration
        accelerated_position = base_position * acceleration
        
        # Cap at 50% of NAV
        max_position = current_nav * 0.50
        final_position = min(accelerated_position, max_position)
        
        details = {
            "base_position": base_position,
            "acceleration_factor": acceleration,
            "accelerated_position": accelerated_position,
            "final_position": final_position,
            "confidence": confidence,
            "milestone_progress": milestone_progress_pct,
            "position_pct_nav": (final_position / current_nav * 100) if current_nav > 0 else 0
        }
        
        return final_position, details
    
    def load_model(self) -> bool:
        """Load trained model from file"""
        try:
            if os.path.exists(self.model_file):
                with open(self.model_file, 'rb') as f:
                    self.model = pickle.load(f)
                self.is_trained = True
                return True
        except Exception as e:
            print(f"Error loading meta-labeling model: {e}")
        return False


