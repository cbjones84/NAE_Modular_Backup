# NAE/tools/profit_algorithms/lstm_predictor.py
"""
LSTM Networks for Price Prediction
Captures temporal dependencies in financial data for better price forecasts
"""

import numpy as np
from typing import List, Optional, Tuple, Dict, Any
import pickle
import os

try:
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("Warning: TensorFlow not available. LSTM predictions will use fallback methods.")

class LSTMPredictor:
    """
    LSTM-based price predictor for financial data
    Captures temporal patterns in price movements
    """
    
    def __init__(self, sequence_length: int = 60, model_file: Optional[str] = None):
        self.sequence_length = sequence_length
        self.model = None
        self.is_trained = False
        self.model_file = model_file or "logs/lstm_price_predictor.h5"
        self.scaler = None  # For feature normalization
        
    def _prepare_sequences(self, prices: List[float], lookback: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare sequences for LSTM training
        Returns (X, y) where X is input sequences and y is target prices
        """
        if lookback is None:
            lookback = self.sequence_length
        
        if len(prices) < lookback + 1:
            return np.array([]), np.array([])
        
        X, y = [], []
        for i in range(len(prices) - lookback):
            X.append(prices[i:i+lookback])
            y.append(prices[i+lookback])
        
        return np.array(X), np.array(y)
    
    def train(self, historical_prices: List[float], epochs: int = 50):
        """
        Train LSTM model on historical price data
        """
        if not TENSORFLOW_AVAILABLE:
            self.is_trained = False
            return False, {"error": "TensorFlow not available"}
        
        try:
            if len(historical_prices) < self.sequence_length + 50:
                return False, {"error": "Insufficient data for training"}
            
            # Normalize prices
            prices_array = np.array(historical_prices)
            self.scaler_mean = np.mean(prices_array)
            self.scaler_std = np.std(prices_array)
            normalized_prices = (prices_array - self.scaler_mean) / self.scaler_std
            
            # Prepare sequences
            X, y = self._prepare_sequences(normalized_prices.tolist())
            
            if len(X) == 0:
                return False, {"error": "Could not create sequences"}
            
            # Reshape for LSTM: [samples, timesteps, features]
            X = X.reshape((X.shape[0], X.shape[1], 1))
            
            # Split train/test
            split = int(len(X) * 0.8)
            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]
            
            # Build LSTM model
            self.model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
                Dropout(0.2),
                LSTM(50, return_sequences=False),
                Dropout(0.2),
                Dense(1)
            ])
            
            self.model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
            
            # Train model
            history = self.model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=32,
                validation_data=(X_test, y_test),
                verbose=0
            )
            
            self.is_trained = True
            
            # Save model
            os.makedirs(os.path.dirname(self.model_file), exist_ok=True)
            self.model.save(self.model_file)
            
            # Calculate training metrics
            train_loss = history.history['loss'][-1]
            val_loss = history.history['val_loss'][-1]
            
            return True, {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "sequences": len(X)
            }
            
        except Exception as e:
            return False, {"error": str(e)}
    
    def predict(self, recent_prices: List[float], horizon: int = 1) -> Optional[float]:
        """
        Predict future price(s) using LSTM model
        Returns predicted price for next period
        """
        if not self.is_trained or self.model is None:
            return self._fallback_prediction(recent_prices)
        
        try:
            if len(recent_prices) < self.sequence_length:
                return self._fallback_prediction(recent_prices)
            
            # Normalize input
            prices_array = np.array(recent_prices[-self.sequence_length:])
            normalized = (prices_array - self.scaler_mean) / self.scaler_std
            
            # Reshape for LSTM
            X = normalized.reshape((1, self.sequence_length, 1))
            
            # Predict
            prediction_normalized = self.model.predict(X, verbose=0)[0][0]
            
            # Denormalize
            prediction = (prediction_normalized * self.scaler_std) + self.scaler_mean
            
            return float(prediction)
            
        except Exception as e:
            return self._fallback_prediction(recent_prices)
    
    def _fallback_prediction(self, recent_prices: List[float]) -> Optional[float]:
        """Fallback prediction method when LSTM not available/trained"""
        if not recent_prices:
            return None
        
        # Simple momentum-based prediction
        if len(recent_prices) < 2:
            return recent_prices[-1]
        
        # Calculate simple moving average and momentum
        window = min(5, len(recent_prices))
        recent = recent_prices[-window:]
        sma = sum(recent) / len(recent)
        
        # Add momentum component
        if len(recent_prices) >= 2:
            momentum = recent_prices[-1] - recent_prices[-2]
            prediction = sma + (momentum * 0.5)  # Dampened momentum
        else:
            prediction = sma
        
        return float(prediction)
    
    def load_model(self) -> bool:
        """Load trained model from file"""
        if not TENSORFLOW_AVAILABLE:
            return False
        
        try:
            if os.path.exists(self.model_file):
                self.model = keras.models.load_model(self.model_file)
                self.is_trained = True
                return True
        except Exception as e:
            print(f"Error loading LSTM model: {e}")
        return False


