"""
Hybrid volatility forecasting ensemble (GARCH + ML models).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

try:
    from arch import arch_model
except ImportError:  # pragma: no cover - optional dependency
    arch_model = None  # type: ignore


class GARCHNotAvailableError(RuntimeError):
    """Raised when the arch library is not available."""


@dataclass
class EnsembleForecast:
    """Container for volatility forecasts."""

    garch_vol: float
    ml_vol: float
    ensemble_vol: float
    weight_ml: float
    weight_garch: float
    components: Dict[str, float]


class VolatilityEnsembleForecaster:
    """
    Combines a GARCH-type model with a boosted tree regressor that ingests
    richer features (realised vol, PCA factors, sentiment).
    """

    def __init__(self, garch_type: str = "GJR-GARCH"):
        self.garch_type = garch_type.upper()
        self.garch_model = None
        self.garch_fit_result = None

        self.scaler = StandardScaler()
        self.ml_model = GradientBoostingRegressor(
            n_estimators=500,
            learning_rate=0.01,
            max_depth=3,
            random_state=42,
        )

        self.meta_weight_ml = 0.5
        self.meta_weight_garch = 0.5

        self._last_realised_vol: Optional[float] = None

    # ------------------------------------------------------------------
    # GARCH training
    # ------------------------------------------------------------------
    def _fit_garch(self, returns: pd.Series) -> None:
        if arch_model is None:
            raise GARCHNotAvailableError(
                "arch package is required for GARCH modelling. "
                "Install with `pip install arch`."
            )

        returns = returns.dropna() * 100  # convert to percentage
        if returns.empty:
            raise ValueError("Returns series is empty after NaN drop.")

        if self.garch_type == "EGARCH":
            model = arch_model(returns, vol="EGARCH", p=1, o=1, q=1, dist="t")
        elif self.garch_type in {"GJR-GARCH", "GJR"}:
            model = arch_model(returns, vol="GARCH", p=1, o=1, q=1, dist="t")
        else:
            model = arch_model(returns, vol="GARCH", p=1, q=1, dist="t")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.garch_fit_result = model.fit(disp="off")
        self.garch_model = model

    def _forecast_garch(self, horizon: int = 1) -> float:
        if self.garch_fit_result is None:
            raise ValueError("GARCH model not fitted.")
        forecast = self.garch_fit_result.forecast(horizon=horizon)
        var = forecast.variance.values[-1, -1] / 10000.0  # convert back from percentage
        return float(np.sqrt(max(var, 0.0)))

    # ------------------------------------------------------------------
    # ML model
    # ------------------------------------------------------------------
    def _prepare_features(self, features: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        if "target_vol" not in features:
            raise ValueError("Feature frame must contain 'target_vol' column.")
        y = features["target_vol"].to_numpy()
        X = features.drop(columns=["target_vol"]).to_numpy()
        return X, y

    def fit(
        self,
        returns: pd.Series,
        feature_frame: pd.DataFrame,
        calibration_window: int = 250,
    ) -> None:
        """
        Fit both components of the ensemble.

        Parameters
        ----------
        returns:
            Asset returns time-series (daily).
        feature_frame:
            DataFrame with engineered features. Must contain 'target_vol'.
        calibration_window:
            Number of most recent observations to use for ML training.
        """
        self._fit_garch(returns)

        if len(feature_frame) < 20:
            raise ValueError("Need at least 20 rows in feature frame for ML.")

        recent = feature_frame.tail(calibration_window)
        X, y = self._prepare_features(recent)
        X_scaled = self.scaler.fit_transform(X)
        self.ml_model.fit(X_scaled, y)
        self._last_realised_vol = float(recent["target_vol"].iloc[-1])

    def forecast(self, feature_row: pd.Series) -> EnsembleForecast:
        """
        Generate hybrid volatility forecast.

        Parameters
        ----------
        feature_row:
            Pandas Series with the same columns used during ML training
            except 'target_vol'.
        """
        if self.garch_model is None or self.garch_fit_result is None:
            raise ValueError("GARCH component has not been fitted.")

        features = feature_row.drop(labels=["target_vol"], errors="ignore")
        X = features.to_numpy().reshape(1, -1)
        X_scaled = self.scaler.transform(X)

        ml_vol = float(max(self.ml_model.predict(X_scaled)[0], 0.0))
        garch_vol = float(max(self._forecast_garch(horizon=1), 0.0))

        ensemble_vol = float(
            self.meta_weight_ml * ml_vol + self.meta_weight_garch * garch_vol
        )

        return EnsembleForecast(
            garch_vol=garch_vol,
            ml_vol=ml_vol,
            ensemble_vol=ensemble_vol,
            weight_ml=self.meta_weight_ml,
            weight_garch=self.meta_weight_garch,
            components={
                "garch": garch_vol,
                "ml": ml_vol,
            },
        )

    def update_meta_weights(
        self,
        realised_vol: float,
        ml_forecast: float,
        garch_forecast: float,
        decay: float = 0.94,
    ) -> None:
        """
        Update ensemble weights using an exponentially weighted performance score.
        """
        realised = max(realised_vol, 1e-8)
        ml_error = (ml_forecast - realised) ** 2
        garch_error = (garch_forecast - realised) ** 2

        if self._last_realised_vol is None:
            self._last_realised_vol = realised

        ml_weight = 1.0 / (ml_error + 1e-8)
        garch_weight = 1.0 / (garch_error + 1e-8)

        self.meta_weight_ml = float(
            decay * self.meta_weight_ml + (1 - decay) * ml_weight
        )
        self.meta_weight_garch = float(
            decay * self.meta_weight_garch + (1 - decay) * garch_weight
        )

        total = self.meta_weight_ml + self.meta_weight_garch
        if total > 0:
            self.meta_weight_ml /= total
            self.meta_weight_garch /= total

        self._last_realised_vol = realised

    def evaluate(self, feature_frame: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate recent performance on a feature frame that includes forecasts and realised vol.
        """
        if "ml_forecast" not in feature_frame or "garch_forecast" not in feature_frame:
            raise ValueError("feature_frame must contain forecast columns for evaluation.")

        realised = feature_frame["target_vol"].to_numpy()
        ml_forecast = feature_frame["ml_forecast"].to_numpy()
        garch_forecast = feature_frame["garch_forecast"].to_numpy()
        ensemble_forecast = (
            self.meta_weight_ml * ml_forecast + self.meta_weight_garch * garch_forecast
        )

        return {
            "rmse_ml": float(np.sqrt(mean_squared_error(realised, ml_forecast))),
            "rmse_garch": float(np.sqrt(mean_squared_error(realised, garch_forecast))),
            "rmse_ensemble": float(np.sqrt(mean_squared_error(realised, ensemble_forecast))),
        }


