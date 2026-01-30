"""
IV Surface modelling utilities for Optimus.

Implements the PCA + Kalman Filter pipeline described in the roadmap
for stable short-term implied volatility forecasting.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class KalmanFilterError(RuntimeError):
    """Raised when the Kalman filter cannot be applied."""


@dataclass
class KalmanConfig:
    """Configuration for the simple Kalman filter."""

    process_noise: float = 1e-4
    observation_noise: float = 1e-3
    initial_covariance: float = 1.0


class SimpleKalmanFilter:
    """
    Lightweight Kalman filter for smoothing PCA factor scores.

    We assume an identity transition and observation matrix. The filter
    primarily serves to denoise the latent factors while preserving
    responsiveness.
    """

    def __init__(self, dimension: int, config: Optional[KalmanConfig] = None):
        self.dimension = dimension
        self.config = config or KalmanConfig()

        self.state_mean_: Optional[np.ndarray] = None
        self.state_cov_: Optional[np.ndarray] = None
        self.history_: List[np.ndarray] = []

        identity = np.eye(dimension)
        self._A = identity  # transition
        self._H = identity  # observation
        self._Q = self.config.process_noise * identity
        self._R = self.config.observation_noise * identity

    def fit(self, observations: np.ndarray) -> "SimpleKalmanFilter":
        """
        Fit the filter to a sequence of observations.

        Parameters
        ----------
        observations:
            Shape (n_observations, dimension).
        """
        if observations.ndim != 2 or observations.shape[1] != self.dimension:
            raise KalmanFilterError(
                f"Expected observations with shape (n, {self.dimension}), "
                f"got {observations.shape}"
            )

        state_mean = observations[0].reshape(-1, 1)
        state_cov = np.eye(self.dimension) * self.config.initial_covariance

        self.history_ = []

        for obs in observations:
            obs_vec = obs.reshape(-1, 1)

            # Predict
            pred_mean = self._A @ state_mean
            pred_cov = self._A @ state_cov @ self._A.T + self._Q

            # Update
            innovation = obs_vec - (self._H @ pred_mean)
            innovation_cov = self._H @ pred_cov @ self._H.T + self._R
            kalman_gain = pred_cov @ self._H.T @ np.linalg.inv(innovation_cov)

            state_mean = pred_mean + kalman_gain @ innovation
            state_cov = (np.eye(self.dimension) - kalman_gain @ self._H) @ pred_cov

            self.history_.append(state_mean.flatten())

        self.state_mean_ = state_mean
        self.state_cov_ = state_cov
        return self

    def forecast(self, steps: int = 1) -> np.ndarray:
        """
        Forecast future latent state(s).

        Returns
        -------
        ndarray of shape (steps, dimension)
        """
        if self.state_mean_ is None or self.state_cov_ is None:
            raise KalmanFilterError("Kalman filter has not been fitted yet.")

        forecasts: List[np.ndarray] = []
        mean = self.state_mean_.copy()
        cov = self.state_cov_.copy()

        for _ in range(steps):
            mean = self._A @ mean
            cov = self._A @ cov @ self._A.T + self._Q
            forecasts.append(mean.flatten())

        return np.vstack(forecasts)

    def smoothed_history(self) -> np.ndarray:
        """Return the filtered state trajectory."""
        if not self.history_:
            raise KalmanFilterError("No history available. Fit the filter first.")
        return np.vstack(self.history_)


@dataclass
class IVSurfaceSnapshot:
    """Represents a single implied volatility surface observation."""

    as_of: _dt.datetime
    surface: pd.DataFrame  # index = maturity (days), columns = strikes


@dataclass
class IVForecastResult:
    """Container for forecast outputs."""

    as_of: _dt.datetime
    forecast_for: _dt.datetime
    surface: pd.DataFrame
    rmse: Optional[float] = None
    directional_accuracy: Optional[float] = None
    metadata: Dict[str, float] = field(default_factory=dict)


class IVSurfaceForecaster:
    """
    Implements the PCA + Kalman filter pipeline for implied volatility surfaces.
    """

    def __init__(
        self,
        n_components: int = 3,
        kalman_config: Optional[KalmanConfig] = None,
    ):
        self.n_components = n_components
        self.kalman_config = kalman_config or KalmanConfig()

        self.scaler: Optional[StandardScaler] = None
        self.pca: Optional[PCA] = None
        self.kalman: Optional[SimpleKalmanFilter] = None

        self._grid_index: Optional[pd.Index] = None
        self._grid_columns: Optional[pd.Index] = None
        self._last_timestamp: Optional[_dt.datetime] = None

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def _flatten_surface(self, surface: pd.DataFrame) -> np.ndarray:
        if self._grid_index is None:
            self._grid_index = surface.index
            self._grid_columns = surface.columns
        else:
            surface = surface.reindex(index=self._grid_index, columns=self._grid_columns)
        return surface.to_numpy().astype(np.float64).flatten()

    def _reshape_to_surface(self, vector: np.ndarray) -> pd.DataFrame:
        if self._grid_index is None or self._grid_columns is None:
            raise ValueError("Surface grid not initialised.")
        matrix = vector.reshape(len(self._grid_index), len(self._grid_columns))
        return pd.DataFrame(matrix, index=self._grid_index, columns=self._grid_columns)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(self, snapshots: Sequence[IVSurfaceSnapshot]) -> "IVSurfaceForecaster":
        if len(snapshots) < self.n_components + 2:
            raise ValueError(
                "Need at least n_components + 2 snapshots to fit the PCA pipeline."
            )

        vectors = np.vstack([self._flatten_surface(s.surface) for s in snapshots])
        self.scaler = StandardScaler()
        scaled = self.scaler.fit_transform(vectors)

        self.pca = PCA(n_components=self.n_components)
        scores = self.pca.fit_transform(scaled)

        self.kalman = SimpleKalmanFilter(self.n_components, self.kalman_config)
        self.kalman.fit(scores)

        self._last_timestamp = snapshots[-1].as_of
        return self

    def update(self, snapshot: IVSurfaceSnapshot) -> None:
        """
        Incrementally update the PCA space with a new snapshot.
        """
        if not all([self.scaler, self.pca, self.kalman]):
            raise ValueError("Model not fitted yet.")
        assert self.scaler is not None
        assert self.pca is not None
        assert self.kalman is not None

        vector = self._flatten_surface(snapshot.surface)
        scaled = self.scaler.transform(vector.reshape(1, -1))
        score = self.pca.transform(scaled)

        # Update Kalman filter with new factor observation
        history = np.vstack([self.kalman.smoothed_history(), score])
        self.kalman.fit(history)
        self._last_timestamp = snapshot.as_of

    def forecast(self, steps: int = 1) -> List[IVForecastResult]:
        if not all([self.scaler, self.pca, self.kalman, self._last_timestamp]):
            raise ValueError("Model not fitted yet.")
        assert self.scaler is not None
        assert self.pca is not None
        assert self.kalman is not None
        assert self._last_timestamp is not None

        factor_forecasts = self.kalman.forecast(steps)
        results: List[IVForecastResult] = []

        for i, factor in enumerate(factor_forecasts, start=1):
            scaled_vector = self.pca.inverse_transform(factor.reshape(1, -1))
            vector = self.scaler.inverse_transform(scaled_vector).flatten()
            surface = self._reshape_to_surface(vector)
            forecast_time = self._last_timestamp + _dt.timedelta(days=i)
            results.append(
                IVForecastResult(
                    as_of=self._last_timestamp,
                    forecast_for=forecast_time,
                    surface=surface,
                )
            )
        return results

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------
    @staticmethod
    def compute_rmse(
        actual_surface: pd.DataFrame, forecast_surface: pd.DataFrame
    ) -> float:
        aligned = actual_surface.reindex_like(forecast_surface).to_numpy()
        predicted = forecast_surface.to_numpy()
        return np.sqrt(np.nanmean((aligned - predicted) ** 2))

    @staticmethod
    def directional_accuracy(
        actual_surface: pd.DataFrame,
        previous_surface: pd.DataFrame,
        forecast_surface: pd.DataFrame,
    ) -> float:
        """
        Measures proportion of grid points where the forecast predicted
        the correct direction of change relative to the previous surface.
        """
        actual = actual_surface.reindex_like(previous_surface).to_numpy()
        previous = previous_surface.to_numpy()
        forecast = forecast_surface.reindex_like(previous_surface).to_numpy()

        actual_change = np.sign(actual - previous)
        forecast_change = np.sign(forecast - previous)

        mask = ~np.isnan(actual_change) & ~np.isnan(forecast_change)
        if mask.sum() == 0:
            return 0.0
        return float(np.mean(actual_change[mask] == forecast_change[mask]))


def build_surface_from_chain(
    option_chain: pd.DataFrame,
    spot_price: float,
    maturities: Optional[Iterable[int]] = None,
    strike_step: float = 5.0,
) -> pd.DataFrame:
    """
    Construct a maturity x strike implied volatility surface from an option chain.

    Parameters
    ----------
    option_chain:
        DataFrame with at least columns ['expiration', 'strike', 'impliedVolatility'].
        Expiration should be datetime-like.
    spot_price:
        Current underlying price used as an anchor for strike selection.
    maturities:
        Optional iterable of target days-to-expiry. If None, maturities are inferred
        from the chain.
    strike_step:
        Strike grid step in absolute dollars.
    """
    if not {"expiration", "strike", "impliedVolatility"}.issubset(option_chain.columns):
        raise ValueError(
            "option_chain must include 'expiration', 'strike', 'impliedVolatility'."
        )

    option_chain = option_chain.copy()
    option_chain["expiration"] = pd.to_datetime(option_chain["expiration"])
    option_chain["dte"] = (
        option_chain["expiration"] - option_chain["expiration"].min()
    ).dt.days

    if maturities is None:
        maturities = sorted(option_chain["dte"].unique())
    else:
        maturities = list(maturities)

    strike_min = max(0.01, spot_price * 0.5)
    strike_max = spot_price * 1.5
    strikes = np.arange(strike_min, strike_max + strike_step, strike_step)

    maturity_index = pd.Index(maturities, name="dte")
    strike_index = pd.Index(strikes, name="strike")
    surface = pd.DataFrame(index=maturity_index, columns=strike_index, dtype=float)

    for maturity in maturities:
        sub_df = option_chain.loc[option_chain["dte"] == maturity]
        if sub_df.empty:
            continue
        interp = np.interp(
            strikes,
            sub_df["strike"].to_numpy(),
            sub_df["impliedVolatility"].to_numpy(),
            left=np.nan,
            right=np.nan,
        )
        surface.loc[maturity] = interp

    surface = surface.astype(float).interpolate(axis=1).ffill(axis=1).bfill(axis=1)
    return surface


