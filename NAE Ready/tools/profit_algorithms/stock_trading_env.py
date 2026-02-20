"""
Multi-Stock Trading Environment for NAE's TD3 Agent

Adapted from the StockEnv in austin-starks/Deep-RL-Stocks with major changes:
  - Feature-vector state instead of chart-image state
  - Accepts price data as numpy arrays (no local CSV dependency)
  - Built-in technical indicators (returns, RSI, moving averages)
  - Configurable transaction costs and reward shaping
  - Works with NAE's market data pipeline

The environment follows an OpenAI-Gym-like interface (reset / step / render)
but does NOT require gym as a dependency.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class EnvConfig:
    """Configuration for the stock trading environment."""

    initial_cash: float = 50_000.0
    transaction_cost_pct: float = 0.001    # 10 bps round-trip
    max_shares_per_action: int = 100       # max shares bought/sold per step
    lookback_window: int = 20              # bars of history in state
    reward_scaling: float = 1e-4           # scale raw dollar P&L
    risk_penalty_lambda: float = 0.5       # Sharpe-style variance penalty
    random_start: bool = True


# ═══════════════════════════════════════════════════════════════════════════
#  Feature engineering helpers
# ═══════════════════════════════════════════════════════════════════════════

def _compute_features(prices: np.ndarray, volumes: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Build a feature matrix from a 1-D price series.

    Returns shape (len(prices), num_features) with:
      [0] normalised price (z-score over trailing window)
      [1] 1-bar return
      [2] 5-bar return
      [3] 20-bar return
      [4] 14-bar RSI (0-1 scaled)
      [5] price / 20-bar SMA ratio
      [6] Bollinger %B (20-bar)
      [7] normalised volume (z-score) — zeros if volumes not provided
    """
    n = len(prices)
    feats = np.zeros((n, 8), dtype=np.float32)

    eps = 1e-8

    # Returns
    ret1 = np.zeros(n, dtype=np.float32)
    ret1[1:] = prices[1:] / (prices[:-1] + eps) - 1.0
    ret5 = np.zeros(n, dtype=np.float32)
    ret5[5:] = prices[5:] / (prices[:-5] + eps) - 1.0
    ret20 = np.zeros(n, dtype=np.float32)
    ret20[20:] = prices[20:] / (prices[:-20] + eps) - 1.0

    # 20-bar SMA and std
    sma20 = np.convolve(prices, np.ones(20) / 20, mode="same")
    std20 = np.array([prices[max(0, i - 19):i + 1].std() for i in range(n)], dtype=np.float32)
    std20[std20 < eps] = eps

    # RSI-14
    deltas = np.diff(prices, prepend=prices[0])
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = np.zeros(n, dtype=np.float32)
    avg_loss = np.zeros(n, dtype=np.float32)
    window = 14
    if n > window:
        avg_gain[window] = gains[1:window + 1].mean()
        avg_loss[window] = losses[1:window + 1].mean()
        for i in range(window + 1, n):
            avg_gain[i] = (avg_gain[i - 1] * (window - 1) + gains[i]) / window
            avg_loss[i] = (avg_loss[i - 1] * (window - 1) + losses[i]) / window

    rs = avg_gain / (avg_loss + eps)
    rsi = 1.0 - 1.0 / (1.0 + rs)  # already 0-1

    # Normalised price (z-score with trailing 20-bar window)
    price_z = (prices - sma20) / (std20 + eps)

    # Bollinger %B
    upper = sma20 + 2 * std20
    lower = sma20 - 2 * std20
    boll_b = (prices - lower) / (upper - lower + eps)

    # SMA ratio
    sma_ratio = prices / (sma20 + eps) - 1.0

    feats[:, 0] = price_z
    feats[:, 1] = ret1
    feats[:, 2] = ret5
    feats[:, 3] = ret20
    feats[:, 4] = rsi
    feats[:, 5] = sma_ratio
    feats[:, 6] = boll_b

    if volumes is not None and len(volumes) == n:
        vol_mean = np.convolve(volumes, np.ones(20) / 20, mode="same")
        vol_std = np.array([volumes[max(0, i - 19):i + 1].std() for i in range(n)], dtype=np.float32)
        vol_std[vol_std < eps] = eps
        feats[:, 7] = (volumes - vol_mean) / (vol_std + eps)

    return feats


# ═══════════════════════════════════════════════════════════════════════════
#  Stock Trading Environment
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class StepResult:
    """Returned by env.step()."""

    state: np.ndarray
    reward: float
    done: bool
    info: Dict[str, Any]


class StockTradingEnv:
    """
    Multi-stock trading environment for the TD3 agent.

    Accepts pre-loaded price data (and optional volumes) so it can work
    with any data source — historical CSVs, broker APIs, or NAE's live feed.

    State vector layout (per stock, repeated for each stock, then portfolio features):
        [stock_features (8 × lookback_window flattened)] × num_stocks
        + [cash_ratio, total_equity_ratio, per-stock weight …]

    Action: float[-1, 1] per stock
        -1 → sell max_shares_per_action shares
         0 → hold
        +1 → buy max_shares_per_action shares
    """

    def __init__(
        self,
        price_data: Dict[str, np.ndarray],
        volume_data: Optional[Dict[str, np.ndarray]] = None,
        config: Optional[EnvConfig] = None,
    ):
        """
        Args:
            price_data: {stock_name: 1-D array of daily close prices}
            volume_data: optional {stock_name: 1-D array of daily volumes}
            config: environment hyper-parameters
        """
        self.cfg = config or EnvConfig()
        self.stock_names = sorted(price_data.keys())
        self.num_stocks = len(self.stock_names)

        # Pre-compute feature matrices for each stock
        self._prices: Dict[str, np.ndarray] = {}
        self._features: Dict[str, np.ndarray] = {}
        min_len = float("inf")
        for name in self.stock_names:
            p = np.asarray(price_data[name], dtype=np.float32)
            v = np.asarray(volume_data[name], dtype=np.float32) if volume_data and name in volume_data else None
            self._prices[name] = p
            self._features[name] = _compute_features(p, v)
            min_len = min(min_len, len(p))

        self._max_t = int(min_len) - 1
        if self._max_t < self.cfg.lookback_window + 10:
            raise ValueError(
                f"Price series too short ({int(min_len)} bars) for lookback={self.cfg.lookback_window}"
            )

        # State / action dimensionality
        self._features_per_stock = 8 * self.cfg.lookback_window
        self._portfolio_features = 1 + self.num_stocks  # cash_ratio + per-stock weight
        self.state_dim = self._features_per_stock * self.num_stocks + self._portfolio_features
        self.action_dim = self.num_stocks

        # Mutable episode state (set in reset)
        self._t = 0
        self._cash = self.cfg.initial_cash
        self._holdings = np.zeros(self.num_stocks, dtype=np.float32)
        self._initial_portfolio_value = self.cfg.initial_cash
        self._prev_portfolio_value = self.cfg.initial_cash
        self._returns_history: List[float] = []

    # ------------------------------------------------------------------
    #  Gym-like interface
    # ------------------------------------------------------------------

    def reset(self) -> np.ndarray:
        """Reset the environment and return the initial state."""
        if self.cfg.random_start:
            earliest = self.cfg.lookback_window
            latest = max(earliest + 1, int(self._max_t * 0.3))
            self._t = np.random.randint(earliest, latest)
        else:
            self._t = self.cfg.lookback_window

        self._cash = self.cfg.initial_cash
        self._holdings = np.zeros(self.num_stocks, dtype=np.float32)
        self._initial_portfolio_value = self.cfg.initial_cash
        self._prev_portfolio_value = self.cfg.initial_cash
        self._returns_history = []
        return self._build_state()

    def step(self, action: np.ndarray) -> StepResult:
        """
        Execute one time step.

        Args:
            action: array of shape (num_stocks,) in [-1, 1]
        """
        action = np.clip(action, -1.0, 1.0)
        prices = self._current_prices()

        # Execute trades
        total_cost = 0.0
        for i, (a, price) in enumerate(zip(action, prices)):
            if price <= 0:
                continue
            shares_delta = int(round(a * self.cfg.max_shares_per_action))

            if shares_delta > 0:
                # Buy
                max_affordable = int(self._cash / (price * (1 + self.cfg.transaction_cost_pct)))
                shares_delta = min(shares_delta, max_affordable)
                cost = shares_delta * price
                fee = cost * self.cfg.transaction_cost_pct
                self._cash -= cost + fee
                self._holdings[i] += shares_delta
                total_cost += fee
            elif shares_delta < 0:
                # Sell
                shares_delta = -min(-shares_delta, int(self._holdings[i]))
                proceeds = -shares_delta * price
                fee = proceeds * self.cfg.transaction_cost_pct
                self._cash += proceeds - fee
                self._holdings[i] += shares_delta
                total_cost += fee

        # Advance time
        self._t += 1
        done = self._t >= self._max_t

        # Compute reward
        new_value = self._portfolio_value()
        step_return = (new_value - self._prev_portfolio_value) / (self._prev_portfolio_value + 1e-8)
        self._returns_history.append(step_return)

        reward = step_return / (self.cfg.reward_scaling + 1e-8)

        # Risk penalty (variance of recent returns → encourages Sharpe-like behavior)
        if len(self._returns_history) > 5:
            recent_std = np.std(self._returns_history[-20:])
            reward -= self.cfg.risk_penalty_lambda * recent_std / (self.cfg.reward_scaling + 1e-8)

        self._prev_portfolio_value = new_value

        state = self._build_state()
        info = {
            "portfolio_value": new_value,
            "cash": self._cash,
            "holdings": self._holdings.copy(),
            "prices": prices,
            "step_return": step_return,
            "total_return": (new_value / self._initial_portfolio_value) - 1.0,
            "transaction_costs": total_cost,
            "timestep": self._t,
        }
        return StepResult(state=state, reward=float(reward), done=done, info=info)

    # ------------------------------------------------------------------
    #  Internals
    # ------------------------------------------------------------------

    def _current_prices(self) -> np.ndarray:
        return np.array([self._prices[n][self._t] for n in self.stock_names], dtype=np.float32)

    def _portfolio_value(self) -> float:
        prices = self._current_prices()
        return float(self._cash + np.dot(self._holdings, prices))

    def _build_state(self) -> np.ndarray:
        """Construct the full state vector for the current timestep."""
        parts: List[np.ndarray] = []

        for name in self.stock_names:
            feat = self._features[name]
            start = max(0, self._t - self.cfg.lookback_window + 1)
            window = feat[start:self._t + 1]
            # Pad if we don't have a full window yet
            if len(window) < self.cfg.lookback_window:
                pad = np.zeros((self.cfg.lookback_window - len(window), window.shape[1]), dtype=np.float32)
                window = np.vstack([pad, window])
            parts.append(window.flatten())

        # Portfolio features
        pv = self._portfolio_value()
        cash_ratio = self._cash / (pv + 1e-8)
        prices = self._current_prices()
        stock_weights = (self._holdings * prices) / (pv + 1e-8)
        parts.append(np.array([cash_ratio], dtype=np.float32))
        parts.append(stock_weights.astype(np.float32))

        return np.concatenate(parts)

    # ------------------------------------------------------------------
    #  Utilities
    # ------------------------------------------------------------------

    @property
    def portfolio_value(self) -> float:
        return self._portfolio_value()

    @property
    def total_return(self) -> float:
        return self._portfolio_value() / self._initial_portfolio_value - 1.0

    def sharpe_ratio(self, risk_free: float = 0.0) -> float:
        if len(self._returns_history) < 2:
            return 0.0
        rets = np.array(self._returns_history)
        excess = rets - risk_free / 252
        std = excess.std()
        if std < 1e-8:
            return 0.0
        return float(np.sqrt(252) * excess.mean() / std)


# ═══════════════════════════════════════════════════════════════════════════
#  Convenience: run a full training episode
# ═══════════════════════════════════════════════════════════════════════════

def run_training_episode(
    env: StockTradingEnv,
    agent: Any,  # TD3StockAgent (avoid circular import)
    max_steps: int = 5000,
    train_every: int = 1,
    max_memory_percent: float = 90.0,
    max_cpu_percent: float = 90.0,
) -> Dict[str, Any]:
    """
    Run one episode: agent interacts with env, stores transitions, trains.

    Auto-stops if system RAM or CPU exceeds thresholds (protects Mac resources).

    Returns summary metrics for the episode.
    """
    from .td3_stock_agent import check_system_resources_ok

    # Use agent config if available
    if hasattr(agent, "cfg"):
        max_memory_percent = getattr(agent.cfg, "max_memory_percent", max_memory_percent)
        max_cpu_percent = getattr(agent.cfg, "max_cpu_percent", max_cpu_percent)

    state = env.reset()
    episode_reward = 0.0
    steps = 0
    train_metrics: List[Dict[str, Any]] = []
    resource_limit_hit = False

    for _ in range(max_steps):
        # Auto-stop if system resources are hindered
        ok, resources = check_system_resources_ok(max_memory_percent, max_cpu_percent)
        if not ok:
            logger.warning(
                "TD3 episode stopped: system resources high (memory=%.1f%%, cpu=%.1f%%)",
                resources.get("memory_percent", 0),
                resources.get("cpu_percent", 0),
            )
            resource_limit_hit = True
            break

        action = agent.select_action(state, add_noise=True)
        result = env.step(action)

        agent.store_transition(state, action, result.state, result.reward, result.done)

        if steps % train_every == 0:
            m = agent.train_step()
            if m.get("status") == "trained":
                train_metrics.append(m)
            elif m.get("reason") == "system_resources_high":
                resource_limit_hit = True
                break

        state = result.state
        episode_reward += result.reward
        steps += 1

        if result.done:
            break

    return {
        "episode_reward": episode_reward,
        "steps": steps,
        "total_return": env.total_return,
        "sharpe_ratio": env.sharpe_ratio(),
        "final_portfolio_value": env.portfolio_value,
        "train_updates": len(train_metrics),
        "avg_critic_loss": float(np.mean([m["critic_loss"] for m in train_metrics])) if train_metrics else 0.0,
        "resource_limit_hit": resource_limit_hit,
    }
