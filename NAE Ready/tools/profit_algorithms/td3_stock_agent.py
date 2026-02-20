"""
TD3 (Twin-Delayed DDPG) Stock Trading Agent for NAE

Ported from austin-starks/Deep-RL-Stocks and adapted for NAE's architecture.

Paper: "Addressing Function Approximation Error in Actor-Critic Methods"
       Fujimoto, Van Hoof & Meger (2018) — arxiv:1802.09477

Key adaptations from the original Deep-RL-Stocks repository:
  - MLP actor/critic instead of CNN (NAE works with feature vectors, not images)
  - Configurable multi-stock action space
  - PyTorch backend for training; numpy fallback for inference-only
  - Integration hooks for OptimusAgent's trade-evaluation pipeline
  - Persistent save/load in NAE's model directory
"""

from __future__ import annotations

import copy
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional PyTorch import — training requires it; inference can fall back
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
    _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except ImportError:
    TORCH_AVAILABLE = False
    _DEVICE = None
    logger.info("PyTorch not installed — TD3 agent available in numpy-inference mode only")

# Module-level availability (Optimus imports this)
TD3_AVAILABLE = True


# ═══════════════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class TD3Config:
    """All hyper-parameters for the TD3 agent."""

    num_stocks: int = 1
    state_dim: int = 32
    hidden_dims: Tuple[int, ...] = (256, 256)

    max_action: float = 1.0
    discount: float = 0.99
    tau: float = 0.005
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_freq: int = 2

    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    grad_clip: float = 1.0

    replay_capacity: int = 200_000
    batch_size: int = 128
    warmup_steps: int = 1_000
    exploration_noise: float = 0.1

    # Resource limits — auto-stop training when exceeded (protects Mac RAM/CPU)
    max_memory_percent: float = 90.0
    max_cpu_percent: float = 90.0

    model_dir: str = ""

    def __post_init__(self):
        if not self.model_dir:
            self.model_dir = str(
                Path(__file__).resolve().parent.parent.parent / "models" / "td3"
            )

    @property
    def action_dim(self) -> int:
        return self.num_stocks


# ═══════════════════════════════════════════════════════════════════════════
#  System resource check — auto-stop training when RAM/CPU is hindered
# ═══════════════════════════════════════════════════════════════════════════

def check_system_resources_ok(
    max_memory_percent: float = 90.0,
    max_cpu_percent: float = 90.0,
) -> Tuple[bool, Dict[str, float]]:
    """
    Check if system RAM and CPU are below thresholds.
    Returns (ok, {"memory_percent": ..., "cpu_percent": ...}).
    If psutil is unavailable, returns (True, {}) to allow training to proceed.
    """
    try:
        import psutil
        mem = psutil.virtual_memory()
        cpu = psutil.cpu_percent(interval=0.1)
        ok = mem.percent < max_memory_percent and cpu < max_cpu_percent
        return ok, {"memory_percent": mem.percent, "cpu_percent": cpu}
    except Exception as e:
        logger.debug("Resource check failed (proceeding): %s", e)
        return True, {}


# ═══════════════════════════════════════════════════════════════════════════
#  Replay Buffer  (numpy-backed, same circular-buffer pattern as the
#  original Deep-RL-Stocks ReplayBuffer but device-agnostic)
# ═══════════════════════════════════════════════════════════════════════════

class TD3ReplayBuffer:
    """Fixed-size circular replay buffer backed by pre-allocated numpy arrays."""

    def __init__(self, state_dim: int, action_dim: int, capacity: int = 200_000):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0

        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.not_dones = np.zeros((capacity, 1), dtype=np.float32)

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        reward: float,
        done: bool,
    ) -> None:
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.next_states[self.ptr] = next_state
        self.rewards[self.ptr] = reward
        self.not_dones[self.ptr] = 1.0 - float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.size, size=batch_size)
        if TORCH_AVAILABLE:
            return (
                torch.FloatTensor(self.states[idx]).to(_DEVICE),
                torch.FloatTensor(self.actions[idx]).to(_DEVICE),
                torch.FloatTensor(self.next_states[idx]).to(_DEVICE),
                torch.FloatTensor(self.rewards[idx]).to(_DEVICE),
                torch.FloatTensor(self.not_dones[idx]).to(_DEVICE),
            )
        return (
            self.states[idx],
            self.actions[idx],
            self.next_states[idx],
            self.rewards[idx],
            self.not_dones[idx],
        )

    def __len__(self) -> int:
        return self.size


# ═══════════════════════════════════════════════════════════════════════════
#  PyTorch Networks (only defined when torch is available)
# ═══════════════════════════════════════════════════════════════════════════

if TORCH_AVAILABLE:

    class _Actor(nn.Module):
        """Deterministic policy: state → action ∈ [-max_action, max_action]."""

        def __init__(self, state_dim: int, action_dim: int, hidden_dims: Tuple[int, ...], max_action: float):
            super().__init__()
            layers: List[nn.Module] = []
            prev = state_dim
            for h in hidden_dims:
                layers += [nn.Linear(prev, h), nn.ReLU()]
                prev = h
            layers.append(nn.Linear(prev, action_dim))
            self.net = nn.Sequential(*layers)
            self.max_action = max_action
            self._init_weights()

        def _init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        def forward(self, state: torch.Tensor) -> torch.Tensor:
            return self.max_action * torch.tanh(self.net(state))

    class _Critic(nn.Module):
        """Twin Q-networks: (state, action) → (Q1, Q2)."""

        def __init__(self, state_dim: int, action_dim: int, hidden_dims: Tuple[int, ...]):
            super().__init__()

            def _build_q(in_dim: int) -> nn.Sequential:
                layers: List[nn.Module] = []
                prev = in_dim
                for h in hidden_dims:
                    layers += [nn.Linear(prev, h), nn.ReLU()]
                    prev = h
                layers.append(nn.Linear(prev, 1))
                return nn.Sequential(*layers)

            sa_dim = state_dim + action_dim
            self.q1 = _build_q(sa_dim)
            self.q2 = _build_q(sa_dim)
            self._init_weights()

        def _init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            sa = torch.cat([state, action], dim=-1)
            return self.q1(sa), self.q2(sa)

        def q1_forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
            sa = torch.cat([state, action], dim=-1)
            return self.q1(sa)


# ═══════════════════════════════════════════════════════════════════════════
#  Numpy-only lightweight policy for inference without PyTorch
# ═══════════════════════════════════════════════════════════════════════════

class _NumpyActor:
    """MLP policy using numpy weights — inference only, no training."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dims: Tuple[int, ...], max_action: float):
        self.max_action = max_action
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []
        prev = state_dim
        for h in hidden_dims:
            self.weights.append(np.random.randn(prev, h).astype(np.float32) * 0.01)
            self.biases.append(np.zeros(h, dtype=np.float32))
            prev = h
        self.weights.append(np.random.randn(prev, action_dim).astype(np.float32) * 0.01)
        self.biases.append(np.zeros(action_dim, dtype=np.float32))

    def predict(self, state: np.ndarray) -> np.ndarray:
        x = state.astype(np.float32)
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            x = np.maximum(0.0, x @ w + b)  # ReLU
        x = x @ self.weights[-1] + self.biases[-1]
        return self.max_action * np.tanh(x)

    def load_from_state_dict(self, state_dict: Dict[str, np.ndarray]) -> None:
        """Load weights from either a PyTorch export or a numpy-mode save."""
        new_weights: List[np.ndarray] = []
        new_biases: List[np.ndarray] = []

        # Try PyTorch Sequential format first: net.0.weight, net.2.weight, …
        layer_idx = 0
        while True:
            w_key = f"net.{layer_idx}.weight"
            b_key = f"net.{layer_idx}.bias"
            if w_key not in state_dict:
                break
            new_weights.append(state_dict[w_key].T)  # torch stores [out, in]
            new_biases.append(state_dict[b_key])
            layer_idx += 2  # skip ReLU entries in Sequential numbering

        # Fallback: numpy-mode format w0, w1, …, b0, b1, …
        if not new_weights:
            i = 0
            while f"w{i}" in state_dict:
                new_weights.append(state_dict[f"w{i}"])
                new_biases.append(state_dict[f"b{i}"])
                i += 1

        if new_weights:
            self.weights = new_weights
            self.biases = new_biases


# ═══════════════════════════════════════════════════════════════════════════
#  TD3 Stock Agent — main public class
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class TD3Signal:
    """Signal returned by the TD3 agent for Optimus consumption."""

    actions: np.ndarray            # raw [-1, 1] per stock
    stock_names: List[str]
    confidence: float              # 0-1 average absolute action magnitude
    recommendation: str            # "buy" | "sell" | "hold"
    details: Dict[str, Any] = field(default_factory=dict)


class TD3StockAgent:
    """
    Twin-Delayed DDPG agent for multi-stock equity trading.

    Usage::

        agent = TD3StockAgent(config)
        signal = agent.get_signal(state_vector, stock_names)
        # ... execute trades based on signal ...
        agent.store_transition(state, action, next_state, reward, done)
        metrics = agent.train_step()
        agent.save()
    """

    def __init__(self, config: Optional[TD3Config] = None):
        self.cfg = config or TD3Config()
        self.total_steps = 0

        # Replay buffer (always available, numpy-backed)
        self.replay_buffer = TD3ReplayBuffer(
            state_dim=self.cfg.state_dim,
            action_dim=self.cfg.action_dim,
            capacity=self.cfg.replay_capacity,
        )

        self._torch_mode = False

        if TORCH_AVAILABLE:
            self._init_torch()
        else:
            self._init_numpy()

        logger.info(
            "TD3StockAgent initialised  stocks=%d  state_dim=%d  torch=%s",
            self.cfg.num_stocks,
            self.cfg.state_dim,
            self._torch_mode,
        )

    # ------------------------------------------------------------------
    #  Initialisation helpers
    # ------------------------------------------------------------------

    def _init_torch(self) -> None:
        c = self.cfg
        self.actor = _Actor(c.state_dim, c.action_dim, c.hidden_dims, c.max_action).to(_DEVICE)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=c.lr_actor)

        self.critic = _Critic(c.state_dim, c.action_dim, c.hidden_dims).to(_DEVICE)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=c.lr_critic)

        self._torch_mode = True

    def _init_numpy(self) -> None:
        self._np_actor = _NumpyActor(
            self.cfg.state_dim,
            self.cfg.action_dim,
            self.cfg.hidden_dims,
            self.cfg.max_action,
        )
        self._torch_mode = False

    # ------------------------------------------------------------------
    #  Action selection
    # ------------------------------------------------------------------

    def select_action(self, state: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """Return continuous action vector ∈ [-max_action, max_action]."""
        if self._torch_mode:
            with torch.no_grad():
                s = torch.FloatTensor(state.reshape(1, -1)).to(_DEVICE)
                action = self.actor(s).cpu().numpy().flatten()
        else:
            action = self._np_actor.predict(state.reshape(1, -1)).flatten()

        if add_noise:
            noise = np.random.normal(0, self.cfg.max_action * self.cfg.exploration_noise, size=action.shape)
            action = (action + noise).clip(-self.cfg.max_action, self.cfg.max_action)

        return action

    def get_signal(self, state: np.ndarray, stock_names: List[str]) -> TD3Signal:
        """
        High-level interface for OptimusAgent.

        Returns a TD3Signal with per-stock recommendations derived from
        the continuous action output.
        """
        actions = self.select_action(state, add_noise=False)
        avg_magnitude = float(np.mean(np.abs(actions)))

        net = float(np.mean(actions))
        if net > 0.15:
            rec = "buy"
        elif net < -0.15:
            rec = "sell"
        else:
            rec = "hold"

        per_stock: Dict[str, float] = {}
        for name, a in zip(stock_names, actions):
            per_stock[name] = float(a)

        return TD3Signal(
            actions=actions,
            stock_names=stock_names,
            confidence=min(1.0, avg_magnitude / self.cfg.max_action),
            recommendation=rec,
            details={
                "per_stock_actions": per_stock,
                "total_steps": self.total_steps,
                "torch_mode": self._torch_mode,
            },
        )

    # ------------------------------------------------------------------
    #  Experience storage
    # ------------------------------------------------------------------

    def store_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        reward: float,
        done: bool,
    ) -> None:
        self.replay_buffer.add(state, action, next_state, reward, done)
        self.total_steps += 1

    # ------------------------------------------------------------------
    #  Training (requires PyTorch)
    # ------------------------------------------------------------------

    def train_step(self) -> Dict[str, Any]:
        """One gradient update on actor + critic.  Returns training metrics."""
        if not self._torch_mode:
            return {"status": "skipped", "reason": "torch_not_available"}
        if len(self.replay_buffer) < self.cfg.warmup_steps:
            return {"status": "warming_up", "buffer_size": len(self.replay_buffer)}

        # Auto-stop if system RAM or CPU is hindered
        ok, resources = check_system_resources_ok(
            self.cfg.max_memory_percent,
            self.cfg.max_cpu_percent,
        )
        if not ok:
            logger.warning(
                "TD3 training paused: system resources high (memory=%.1f%%, cpu=%.1f%%)",
                resources.get("memory_percent", 0),
                resources.get("cpu_percent", 0),
            )
            return {
                "status": "skipped",
                "reason": "system_resources_high",
                "memory_percent": resources.get("memory_percent"),
                "cpu_percent": resources.get("cpu_percent"),
            }

        c = self.cfg
        state, action, next_state, reward, not_done = self.replay_buffer.sample(c.batch_size)

        with torch.no_grad():
            noise = (torch.randn_like(action) * c.policy_noise).clamp(-c.noise_clip, c.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-c.max_action, c.max_action)

            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward + not_done * c.discount * target_q

        current_q1, current_q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), c.grad_clip)
        self.critic_optimizer.step()

        metrics: Dict[str, Any] = {
            "status": "trained",
            "critic_loss": critic_loss.item(),
            "buffer_size": len(self.replay_buffer),
        }

        # Delayed policy update (core TD3 trick)
        if self.total_steps % c.policy_freq == 0:
            actor_loss = -self.critic.q1_forward(state, self.actor(state)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), c.grad_clip)
            self.actor_optimizer.step()

            # Soft-update target networks
            for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
                tp.data.copy_(c.tau * p.data + (1 - c.tau) * tp.data)
            for p, tp in zip(self.actor.parameters(), self.actor_target.parameters()):
                tp.data.copy_(c.tau * p.data + (1 - c.tau) * tp.data)

            metrics["actor_loss"] = actor_loss.item()

        return metrics

    # ------------------------------------------------------------------
    #  Persistence
    # ------------------------------------------------------------------

    def save(self, path: Optional[str] = None) -> str:
        """Save model weights.  Returns the directory used."""
        save_dir = path or self.cfg.model_dir
        os.makedirs(save_dir, exist_ok=True)

        if self._torch_mode:
            torch.save(self.actor.state_dict(), os.path.join(save_dir, "td3_actor.pt"))
            torch.save(self.critic.state_dict(), os.path.join(save_dir, "td3_critic.pt"))
            torch.save(self.actor_optimizer.state_dict(), os.path.join(save_dir, "td3_actor_opt.pt"))
            torch.save(self.critic_optimizer.state_dict(), os.path.join(save_dir, "td3_critic_opt.pt"))

            # Also export numpy-compatible weights for inference fallback
            np_state = {k: v.cpu().numpy() for k, v in self.actor.state_dict().items()}
            np.savez(os.path.join(save_dir, "td3_actor_np.npz"), **np_state)
        else:
            np.savez(
                os.path.join(save_dir, "td3_actor_np.npz"),
                **{f"w{i}": w for i, w in enumerate(self._np_actor.weights)},
                **{f"b{i}": b for i, b in enumerate(self._np_actor.biases)},
            )

        np.save(os.path.join(save_dir, "td3_config.npy"), {
            "total_steps": self.total_steps,
            "state_dim": self.cfg.state_dim,
            "action_dim": self.cfg.action_dim,
            "num_stocks": self.cfg.num_stocks,
        })
        logger.info("TD3 model saved to %s", save_dir)
        return save_dir

    def load(self, path: Optional[str] = None) -> bool:
        """Load model weights. Returns True on success."""
        load_dir = path or self.cfg.model_dir

        try:
            if self._torch_mode:
                actor_path = os.path.join(load_dir, "td3_actor.pt")
                if os.path.exists(actor_path):
                    self.actor.load_state_dict(torch.load(actor_path, map_location=_DEVICE))
                    self.actor_target = copy.deepcopy(self.actor)
                    critic_path = os.path.join(load_dir, "td3_critic.pt")
                    if os.path.exists(critic_path):
                        self.critic.load_state_dict(torch.load(critic_path, map_location=_DEVICE))
                        self.critic_target = copy.deepcopy(self.critic)
                    logger.info("TD3 torch model loaded from %s", load_dir)
                    return True

            # Fallback: load numpy weights
            np_path = os.path.join(load_dir, "td3_actor_np.npz")
            if os.path.exists(np_path):
                data = dict(np.load(np_path, allow_pickle=True))
                if not self._torch_mode:
                    if hasattr(self, "_np_actor"):
                        self._np_actor.load_from_state_dict(data)
                logger.info("TD3 numpy model loaded from %s", load_dir)
                return True

        except Exception as e:
            logger.warning("TD3 model load failed: %s", e)

        return False


# ═══════════════════════════════════════════════════════════════════════════
#  Convenience constructors
# ═══════════════════════════════════════════════════════════════════════════

def create_td3_agent(
    num_stocks: int = 1,
    state_dim: int = 32,
    model_dir: str = "",
    **overrides: Any,
) -> TD3StockAgent:
    """Factory that builds a TD3StockAgent with sensible defaults."""
    cfg = TD3Config(num_stocks=num_stocks, state_dim=state_dim, model_dir=model_dir, **overrides)
    agent = TD3StockAgent(cfg)
    agent.load()  # try to resume from disk
    return agent
