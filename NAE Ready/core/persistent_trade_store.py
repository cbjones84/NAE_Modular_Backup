#!/usr/bin/env python3
"""
Persistent Trade Store - Ensures Optimus never loses trade history on restart.

Saves trade_history, open_positions, win rates, P&L, and decision journal
to disk (JSON) so Optimus can learn from weeks/months of past trading.

Also includes:
- Position P&L logger (per-position tracking over time)
- Trade decision journal (WHY each trade was made)
- Goal progress tracker (dynamic strategy adaptation)
"""

import os
import json
import datetime
import threading
import hashlib
from typing import Dict, Any, List, Optional
from pathlib import Path


class PersistentTradeStore:
    """
    JSON-based persistent storage for Optimus trade data.
    Survives restarts so Optimus can learn from past performance.
    """

    def __init__(self, data_dir: Optional[str] = None):
        if data_dir:
            self.data_dir = Path(data_dir)
        else:
            # Default: NAE Ready/data/optimus/
            self.data_dir = Path(__file__).parent.parent / "data" / "optimus"
        
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # File paths
        self.trade_history_file = self.data_dir / "trade_history.json"
        self.positions_file = self.data_dir / "open_positions.json"
        self.win_rates_file = self.data_dir / "win_rates.json"
        self.pnl_file = self.data_dir / "pnl_tracker.json"
        self.decision_journal_file = self.data_dir / "decision_journal.json"
        self.position_pnl_log_file = self.data_dir / "position_pnl_log.json"
        self.goal_progress_file = self.data_dir / "goal_progress.json"
        self.session_file = self.data_dir / "session_state.json"
        
        # Thread lock for concurrent writes
        self._lock = threading.Lock()
        
        # Initialize data structures (load from disk or create fresh)
        self.trade_history: List[Dict[str, Any]] = self._load_json(self.trade_history_file, [])
        self.open_positions: Dict[str, Dict[str, Any]] = self._load_json(self.positions_file, {})
        self.win_rates: Dict[str, Any] = self._load_json(self.win_rates_file, {
            "overall_win_rate": 0.5,
            "recent_win_rate": 0.5,
            "strategy_win_rates": {},
            "total_wins": 0,
            "total_losses": 0,
            "total_trades": 0
        })
        self.pnl_tracker: Dict[str, Any] = self._load_json(self.pnl_file, {
            "realized_pnl": 0.0,
            "unrealized_pnl": 0.0,
            "daily_pnl": 0.0,
            "weekly_pnl": 0.0,
            "monthly_pnl": 0.0,
            "all_time_pnl": 0.0,
            "peak_nav": 0.0,
            "current_drawdown_pct": 0.0,
            "last_reset_date": datetime.date.today().isoformat()
        })
        self.decision_journal: List[Dict[str, Any]] = self._load_json(self.decision_journal_file, [])
        self.position_pnl_log: List[Dict[str, Any]] = self._load_json(self.position_pnl_log_file, [])
        self.goal_progress: Dict[str, Any] = self._load_json(self.goal_progress_file, {
            "start_date": datetime.date.today().isoformat(),
            "starting_capital": 100.0,
            "current_nav": 0.0,
            "target_year_1": 9411.0,
            "target_5m": 5000000.0,
            "progress_history": [],
            "strategy_adjustments": []
        })

    # ========================================================================
    # Low-level JSON I/O
    # ========================================================================
    def _load_json(self, filepath: Path, default: Any) -> Any:
        """Load JSON file, return default if missing or corrupt."""
        try:
            if filepath.exists() and filepath.stat().st_size > 0:
                with open(filepath, "r") as f:
                    return json.load(f)
        except (json.JSONDecodeError, IOError, OSError) as e:
            # Corrupt file - back it up and start fresh
            backup = filepath.with_suffix(f".backup_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            try:
                if filepath.exists():
                    filepath.rename(backup)
            except Exception:
                pass
        return default if not isinstance(default, (list, dict)) else type(default)(default)

    def _save_json(self, filepath: Path, data: Any):
        """Atomically save JSON to disk (write to tmp, then rename)."""
        with self._lock:
            tmp = filepath.with_suffix(".tmp")
            try:
                with open(tmp, "w") as f:
                    json.dump(data, f, indent=2, default=str)
                tmp.replace(filepath)
            except Exception:
                if tmp.exists():
                    tmp.unlink()
                raise

    # ========================================================================
    # Trade History (persistent, survives restarts)
    # ========================================================================
    def record_trade(self, trade: Dict[str, Any]):
        """
        Record a completed trade to persistent history.
        Keeps full history (not just last 50).
        """
        trade_record = {
            "id": hashlib.md5(f"{trade.get('symbol', '')}_{datetime.datetime.now().isoformat()}".encode()).hexdigest()[:12],
            "timestamp": datetime.datetime.now().isoformat(),
            "symbol": trade.get("symbol", "UNKNOWN"),
            "side": trade.get("side", "unknown"),
            "quantity": trade.get("quantity", 0),
            "price": trade.get("price", 0),
            "strategy": trade.get("strategy_name", trade.get("strategy", "default")),
            "pnl": trade.get("pnl", 0),
            "is_win": trade.get("pnl", 0) > 0,
            "status": trade.get("status", "unknown"),
            "entry_price": trade.get("entry_price", 0),
            "exit_price": trade.get("exit_price", trade.get("price", 0)),
            "holding_period_hours": trade.get("holding_period_hours", 0),
            "reason": trade.get("reason", ""),
            "confidence": trade.get("confidence", 0),
            "market_conditions": trade.get("market_conditions", {}),
        }
        self.trade_history.append(trade_record)
        self._save_json(self.trade_history_file, self.trade_history)
        
        # Update win rates
        self._update_win_rates(trade_record)
        
        # Update P&L tracker
        self._update_pnl(trade_record)

    def get_trade_history(self, limit: int = 0, symbol: Optional[str] = None, strategy: Optional[str] = None) -> List[Dict]:
        """Get trade history with optional filters."""
        history = self.trade_history
        if symbol:
            history = [t for t in history if t.get("symbol") == symbol]
        if strategy:
            history = [t for t in history if t.get("strategy") == strategy]
        if limit > 0:
            history = history[-limit:]
        return history

    def get_recent_trades(self, count: int = 20) -> List[Dict]:
        """Get the N most recent trades."""
        return self.trade_history[-count:] if self.trade_history else []

    # ========================================================================
    # Win Rate Tracking (persistent)
    # ========================================================================
    def _update_win_rates(self, trade: Dict[str, Any]):
        """Update persistent win rate tracking."""
        is_win = trade.get("is_win", False)
        strategy = trade.get("strategy", "default")
        
        self.win_rates["total_trades"] = self.win_rates.get("total_trades", 0) + 1
        if is_win:
            self.win_rates["total_wins"] = self.win_rates.get("total_wins", 0) + 1
        else:
            self.win_rates["total_losses"] = self.win_rates.get("total_losses", 0) + 1
        
        total = self.win_rates["total_trades"]
        self.win_rates["overall_win_rate"] = self.win_rates["total_wins"] / total if total > 0 else 0.5
        
        # Recent win rate (last 20 trades)
        recent = self.trade_history[-20:] if len(self.trade_history) >= 20 else self.trade_history
        if recent:
            recent_wins = sum(1 for t in recent if t.get("is_win", False))
            self.win_rates["recent_win_rate"] = recent_wins / len(recent)
        
        # Strategy-specific
        strategy_rates = self.win_rates.get("strategy_win_rates", {})
        strategy_trades = [t for t in self.trade_history if t.get("strategy") == strategy]
        if strategy_trades:
            strat_wins = sum(1 for t in strategy_trades if t.get("is_win", False))
            strategy_rates[strategy] = {
                "win_rate": strat_wins / len(strategy_trades),
                "total": len(strategy_trades),
                "wins": strat_wins
            }
        self.win_rates["strategy_win_rates"] = strategy_rates
        
        self._save_json(self.win_rates_file, self.win_rates)

    def get_win_rate(self, strategy: Optional[str] = None) -> float:
        """Get win rate (overall or for a specific strategy)."""
        if strategy:
            strat_data = self.win_rates.get("strategy_win_rates", {}).get(strategy, {})
            return strat_data.get("win_rate", 0.5)
        return self.win_rates.get("recent_win_rate", 0.5)

    # ========================================================================
    # P&L Tracking (persistent)
    # ========================================================================
    def _update_pnl(self, trade: Dict[str, Any]):
        """Update P&L tracking from a trade."""
        pnl = trade.get("pnl", 0)
        self.pnl_tracker["realized_pnl"] = self.pnl_tracker.get("realized_pnl", 0) + pnl
        self.pnl_tracker["daily_pnl"] = self.pnl_tracker.get("daily_pnl", 0) + pnl
        self.pnl_tracker["weekly_pnl"] = self.pnl_tracker.get("weekly_pnl", 0) + pnl
        self.pnl_tracker["monthly_pnl"] = self.pnl_tracker.get("monthly_pnl", 0) + pnl
        self.pnl_tracker["all_time_pnl"] = self.pnl_tracker.get("all_time_pnl", 0) + pnl
        
        # Reset daily P&L if new day
        today = datetime.date.today().isoformat()
        if self.pnl_tracker.get("last_reset_date") != today:
            self.pnl_tracker["daily_pnl"] = pnl  # First trade of the day
            self.pnl_tracker["last_reset_date"] = today
        
        self._save_json(self.pnl_file, self.pnl_tracker)

    def update_nav(self, nav: float):
        """Update NAV in persistent tracker (call after each balance sync)."""
        peak = self.pnl_tracker.get("peak_nav", 0)
        if nav > peak:
            self.pnl_tracker["peak_nav"] = nav
        if peak > 0:
            self.pnl_tracker["current_drawdown_pct"] = (peak - nav) / peak
        self.pnl_tracker["current_nav"] = nav
        self.pnl_tracker["last_updated"] = datetime.datetime.now().isoformat()
        self._save_json(self.pnl_file, self.pnl_tracker)

    # ========================================================================
    # Position P&L Logger (tracks each position over time)
    # ========================================================================
    def log_position_snapshot(self, positions: List[Dict[str, Any]]):
        """
        Log a snapshot of all open positions with current P&L.
        Call this periodically (e.g., every cycle).
        """
        timestamp = datetime.datetime.now().isoformat()
        snapshot = {
            "timestamp": timestamp,
            "positions": []
        }
        for pos in positions:
            snapshot["positions"].append({
                "symbol": pos.get("symbol", ""),
                "entry_price": pos.get("entry_price", 0),
                "current_price": pos.get("current_price", 0),
                "quantity": pos.get("quantity", 0),
                "pnl_dollars": pos.get("unrealized_pnl", 0),
                "pnl_pct": (
                    (pos.get("current_price", 0) - pos.get("entry_price", 1)) / pos.get("entry_price", 1)
                    if pos.get("entry_price", 0) > 0 else 0
                ),
                "holding_hours": pos.get("holding_hours", 0),
                "cost_basis": pos.get("cost_basis", 0),
            })
        
        self.position_pnl_log.append(snapshot)
        
        # Keep last 500 snapshots (at 30s intervals = ~4 hours of data)
        if len(self.position_pnl_log) > 500:
            self.position_pnl_log = self.position_pnl_log[-500:]
        
        self._save_json(self.position_pnl_log_file, self.position_pnl_log)

    def get_position_history(self, symbol: str, limit: int = 50) -> List[Dict]:
        """Get P&L history for a specific position over time."""
        history = []
        for snapshot in self.position_pnl_log:
            for pos in snapshot.get("positions", []):
                if pos.get("symbol") == symbol:
                    history.append({
                        "timestamp": snapshot["timestamp"],
                        **pos
                    })
        return history[-limit:] if limit > 0 else history

    # ========================================================================
    # Trade Decision Journal (WHY trades were made)
    # ========================================================================
    def journal_decision(self, decision: Dict[str, Any]):
        """
        Record a trade decision with full reasoning.
        
        Args:
            decision: Dict with keys like:
                - action: "buy" / "sell" / "hold"
                - symbol: ticker
                - reason: human-readable reason
                - signals: list of signals that triggered the decision
                - confidence: 0-1 confidence score
                - market_context: current market conditions
                - strategy_used: which strategy generated this
                - outcome: (filled in later) what happened
        """
        entry = {
            "id": hashlib.md5(f"{decision.get('symbol', '')}_{datetime.datetime.now().isoformat()}".encode()).hexdigest()[:12],
            "timestamp": datetime.datetime.now().isoformat(),
            "action": decision.get("action", "unknown"),
            "symbol": decision.get("symbol", "UNKNOWN"),
            "reason": decision.get("reason", ""),
            "signals": decision.get("signals", []),
            "confidence": decision.get("confidence", 0),
            "strategy_used": decision.get("strategy_used", ""),
            "market_context": decision.get("market_context", {}),
            "entry_price": decision.get("entry_price", 0),
            "target_price": decision.get("target_price", 0),
            "stop_loss": decision.get("stop_loss", 0),
            "expected_return_pct": decision.get("expected_return_pct", 0),
            "outcome": decision.get("outcome", "pending"),
            "outcome_pnl": decision.get("outcome_pnl", 0),
        }
        self.decision_journal.append(entry)
        
        # Keep last 500 decisions
        if len(self.decision_journal) > 500:
            self.decision_journal = self.decision_journal[-500:]
        
        self._save_json(self.decision_journal_file, self.decision_journal)
        return entry["id"]

    def update_decision_outcome(self, decision_id: str, outcome: str, pnl: float = 0):
        """Update a decision journal entry with its outcome."""
        for entry in reversed(self.decision_journal):
            if entry.get("id") == decision_id:
                entry["outcome"] = outcome
                entry["outcome_pnl"] = pnl
                entry["outcome_timestamp"] = datetime.datetime.now().isoformat()
                self._save_json(self.decision_journal_file, self.decision_journal)
                return True
        return False

    def get_decision_patterns(self) -> Dict[str, Any]:
        """Analyze decision journal for patterns (what works vs what doesn't)."""
        if not self.decision_journal:
            return {"patterns": [], "total_decisions": 0}
        
        completed = [d for d in self.decision_journal if d.get("outcome") not in ("pending", None)]
        if not completed:
            return {"patterns": [], "total_decisions": len(self.decision_journal), "completed": 0}
        
        # Analyze by strategy
        strategy_results: Dict[str, Dict] = {}
        for d in completed:
            strat = d.get("strategy_used", "unknown")
            if strat not in strategy_results:
                strategy_results[strat] = {"wins": 0, "losses": 0, "total_pnl": 0, "avg_confidence": 0, "count": 0}
            strategy_results[strat]["count"] += 1
            strategy_results[strat]["total_pnl"] += d.get("outcome_pnl", 0)
            strategy_results[strat]["avg_confidence"] += d.get("confidence", 0)
            if d.get("outcome_pnl", 0) > 0:
                strategy_results[strat]["wins"] += 1
            else:
                strategy_results[strat]["losses"] += 1
        
        for strat, data in strategy_results.items():
            if data["count"] > 0:
                data["win_rate"] = data["wins"] / data["count"]
                data["avg_confidence"] /= data["count"]
                data["avg_pnl"] = data["total_pnl"] / data["count"]
        
        # Best and worst strategies
        sorted_strats = sorted(strategy_results.items(), key=lambda x: x[1].get("win_rate", 0), reverse=True)
        
        return {
            "total_decisions": len(self.decision_journal),
            "completed": len(completed),
            "strategy_results": strategy_results,
            "best_strategy": sorted_strats[0][0] if sorted_strats else None,
            "worst_strategy": sorted_strats[-1][0] if len(sorted_strats) > 1 else None,
        }

    # ========================================================================
    # Goal Progress Tracking (dynamic strategy adaptation)
    # ========================================================================
    def update_goal_progress(self, nav: float, starting_capital: float = 100.0):
        """
        Update goal progress and determine if strategy needs adjustment.
        
        Returns strategy adjustment recommendations.
        """
        today = datetime.date.today()
        start_date = datetime.date.fromisoformat(self.goal_progress.get("start_date", today.isoformat()))
        days_active = max(1, (today - start_date).days)
        
        # Year 1 target: $9,411 from $100 starting capital
        year_1_target = 9411.0
        year_1_days = 365
        
        # Expected progress (linear interpolation for simplicity)
        expected_nav_today = starting_capital + (year_1_target - starting_capital) * min(1.0, days_active / year_1_days)
        
        # How far ahead/behind are we?
        progress_pct = (nav / year_1_target * 100) if year_1_target > 0 else 0
        gap_dollars = expected_nav_today - nav
        gap_pct = (gap_dollars / expected_nav_today * 100) if expected_nav_today > 0 else 0
        
        # Determine required daily return to get back on track
        days_remaining = max(1, year_1_days - days_active)
        required_total_return = (year_1_target / max(nav, 1)) - 1
        required_daily_return = (1 + required_total_return) ** (1 / days_remaining) - 1
        
        # Strategy adjustment recommendation
        adjustment = self._recommend_strategy_adjustment(
            nav=nav,
            expected_nav=expected_nav_today,
            gap_pct=gap_pct,
            required_daily_return=required_daily_return,
            days_active=days_active,
            recent_win_rate=self.win_rates.get("recent_win_rate", 0.5)
        )
        
        # Log progress
        progress_entry = {
            "date": today.isoformat(),
            "nav": nav,
            "expected_nav": round(expected_nav_today, 2),
            "gap_dollars": round(gap_dollars, 2),
            "gap_pct": round(gap_pct, 2),
            "progress_pct": round(progress_pct, 4),
            "days_active": days_active,
            "required_daily_return_pct": round(required_daily_return * 100, 4),
            "adjustment": adjustment["recommendation"]
        }
        
        # Only log once per day
        history = self.goal_progress.get("progress_history", [])
        if not history or history[-1].get("date") != today.isoformat():
            history.append(progress_entry)
            # Keep last 365 days
            if len(history) > 365:
                history = history[-365:]
            self.goal_progress["progress_history"] = history
        
        self.goal_progress["current_nav"] = nav
        self.goal_progress["last_updated"] = datetime.datetime.now().isoformat()
        self._save_json(self.goal_progress_file, self.goal_progress)
        
        return adjustment

    def _recommend_strategy_adjustment(
        self, nav: float, expected_nav: float, gap_pct: float,
        required_daily_return: float, days_active: int, recent_win_rate: float
    ) -> Dict[str, Any]:
        """
        Generate strategy adjustment recommendations based on goal progress.
        
        Returns dict with:
            - recommendation: str description
            - risk_multiplier: float (1.0 = normal, >1 = more aggressive, <1 = more conservative)
            - position_size_adjustment: float multiplier
            - strategy_preference: str (which strategy to favor)
            - urgency: str ("low", "medium", "high", "critical")
        """
        # Behind schedule
        if gap_pct > 50:
            return {
                "recommendation": f"CRITICALLY BEHIND ({gap_pct:.0f}% behind). Max aggressive: larger positions, higher-frequency trades, momentum strategies.",
                "risk_multiplier": 1.5,
                "position_size_adjustment": 1.3,
                "strategy_preference": "momentum_scalping",
                "urgency": "critical",
                "max_position_pct": 0.30,
                "trade_frequency": "high",
            }
        elif gap_pct > 20:
            return {
                "recommendation": f"Behind schedule ({gap_pct:.0f}%). Increase position sizes and trade frequency.",
                "risk_multiplier": 1.3,
                "position_size_adjustment": 1.2,
                "strategy_preference": "aggressive_momentum",
                "urgency": "high",
                "max_position_pct": 0.25,
                "trade_frequency": "medium-high",
            }
        elif gap_pct > 5:
            return {
                "recommendation": f"Slightly behind ({gap_pct:.0f}%). Maintain aggressive stance, optimize entries.",
                "risk_multiplier": 1.1,
                "position_size_adjustment": 1.1,
                "strategy_preference": "balanced_aggressive",
                "urgency": "medium",
                "max_position_pct": 0.20,
                "trade_frequency": "medium",
            }
        elif gap_pct > -10:
            # On track or slightly ahead
            return {
                "recommendation": f"On track ({abs(gap_pct):.0f}% {'ahead' if gap_pct < 0 else 'behind'}). Maintain current strategy.",
                "risk_multiplier": 1.0,
                "position_size_adjustment": 1.0,
                "strategy_preference": "current",
                "urgency": "low",
                "max_position_pct": 0.20,
                "trade_frequency": "normal",
            }
        else:
            # Ahead of schedule
            return {
                "recommendation": f"AHEAD of schedule ({abs(gap_pct):.0f}% ahead). Can afford slightly conservative approach to lock in gains.",
                "risk_multiplier": 0.9,
                "position_size_adjustment": 0.9,
                "strategy_preference": "capital_preservation",
                "urgency": "low",
                "max_position_pct": 0.15,
                "trade_frequency": "normal",
            }

    # ========================================================================
    # Open Positions (persistent)
    # ========================================================================
    def save_positions(self, positions: Dict[str, Dict[str, Any]]):
        """Save current open positions to disk."""
        self.open_positions = positions
        self._save_json(self.positions_file, self.open_positions)

    def load_positions(self) -> Dict[str, Dict[str, Any]]:
        """Load open positions from disk."""
        return self.open_positions

    # ========================================================================
    # Session State (save/restore full Optimus state)
    # ========================================================================
    def save_session(self, state: Dict[str, Any]):
        """Save current session state (NAV, starting NAV, phase, etc.)."""
        session = {
            "timestamp": datetime.datetime.now().isoformat(),
            "nav": state.get("nav", 0),
            "starting_nav": state.get("starting_nav", 0),
            "peak_nav": state.get("peak_nav", 0),
            "realized_pnl": state.get("realized_pnl", 0),
            "unrealized_pnl": state.get("unrealized_pnl", 0),
            "daily_pnl": state.get("daily_pnl", 0),
            "monthly_realized_profit": state.get("monthly_realized_profit", 0),
            "current_phase": state.get("current_phase", "accumulation"),
            "total_trades_executed": state.get("total_trades_executed", 0),
            "trading_enabled": state.get("trading_enabled", True),
            "accelerator_enabled": state.get("accelerator_enabled", True),
            "recent_win_rate": state.get("recent_win_rate", 0.5),
            "consecutive_losses": state.get("consecutive_losses", 0),
        }
        self._save_json(self.session_file, session)

    def load_session(self) -> Optional[Dict[str, Any]]:
        """Load previous session state."""
        data = self._load_json(self.session_file, None)
        return data if data else None

    # ========================================================================
    # Summary / Analytics
    # ========================================================================
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get a comprehensive performance summary from persistent data."""
        total_trades = len(self.trade_history)
        if total_trades == 0:
            return {
                "total_trades": 0,
                "message": "No trade history yet"
            }
        
        wins = sum(1 for t in self.trade_history if t.get("is_win", False))
        losses = total_trades - wins
        total_pnl = sum(t.get("pnl", 0) for t in self.trade_history)
        avg_win = 0
        avg_loss = 0
        
        winning_trades = [t for t in self.trade_history if t.get("pnl", 0) > 0]
        losing_trades = [t for t in self.trade_history if t.get("pnl", 0) <= 0]
        
        if winning_trades:
            avg_win = sum(t["pnl"] for t in winning_trades) / len(winning_trades)
        if losing_trades:
            avg_loss = sum(t["pnl"] for t in losing_trades) / len(losing_trades)
        
        # Profit factor
        gross_profit = sum(t["pnl"] for t in winning_trades) if winning_trades else 0
        gross_loss = abs(sum(t["pnl"] for t in losing_trades)) if losing_trades else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        return {
            "total_trades": total_trades,
            "wins": wins,
            "losses": losses,
            "win_rate": wins / total_trades,
            "total_pnl": round(total_pnl, 2),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "profit_factor": round(profit_factor, 2),
            "best_trade": max(self.trade_history, key=lambda t: t.get("pnl", 0)),
            "worst_trade": min(self.trade_history, key=lambda t: t.get("pnl", 0)),
            "strategies_used": list(set(t.get("strategy", "unknown") for t in self.trade_history)),
            "symbols_traded": list(set(t.get("symbol", "UNKNOWN") for t in self.trade_history)),
        }
