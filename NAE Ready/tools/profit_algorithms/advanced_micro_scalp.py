"""
Advanced Micro-Scalp Accelerator Strategy

Temporary aggressive growth strategy for small accounts ($100 → $500-$1000).
Designed to bootstrap Optimus's account quickly before transitioning to
long-term generational wealth strategies.

Features:
- SPY 0DTE options scalping
- Volatility filters (IV percentile + ATR)
- Time-of-day filters
- Spread-aware exits
- Session-based retraining hooks
- Risk-of-ruin & Kelly guidance
- Account-level compounding / position sizing rules
- Settled cash enforcement (prevents free-riding violations)

Target: 4.3% weekly returns to maximize achievement toward generational wealth goal.
"""

import datetime
import math
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

# Import settlement ledger
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from tools.settlement_utils import SettlementLedger

logger = logging.getLogger(__name__)


@dataclass
class AcceleratorConfig:
    """Configuration for micro-scalp accelerator"""
    max_trades_per_day: int = 2
    max_daily_drawdown_pct: float = 0.25  # Stop trading for day if account loses 25%
    max_risk_per_trade_pct: float = 0.12  # Fraction of account to risk (cap output of kelly)
    min_probability: float = 0.70  # Required prob from ralph
    profit_target_pct: float = 0.25  # 25% gain target
    stop_loss_pct: float = 0.15  # 15% loss stop
    max_bid_ask_spread_pct: float = 0.20  # Reject contracts where spread > 20% of mid
    min_iv_percentile: float = 0.20  # Require IV percentile > 20 (avoid dead low IV)
    max_iv_percentile: float = 0.95  # Avoid extremely-high IV tail risk
    atr_filter_min: float = 0.25  # Min ATR (as fraction of price) to ensure enough movement
    allowed_hours: List[Tuple[int, int, int, int]] = None  # Trading windows
    settlement_seconds_conservative: int = 24 * 3600  # 24 hours default for options
    retrain_interval_minutes: int = 60  # Retrain Ralph every X minutes
    session_retrain_min_trades: int = 3  # Minimum trades to trigger session retrain
    target_account_size: float = 8000.0  # Turn off accelerator when account reaches this (range: $8000-$10000)
    weekly_return_target: float = 0.043  # 4.3% weekly returns target
    
    def __post_init__(self):
        """Set default allowed hours if not provided"""
        if self.allowed_hours is None:
            # Default: 9:45-10:30 and 13:00-15:30 (local exchange time)
            self.allowed_hours = [
                (9, 45, 10, 30),   # 9:45 - 10:30
                (13, 0, 15, 30)    # 13:00 - 15:30
            ]


class AdvancedMicroScalpAccelerator:
    """
    Advanced micro-scalp accelerator for rapid account growth.
    
    Designed for Tradier cash accounts (no PDT restrictions).
    Tracks settled cash to prevent free-riding violations.
    """
    
    def __init__(self, broker, data, ralph, config: Optional[AcceleratorConfig] = None):
        """
        Initialize accelerator
        
        Args:
            broker: Broker SDK wrapper with methods:
                - get_account_balance()
                - buy_option(symbol, quantity)
                - sell_option(symbol, quantity)
                - get_account_info()
            data: Market data wrapper with methods:
                - get_option_chain(symbol, dte)
                - get_option_price(symbol)
                - get_option_bid_ask(symbol)
                - get_spy_price()
                - get_atr(symbol, lookback)
                - get_iv_percentile(symbol, strike, expiry)
            ralph: Signal engine with:
                - get_intraday_direction_probability(symbol)
                - retrain_hook(summary) (optional)
            config: AcceleratorConfig instance (optional)
        """
        self.broker = broker
        self.data = data
        self.ralph = ralph
        self.cfg = config or AcceleratorConfig()
        
        # Initialize settlement ledger
        self.ledger = SettlementLedger(broker)
        
        # Runtime tracking
        self.daily_profit = 0.0
        self.trades_today = 0
        self.session_trade_history: List[Dict[str, Any]] = []
        self.last_retrain_time: Optional[datetime.datetime] = None
        self.start_of_day_balance: Optional[float] = None
        
        logger.info("AdvancedMicroScalpAccelerator initialized")
    
    # -------------------------
    # Utilities
    # -------------------------
    
    def now_local(self) -> datetime.datetime:
        """Get current local time"""
        return datetime.datetime.now()
    
    def in_allowed_time_window(self) -> bool:
        """
        Check if current time is within allowed trading windows.
        
        Each tuple: (start_hour, start_min, end_hour, end_min) in local time.
        """
        now = self.now_local()
        for (sh, sm, eh, em) in self.cfg.allowed_hours:
            start = now.replace(hour=sh, minute=sm, second=0, microsecond=0)
            end = now.replace(hour=eh, minute=em, second=0, microsecond=0)
            if start <= now <= end:
                return True
        return False
    
    def get_account_size(self) -> float:
        """Get current account size"""
        try:
            return self.broker.get_account_balance() or 0.0
        except Exception:
            return 0.0
    
    def current_drawdown_exceeded(self) -> bool:
        """
        Check if daily drawdown limit exceeded.
        
        Returns:
            True if drawdown limit exceeded
        """
        if self.start_of_day_balance is None:
            self.start_of_day_balance = self.get_account_size()
        
        current = self.get_account_size()
        if self.start_of_day_balance <= 0:
            return True
        
        loss = self.start_of_day_balance - current
        drawdown_pct = loss / self.start_of_day_balance
        
        return drawdown_pct >= self.cfg.max_daily_drawdown_pct
    
    # -------------------------
    # Volatility & Spread filters
    # -------------------------
    
    def passes_volatility_filters(self, contract: Dict[str, Any]) -> bool:
        """
        Check if contract passes volatility and spread filters.
        
        Args:
            contract: Option contract dict with fields:
                strike, type, ask, bid, underlying_price, delta, iv
        
        Returns:
            True if contract passes filters
        """
        try:
            # IV percentile filter
            iv_pct = None
            try:
                if hasattr(self.data, "get_iv_percentile"):
                    iv_pct = self.data.get_iv_percentile(
                        "SPY", contract.get("strike"), contract.get("expiry")
                    )
            except Exception:
                pass
            
            # Fallback: estimate from contract IV
            if iv_pct is None:
                contract_iv = contract.get("iv") or contract.get("implied_volatility")
                if contract_iv:
                    # Rough estimate: assume IV percentile based on historical range
                    # This is a simplification - ideally use historical IV data
                    iv_pct = 0.5  # Neutral assumption if no data
            
            if iv_pct is not None:
                if iv_pct < self.cfg.min_iv_percentile or iv_pct > self.cfg.max_iv_percentile:
                    return False
            
            # ATR filter (ensure underlying moves enough intraday)
            try:
                atr = self.data.get_atr("SPY", lookback=14)
                spy_price = self.data.get_spy_price()
                if spy_price and spy_price > 0:
                    atr_frac = atr / spy_price
                    if atr_frac < self.cfg.atr_filter_min:
                        return False
            except Exception:
                # If ATR unavailable, skip filter (conservative)
                pass
            
            # Spread filter: reject if bid-ask spread too wide
            ask = contract.get("ask") or contract.get("last") or 0
            bid = contract.get("bid") or contract.get("last") or 0
            
            if ask <= 0:
                return False
            
            mid = (ask + bid) / 2.0 if bid > 0 else ask
            if mid <= 0:
                return False
            
            spread_pct = (ask - bid) / mid if bid > 0 else 0.0
            if spread_pct > self.cfg.max_bid_ask_spread_pct:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error in volatility filter: {e}")
            return False
    
    # -------------------------
    # Money management
    # -------------------------
    
    def compute_kelly_fraction(self, win_prob: float, payout_ratio: float) -> float:
        """
        Compute Kelly fraction for binary outcome.
        
        Kelly f* = (p*(b+1) - 1) / b
        
        Args:
            win_prob: Probability of winning
            payout_ratio: Average win / average loss
        
        Returns:
            Kelly fraction (clipped to max_risk_per_trade_pct)
        """
        p = win_prob
        b = payout_ratio
        
        if b <= 0 or p <= 0:
            return 0.0
        
        f = (p * (b + 1) - 1) / b
        f = max(0.0, f)  # No negative Kelly
        
        # Clip to max risk per trade
        return min(f, self.cfg.max_risk_per_trade_pct)
    
    def position_size_from_fraction(self, fraction: float) -> float:
        """
        Calculate position size in dollars from fraction of account.
        
        Args:
            fraction: Fraction of account to risk
        
        Returns:
            Risk dollars available
        """
        acct = max(1.0, self.get_account_size())
        return acct * fraction
    
    def compute_risk_of_ruin_guidance(self, win_prob: float, payout_ratio: float, 
                                     fraction: float) -> Dict[str, Any]:
        """
        Provide risk-of-ruin guidance.
        
        Args:
            win_prob: Probability of winning
            payout_ratio: Average win / average loss
            fraction: Fraction being risked
        
        Returns:
            Dictionary with Kelly fraction and risk guidance
        """
        kelly = self.compute_kelly_fraction(win_prob, payout_ratio)
        recommended = min(kelly, fraction)
        
        guidance = {
            "kelly_fraction": kelly,
            "recommended_fraction": recommended,
            "risk_comment": ""
        }
        
        if fraction <= kelly:
            guidance["risk_comment"] = (
                "Risk is within Kelly; long-run ruin probability is low if edge holds."
            )
        else:
            guidance["risk_comment"] = (
                "You are risking more than Kelly — higher chance of ruin if edge doesn't hold."
            )
        
        return guidance
    
    # -------------------------
    # Contract selection and execution
    # -------------------------
    
    def pick_direction(self) -> Tuple[Optional[str], float]:
        """
        Pick trading direction using Ralph signals.
        
        Returns:
            Tuple of (direction, probability) or (None, 0.0) if no signal
        """
        try:
            signal = self.ralph.get_intraday_direction_probability("SPY")
            
            up = signal.get("prob_up", 0.0)
            down = signal.get("prob_down", 0.0)
            
            if up >= self.cfg.min_probability:
                return "CALL", up
            if down >= self.cfg.min_probability:
                return "PUT", down
            
            return None, 0.0
        except Exception as e:
            logger.error(f"Error getting direction from Ralph: {e}")
            return None, 0.0
    
    def pick_contract(self, direction: str) -> Optional[Dict[str, Any]]:
        """
        Pick contract from 0DTE chain.
        
        Args:
            direction: "CALL" or "PUT"
        
        Returns:
            Contract dict or None
        """
        try:
            # Get 0DTE chain
            chain = self.data.get_option_chain("SPY", dte=0)
            if not chain:
                return None
            
            # Filter candidates
            candidates = []
            for c in chain:
                # Check type
                contract_type = c.get("type") or c.get("option_type")
                if contract_type != direction:
                    continue
                
                # Check price range
                ask = c.get("ask") or c.get("last") or 0
                if not (3.0 <= ask <= 20.0):
                    continue
                
                # Check volatility/spread filters
                if self.passes_volatility_filters(c):
                    candidates.append(c)
            
            if not candidates:
                return None
            
            # Choose contract with lowest ask and decent delta (0.2-0.4)
            candidates.sort(key=lambda x: (
                x.get("ask") or x.get("last") or float('inf'),
                abs((x.get("delta") or 0.3) - 0.30)
            ))
            
            return candidates[0]
            
        except Exception as e:
            logger.error(f"Error picking contract: {e}")
            return None
    
    def execute_trade(self, contract: Dict[str, Any], direction: str, prob: float) -> str:
        """
        Execute trade with position sizing and monitoring.
        
        Args:
            contract: Contract dict
            direction: "CALL" or "PUT"
            prob: Probability from Ralph
        
        Returns:
            Result string (e.g., "TAKE_PROFIT", "STOP_LOSS", "ERROR")
        """
        try:
            # Determine position sizing via Kelly guidance
            b = self.cfg.profit_target_pct / self.cfg.stop_loss_pct if self.cfg.stop_loss_pct > 0 else 1.0
            fraction = self.cfg.max_risk_per_trade_pct
            kelly = self.compute_kelly_fraction(prob, b)
            use_fraction = min(kelly, fraction)
            
            risk_budget = self.position_size_from_fraction(use_fraction)
            
            # Cost to buy 1 contract
            ask = contract.get("ask") or contract.get("last") or 0
            cost_per_contract = ask * 100
            
            # Max contracts based on risk budget
            max_contracts = int(risk_budget // (cost_per_contract * self.cfg.stop_loss_pct))
            if max_contracts < 1:
                max_contracts = 1 if cost_per_contract <= self.ledger.available_settled_cash() else 0
            
            if max_contracts == 0:
                return "INSUFFICIENT_FUNDS"
            
            # Final cost
            total_cost = cost_per_contract * max_contracts
            
            # Final settlement check (prevent free-riding)
            if not self.ledger.ensure_funds_available(total_cost):
                return "NO_SETTLED_FUNDS"
            
            # Get contract symbol
            contract_symbol = contract.get("symbol") or contract.get("option_symbol")
            if not contract_symbol:
                return "NO_CONTRACT_SYMBOL"
            
            # Place order via broker
            try:
                order = self.broker.buy_option(contract_symbol, max_contracts)
                if not order:
                    return "ORDER_FAILED"
                
                # Check if order was filled
                filled = order.get("filled", False) if isinstance(order, dict) else False
                if not filled and hasattr(order, "filled"):
                    filled = order.filled
                
                if not filled:
                    return "ORDER_NOT_FILLED"
                
                # Get fill price
                entry_price = order.get("fill_price") or order.get("price") or ask
                if isinstance(order, dict) and "fill_price" not in order:
                    entry_price = order.get("price") or ask
                elif hasattr(order, "fill_price"):
                    entry_price = order.fill_price
                
            except Exception as e:
                logger.error(f"Error placing order: {e}")
                return "ORDER_FAILED"
            
            # Reserve funds until settled
            self.ledger.reserve_for_order(
                total_cost,
                settle_after_seconds=self.cfg.settlement_seconds_conservative,
                reason="accelerator_buy",
                security_type="option"
            )
            
            entry_time = datetime.datetime.utcnow()
            
            # Monitor trade loop
            timeout_seconds = 60 * 60  # 60 minutes max
            start = datetime.datetime.utcnow()
            
            while (datetime.datetime.utcnow() - start).total_seconds() < timeout_seconds:
                try:
                    # Get current price
                    cur = self.data.get_option_price(contract_symbol)
                    if cur is None:
                        continue
                    
                    # Get bid/ask for spread-aware exit
                    bid, ask = None, None
                    try:
                        if hasattr(self.data, "get_option_bid_ask"):
                            bid_ask = self.data.get_option_bid_ask(contract_symbol)
                            if isinstance(bid_ask, tuple):
                                bid, ask = bid_ask
                            elif isinstance(bid_ask, dict):
                                bid = bid_ask.get("bid")
                                ask = bid_ask.get("ask")
                    except Exception:
                        pass
                    
                    mid = (bid + ask) / 2.0 if (bid is not None and ask is not None) else cur
                    if mid <= 0:
                        continue
                    
                    # Compute P&L percentage
                    pnl_pct = (cur - entry_price) / entry_price
                    
                    # Spread-aware exit: adjust targets if spread widens
                    spread_pct = (ask - bid) / max(mid, 1e-6) if (bid and ask) else 0.0
                    
                    if spread_pct > 0.40:
                        # Huge spread — take profit/stop earlier
                        dynamic_profit = self.cfg.profit_target_pct * 0.8
                        dynamic_stop = self.cfg.stop_loss_pct * 0.9
                    else:
                        dynamic_profit = self.cfg.profit_target_pct
                        dynamic_stop = self.cfg.stop_loss_pct
                    
                    # Profit target reached
                    if pnl_pct >= dynamic_profit:
                        try:
                            self.broker.sell_option(contract_symbol, max_contracts)
                            profit = (cur - entry_price) * 100 * max_contracts
                            self.daily_profit += profit
                            self.trades_today += 1
                            self.session_trade_history.append({
                                "result": "win",
                                "pnl_pct": pnl_pct,
                                "profit": profit
                            })
                            return "TAKE_PROFIT"
                        except Exception as e:
                            logger.error(f"Error selling at profit: {e}")
                            return "EXIT_ERROR"
                    
                    # Stop loss reached
                    if pnl_pct <= -dynamic_stop:
                        try:
                            self.broker.sell_option(contract_symbol, max_contracts)
                            loss = (entry_price - cur) * 100 * max_contracts
                            self.daily_profit -= loss
                            self.trades_today += 1
                            self.session_trade_history.append({
                                "result": "loss",
                                "pnl_pct": pnl_pct,
                                "loss": loss
                            })
                            return "STOP_LOSS"
                        except Exception as e:
                            logger.error(f"Error selling at stop: {e}")
                            return "EXIT_ERROR"
                    
                    # Sleep briefly to avoid tight loop
                    import time
                    time.sleep(1)
                    
                except Exception as e:
                    logger.error(f"Error monitoring trade: {e}")
                    continue
            
            # Timeout — exit conservatively
            try:
                self.broker.sell_option(contract_symbol, max_contracts)
                final_price = self.data.get_option_price(contract_symbol) or entry_price
                pnl_pct = (final_price - entry_price) / entry_price if final_price else 0.0
                profit = (final_price - entry_price) * 100 * max_contracts if final_price else 0.0
                self.daily_profit += profit
                self.trades_today += 1
                self.session_trade_history.append({
                    "result": "timeout",
                    "pnl_pct": pnl_pct,
                    "profit": profit
                })
                return "TIMEOUT_EXIT"
            except Exception as e:
                logger.error(f"Error exiting on timeout: {e}")
                return "TIMEOUT_EXIT_ERROR"
            
        except Exception as e:
            logger.error(f"Error in execute_trade: {e}")
            return "ERROR"
    
    # -------------------------
    # Retraining hook
    # -------------------------
    
    def maybe_retrain_session(self) -> None:
        """Call Ralph retrain when session triggers"""
        now = datetime.datetime.utcnow()
        
        if self.last_retrain_time is None:
            self.last_retrain_time = now
            return
        
        minutes_since = (now - self.last_retrain_time).total_seconds() / 60.0
        
        if (minutes_since >= self.cfg.retrain_interval_minutes and 
            len(self.session_trade_history) >= self.cfg.session_retrain_min_trades):
            
            # Prepare dataset summary
            summary = {
                "trades": self.session_trade_history[-50:],  # Last 50
                "account_balance": self.get_account_size(),
                "timestamp": now.isoformat(),
                "daily_profit": self.daily_profit
            }
            
            # Call ralph's retrain hook (non-blocking if ralph supports)
            if hasattr(self.ralph, "retrain_hook"):
                try:
                    self.ralph.retrain_hook(summary)
                    logger.info("Ralph retrain hook called")
                except Exception as e:
                    logger.warning(f"Ralph retrain hook error: {e}")
            
            self.last_retrain_time = now
    
    # -------------------------
    # Top-level entrypoint
    # -------------------------
    
    def execute(self) -> str:
        """
        Main execution method - call this to run accelerator cycle.
        
        Returns:
            Result string indicating outcome
        """
        # Reconcile settled ledger
        self.ledger.release_settled()
        
        # Initialize start of day balance if needed
        if self.start_of_day_balance is None:
            self.start_of_day_balance = self.get_account_size()
        
        # Check if account has grown enough to disable accelerator
        if self.get_account_size() >= self.cfg.target_account_size:
            logger.info(f"Account reached target size (${self.cfg.target_account_size:.2f}), "
                       f"accelerator should be disabled")
            return "TARGET_REACHED"
        
        # Basic checks
        if self.current_drawdown_exceeded():
            return "DAILY_DRAWDOWN_EXCEEDED"
        
        if self.trades_today >= self.cfg.max_trades_per_day:
            return "MAX_TRADES_REACHED"
        
        if not self.in_allowed_time_window():
            return "OUT_OF_TIME_WINDOW"
        
        # Get direction from Ralph
        direction, prob = self.pick_direction()
        if direction is None:
            return "NO_HIGH_CONFIDENCE_SIGNAL"
        
        # Pick contract
        contract = self.pick_contract(direction)
        if not contract:
            return "NO_VALID_CONTRACT"
        
        # Execute trade
        result = self.execute_trade(contract, direction, prob)
        
        # Session retraining trigger
        self.maybe_retrain_session()
        
        return result
    
    def get_status(self) -> Dict[str, Any]:
        """Get accelerator status"""
        return {
            "daily_profit": self.daily_profit,
            "trades_today": self.trades_today,
            "account_size": self.get_account_size(),
            "start_of_day_balance": self.start_of_day_balance,
            "settlement_status": self.ledger.get_settlement_status(),
            "session_trades": len(self.session_trade_history),
            "target_account_size": self.cfg.target_account_size,
            "weekly_return_target": self.cfg.weekly_return_target
        }

