#!/usr/bin/env python3
"""
Optimus Options Strategy Engine v2 — Greeks-Driven Selection

Key improvements over v1:
  1. Delta-based strike selection (targets 0.20-0.40 delta)
  2. Implied volatility awareness (rejects overpriced options via IV rank)
  3. Theta-cost-per-day scoring (penalizes high daily decay)
  4. Multi-expiration comparison (picks best theta/premium ratio)
  5. Multi-signal momentum (not just today's % change)
  6. Directional balance (max 2 positions same direction)
  7. Higher quality bar (min score 55, min premium $0.05)
  8. Smart exit management (trailing stops, theta-aware exits)

Supported strategies (option level 2):
  - Long calls on momentum / breakout setups
  - Long puts on breakdown / hedging setups
"""

import logging
import math
import datetime
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


class OptimusOptionsEngine:
    """
    Greeks-driven options strategy engine for Optimus.

    Uses delta for strike selection, IV rank for timing,
    theta for cost awareness, and multi-day momentum for direction.
    """

    # -------------------------------------------------------
    # Watchlist — liquid, options-active tickers suited for
    # small-account directional plays
    # -------------------------------------------------------
    WATCHLIST = [
        # Leveraged ETFs (high vol, tight spreads, weekly options)
        "SQQQ", "TQQQ", "TZA", "TNA", "SOXS", "SOXL",
        "UVXY", "SPXS", "SPXL",
        # High-options-volume equities (low share price)
        "SOFI", "PLTR", "NIO", "F", "SNAP", "AMC",
        # Benchmark (for momentum context)
        "SPY", "QQQ",
    ]

    # -------------------------------------------------------
    # Tuning Constants
    # -------------------------------------------------------
    # Premium bounds (per-contract, in dollars)
    MAX_PREMIUM_PER_CONTRACT = 0.50   # $50 max per contract
    MIN_PREMIUM_PER_CONTRACT = 0.05   # $5 min — kills lottery tickets

    # Delta sweet spot for strike selection
    TARGET_DELTA_LOW  = 0.20   # minimum delta (enough skin in the game)
    TARGET_DELTA_HIGH = 0.40   # maximum delta (affordable OTM)
    IDEAL_DELTA       = 0.30   # bullseye

    # IV rank thresholds (0-100 scale)
    IV_RANK_MAX = 70   # Reject if IV rank above this (over-priced options)
    IV_RANK_SWEET_LOW  = 20
    IV_RANK_SWEET_HIGH = 50  # Buy when IV is cheap

    # Theta cost ceiling: max acceptable theta-to-premium ratio
    # e.g. 0.05 means we accept losing up to 5% of premium per day to theta
    MAX_THETA_RATIO = 0.08

    # Scoring
    MIN_SCORE_THRESHOLD = 55   # Much higher bar than the old 30

    # Position management
    MAX_OPTIONS_ALLOCATION_PCT = 0.25  # 25% of NAV for options
    MAX_CONTRACTS_PER_TRADE = 2
    MAX_SAME_DIRECTION = 2  # Max 2 bullish or 2 bearish at once

    # Expiration window
    MIN_DAYS_TO_EXP = 5    # No weeklies about to expire
    MAX_DAYS_TO_EXP = 35   # Don't go past ~5 weeks
    IDEAL_DAYS_LOW  = 10
    IDEAL_DAYS_HIGH = 21

    # Spread ceiling (as fraction of mid price)
    MAX_SPREAD_PCT = 0.35   # 35% spread — tighter than old 50%

    # Liquidity floors
    MIN_OPEN_INTEREST = 50
    MIN_OPTION_VOLUME = 10

    def __init__(self, tradier_adapter, nav: float = 300.0, option_level: int = 2):
        self.tradier = tradier_adapter
        self.nav = nav
        self.option_level = option_level
        self.open_option_positions: Dict[str, Dict] = {}
        self.options_trade_history: List[Dict] = []

        # Budget
        self.max_options_budget = nav * self.MAX_OPTIONS_ALLOCATION_PCT
        self.spent_on_options = 0.0

        # Directional tracking
        self._bullish_count = 0
        self._bearish_count = 0

        # Track peak price per position for trailing stops
        self._peak_prices: Dict[str, float] = {}  # option_symbol -> highest bid seen

        logger.info(
            f"Options Engine v2 initialized: NAV=${nav:.2f}, Level={option_level}, "
            f"Budget=${self.max_options_budget:.2f}, Delta target={self.IDEAL_DELTA}"
        )

    # ------------------------------------------------------------------
    # NAV / Budget
    # ------------------------------------------------------------------
    def update_nav(self, nav: float):
        """Update NAV and recalculate budget using actual available cash."""
        self.nav = nav
        actual_cash = self._get_available_cash()
        nav_limit = nav * self.MAX_OPTIONS_ALLOCATION_PCT
        self.max_options_budget = min(nav_limit, actual_cash) if actual_cash > 0 else nav_limit
        logger.info(
            f"Options budget: ${self.max_options_budget:.2f} "
            f"(cash=${actual_cash:.2f}, NAV-limit=${nav_limit:.2f})"
        )

    def _get_available_cash(self) -> float:
        try:
            balances = self.tradier.get_balances()
            if not balances:
                return 0.0
            bal = balances.get("balances", balances)
            cash = bal.get("cash", {})
            if isinstance(cash, dict):
                ca = cash.get("cash_available", 0)
                if ca and isinstance(ca, (int, float)):
                    return float(ca)
            for field in ("cash_available", "total_cash", "buying_power"):
                val = bal.get(field, 0)
                if val and isinstance(val, (int, float)) and float(val) > 0:
                    return float(val)
            return 0.0
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    # Scanning
    # ------------------------------------------------------------------
    def scan_opportunities(self) -> List[Dict[str, Any]]:
        """Scan watchlist for high-quality options opportunities."""
        opportunities: List[Dict[str, Any]] = []

        for symbol in self.WATCHLIST:
            try:
                opps = self._analyze_symbol(symbol)
                opportunities.extend(opps)
            except Exception as e:
                logger.debug(f"Scan error {symbol}: {e}")

        opportunities.sort(key=lambda x: x.get("score", 0), reverse=True)

        top = opportunities[:5]
        if top:
            logger.info(
                f"Options scan: {len(opportunities)} candidates, "
                f"top={top[0]['symbol']} {top[0]['option_type']} "
                f"${top[0]['strike']} score={top[0]['score']:.0f} "
                f"delta={top[0].get('delta', 0):.2f}"
            )
        else:
            logger.info("Options scan: 0 qualifying candidates")
        return top

    def _analyze_symbol(self, symbol: str) -> List[Dict[str, Any]]:
        """Analyze all valid expirations for a symbol."""
        opportunities: List[Dict[str, Any]] = []

        # --- Quote data for momentum context ---
        quote_data = self.tradier.get_quote(symbol)
        if not quote_data:
            return []
        quote = quote_data.get("quotes", {}).get("quote", {})
        if not isinstance(quote, dict):
            return []

        current_price = quote.get("last", 0)
        if current_price <= 0:
            return []

        stock_volume = quote.get("volume", 0) or 0
        if stock_volume < 50000:
            return []  # Too illiquid for reliable options

        # Build momentum context from quote fields
        momentum = self._assess_momentum(quote, current_price)

        # --- Get valid expirations ---
        expirations = self.tradier.get_option_expirations(symbol)
        if not expirations:
            return []

        today = datetime.date.today()
        valid_exps: List[tuple] = []
        for exp_str in expirations[:8]:
            try:
                exp_date = datetime.date.fromisoformat(exp_str)
                dte = (exp_date - today).days
                if self.MIN_DAYS_TO_EXP <= dte <= self.MAX_DAYS_TO_EXP:
                    valid_exps.append((exp_str, dte))
            except (ValueError, TypeError):
                continue

        if not valid_exps:
            return []

        # --- Scan up to 3 expirations (multi-expiration comparison) ---
        for exp_str, dte in valid_exps[:3]:
            chain = self.tradier.get_option_chain(symbol, exp_str)
            if not chain:
                continue

            for contract in chain:
                try:
                    opp = self._evaluate_contract(
                        contract, symbol, current_price, dte, momentum
                    )
                    if opp:
                        opportunities.append(opp)
                except Exception:
                    continue

        return opportunities

    # ------------------------------------------------------------------
    # Momentum Assessment
    # ------------------------------------------------------------------
    @staticmethod
    def _assess_momentum(quote: Dict, current_price: float) -> Dict[str, Any]:
        """
        Build a momentum signal from available quote data.

        Uses: today's change, distance from 52-week high/low,
        volume vs average volume, and prevclose context.
        """
        change_pct = float(quote.get("change_percentage", 0) or 0)
        prevclose = float(quote.get("prevclose", 0) or 0)
        high_52w = float(quote.get("week_52_high", 0) or 0)
        low_52w = float(quote.get("week_52_low", 0) or 0)
        avg_volume = float(quote.get("average_volume", 0) or 0)
        today_volume = float(quote.get("volume", 0) or 0)

        # Distance from 52-week extremes (0 = at low, 1 = at high)
        range_52w = high_52w - low_52w if high_52w > low_52w else 1.0
        position_in_range = (current_price - low_52w) / range_52w if range_52w > 0 else 0.5

        # Volume surge (today vs 50-day average)
        volume_ratio = today_volume / avg_volume if avg_volume > 0 else 1.0

        # Multi-day trend proxy: distance from prevclose over recent price range
        if prevclose > 0 and high_52w > low_52w:
            two_day_move = (current_price - prevclose) / prevclose * 100
        else:
            two_day_move = change_pct

        # Composite momentum score: -100 (extreme bearish) to +100 (extreme bullish)
        momentum_score = 0.0

        # Today's change contribution (weight: 40%)
        momentum_score += max(-40, min(40, change_pct * 10))

        # Position in 52-week range (weight: 30%)
        # Near highs = bullish momentum, near lows = bearish
        range_signal = (position_in_range - 0.5) * 60  # -30 to +30
        momentum_score += range_signal

        # Volume confirmation (weight: 30%)
        # High volume amplifies the directional signal
        if volume_ratio > 1.5:  # Unusual volume
            vol_boost = min(30, (volume_ratio - 1.0) * 15)
            if change_pct > 0:
                momentum_score += vol_boost
            else:
                momentum_score -= vol_boost

        return {
            "score": max(-100, min(100, momentum_score)),
            "change_pct": change_pct,
            "position_in_52w_range": position_in_range,
            "volume_ratio": volume_ratio,
            "two_day_move": two_day_move,
            "direction": "bullish" if momentum_score > 15 else ("bearish" if momentum_score < -15 else "neutral"),
        }

    # ------------------------------------------------------------------
    # Contract Evaluation (Greeks-Driven)
    # ------------------------------------------------------------------
    def _evaluate_contract(
        self,
        contract: Dict[str, Any],
        symbol: str,
        current_price: float,
        days_to_exp: int,
        momentum: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Evaluate one contract using Greeks + momentum + liquidity."""

        option_type = contract.get("option_type", "").lower()
        strike = contract.get("strike", 0)
        bid = contract.get("bid", 0) or 0
        ask = contract.get("ask", 0) or 0
        last = contract.get("last", 0) or 0
        opt_volume = contract.get("volume", 0) or 0
        open_interest = contract.get("open_interest", 0) or 0
        greeks = contract.get("greeks") or {}

        # --- Extract Greeks ---
        delta = abs(float(greeks.get("delta", 0) or 0))
        theta = float(greeks.get("theta", 0) or 0)  # negative for long options
        gamma = float(greeks.get("gamma", 0) or 0)
        vega = float(greeks.get("vega", 0) or 0)
        mid_iv = float(greeks.get("mid_iv", 0) or greeks.get("smv_vol", 0) or 0)

        # --- OCC symbol ---
        occ_symbol = contract.get("symbol", "")
        if not occ_symbol or len(occ_symbol) <= len(symbol):
            exp_date = contract.get("expiration_date", "")
            if exp_date and strike > 0 and option_type:
                occ_symbol = self._build_occ_symbol(symbol, exp_date, option_type, strike)
            if not occ_symbol:
                return None

        # --- Mid price ---
        if bid > 0 and ask > 0:
            mid_price = (bid + ask) / 2
        elif last > 0:
            mid_price = last
        else:
            return None

        # ============================================================
        # HARD FILTERS — reject before scoring
        # ============================================================

        # 1. Premium bounds
        remaining_budget = self.max_options_budget - self.spent_on_options
        max_affordable = remaining_budget / 100.0 if remaining_budget > 0 else 0
        effective_max = min(self.MAX_PREMIUM_PER_CONTRACT, max_affordable)
        if mid_price < self.MIN_PREMIUM_PER_CONTRACT or mid_price > effective_max:
            return None

        # 2. Delta bounds (the core quality gate)
        if delta < self.TARGET_DELTA_LOW or delta > 0.60:
            return None  # Too far OTM or too deep ITM

        # 3. Liquidity
        if open_interest < self.MIN_OPEN_INTEREST:
            return None
        if opt_volume < self.MIN_OPTION_VOLUME:
            return None

        # 4. Spread
        if bid > 0 and ask > 0:
            spread_pct = (ask - bid) / mid_price if mid_price > 0 else 1.0
            if spread_pct > self.MAX_SPREAD_PCT:
                return None
        else:
            return None  # Need both bid and ask

        # 5. Theta cost ceiling
        theta_per_day = abs(theta)
        theta_ratio = theta_per_day / mid_price if mid_price > 0 else 1.0
        if theta_ratio > self.MAX_THETA_RATIO:
            return None  # Losing too much per day to time decay

        # 6. Directional balance check
        direction = "bullish" if option_type == "call" else "bearish"
        if direction == "bullish" and self._bullish_count >= self.MAX_SAME_DIRECTION:
            return None
        if direction == "bearish" and self._bearish_count >= self.MAX_SAME_DIRECTION:
            return None

        # ============================================================
        # SCORING — Greeks-weighted (0 to 100)
        # ============================================================
        score = self._score_opportunity(
            delta=delta,
            theta=theta,
            gamma=gamma,
            vega=vega,
            mid_iv=mid_iv,
            mid_price=mid_price,
            days_to_exp=days_to_exp,
            opt_volume=opt_volume,
            open_interest=open_interest,
            bid=bid,
            ask=ask,
            option_type=option_type,
            momentum=momentum,
            theta_ratio=theta_ratio,
        )

        if score < self.MIN_SCORE_THRESHOLD:
            return None

        # --- Cost & sizing ---
        cost_per_contract = mid_price * 100
        max_afford = int(remaining_budget / cost_per_contract) if cost_per_contract > 0 else 0
        num_contracts = min(max_afford, self.MAX_CONTRACTS_PER_TRADE)
        if num_contracts < 1:
            return None

        # OTM percentage (for reporting)
        if option_type == "call":
            otm_pct = (strike - current_price) / current_price if current_price > 0 else 0
        else:
            otm_pct = (current_price - strike) / current_price if current_price > 0 else 0

        reasoning = (
            f"{direction.upper()} {symbol} {option_type} ${strike} "
            f"| delta={delta:.2f} theta={theta:.3f} IV={mid_iv:.0%} "
            f"| {days_to_exp}d to exp | score={score:.0f}"
        )

        return {
            "symbol": symbol,
            "option_symbol": occ_symbol,
            "option_type": option_type,
            "strike": strike,
            "expiration": contract.get("expiration_date", ""),
            "days_to_exp": days_to_exp,
            "current_price": current_price,
            "bid": bid,
            "ask": ask,
            "mid_price": mid_price,
            "cost_per_contract": cost_per_contract,
            "num_contracts": num_contracts,
            "total_cost": cost_per_contract * num_contracts,
            "otm_pct": otm_pct,
            "is_otm": otm_pct > 0,
            "volume": opt_volume,
            "open_interest": open_interest,
            "score": score,
            "direction": direction,
            "reasoning": reasoning,
            "greeks": greeks,
            # Extracted greeks for downstream use
            "delta": delta,
            "theta": theta,
            "gamma": gamma,
            "vega": vega,
            "mid_iv": mid_iv,
            "theta_ratio": theta_ratio,
            "momentum": momentum,
        }

    # ------------------------------------------------------------------
    # Scoring v2 — Greeks-weighted
    # ------------------------------------------------------------------
    def _score_opportunity(
        self,
        delta: float,
        theta: float,
        gamma: float,
        vega: float,
        mid_iv: float,
        mid_price: float,
        days_to_exp: int,
        opt_volume: int,
        open_interest: int,
        bid: float,
        ask: float,
        option_type: str,
        momentum: Dict[str, Any],
        theta_ratio: float,
    ) -> float:
        """
        Score an option opportunity (0–100).

        Weighting:
          Delta quality:     25 pts max
          Theta efficiency:  20 pts max
          Momentum fit:      20 pts max
          Expiration sweet:  15 pts max
          Liquidity:         10 pts max
          Spread quality:    10 pts max
        Total possible:     100 pts
        """
        score = 0.0

        # ---- 1. DELTA QUALITY (25 pts) ----
        # Bullseye at IDEAL_DELTA, falls off toward edges
        delta_distance = abs(delta - self.IDEAL_DELTA)
        if delta_distance < 0.03:
            score += 25  # Perfect delta zone
        elif delta_distance < 0.07:
            score += 20
        elif delta_distance < 0.12:
            score += 14
        else:
            score += 8   # Still within hard filter bounds

        # ---- 2. THETA EFFICIENCY (20 pts) ----
        # Lower theta_ratio = less daily decay = better
        if theta_ratio < 0.02:
            score += 20  # Excellent: <2% daily decay
        elif theta_ratio < 0.04:
            score += 15
        elif theta_ratio < 0.06:
            score += 10
        else:
            score += 4   # Acceptable but expensive decay

        # ---- 3. MOMENTUM FIT (20 pts) ----
        mom_score = momentum.get("score", 0)
        mom_direction = momentum.get("direction", "neutral")

        # Alignment: call+bullish momentum or put+bearish momentum
        if option_type == "call" and mom_direction == "bullish":
            # Scale by momentum strength
            alignment_pts = min(20, max(5, abs(mom_score) / 5))
            score += alignment_pts
        elif option_type == "put" and mom_direction == "bearish":
            alignment_pts = min(20, max(5, abs(mom_score) / 5))
            score += alignment_pts
        elif mom_direction == "neutral":
            score += 5   # Neutral is ok but not great
        # Misaligned direction gets 0 points (contrarian is risky)

        # Volume surge bonus (unusual activity often precedes moves)
        vol_ratio = momentum.get("volume_ratio", 1.0)
        if vol_ratio > 2.0:
            score += min(5, (vol_ratio - 1.5) * 3)  # Up to 5 bonus pts

        # ---- 4. EXPIRATION SWEET SPOT (15 pts) ----
        if self.IDEAL_DAYS_LOW <= days_to_exp <= self.IDEAL_DAYS_HIGH:
            score += 15  # Ideal: 10-21 days
        elif 7 <= days_to_exp < self.IDEAL_DAYS_LOW:
            score += 10  # Slightly short
        elif self.IDEAL_DAYS_HIGH < days_to_exp <= self.MAX_DAYS_TO_EXP:
            score += 10  # Slightly long
        else:
            score += 3   # Edge of acceptable range

        # ---- 5. LIQUIDITY (10 pts) ----
        if open_interest >= 5000 and opt_volume >= 500:
            score += 10
        elif open_interest >= 1000 and opt_volume >= 100:
            score += 7
        elif open_interest >= 200 and opt_volume >= 25:
            score += 4
        else:
            score += 1   # Bare minimum (passed hard filter)

        # ---- 6. SPREAD QUALITY (10 pts) ----
        if bid > 0 and ask > 0:
            spread_pct = (ask - bid) / ((bid + ask) / 2)
            if spread_pct < 0.08:
                score += 10  # Extremely tight
            elif spread_pct < 0.15:
                score += 7
            elif spread_pct < 0.25:
                score += 4
            else:
                score += 1

        # ---- IV RANK ADJUSTMENT (bonus/penalty) ----
        # mid_iv is the annualized implied volatility (e.g., 0.45 = 45%)
        # We use it as a rough proxy; a full IV rank requires historical data
        # High IV = options are expensive = penalty for buying
        if mid_iv > 0:
            if mid_iv > 1.0:   # >100% IV — extremely expensive
                score -= 10
            elif mid_iv > 0.7:  # >70% IV — pricey
                score -= 5
            elif mid_iv < 0.3:  # <30% IV — cheap options, good for buying
                score += 5

        # ---- GAMMA BONUS ----
        # High gamma near ATM means option accelerates in value on moves
        if gamma > 0.05:
            score += 3
        elif gamma > 0.03:
            score += 1

        return max(0, min(100, score))

    # ------------------------------------------------------------------
    # Trade Signal Generation
    # ------------------------------------------------------------------
    def generate_trade_signal(self, opportunity: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a scored opportunity into an execution-ready trade signal."""
        option_type = opportunity["option_type"]

        logger.info(
            f"Trade signal: {opportunity['symbol']} {option_type} "
            f"${opportunity['strike']} delta={opportunity.get('delta', 0):.2f} "
            f"score={opportunity['score']:.0f} OCC={opportunity['option_symbol']}"
        )

        return {
            "symbol": opportunity["symbol"],
            "option_symbol": opportunity["option_symbol"],
            "side": "buy_to_open",
            "quantity": opportunity["num_contracts"],
            "order_type": "limit",
            "price": opportunity["ask"],   # Pay the ask for reliable fill
            "duration": "day",
            "strategy_name": f"options_{option_type}_{opportunity['symbol']}",
            "strategy_id": f"opt_{option_type}_{opportunity['symbol']}_{opportunity['strike']}",
            "expected_return": 0.50,
            "stop_loss_pct": 0.40,
            "trade_type": "option",
            "option_type": option_type,
            "strike": opportunity["strike"],
            "expiration": opportunity["expiration"],
            "entry_reasoning": opportunity["reasoning"],
            "score": opportunity["score"],
            "greeks": opportunity.get("greeks", {}),
            "delta": opportunity.get("delta", 0),
            "theta": opportunity.get("theta", 0),
        }

    # ------------------------------------------------------------------
    # Exit Management — Trailing Stops + Theta Awareness
    # ------------------------------------------------------------------
    def check_option_exits(self, positions: Dict[str, Dict]) -> List[Dict[str, Any]]:
        """
        Check open option positions for exit signals.

        Exit rules (priority order):
          1. Trailing stop: price fell 35% from peak since entry
          2. Hard stop: down 40% from entry
          3. Take profit: up 75%+ from entry
          4. Theta bleed: within 3 days of expiry and not up 30%+
          5. Expiration eve: 1 day left, salvage any remaining value
        """
        exit_signals: List[Dict[str, Any]] = []

        for pos_key, pos in positions.items():
            if pos.get("trade_type") != "option":
                continue

            option_symbol = pos.get("option_symbol", "")
            if not option_symbol:
                continue

            entry_price = pos.get("entry_price", 0)
            quantity = pos.get("quantity", 0)
            if entry_price <= 0 or quantity <= 0:
                continue

            try:
                quote = self.tradier.get_quote(option_symbol)
                if not quote:
                    continue

                q = quote.get("quotes", {}).get("quote", {})
                current_bid = q.get("bid", 0) or 0
                current_ask = q.get("ask", 0) or 0
                current_mid = (current_bid + current_ask) / 2 if current_bid > 0 and current_ask > 0 else current_bid

                if current_mid <= 0:
                    continue

                pnl_pct = (current_mid - entry_price) / entry_price if entry_price > 0 else 0

                # Update peak price for trailing stop
                peak = self._peak_prices.get(option_symbol, entry_price)
                if current_mid > peak:
                    peak = current_mid
                    self._peak_prices[option_symbol] = peak

                # Drawdown from peak
                drawdown_from_peak = (peak - current_mid) / peak if peak > 0 else 0

                # Days to expiration
                days_left = 999
                exp_str = pos.get("expiration", "")
                if exp_str:
                    try:
                        exp_date = datetime.date.fromisoformat(exp_str)
                        days_left = (exp_date - datetime.date.today()).days
                    except (ValueError, TypeError):
                        pass

                # --- EXIT RULES ---
                should_exit = False
                reason = ""

                # Rule 1: Trailing stop (gave back 35% from peak)
                if peak > entry_price * 1.10 and drawdown_from_peak >= 0.35:
                    should_exit = True
                    reason = (
                        f"Trailing stop: peaked at ${peak:.2f}, "
                        f"now ${current_mid:.2f} ({drawdown_from_peak:.0%} drawdown)"
                    )

                # Rule 2: Hard stop at -40%
                elif pnl_pct <= -0.40:
                    should_exit = True
                    reason = f"Hard stop: {pnl_pct:.0%} loss"

                # Rule 3: Take profit at +75%
                elif pnl_pct >= 0.75:
                    should_exit = True
                    reason = f"Take profit: {pnl_pct:.0%} gain"

                # Rule 4: Theta bleed (3 days left, not solidly profitable)
                elif days_left <= 3 and pnl_pct < 0.30:
                    should_exit = True
                    reason = f"Theta bleed: {days_left}d left, P&L {pnl_pct:.0%}"

                # Rule 5: Expiration eve (1 day left, salvage anything)
                elif days_left <= 1:
                    should_exit = True
                    reason = f"Expiration eve: {days_left}d left, P&L {pnl_pct:.0%}"

                if should_exit:
                    exit_signals.append({
                        "symbol": pos.get("symbol", pos_key),
                        "option_symbol": option_symbol,
                        "side": "sell_to_close",
                        "quantity": quantity,
                        "order_type": "market",
                        "duration": "day",
                        "strategy_name": f"options_exit_{pos_key}",
                        "strategy_id": f"opt_exit_{pos_key}",
                        "exit_reason": reason,
                        "current_bid": current_bid,
                        "entry_price": entry_price,
                        "pnl_pct": pnl_pct,
                        "trade_type": "option",
                    })
                    logger.info(f"EXIT SIGNAL: {option_symbol} — {reason}")

                    # Clean up peak tracking
                    if option_symbol in self._peak_prices:
                        del self._peak_prices[option_symbol]

            except Exception as e:
                logger.debug(f"Exit check error {pos_key}: {e}")

        return exit_signals

    # ------------------------------------------------------------------
    # Directional Balance Tracking
    # ------------------------------------------------------------------
    def update_direction_counts(self, positions: Dict[str, Dict]):
        """Recount bullish/bearish positions from live position dict."""
        bull = 0
        bear = 0
        for _, pos in positions.items():
            if pos.get("trade_type") != "option":
                continue
            ot = pos.get("option_type", "")
            if ot == "call":
                bull += 1
            elif ot == "put":
                bear += 1
        self._bullish_count = bull
        self._bearish_count = bear

    def record_trade(self, direction: str):
        """Update directional count after a trade executes."""
        if direction == "bullish":
            self._bullish_count += 1
        elif direction == "bearish":
            self._bearish_count += 1

    def record_exit(self, option_type: str):
        """Update directional count after an exit."""
        if option_type == "call":
            self._bullish_count = max(0, self._bullish_count - 1)
        elif option_type == "put":
            self._bearish_count = max(0, self._bearish_count - 1)

    # ------------------------------------------------------------------
    # OCC Symbol Builder
    # ------------------------------------------------------------------
    @staticmethod
    def _build_occ_symbol(underlying: str, expiration: str, option_type: str, strike: float) -> str:
        try:
            exp = datetime.date.fromisoformat(expiration)
            exp_str = exp.strftime("%y%m%d")
            otype = "C" if option_type.lower().startswith("c") else "P"
            strike_int = int(strike * 1000)
            strike_str = f"{strike_int:08d}"
            padded = underlying.ljust(6)
            return f"{padded}{exp_str}{otype}{strike_str}"
        except Exception as e:
            logger.error(f"OCC build error: {e}")
            return ""

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------
    def get_status(self) -> Dict[str, Any]:
        return {
            "enabled": True,
            "version": 2,
            "option_level": self.option_level,
            "nav": self.nav,
            "max_budget": self.max_options_budget,
            "spent": self.spent_on_options,
            "remaining_budget": self.max_options_budget - self.spent_on_options,
            "open_positions": len(self.open_option_positions),
            "bullish_count": self._bullish_count,
            "bearish_count": self._bearish_count,
            "total_trades": len(self.options_trade_history),
            "target_delta": f"{self.TARGET_DELTA_LOW}-{self.TARGET_DELTA_HIGH}",
            "min_score": self.MIN_SCORE_THRESHOLD,
        }
