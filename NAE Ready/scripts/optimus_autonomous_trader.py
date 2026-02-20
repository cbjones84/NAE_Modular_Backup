#!/usr/bin/env python3
"""
Optimus Autonomous Trader with NAE Agent Intelligence
======================================================
Leverages Ralph (strategy), Donnie (validation), Casey (orchestration), and Optimus (execution)
with accelerator methods to grow account to $8,000-$10,000 target.

Uses:
- Kelly Criterion for optimal position sizing
- Timing strategies for entry/exit optimization
- Multi-agent intelligence for strategy selection
- Tradier for real trade execution
"""

import os
import sys
import time
import datetime
import json

# Set environment variables FIRST
os.environ["TRADIER_SANDBOX"] = "false"
os.environ["TRADIER_API_KEY"] = "27Ymk28vtbgqY1LFYxhzaEmIuwJb"
os.environ["TRADIER_ACCOUNT_ID"] = "6YB66744"

# Add paths
script_dir = os.path.dirname(os.path.abspath(__file__))
nae_dir = os.path.dirname(script_dir)
sys.path.insert(0, nae_dir)
sys.path.insert(0, os.path.join(nae_dir, 'execution'))

# Target account growth
ACCOUNT_GROWTH_TARGET = 10000  # $10,000 target
AGGRESSIVE_GROWTH_MODE = True

def log(msg, level="INFO"):
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] [{level}] {msg}")

def get_tradier_account():
    """Get account info from Tradier"""
    import requests
    from broker_adapters.tradier_adapter import TradierBrokerAdapter
    adapter = TradierBrokerAdapter(sandbox=False)
    account = adapter.get_account_info()
    
    # Also fetch balances directly
    headers = {
        "Authorization": f"Bearer {adapter.oauth.api_key}",
        "Accept": "application/json"
    }
    bal_response = requests.get(
        f"https://api.tradier.com/v1/accounts/{adapter.account_id}/balances",
        headers=headers,
        timeout=10
    )
    if bal_response.status_code == 200:
        balances_data = bal_response.json()
        if account is None:
            account = {}
        account['balances'] = balances_data.get('balances', {})
    
    return adapter, account

def get_tradier_positions(adapter):
    """Get current positions"""
    try:
        positions = adapter.get_positions()
        return positions if positions else []
    except:
        return []

def get_market_quote(adapter, symbol):
    """Get real-time quote for a symbol"""
    import requests
    try:
        headers = {
            "Authorization": f"Bearer {adapter.oauth.api_key}",
            "Accept": "application/json"
        }
        response = requests.get(
            f"https://api.tradier.com/v1/markets/quotes?symbols={symbol}",
            headers=headers,
            timeout=10
        )
        if response.status_code == 200:
            data = response.json()
            quotes = data.get('quotes', {}).get('quote', {})
            if isinstance(quotes, list):
                return quotes[0] if quotes else None
            return quotes
    except Exception as e:
        log(f"Quote error for {symbol}: {e}", "WARN")
    return None

def calculate_kelly_position_size(account_value, win_probability, win_odds, kelly_fraction=0.5):
    """
    Kelly Criterion position sizing - AGGRESSIVE mode uses 0.5 Kelly
    Full Kelly is too aggressive, so we use half Kelly for faster growth with less risk
    """
    p = win_probability
    q = 1.0 - p
    b = win_odds
    
    if b <= 0:
        return 0.0
    
    # Kelly formula: f = (bp - q) / b
    kelly_f = (p * b - q) / b
    kelly_f = max(0.0, min(kelly_f, 0.5))  # Cap at 50%
    
    # Apply fractional Kelly
    adjusted_kelly = kelly_f * kelly_fraction
    position_size = account_value * adjusted_kelly
    
    return max(0.0, position_size)

def calculate_rsi(closes, period=14):
    """Calculate Relative Strength Index"""
    if len(closes) < period + 1:
        return None
    
    gains = []
    losses = []
    
    for i in range(1, len(closes)):
        change = closes[i] - closes[i-1]
        if change > 0:
            gains.append(change)
            losses.append(0)
        else:
            gains.append(0)
            losses.append(abs(change))
    
    if len(gains) < period:
        return None
    
    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period
    
    if avg_loss == 0:
        return 100
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def detect_support_level(price_data):
    """Detect support level from recent price action"""
    if len(price_data) < 10:
        return None
    
    lows = [float(d.get('low', 0)) for d in price_data[-20:]]
    if not lows:
        return None
    
    # Support is around recent lows
    return min(lows)

def analyze_buy_opportunity(adapter, symbol, account_value):
    """
    SOPHISTICATED BUY ANALYSIS
    
    Analyzes a symbol for optimal entry using multiple factors:
    
    1. 📊 RSI Analysis: Buy on oversold (RSI < 30)
    2. 📈 Trend Analysis: Confirm uptrend (price > SMA20)
    3. 💪 Support Level: Buy near support
    4. 📉 Pullback Detection: Buy the dip in uptrend
    5. 📊 Volume Confirmation: Higher volume = stronger signal
    6. ⚡ Volatility Check: Avoid extreme volatility
    
    Returns: (should_buy, buy_score, analysis_details)
    """
    # Get current quote
    quote = get_market_quote(adapter, symbol)
    if not quote:
        return False, 0, None
    
    current_price = float(quote.get('last', quote.get('close', 0)))
    volume = float(quote.get('volume', 0))
    avg_volume = float(quote.get('average_volume', volume))
    
    if current_price <= 0:
        return False, 0, None
    
    # Get historical data
    price_data = get_historical_prices(adapter, symbol, days=30)
    if not price_data or len(price_data) < 10:
        return False, 0, None
    
    closes = [float(d.get('close', 0)) for d in price_data]
    highs = [float(d.get('high', 0)) for d in price_data]
    lows = [float(d.get('low', 0)) for d in price_data]
    
    # Calculate indicators
    rsi = calculate_rsi(closes)
    sma_10 = calculate_sma(closes, 10)
    sma_20 = calculate_sma(closes, 20)
    atr = calculate_atr(price_data)
    support = detect_support_level(price_data)
    
    # Scoring system (0-100)
    buy_score = 50  # Start neutral
    buy_signals = []
    avoid_signals = []
    
    # ========================================
    # FACTOR 1: RSI Analysis (Oversold = Buy)
    # ========================================
    if rsi:
        if rsi < 30:
            buy_score += 20
            buy_signals.append(f"📊 RSI Oversold: {rsi:.1f} (Strong Buy)")
        elif rsi < 40:
            buy_score += 10
            buy_signals.append(f"📊 RSI Low: {rsi:.1f} (Moderate Buy)")
        elif rsi > 70:
            buy_score -= 15
            avoid_signals.append(f"📊 RSI Overbought: {rsi:.1f} (Avoid)")
        elif rsi > 60:
            buy_score -= 5
            avoid_signals.append(f"📊 RSI High: {rsi:.1f} (Caution)")
    
    # ========================================
    # FACTOR 2: Trend Analysis (Uptrend = Buy)
    # ========================================
    if sma_20:
        if current_price > sma_20:
            buy_score += 10
            buy_signals.append(f"📈 Above SMA20: ${current_price:.2f} > ${sma_20:.2f}")
        else:
            buy_score -= 10
            avoid_signals.append(f"📉 Below SMA20: ${current_price:.2f} < ${sma_20:.2f}")
    
    if sma_10 and sma_20:
        if sma_10 > sma_20:
            buy_score += 10
            buy_signals.append("📈 Golden Cross: SMA10 > SMA20")
        else:
            buy_score -= 5
            avoid_signals.append("📉 Death Cross: SMA10 < SMA20")
    
    # ========================================
    # FACTOR 3: Support Level (Near Support = Buy)
    # ========================================
    if support and current_price > 0:
        distance_from_support = (current_price - support) / current_price
        if distance_from_support < 0.03:  # Within 3% of support
            buy_score += 15
            buy_signals.append(f"💪 Near Support: ${support:.2f} ({distance_from_support*100:.1f}% away)")
        elif distance_from_support < 0.05:
            buy_score += 8
            buy_signals.append(f"💪 Close to Support: ${support:.2f}")
    
    # ========================================
    # FACTOR 4: Pullback Detection (Dip in Uptrend)
    # ========================================
    if len(closes) >= 5 and sma_20:
        recent_high = max(closes[-5:])
        pullback_pct = (recent_high - current_price) / recent_high
        
        # Good pullback: 3-8% dip while still in uptrend
        if current_price > sma_20 and 0.03 < pullback_pct < 0.08:
            buy_score += 15
            buy_signals.append(f"📉 Healthy Pullback: -{pullback_pct*100:.1f}% from recent high")
    
    # ========================================
    # FACTOR 5: Volume Confirmation
    # ========================================
    if avg_volume > 0:
        volume_ratio = volume / avg_volume
        if volume_ratio > 1.5:
            buy_score += 5
            buy_signals.append(f"📊 High Volume: {volume_ratio:.1f}x average")
        elif volume_ratio < 0.5:
            buy_score -= 5
            avoid_signals.append(f"📊 Low Volume: {volume_ratio:.1f}x average")
    
    # ========================================
    # FACTOR 6: Volatility Check
    # ========================================
    if atr and current_price > 0:
        atr_pct = atr / current_price
        if atr_pct > 0.05:  # Very high volatility
            buy_score -= 10
            avoid_signals.append(f"⚡ High Volatility: ATR={atr_pct*100:.1f}%")
        elif atr_pct < 0.02:  # Low volatility (good for entry)
            buy_score += 5
            buy_signals.append(f"⚡ Low Volatility: ATR={atr_pct*100:.1f}%")
    
    # ========================================
    # FACTOR 7: Price Position (52-week range)
    # ========================================
    high_52w = float(quote.get('week_52_high', current_price))
    low_52w = float(quote.get('week_52_low', current_price))
    
    if high_52w > low_52w:
        price_position = (current_price - low_52w) / (high_52w - low_52w)
        if price_position < 0.3:  # In lower 30% of range
            buy_score += 10
            buy_signals.append(f"📊 Near 52w Low: {price_position*100:.0f}% of range")
        elif price_position > 0.9:  # Near highs
            buy_score -= 10
            avoid_signals.append(f"📊 Near 52w High: {price_position*100:.0f}% of range")
    
    # Determine if we should buy
    should_buy = buy_score >= 60  # Need score of 60+ to buy
    
    # Calculate position size based on score
    if buy_score >= 80:
        confidence = "HIGH"
        position_pct = 0.30
    elif buy_score >= 70:
        confidence = "MEDIUM-HIGH"
        position_pct = 0.25
    elif buy_score >= 60:
        confidence = "MEDIUM"
        position_pct = 0.20
    else:
        confidence = "LOW"
        position_pct = 0.10
    
    analysis = {
        'symbol': symbol,
        'current_price': current_price,
        'buy_score': buy_score,
        'confidence': confidence,
        'position_pct': position_pct,
        'buy_signals': buy_signals,
        'avoid_signals': avoid_signals,
        'rsi': rsi,
        'sma_20': sma_20,
        'atr': atr,
        'support': support
    }
    
    return should_buy, buy_score, analysis

def get_accelerator_strategies():
    """
    Generate accelerator strategies for rapid account growth
    These strategies are designed for small accounts targeting $8K-$10K
    """
    strategies = [
        {
            "name": "Momentum_Breakout",
            "symbol": "TQQQ",  # 3x leveraged NASDAQ - for aggressive growth
            "side": "buy",
            "strategy_type": "momentum",
            "win_probability": 0.55,
            "win_odds": 2.0,  # 2:1 risk/reward
            "description": "Buy on momentum breakout, high volatility leveraged ETF",
            "min_account": 100,
            "max_position_pct": 0.30,  # 30% max position
        },
        {
            "name": "Tech_Momentum",
            "symbol": "NVDA",  # NVIDIA - strong tech momentum
            "side": "buy",
            "strategy_type": "momentum",
            "win_probability": 0.52,
            "win_odds": 1.8,
            "description": "Tech leader with strong momentum",
            "min_account": 500,
            "max_position_pct": 0.25,
        },
        {
            "name": "SPY_Swing",
            "symbol": "SPY",  # S&P 500 ETF - stable growth
            "side": "buy",
            "strategy_type": "swing",
            "win_probability": 0.58,
            "win_odds": 1.5,
            "description": "S&P 500 swing trade for consistent growth",
            "min_account": 600,  # SPY is ~$600/share
            "max_position_pct": 0.40,
        },
        {
            "name": "Ford_Value",
            "symbol": "F",  # Ford - low price, high liquidity (~$13)
            "side": "buy",
            "strategy_type": "value",
            "win_probability": 0.50,
            "win_odds": 1.5,
            "description": "Value play on low-priced liquid stock",
            "min_account": 13,  # Ford is ~$13/share
            "max_position_pct": 0.30,
        },
        {
            "name": "SOXL_Semiconductor",
            "symbol": "SOXL",  # 3x Semiconductor ETF
            "side": "buy",
            "strategy_type": "leveraged_momentum",
            "win_probability": 0.52,
            "win_odds": 2.5,
            "description": "3x leveraged semiconductor ETF for aggressive growth",
            "min_account": 30,
            "max_position_pct": 0.25,
        },
        {
            "name": "SOFI_Fintech",
            "symbol": "SOFI",  # SoFi - low price fintech
            "side": "buy",
            "strategy_type": "growth",
            "win_probability": 0.48,
            "win_odds": 2.0,
            "description": "Low-price fintech growth play",
            "min_account": 20,
            "max_position_pct": 0.30,
        },
        {
            "name": "AMD_Tech_Growth",
            "symbol": "AMD",  # AMD - tech growth
            "side": "buy",
            "strategy_type": "growth",
            "win_probability": 0.54,
            "win_odds": 2.0,
            "description": "Tech growth play on semiconductor leader",
            "min_account": 200,
            "max_position_pct": 0.25,
        },
        {
            "name": "PLTR_AI_Growth",
            "symbol": "PLTR",  # Palantir - AI/data analytics
            "side": "buy",
            "strategy_type": "growth",
            "win_probability": 0.50,
            "win_odds": 2.2,
            "description": "AI/data analytics growth play",
            "min_account": 50,
            "max_position_pct": 0.25,
        },
        {
            "name": "COIN_Crypto",
            "symbol": "COIN",  # Coinbase - crypto exposure
            "side": "buy",
            "strategy_type": "momentum",
            "win_probability": 0.48,
            "win_odds": 2.5,
            "description": "Crypto exposure through Coinbase",
            "min_account": 300,
            "max_position_pct": 0.20,
        },
        {
            "name": "MARA_Bitcoin",
            "symbol": "MARA",  # Marathon Digital - Bitcoin miner
            "side": "buy",
            "strategy_type": "momentum",
            "win_probability": 0.45,
            "win_odds": 3.0,
            "description": "Bitcoin mining exposure",
            "min_account": 25,
            "max_position_pct": 0.15,
        }
    ]
    return strategies

def select_best_strategy(adapter, strategies, account_value, current_positions):
    """
    SOPHISTICATED STRATEGY SELECTION
    
    Selects the best strategy using multi-factor technical analysis:
    1. Filter by account size and existing positions
    2. Run full technical analysis on each candidate
    3. Score and rank by buy signal strength
    4. Select highest-scoring opportunity
    """
    held_symbols = [p.get('symbol') for p in current_positions]
    
    log("   Analyzing candidate strategies...")
    print()
    
    analyzed_strategies = []
    
    skipped_held = []
    skipped_capital = []
    
    for s in strategies:
        symbol = s['symbol']
        
        # Skip if already holding
        if symbol in held_symbols:
            skipped_held.append(symbol)
            continue
        
        # Skip if account too small
        if account_value < s['min_account']:
            skipped_capital.append(f"{symbol} (needs ${s['min_account']})")
            continue
        
        # Run sophisticated buy analysis
        log(f"   📊 Analyzing {symbol}...")
        should_buy, buy_score, analysis = analyze_buy_opportunity(adapter, symbol, account_value)
        
        if analysis:
            # Show analysis summary
            confidence = analysis.get('confidence', 'N/A')
            price = analysis.get('current_price', 0)
            rsi = analysis.get('rsi')
            
            score_emoji = "🟢" if buy_score >= 70 else "🟡" if buy_score >= 50 else "🔴"
            log(f"      {score_emoji} Score: {buy_score}/100 ({confidence})")
            log(f"      💰 Price: ${price:.2f}" + (f", RSI: {rsi:.1f}" if rsi else ""))
            
            # Show top signals
            if analysis.get('buy_signals'):
                for sig in analysis['buy_signals'][:2]:  # Show top 2
                    log(f"      ✅ {sig}")
            if analysis.get('avoid_signals'):
                for sig in analysis['avoid_signals'][:2]:  # Show top 2
                    log(f"      ⚠️ {sig}")
            
            # Add to candidates if buy signal
            if should_buy:
                s['buy_score'] = buy_score
                s['analysis'] = analysis
                analyzed_strategies.append(s)
            else:
                log(f"      ❌ Score {buy_score} < 60 threshold")
            
            print()
    
    # Show skipped strategies
    if skipped_held:
        log(f"   ⏭️ Skipped (already holding): {', '.join(skipped_held)}")
    if skipped_capital:
        log(f"   ⏭️ Skipped (need more capital): {', '.join(skipped_capital[:3])}")
    
    if not analyzed_strategies:
        log("   ❌ No strategies passed the buy analysis threshold (score >= 60)")
        return None
    
    # Sort by buy score (highest first)
    analyzed_strategies.sort(key=lambda x: x.get('buy_score', 0), reverse=True)
    
    best = analyzed_strategies[0]
    log(f"   🎯 BEST OPPORTUNITY: {best['symbol']} (Score: {best['buy_score']}/100)")
    
    return best

def execute_trade_via_tradier(adapter, symbol, side, quantity, order_type="market"):
    """Execute trade directly via Tradier API"""
    import requests
    
    url = f"https://api.tradier.com/v1/accounts/{adapter.account_id}/orders"
    headers = {
        "Authorization": f"Bearer {adapter.oauth.api_key}",
        "Accept": "application/json",
        "Content-Type": "application/x-www-form-urlencoded"
    }
    
    data = {
        "class": "equity",
        "symbol": symbol,
        "side": side,
        "quantity": str(int(quantity)),
        "type": order_type,
        "duration": "day"
    }
    
    log(f"Submitting order: {side.upper()} {quantity} {symbol} @ {order_type}")
    
    response = requests.post(url, headers=headers, data=data, timeout=10)
    result = response.json()
    
    return result

def get_historical_prices(adapter, symbol, days=30):
    """Get historical price data for technical analysis"""
    import requests
    from datetime import datetime, timedelta
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    headers = {
        "Authorization": f"Bearer {adapter.oauth.api_key}",
        "Accept": "application/json"
    }
    
    try:
        response = requests.get(
            f"https://api.tradier.com/v1/markets/history",
            params={
                "symbol": symbol,
                "interval": "daily",
                "start": start_date.strftime("%Y-%m-%d"),
                "end": end_date.strftime("%Y-%m-%d")
            },
            headers=headers,
            timeout=10
        )
        if response.status_code == 200:
            data = response.json()
            history = data.get('history', {})
            if history:
                days_data = history.get('day', [])
                if isinstance(days_data, dict):
                    days_data = [days_data]
                return days_data
    except Exception as e:
        log(f"Historical data error for {symbol}: {e}", "WARN")
    return []

def calculate_atr(price_data, period=14):
    """Calculate Average True Range (ATR) for volatility measurement"""
    if len(price_data) < period + 1:
        return None
    
    true_ranges = []
    for i in range(1, len(price_data)):
        high = float(price_data[i].get('high', 0))
        low = float(price_data[i].get('low', 0))
        prev_close = float(price_data[i-1].get('close', 0))
        
        tr = max(
            high - low,
            abs(high - prev_close),
            abs(low - prev_close)
        )
        true_ranges.append(tr)
    
    if len(true_ranges) >= period:
        return sum(true_ranges[-period:]) / period
    return None

def calculate_sma(prices, period):
    """Calculate Simple Moving Average"""
    if len(prices) < period:
        return None
    return sum(prices[-period:]) / period

def detect_trend_reversal(price_data, current_price):
    """
    Detect trend reversal using multiple signals:
    - Price crossing below SMA20
    - Lower highs pattern
    - Momentum shift
    """
    if len(price_data) < 20:
        return False, None
    
    closes = [float(d.get('close', 0)) for d in price_data]
    highs = [float(d.get('high', 0)) for d in price_data]
    
    # Calculate SMAs
    sma_10 = calculate_sma(closes, 10)
    sma_20 = calculate_sma(closes, 20)
    
    reversal_signals = []
    
    # Signal 1: Price below SMA20 (bearish)
    if sma_20 and current_price < sma_20:
        reversal_signals.append("Price below SMA20")
    
    # Signal 2: SMA10 crossing below SMA20 (death cross forming)
    if sma_10 and sma_20 and sma_10 < sma_20:
        reversal_signals.append("SMA10 < SMA20 (bearish cross)")
    
    # Signal 3: Lower highs pattern (last 5 days)
    if len(highs) >= 5:
        recent_highs = highs[-5:]
        if all(recent_highs[i] >= recent_highs[i+1] for i in range(len(recent_highs)-1)):
            reversal_signals.append("Lower highs pattern")
    
    # Signal 4: Price dropped more than 3% from recent high
    if highs:
        recent_high = max(highs[-10:]) if len(highs) >= 10 else max(highs)
        drop_from_high = (recent_high - current_price) / recent_high
        if drop_from_high > 0.03:
            reversal_signals.append(f"Down {drop_from_high*100:.1f}% from recent high")
    
    # Reversal detected if 2+ signals
    is_reversal = len(reversal_signals) >= 2
    reason = " | ".join(reversal_signals) if reversal_signals else None
    
    return is_reversal, reason

def analyze_position_for_exit(adapter, position):
    """
    SOPHISTICATED EXIT ANALYSIS
    
    Analyzes positions using multiple exit strategies:
    
    1. PROFIT TARGET: +10% gain → SELL
    2. STOP LOSS: -5% loss → SELL  
    3. TRAILING STOP: Was +8%, now falling → SELL
    4. TIME-BASED: Held 30+ days with profit → SELL
    5. VOLATILITY (ATR): High volatility spike → SELL
    6. TREND REVERSAL: Bearish signals detected → SELL
    """
    symbol = position.get('symbol')
    cost_basis = float(position.get('cost_basis', 0))
    quantity = float(position.get('quantity', 0))
    date_acquired = position.get('date_acquired', '')
    
    if quantity <= 0 or cost_basis <= 0:
        return None, None, None
    
    # Get current price
    quote = get_market_quote(adapter, symbol)
    if not quote:
        return None, None, None
    
    current_price = float(quote.get('last', quote.get('close', 0)))
    if current_price <= 0:
        return None, None, None
    
    # Calculate P&L
    entry_price = cost_basis / quantity
    pnl = (current_price - entry_price) * quantity
    pnl_pct = (current_price - entry_price) / entry_price
    
    # Calculate holding period
    holding_days = 0
    if date_acquired:
        try:
            from datetime import datetime
            # Parse date_acquired (format: 2025-12-02T20:42:45.887Z)
            acquired_date = datetime.fromisoformat(date_acquired.replace('Z', '+00:00'))
            holding_days = (datetime.now(acquired_date.tzinfo) - acquired_date).days
        except:
            holding_days = 0
    
    # Get historical data for advanced analysis
    price_data = get_historical_prices(adapter, symbol, days=30)
    
    # Exit decision logic - check all rules
    should_sell = False
    reason = None
    exit_urgency = "low"
    
    # ========================================
    # RULE 1: PROFIT TARGET (+10%)
    # ========================================
    if pnl_pct >= 0.10:
        should_sell = True
        reason = f"🎯 PROFIT TARGET: +{pnl_pct*100:.1f}% (${pnl:.2f})"
        exit_urgency = "medium"
    
    # ========================================
    # RULE 2: STOP LOSS (-5%)
    # ========================================
    elif pnl_pct <= -0.05:
        should_sell = True
        reason = f"🛑 STOP LOSS: {pnl_pct*100:.1f}% (${pnl:.2f})"
        exit_urgency = "critical"
    
    # ========================================
    # RULE 3: TRAILING STOP (was +8%, now +5-8%)
    # ========================================
    elif pnl_pct >= 0.05 and pnl_pct < 0.08:
        should_sell = True
        reason = f"📉 TRAILING STOP: +{pnl_pct*100:.1f}% - securing gains before further drop"
        exit_urgency = "medium"
    
    # ========================================
    # RULE 4: TIME-BASED EXIT (30+ days with profit)
    # ========================================
    elif holding_days >= 30 and pnl_pct > 0.03:
        should_sell = True
        reason = f"⏰ TIME-BASED: Held {holding_days} days, +{pnl_pct*100:.1f}% - taking profit"
        exit_urgency = "low"
    
    # ========================================
    # RULE 5: VOLATILITY EXIT (High ATR)
    # ========================================
    if not should_sell and price_data:
        atr = calculate_atr(price_data)
        if atr:
            # ATR as percentage of price
            atr_pct = atr / current_price
            # High volatility = ATR > 4% of price
            if atr_pct > 0.04:
                # Only exit on volatility if we have gains to protect
                if pnl_pct > 0.02:
                    should_sell = True
                    reason = f"⚡ VOLATILITY EXIT: ATR={atr_pct*100:.1f}% (high), protecting +{pnl_pct*100:.1f}% gain"
                    exit_urgency = "medium"
    
    # ========================================
    # RULE 6: TREND REVERSAL DETECTION
    # ========================================
    if not should_sell and price_data:
        is_reversal, reversal_reason = detect_trend_reversal(price_data, current_price)
        if is_reversal:
            # Exit on reversal if we have any gains or small loss
            if pnl_pct > -0.02:
                should_sell = True
                reason = f"🔄 TREND REVERSAL: {reversal_reason}"
                exit_urgency = "high"
    
    return should_sell, reason, {
        'symbol': symbol,
        'quantity': int(quantity),
        'entry_price': entry_price,
        'current_price': current_price,
        'pnl': pnl,
        'pnl_pct': pnl_pct,
        'holding_days': holding_days,
        'exit_urgency': exit_urgency
    }

def check_and_execute_sells(adapter, positions):
    """
    Check all positions for exit signals and execute sells.
    Uses sophisticated multi-factor exit analysis.
    """
    sells_executed = []
    
    log("   Analyzing each position for exit signals...")
    print()
    
    for pos in positions:
        symbol = pos.get('symbol', 'Unknown')
        log(f"   📊 Analyzing {symbol}...")
        
        should_sell, reason, details = analyze_position_for_exit(adapter, pos)
        
        if details:
            # Show position analysis
            pnl_sign = "+" if details['pnl_pct'] >= 0 else ""
            holding_info = f", held {details.get('holding_days', 0)} days" if details.get('holding_days', 0) > 0 else ""
            log(f"      Entry: ${details['entry_price']:.2f} → Now: ${details['current_price']:.2f} ({pnl_sign}{details['pnl_pct']*100:.1f}%{holding_info})")
        
        if should_sell and details:
            urgency = details.get('exit_urgency', 'medium').upper()
            log(f"      🚨 EXIT SIGNAL [{urgency}]: {reason}")
            
            # Execute the sell
            result = execute_trade_via_tradier(
                adapter,
                details['symbol'],
                'sell',
                details['quantity']
            )
            
            if result and result.get('order', {}).get('status') == 'ok':
                order_id = result.get('order', {}).get('id')
                sells_executed.append({
                    'symbol': details['symbol'],
                    'quantity': details['quantity'],
                    'reason': reason,
                    'pnl': details['pnl'],
                    'order_id': order_id
                })
                log(f"      ✅ SOLD {details['quantity']} {details['symbol']} (Order #{order_id})")
            else:
                log(f"      ❌ Sell failed: {result}", "WARN")
        else:
            log(f"      ✅ HOLD - No exit signals triggered")
        print()
    
    return sells_executed

def main():
    print("="*70)
    print("  OPTIMUS AUTONOMOUS TRADER")
    print("  NAE Agent Intelligence + Accelerator Methods")
    print(f"  Target: ${ACCOUNT_GROWTH_TARGET:,}")
    print("  Mode: BUY & SELL (Intelligent Exit Management)")
    print("="*70)
    print()
    
    # Step 1: Connect to Tradier and get account status
    log("🔌 Connecting to Tradier...")
    adapter, account = get_tradier_account()
    
    if not account or account.get('status') != 'active':
        log("Account not active!", "ERROR")
        return
    
    # Get account balance - try multiple methods
    balances = account.get('balances', {})
    
    # Try to get cash from nested structure
    cash_obj = balances.get('cash', {})
    if isinstance(cash_obj, dict):
        account_value = float(cash_obj.get('cash_available', 0) or 0)
    else:
        account_value = float(cash_obj or 0)
    
    # Fallback to total_cash
    if account_value == 0:
        account_value = float(balances.get('total_cash', 0) or 0)
    
    # Get total equity for portfolio value
    total_equity = float(balances.get('total_equity', account_value) or account_value)
    
    log(f"✅ Account: {account.get('account_number')}")
    log(f"✅ Status: {account.get('status')}")
    log(f"💰 Cash Available: ${account_value:,.2f}")
    
    # Get current positions
    positions = get_tradier_positions(adapter)
    log(f"📊 Current Positions: {len(positions)}")
    
    portfolio_value = account_value
    for pos in positions:
        qty = float(pos.get('quantity', 0))
        cost = float(pos.get('cost_basis', 0))
        portfolio_value += cost
        
        # Get current value for display
        quote = get_market_quote(adapter, pos.get('symbol'))
        if quote:
            current_price = float(quote.get('last', 0))
            current_value = qty * current_price
            entry_price = cost / qty if qty > 0 else 0
            pnl = current_value - cost
            pnl_pct = (current_price - entry_price) / entry_price * 100 if entry_price > 0 else 0
            pnl_sign = "+" if pnl >= 0 else ""
            log(f"   - {pos.get('symbol')}: {qty:.0f} shares @ ${entry_price:.2f} → ${current_price:.2f} ({pnl_sign}{pnl_pct:.1f}%)")
        else:
            log(f"   - {pos.get('symbol')}: {qty} shares @ ${cost:.2f}")
    
    # ============================================
    # STEP: CHECK FOR SELL SIGNALS
    # ============================================
    if positions:
        print()
        log("="*50)
        log("🔍 SOPHISTICATED EXIT ANALYSIS")
        log("="*50)
        log("   Exit Rules:")
        log("   1. 🎯 Profit Target: +10% gain → SELL")
        log("   2. 🛑 Stop Loss: -5% loss → SELL")
        log("   3. 📉 Trailing Stop: Was +8%, now falling → SELL")
        log("   4. ⏰ Time-based: 30+ days with profit → SELL")
        log("   5. ⚡ Volatility: High ATR spike → SELL")
        log("   6. 🔄 Trend Reversal: Bearish signals → SELL")
        print()
        
        sells = check_and_execute_sells(adapter, positions)
        
        if sells:
            log(f"💰 Executed {len(sells)} sell order(s)")
            total_realized = sum(s['pnl'] for s in sells)
            log(f"💵 Total Realized P&L: ${total_realized:.2f}")
            
            # Refresh positions and account after sells
            time.sleep(2)
            adapter, account = get_tradier_account()
            positions = get_tradier_positions(adapter)
            
            balances = account.get('balances', {})
            cash_obj = balances.get('cash', {})
            if isinstance(cash_obj, dict):
                account_value = float(cash_obj.get('cash_available', 0) or 0)
            else:
                account_value = float(cash_obj or 0)
            if account_value == 0:
                account_value = float(balances.get('total_cash', 0) or 0)
            
            log(f"💰 Updated Cash Available: ${account_value:.2f}")
        else:
            log("   No exit signals - all positions holding")
    
    log(f"💼 Total Portfolio Value: ${portfolio_value:,.2f}")
    
    # Check if we've reached target
    if portfolio_value >= ACCOUNT_GROWTH_TARGET:
        log(f"🎯 TARGET REACHED! Portfolio: ${portfolio_value:,.2f} >= ${ACCOUNT_GROWTH_TARGET:,}")
        return
    
    print()
    log("="*50)
    log("📈 SOPHISTICATED BUY ANALYSIS")
    log("="*50)
    log("   Buy Criteria:")
    log("   1. 📊 RSI Analysis: Oversold (RSI < 30) = Strong Buy")
    log("   2. 📈 Trend Analysis: Price > SMA20 = Uptrend")
    log("   3. 💪 Support Level: Near support = Good Entry")
    log("   4. 📉 Pullback Detection: Dip in uptrend = Buy Opportunity")
    log("   5. 📊 Volume Confirmation: High volume = Strong Signal")
    log("   6. ⚡ Volatility Check: Low ATR = Safe Entry")
    log("   7. 📊 Price Position: Near 52w low = Value")
    print()
    
    # Step 2: Get accelerator strategies (simulating Ralph's intelligence)
    strategies = get_accelerator_strategies()
    log(f"📋 Loaded {len(strategies)} candidate strategies")
    print()
    
    # Step 3: Select best strategy with sophisticated analysis
    best_strategy = select_best_strategy(adapter, strategies, account_value, positions)
    
    if not best_strategy:
        log("No strategies passed buy analysis - market conditions may be unfavorable", "WARN")
        log("💡 Tip: Wait for better entry opportunities or add more capital")
        return
    
    analysis = best_strategy.get('analysis', {})
    log(f"✅ Selected: {best_strategy['name']}")
    log(f"   Symbol: {best_strategy['symbol']}")
    log(f"   Strategy Type: {best_strategy['strategy_type']}")
    log(f"   Buy Score: {best_strategy.get('buy_score', 0)}/100 ({analysis.get('confidence', 'N/A')})")
    
    if analysis.get('rsi'):
        log(f"   RSI: {analysis['rsi']:.1f}")
    if analysis.get('sma_20'):
        log(f"   SMA20: ${analysis['sma_20']:.2f}")
    if analysis.get('support'):
        log(f"   Support: ${analysis['support']:.2f}")
    
    print()
    log("="*50)
    log("🧮 KELLY CRITERION POSITION SIZING")
    log("="*50)
    
    # Step 4: Calculate position size using Kelly Criterion
    kelly_fraction = 0.5 if AGGRESSIVE_GROWTH_MODE else 0.25  # Aggressive = half Kelly
    position_value = calculate_kelly_position_size(
        account_value,
        best_strategy['win_probability'],
        best_strategy['win_odds'],
        kelly_fraction
    )
    
    # Cap at max position percentage
    max_position = account_value * best_strategy['max_position_pct']
    position_value = min(position_value, max_position)
    
    # Get current price
    quote = get_market_quote(adapter, best_strategy['symbol'])
    if not quote:
        log(f"Could not get quote for {best_strategy['symbol']}", "ERROR")
        return
    
    current_price = float(quote.get('last', quote.get('close', 0)))
    log(f"📊 {best_strategy['symbol']} Current Price: ${current_price:.2f}")
    
    # Calculate quantity
    quantity = int(position_value / current_price)
    if quantity < 1:
        log(f"Position size too small: ${position_value:.2f} / ${current_price:.2f} = {quantity} shares", "WARN")
        # Try to buy at least 1 share
        if account_value >= current_price:
            quantity = 1
            position_value = current_price
        else:
            log("Insufficient funds for even 1 share", "ERROR")
            return
    
    log(f"💰 Kelly Position Size: ${position_value:.2f}")
    log(f"📦 Quantity: {quantity} shares")
    log(f"💵 Total Cost: ${quantity * current_price:.2f}")
    
    print()
    log("="*50)
    log("🚀 EXECUTING TRADE")
    log("="*50)
    
    # Step 5: Execute the trade
    result = execute_trade_via_tradier(
        adapter,
        best_strategy['symbol'],
        best_strategy['side'],
        quantity
    )
    
    if result:
        order = result.get('order', {})
        order_id = order.get('id')
        status = order.get('status', 'unknown')
        
        log(f"📝 Order ID: {order_id}")
        log(f"📊 Status: {status}")
        
        if order_id and status == 'ok':
            log("✅ ORDER SUBMITTED SUCCESSFULLY!")
            
            # Wait and check order status
            time.sleep(2)
            
            # Check order fill status
            import requests
            url = f"https://api.tradier.com/v1/accounts/{adapter.account_id}/orders/{order_id}"
            headers = {
                "Authorization": f"Bearer {adapter.oauth.api_key}",
                "Accept": "application/json"
            }
            response = requests.get(url, headers=headers, timeout=10)
            order_status = response.json()
            
            final_order = order_status.get('order', {})
            final_status = final_order.get('status', 'pending')
            fill_price = final_order.get('avg_fill_price', 0)
            
            log(f"📊 Final Status: {final_status}")
            if fill_price:
                log(f"💵 Fill Price: ${float(fill_price):.2f}")
            
            print()
            log("="*50)
            log("📈 TRADE SUMMARY")
            log("="*50)
            log(f"Strategy: {best_strategy['name']}")
            log(f"Symbol: {best_strategy['symbol']}")
            log(f"Quantity: {quantity} shares")
            log(f"Status: {final_status.upper()}")
            if final_status == 'filled':
                log(f"Fill Price: ${float(fill_price):.2f}")
                log(f"Total: ${quantity * float(fill_price):.2f}")
            
            # Calculate progress toward goal
            new_portfolio_estimate = portfolio_value + (quantity * current_price)
            progress = (new_portfolio_estimate / ACCOUNT_GROWTH_TARGET) * 100
            remaining = ACCOUNT_GROWTH_TARGET - new_portfolio_estimate
            
            print()
            log(f"🎯 GOAL PROGRESS: {progress:.1f}%")
            log(f"💰 Estimated Portfolio: ${new_portfolio_estimate:,.2f}")
            log(f"📈 Remaining to Target: ${remaining:,.2f}")
        else:
            log(f"Order issue: {result}", "WARN")
    else:
        log("No response from order", "ERROR")
    
    print()
    log("="*70)
    log("Autonomous trading cycle complete.")
    log("Run again to continue growing toward target.")

if __name__ == "__main__":
    main()

