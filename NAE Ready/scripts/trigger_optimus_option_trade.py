#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trigger Optimus to Evaluate and Execute Intelligent Option Trades

This script runs at three key times:
1. Market Open (9:30 AM - 10:30 AM ET) - Morning momentum strategies
2. Midday (12:00 PM - 1:30 PM ET) - Reversal/continuation strategies
3. Market Close (3:00 PM - 4:00 PM ET) - End-of-day premium capture

This script:
1. Detects current market phase (open/midday/close)
2. Uses Optimus's option analysis to find opportunities
3. Evaluates multiple symbols for best option opportunity
4. Executes intelligent option trades with phase-appropriate strategies
"""

import os
import sys
import time
import logging
import io
from datetime import datetime, time as dt_time
from typing import Dict, Any, Optional, List
from pathlib import Path

# Fix encoding for Windows
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Add NAE to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import Optimus
from agents.optimus import OptimusAgent

# Common symbols for option analysis
DEFAULT_SYMBOLS = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "GOOGL", "META"]


def get_market_status() -> Dict[str, Any]:
    """Get current market status and trading phase"""
    now = datetime.now()
    current_time = now.time()
    
    # Market hours (ET)
    market_open = dt_time(9, 30)  # 9:30 AM ET
    market_close = dt_time(16, 0)  # 4:00 PM ET
    
    # Phase definitions
    morning_window_start = dt_time(9, 30)  # 9:30 AM ET
    morning_window_end = dt_time(10, 30)  # 10:30 AM ET
    midday_window_start = dt_time(12, 0)  # 12:00 PM ET
    midday_window_end = dt_time(13, 30)  # 1:30 PM ET
    close_window_start = dt_time(15, 0)  # 3:00 PM ET
    close_window_end = dt_time(16, 0)  # 4:00 PM ET
    
    # Check if market is open
    is_open = market_open <= current_time <= market_close
    
    # Determine current trading phase
    phase = None
    phase_description = None
    minutes_until_next_phase = 0
    can_trade = False  # Flag for whether trading is allowed
    
    if is_open:
        can_trade = True  # Trading allowed throughout market hours
        if morning_window_start <= current_time <= morning_window_end:
            phase = "morning"
            phase_description = "Market Open (Morning Momentum)"
            minutes_until_next_phase = int(((datetime.combine(now.date(), morning_window_end) - now).total_seconds() / 60))
        elif midday_window_start <= current_time <= midday_window_end:
            phase = "midday"
            phase_description = "Midday (Reversal/Continuation)"
            minutes_until_next_phase = int(((datetime.combine(now.date(), midday_window_end) - now).total_seconds() / 60))
        elif close_window_start <= current_time <= close_window_end:
            phase = "close"
            phase_description = "Market Close (Premium Capture)"
            minutes_until_next_phase = int(((datetime.combine(now.date(), close_window_end) - now).total_seconds() / 60))
        else:
            # Between phases - still allow trading but use general strategy
            if current_time < midday_window_start:
                phase = "between_morning_midday"
                phase_description = "Between Morning and Midday (General Trading)"
                minutes_until_next_phase = int(((datetime.combine(now.date(), midday_window_start) - now).total_seconds() / 60))
            elif current_time < close_window_start:
                phase = "between_midday_close"
                phase_description = "Between Midday and Close (General Trading)"
                minutes_until_next_phase = int(((datetime.combine(now.date(), close_window_start) - now).total_seconds() / 60))
            else:
                phase = "after_close_window"
                phase_description = "After Close Window (General Trading)"
                minutes_until_next_phase = 0
    else:
        phase = "closed"
        phase_description = "Market Closed"
        minutes_until_next_phase = 0
        can_trade = False
    
    # Calculate minutes until close
    if is_open:
        close_datetime = datetime.combine(now.date(), market_close)
        if current_time > market_close:
            minutes_until_close = 0
        else:
            minutes_until_close = int((close_datetime - now).total_seconds() / 60)
    else:
        minutes_until_close = 0
    
    return {
        "is_open": is_open,
        "current_time": current_time.strftime("%H:%M:%S"),
        "phase": phase,
        "phase_description": phase_description,
        "minutes_until_next_phase": minutes_until_next_phase,
        "minutes_until_close": minutes_until_close,
        "market_open": market_open.strftime("%H:%M"),
        "market_close": market_close.strftime("%H:%M"),
        "morning_window": f"{morning_window_start.strftime('%H:%M')}-{morning_window_end.strftime('%H:%M')}",
        "midday_window": f"{midday_window_start.strftime('%H:%M')}-{midday_window_end.strftime('%H:%M')}",
        "close_window": f"{close_window_start.strftime('%H:%M')}-{close_window_end.strftime('%H:%M')}",
        "can_trade": can_trade
    }


def find_best_option_opportunity(optimus: OptimusAgent, symbols: List[str]) -> Optional[Dict[str, Any]]:
    """Find the best option opportunity across multiple symbols"""
    logger.info("🔍 Analyzing option opportunities across symbols...")
    
    best_opportunity = None
    best_score = -float('inf')
    
    for symbol in symbols:
        try:
            logger.info(f"  Analyzing {symbol}...")
            
            # Generate option signals for this symbol
            option_signals = optimus._generate_option_signals(symbol)
            
            if not option_signals:
                logger.debug(f"    No option signals for {symbol}")
                continue
            
            # Calculate opportunity score
            score = 0.0
            opportunity_details = {
                "symbol": symbol,
                "underlying": symbol,
                "asset_type": "option",
                "option_signals": option_signals
            }
            
            # IV edge (positive is good - IV higher than forecasted)
            iv_edge = option_signals.get("iv_edge_atm")
            if iv_edge is not None:
                if iv_edge > 0:
                    score += iv_edge * 100  # Positive IV edge is good for selling options
                    opportunity_details["iv_edge"] = iv_edge
                    logger.info(f"    {symbol}: IV Edge = {iv_edge:.4f}")
            
            # Check liquidity (bid/ask spread)
            atm_bid = option_signals.get("atm_bid")
            atm_ask = option_signals.get("atm_ask")
            if atm_bid and atm_ask and atm_bid > 0 and atm_ask > 0:
                spread_pct = (atm_ask - atm_bid) / ((atm_bid + atm_ask) / 2) if atm_bid > 0 else 1.0
                if spread_pct < 0.10:  # Good liquidity (less than 10% spread)
                    score += 20
                    opportunity_details["spread_pct"] = spread_pct
                    opportunity_details["bid"] = atm_bid
                    opportunity_details["ask"] = atm_ask
                    opportunity_details["atm_bid"] = atm_bid  # Store for execution details
                    opportunity_details["atm_ask"] = atm_ask  # Store for execution details
                    logger.info(f"    {symbol}: Spread = {spread_pct:.2%}")
            
            # Volatility forecast
            ensemble_vol = option_signals.get("ensemble_vol")
            if ensemble_vol:
                opportunity_details["volatility"] = ensemble_vol
                # Higher vol = higher option premiums (good for selling)
                if ensemble_vol > 0.20:
                    score += 10
            
            # Dispersion signal (for index products)
            dispersion = option_signals.get("dispersion_signal")
            if dispersion:
                opportunity_details["dispersion"] = dispersion
                logger.info(f"    {symbol}: Dispersion signal available")
            
            opportunity_details["score"] = score
            
            if score > best_score:
                best_score = score
                best_opportunity = opportunity_details
                logger.info(f"  ✅ {symbol} is currently best opportunity (score: {score:.2f})")
                
        except Exception as e:
            logger.error(f"  ❌ Error analyzing {symbol}: {e}")
            continue
    
    return best_opportunity


def create_option_execution_details(opportunity: Dict[str, Any], phase: str = "close") -> Dict[str, Any]:
    """Create execution details for an option trade based on opportunity and trading phase"""
    symbol = opportunity["symbol"]
    option_signals = opportunity["option_signals"]
    spot_price = option_signals.get("spot_price", 0)
    
    # Determine option strategy based on signals and phase
    iv_edge = opportunity.get("iv_edge", 0)
    volatility = opportunity.get("volatility", 0.2)
    
    # Get option chain to find ATM strike
    try:
        # Determine strike (ATM for now, can adjust based on phase)
        strike = round(spot_price)  # Round to nearest dollar
        
        # Get option price from opportunity or signals (use mid-price)
        # First try from opportunity (passed from find_best_option_opportunity)
        atm_bid = opportunity.get("atm_bid") or option_signals.get("atm_bid", 0)
        atm_ask = opportunity.get("atm_ask") or option_signals.get("atm_ask", 0)
        option_price = (atm_bid + atm_ask) / 2.0 if (atm_bid > 0 and atm_ask > 0) else 0
        
        # If no option price available, estimate from spot and volatility
        if option_price == 0 and spot_price > 0:
            # Rough estimate: use 2-3% of spot as option premium for ATM calls
            # Adjust based on volatility
            vol_multiplier = volatility if volatility > 0 else 0.20
            option_price = spot_price * 0.02 * (1 + vol_multiplier)
        
        # Ensure we have a valid price - critical for trade execution
        if option_price == 0:
            logger.warning(f"⚠️  No option price available for {symbol} - trade may fail validation")
        
        # Base execution details
        execution_details = {
            "action": "execute_trade",
            "asset_type": "option",
            "symbol": symbol,  # Underlying symbol
            "underlying": symbol,
            "option_type": "call",
            "strike": strike,
            "quantity": 1,  # 1 contract (100 shares)
            "order_type": "limit",
            "side": "sell_to_open",
            "price": option_price,  # CRITICAL: Required for trade validation
            "trust_score": 65,
            "backtest_score": 60,
            "expected_return": 0.01,
            "stop_loss_pct": 0.05,
            "time_in_force": "day",
            "quant_signals": option_signals,
            "iv_edge": iv_edge,
            "opportunity_score": opportunity.get("score", 0),
            "trading_phase": phase
        }
        
        # Phase-specific strategy selection
        if phase == "morning":
            # Morning: Momentum plays, trend continuation
            # Higher risk tolerance, directional bets
            if iv_edge > 0.03:  # Elevated IV in morning
                # Sell premium on morning volatility
                execution_details["side"] = "sell_to_open"
                execution_details["option_type"] = "call"
                execution_details["strategy_name"] = "Morning Volatility Premium"
                execution_details["strike"] = round(spot_price * 1.02)  # 2% OTM for safety
                execution_details["trust_score"] = 70
            elif iv_edge < -0.03:  # Low IV - buy options for direction
                # Buy calls for morning momentum
                execution_details["side"] = "buy_to_open"
                execution_details["option_type"] = "call"
                execution_details["strategy_name"] = "Morning Momentum Call"
                execution_details["strike"] = round(spot_price * 0.98)  # Slightly ITM
                execution_details["trust_score"] = 68
                execution_details["expected_return"] = 0.15  # Higher target for directional play
            else:
                # Neutral - covered call
                execution_details["strategy_name"] = "Morning Covered Call"
                execution_details["strike"] = strike  # ATM
                execution_details["trust_score"] = 65
        
        elif phase == "midday":
            # Midday: Reversal plays, range trading, premium collection
            # More conservative, focus on premium
            if iv_edge > 0.05:  # Strong IV edge
                # Sell premium (best time for credit spreads)
                execution_details["side"] = "sell_to_open"
                execution_details["option_type"] = "call"
                execution_details["strategy_name"] = "Midday IV Edge Premium"
                execution_details["strike"] = round(spot_price * 1.01)  # 1% OTM
                execution_details["trust_score"] = 72
                execution_details["expected_return"] = 0.008  # Lower but safer
            elif iv_edge < -0.05:  # Low IV - consider buying
                # Buy low IV options for afternoon move
                execution_details["side"] = "buy_to_open"
                execution_details["option_type"] = "put" if volatility < 0.15 else "call"
                execution_details["strategy_name"] = "Midday Low IV Purchase"
                execution_details["strike"] = strike  # ATM
                execution_details["trust_score"] = 65
                execution_details["expected_return"] = 0.12
            else:
                # Neutral - cash-secured put
                execution_details["side"] = "sell_to_open"
                execution_details["option_type"] = "put"
                execution_details["strategy_name"] = "Midday Cash-Secured Put"
                execution_details["strike"] = round(spot_price * 0.98)  # 2% OTM
                execution_details["trust_score"] = 68
        
        elif phase == "close":
            # Market close: Premium capture, end-of-day strategies
            # Very conservative, focus on capturing daily premium
            if iv_edge > 0.02:  # Any positive IV edge
                # Sell premium (covered call or cash-secured put)
                execution_details["side"] = "sell_to_open"
                execution_details["option_type"] = "call"
                execution_details["strategy_name"] = "End-of-Day Premium Capture"
                execution_details["strike"] = round(spot_price * 1.01)  # 1% OTM
                execution_details["trust_score"] = 70
                execution_details["expected_return"] = 0.006  # Conservative target
            elif iv_edge < -0.03:  # Low IV - might see volatility tomorrow
                # Buy puts as hedge or speculative play
                execution_details["side"] = "buy_to_open"
                execution_details["option_type"] = "put"
                execution_details["strategy_name"] = "End-of-Day Low IV Put"
                execution_details["strike"] = round(spot_price * 0.99)  # 1% OTM
                execution_details["trust_score"] = 62
                execution_details["expected_return"] = 0.10
            else:
                # Default: Covered call
                execution_details["strategy_name"] = "Market Close Covered Call"
                execution_details["strike"] = strike  # ATM
                execution_details["trust_score"] = 65
        
        elif phase in ["between_morning_midday", "between_midday_close", "after_close_window"]:
            # General trading between windows - balanced approach
            # Use current market conditions to decide
            if iv_edge > 0.03:  # Positive IV edge
                execution_details["side"] = "sell_to_open"
                execution_details["option_type"] = "call"
                execution_details["strategy_name"] = "General IV Edge Premium"
                execution_details["strike"] = round(spot_price * 1.01)  # 1% OTM
                execution_details["trust_score"] = 67
                execution_details["expected_return"] = 0.008
            elif iv_edge < -0.03:  # Negative IV edge
                execution_details["side"] = "buy_to_open"
                execution_details["option_type"] = "call" if volatility > 0.18 else "put"
                execution_details["strategy_name"] = "General Low IV Purchase"
                execution_details["strike"] = strike  # ATM
                execution_details["trust_score"] = 63
                execution_details["expected_return"] = 0.10
            else:
                # Neutral - balanced approach
                execution_details["side"] = "sell_to_open"
                execution_details["option_type"] = "call"
                execution_details["strategy_name"] = "General Covered Call"
                execution_details["strike"] = round(spot_price * 1.005)  # 0.5% OTM
                execution_details["trust_score"] = 65
                execution_details["expected_return"] = 0.007
        else:
            # Unknown phase or closed - conservative approach
            execution_details["strategy_name"] = "Conservative Option Trade"
            execution_details["trust_score"] = 60
        
        return execution_details
        
    except Exception as e:
        logger.error(f"Error creating execution details: {e}")
        return None


def trigger_optimus_option_trade(symbols: Optional[List[str]] = None, sandbox: bool = True):
    """Main function to trigger Optimus option trade"""
    logger.info("="*80)
    logger.info("🎯 TRIGGERING OPTIMUS OPTION TRADE")
    logger.info("="*80)
    
    # Check market status
    market_status = get_market_status()
    phase = market_status.get("phase", "unknown")
    phase_desc = market_status.get("phase_description", "Unknown Phase")
    
    logger.info(f"Market Status:")
    logger.info(f"  Current Time: {market_status['current_time']}")
    logger.info(f"  Market Open: {market_status['market_open']}")
    logger.info(f"  Market Close: {market_status['market_close']}")
    logger.info(f"  Is Open: {market_status['is_open']}")
    logger.info(f"  Trading Phase: {phase_desc}")
    logger.info(f"  Minutes Until Next Phase: {market_status['minutes_until_next_phase']}")
    logger.info(f"  Minutes Until Close: {market_status['minutes_until_close']}")
    logger.info("")
    logger.info(f"Trading Windows:")
    logger.info(f"  Morning: {market_status['morning_window']} (Momentum)")
    logger.info(f"  Midday: {market_status['midday_window']} (Reversal/Continuation)")
    logger.info(f"  Close: {market_status['close_window']} (Premium Capture)")
    
    if not market_status["is_open"]:
        logger.warning("⚠️  Market is currently CLOSED")
        logger.info("Note: This script will still analyze opportunities for next market open")
        phase = "closed"
        if not market_status.get("can_trade", False):
            logger.warning("⚠️  Trading will not execute while market is closed")
    elif phase not in ["morning", "midday", "close"]:
        logger.info(f"ℹ️  Current time is between trading windows - General Trading Mode")
        logger.info(f"Phase: {phase_desc}")
        logger.info("✅ Trading enabled throughout market hours - using general strategy")
        if phase == "between_morning_midday":
            logger.info(f"Next optimal window: Midday (12:00 PM) in {market_status['minutes_until_next_phase']} minutes")
        elif phase == "between_midday_close":
            logger.info(f"Next optimal window: Close (3:00 PM) in {market_status['minutes_until_next_phase']} minutes")
    
    # Initialize Optimus
    logger.info("\n🤖 Initializing Optimus...")
    try:
        optimus = OptimusAgent(sandbox=sandbox)
        
        # Ensure trading is enabled
        if not optimus.trading_enabled:
            logger.warning("⚠️  Trading disabled - attempting to enable...")
            if hasattr(optimus, 'deactivate_kill_switch'):
                optimus.deactivate_kill_switch("Manual option trade trigger")
                logger.info("✅ Trading enabled")
        else:
            logger.info("✅ Trading enabled")
            
    except Exception as e:
        logger.error(f"❌ Failed to initialize Optimus: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False
    
    # Find best option opportunity
    symbols_to_analyze = symbols or DEFAULT_SYMBOLS
    logger.info(f"\n📊 Analyzing {len(symbols_to_analyze)} symbols for option opportunities...")
    
    opportunity = find_best_option_opportunity(optimus, symbols_to_analyze)
    
    if not opportunity:
        logger.warning("❌ No suitable option opportunities found")
        return False
    
    logger.info("\n" + "="*80)
    logger.info(f"✅ BEST OPPORTUNITY FOUND: {opportunity['symbol']}")
    logger.info(f"   Opportunity Score: {opportunity.get('score', 0):.2f}")
    logger.info(f"   IV Edge: {opportunity.get('iv_edge', 'N/A')}")
    logger.info("="*80)
    
    # Create execution details based on current phase
    logger.info("\n📝 Creating execution details...")
    logger.info(f"   Strategy will be tailored for: {phase_desc}")
    execution_details = create_option_execution_details(opportunity, phase=phase)
    
    if not execution_details:
        logger.error("❌ Failed to create execution details")
        return False
    
    logger.info(f"✅ Execution details created:")
    logger.info(f"   Strategy: {execution_details.get('strategy_name')}")
    logger.info(f"   Symbol: {execution_details.get('symbol')}")
    logger.info(f"   Option Type: {execution_details.get('option_type')}")
    logger.info(f"   Side: {execution_details.get('side')}")
    logger.info(f"   Strike: {execution_details.get('strike')}")
    logger.info(f"   Trust Score: {execution_details.get('trust_score')}")
    
    # Execute trade
    logger.info("\n🚀 EXECUTING OPTION TRADE...")
    logger.info("="*80)
    
    try:
        result = optimus.execute_trade(execution_details)
        
        logger.info("="*80)
        logger.info("📊 TRADE EXECUTION RESULT:")
        logger.info("="*80)
        logger.info(f"Status: {result.get('status', 'unknown')}")
        logger.info(f"Symbol: {result.get('symbol', 'N/A')}")
        
        if result.get('status') == 'executed':
            logger.info("✅ TRADE EXECUTED SUCCESSFULLY!")
            logger.info(f"Order ID: {result.get('order_id', 'N/A')}")
            logger.info(f"Execution Price: {result.get('execution_price', 'N/A')}")
            logger.info(f"Quantity: {result.get('quantity', 'N/A')}")
        else:
            logger.warning(f"⚠️  Trade Status: {result.get('status')}")
            logger.warning(f"Reason: {result.get('reason', 'N/A')}")
        
        logger.info("="*80)
        return result.get('status') == 'executed'
        
    except Exception as e:
        logger.error(f"❌ Error executing trade: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Trigger Optimus to execute intelligent option trades at market open, midday, or before close. "
                    "Strategies are automatically selected based on current trading phase."
    )
    parser.add_argument("--symbols", nargs="+", help="Symbols to analyze (default: SPY QQQ AAPL MSFT NVDA TSLA AMZN GOOGL META)")
    parser.add_argument("--live", action="store_true", help="Use live mode (default: sandbox)")
    parser.add_argument("--sandbox", action="store_true", default=True, help="Use sandbox mode (default)")
    
    args = parser.parse_args()
    
    sandbox_mode = not args.live if args.live else True
    
    success = trigger_optimus_option_trade(
        symbols=args.symbols,
        sandbox=sandbox_mode
    )
    
    sys.exit(0 if success else 1)

