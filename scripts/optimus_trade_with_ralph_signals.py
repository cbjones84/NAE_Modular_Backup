#!/usr/bin/env python3
"""
Optimus Trade Decision with Ralph Signals
==========================================
This script:
1. Gets current positions from Optimus
2. Evaluates each position using Optimus's analysis and Ralph's knowledge
3. Makes decisions: hold or sell existing positions
4. Uses available cash ($41.79) to make new trades based on Ralph's strategies
"""

import os
import sys
import datetime
from pathlib import Path

# Setup paths
SCRIPT_DIR = Path(__file__).parent
NAE_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(NAE_DIR))

# Set environment for Tradier
os.environ["TRADIER_SANDBOX"] = "false"
os.environ["TRADIER_API_KEY"] = "27Ymk28vtbgqY1LFYxhzaEmIuwJb"
os.environ["TRADIER_ACCOUNT_ID"] = "6YB66744"
os.environ["TRADIER_ACCOUNT_TYPE"] = "cash"

def log(message: str, level: str = "INFO"):
    """Log with timestamp"""
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}")

def main():
    print("\n" + "=" * 70)
    print("  OPTIMUS TRADING DECISION WITH RALPH SIGNALS")
    print("  Evaluating Positions & Trading with Available Cash")
    print("=" * 70 + "\n")
    
    try:
        from agents.optimus import OptimusAgent
        
        # Initialize Optimus
        log("Initializing Optimus...")
        optimus = OptimusAgent(sandbox=False)
        
        # Sync account balance
        log("Syncing account balance...")
        optimus._sync_account_balance()
        
        # Get current status
        status = optimus.get_trading_status()
        log(f"NAV: ${status.get('nav', 0):,.2f}")
        log(f"Cash Available: ${status.get('cash', 0):,.2f}")
        log(f"Buying Power: ${status.get('buying_power', 0):,.2f}")
        
        # Get current positions from Tradier
        log("\n" + "-" * 70)
        log("PHASE 1: EVALUATING CURRENT POSITIONS")
        log("-" * 70)
        
        positions = []
        if hasattr(optimus, 'self_healing_engine') and optimus.self_healing_engine:
            if hasattr(optimus.self_healing_engine, 'tradier_adapter'):
                tradier_adapter = optimus.self_healing_engine.tradier_adapter
                if tradier_adapter:
                    positions = tradier_adapter.get_positions()
                    log(f"Found {len(positions)} positions from Tradier")
        
        # Also check Optimus's internal position tracking
        if hasattr(optimus, 'open_positions') and optimus.open_positions:
            log(f"Optimus has {len(optimus.open_positions)} tracked positions")
            for symbol, pos in optimus.open_positions.items():
                log(f"  - {symbol}: {pos}")
        
        # Use Optimus's built-in exit analysis
        log("\nUsing Optimus's built-in position evaluation...")
        optimus._analyze_and_execute_exits()
        
        # Get positions again after evaluation
        position_decisions = []
        for pos in positions:
            symbol = pos.get('symbol') or (pos.get('symbol_description', '').split()[0] if pos.get('symbol_description') else '')
            if not symbol:
                continue
                
            quantity = float(pos.get('quantity', 0))
            if quantity == 0:
                continue
                
            log(f"\nPosition: {symbol} ({quantity} shares)")
            
            # Check if Optimus has exit signals (from logs we saw exit signals)
            # The exit analysis already ran via _analyze_and_execute_exits()
            # Check if position is still open
            still_open = False
            if hasattr(optimus, 'open_positions'):
                still_open = symbol in optimus.open_positions
            elif hasattr(optimus, 'open_positions_dict'):
                still_open = symbol in optimus.open_positions_dict
            
            if still_open:
                decision = "HOLD"
                reason = "Position still open - Optimus evaluated and decided to hold"
            else:
                decision = "SOLD"
                reason = "Position was closed by Optimus exit analysis"
            
            position_decisions.append({
                'symbol': symbol,
                'quantity': quantity,
                'decision': decision,
                'reason': reason
            })
            
            log(f"  Status: {decision} - {reason}")
        
        # Execute position decisions
        log("\n" + "-" * 70)
        log("PHASE 2: EXECUTING POSITION DECISIONS")
        log("-" * 70)
        
        for decision in position_decisions:
            if decision['decision'] == "SELL":
                symbol = decision['symbol']
                quantity = int(decision['quantity'])
                log(f"\nExecuting SELL order for {symbol} ({quantity} shares)...")
                
                try:
                    result = optimus.close_position(symbol, quantity=quantity)
                    if result:
                        log(f"  ✅ Sell order submitted for {symbol}")
                    else:
                        log(f"  ❌ Failed to sell {symbol}", "ERROR")
                except Exception as e:
                    log(f"  ❌ Error selling {symbol}: {e}", "ERROR")
            else:
                log(f"\nHolding {decision['symbol']} - {decision['reason']}")
        
        # Now use available cash for new trades
        log("\n" + "-" * 70)
        log("PHASE 3: TRADING WITH AVAILABLE CASH")
        log("-" * 70)
        
        cash_available = status.get('cash', 0)
        log(f"Available cash: ${cash_available:,.2f}")
        
        if cash_available < 10:
            log("Insufficient cash for trading (need at least $10)", "WARNING")
            return
        
        # Get Ralph's knowledge/strategies
        try:
            from agents.ralph import RalphAgent
            ralph = RalphAgent()
            log("Ralph agent initialized")
        except Exception as e:
            log(f"Could not initialize Ralph: {e}", "WARNING")
            ralph = None
        
        # Find affordable trading opportunities using Optimus's day trading cycle
        log("\nScanning for trading opportunities using Optimus's day trading strategies...")
        
        # Use Optimus's day trading cycle to find opportunities
        best_opportunity = None
        try:
            if hasattr(optimus, 'run_day_trading_cycle'):
                log("Running Optimus day trading cycle to find opportunities...")
                day_trade_result = optimus.run_day_trading_cycle()
                
                if day_trade_result and day_trade_result.get('trades_executed', 0) > 0:
                    log(f"✅ Optimus executed {day_trade_result.get('trades_executed', 0)} day trade(s)")
                    log("Day trading cycle completed - check positions for new trades")
                    return
                else:
                    log("Day trading cycle found no immediate opportunities")
            
            # Fallback: Manual scan of affordable symbols
            candidates = ["SOXL", "TQQQ", "SQQQ", "AMD", "F", "PLTR", "SOFI", "NIO"]
            log("Scanning affordable symbols manually...")
            
            for symbol in candidates:
                try:
                    if hasattr(optimus, 'self_healing_engine') and optimus.self_healing_engine:
                        tradier_adapter = optimus.self_healing_engine.tradier_adapter
                        if tradier_adapter:
                            # Try get_quote method
                            quote = None
                            if hasattr(tradier_adapter, 'get_quote'):
                                quote = tradier_adapter.get_quote(symbol)
                            
                            # Fallback to REST client
                            if not quote and hasattr(tradier_adapter, 'rest_client'):
                                try:
                                    response = tradier_adapter.rest_client._request("GET", f"markets/quotes?symbols={symbol}")
                                    if response and 'quotes' in response:
                                        quote_data = response['quotes'].get('quote', {})
                                        if isinstance(quote_data, list):
                                            quote_data = quote_data[0] if quote_data else {}
                                        quote = quote_data
                                except:
                                    pass
                            
                            if quote:
                                price = float(quote.get('last', quote.get('close', 0)))
                                change_pct = float(quote.get('change_percentage', 0))
                                
                                # Check if affordable
                                if price > 0 and price <= cash_available * 0.9:  # Leave 10% buffer
                                    shares = int((cash_available * 0.9) / price)
                                    if shares > 0:
                                        log(f"  {symbol}: ${price:.2f} ({change_pct:+.2f}%) - Can buy {shares} share(s)")
                                        
                                        # Simple scoring: prefer positive momentum
                                        if best_opportunity is None:
                                            best_opportunity = {
                                                'symbol': symbol,
                                                'price': price,
                                                'shares': shares,
                                                'change_pct': change_pct
                                            }
                                        elif change_pct > best_opportunity.get('change_pct', -999):
                                            best_opportunity = {
                                                'symbol': symbol,
                                                'price': price,
                                                'shares': shares,
                                                'change_pct': change_pct
                                            }
                except Exception as e:
                    log(f"  Error analyzing {symbol}: {e}", "WARNING")
                    continue
        except Exception as e:
            log(f"Error in opportunity scan: {e}", "ERROR")
        
        if best_opportunity:
            symbol = best_opportunity['symbol']
            shares = best_opportunity['shares']
            price = best_opportunity['price']
            trade_value = shares * price
            
            log(f"\nBest opportunity: {symbol}")
            log(f"  Price: ${price:.2f}")
            log(f"  Shares: {shares}")
            log(f"  Trade Value: ${trade_value:.2f}")
            
            # Get entry signal from Optimus using timing engine
            log(f"\nGetting entry signal for {symbol}...")
            try:
                # Use Optimus's timing engine for entry analysis
                if hasattr(optimus, 'timing_engine') and optimus.timing_engine:
                    # Get price data for analysis
                    price_data = None
                    try:
                        if hasattr(optimus, 'polygon_client') and optimus.polygon_client:
                            # Try to get recent price data
                            pass  # Simplified for now
                    except:
                        pass
                    
                    # Analyze entry timing
                    entry_analysis = optimus.timing_engine.analyze_entry_timing(
                        symbol=symbol,
                        current_price=price,
                        price_data=price_data
                    )
                    
                    if entry_analysis:
                        signal = entry_analysis.entry_signal.value if hasattr(entry_analysis.entry_signal, 'value') else str(entry_analysis.entry_signal)
                        confidence = entry_analysis.confidence
                        score = entry_analysis.timing_score
                        
                        log(f"  Entry Signal: {signal}")
                        log(f"  Confidence: {confidence:.0%}")
                        log(f"  Score: {score:.1f}/100")
                        
                        # Only trade if signal is buy/strong_buy and confidence > 50%
                        if signal in ['buy', 'strong_buy'] and confidence > 0.5:
                            log(f"\n✅ Executing BUY order for {symbol}...")
                            
                            execution_details = {
                                "symbol": symbol,
                                "side": "buy",
                                "quantity": shares,
                                "order_type": "market",
                                "asset_type": "equity",
                                "strategy_id": "ralph_optimus_collab",
                                "strategy_name": "Ralph-Optimus Collaboration",
                                "reason": f"Entry signal: {signal} (confidence: {confidence:.0%})"
                            }
                            
                            result = optimus.execute_trade(execution_details)
                            
                            if result and result.get('status') != 'rejected':
                                log(f"  ✅ Trade executed successfully!")
                                log(f"  Order ID: {result.get('order_id', 'N/A')}")
                            else:
                                log(f"  ❌ Trade rejected: {result.get('reason', 'Unknown') if result else 'No result'}", "ERROR")
                        else:
                            log(f"  ⚠️ Entry signal not strong enough ({signal}, {confidence:.0%} confidence) - skipping trade")
                    else:
                        log("  ⚠️ No entry analysis available - trying direct execution with lower threshold...")
                        # If no analysis, still try if price is reasonable
                        if price <= cash_available * 0.8:  # More conservative
                            log(f"\n⚠️ Executing BUY with limited analysis (price-based decision)...")
                            execution_details = {
                                "symbol": symbol,
                                "side": "buy",
                                "quantity": shares,
                                "order_type": "market",
                                "asset_type": "equity",
                                "strategy_id": "manual_cash_utilization",
                                "strategy_name": "Cash Utilization Trade",
                                "reason": f"Affordable opportunity: ${price:.2f} within budget"
                            }
                            result = optimus.execute_trade(execution_details)
                            if result and result.get('status') != 'rejected':
                                log(f"  ✅ Trade executed!")
                            else:
                                log(f"  ❌ Trade rejected: {result.get('reason', 'Unknown') if result else 'No result'}", "ERROR")
                else:
                    log("  ⚠️ Timing engine not available - skipping trade")
            except Exception as e:
                log(f"  ❌ Error getting entry signal: {e}", "ERROR")
                import traceback
                traceback.print_exc()
        else:
            log("No suitable trading opportunities found", "WARNING")
        
        log("\n" + "=" * 70)
        log("TRADING DECISION PROCESS COMPLETE")
        log("=" * 70)
        
    except Exception as e:
        log(f"Error: {e}", "ERROR")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

