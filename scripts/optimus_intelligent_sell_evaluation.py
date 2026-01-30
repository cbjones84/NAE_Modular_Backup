#!/usr/bin/env python3
"""
Optimus Intelligent Sell Evaluation
===================================
This script evaluates all open positions and makes intelligent sell decisions
based on:
1. Optimus's own feedback loop and performance data
2. Ralph's trading strategies and exit rules
3. Technical analysis and timing signals
4. Profit targets and risk management

NO SHORT SELLING - Only sells shares that are owned.
"""

import os
import sys
import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

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

def load_ralph_knowledge() -> Optional[Dict[str, Any]]:
    """Load Ralph's knowledge package"""
    try:
        from scripts.ralph_optimus_knowledge_trade import RalphKnowledgeExporter
        
        ralph = RalphKnowledgeExporter()
        if ralph.load_knowledge():
            knowledge = ralph.export_for_optimus()
            log("✅ Loaded Ralph's knowledge package")
            return knowledge
        else:
            log("⚠️ Could not load Ralph's knowledge", "WARNING")
            return None
    except Exception as e:
        log(f"⚠️ Error loading Ralph knowledge: {e}", "WARNING")
        return None

def main():
    print("\n" + "=" * 70)
    print("  OPTIMUS INTELLIGENT SELL EVALUATION")
    print("  Using Feedback Loop + Ralph Knowledge")
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
        log(f"Realized P&L: ${status.get('realized_pnl', 0):,.2f}")
        log(f"Unrealized P&L: ${status.get('unrealized_pnl', 0):,.2f}")
        
        # Load Ralph's knowledge
        log("\n" + "-" * 70)
        log("Loading Ralph's Knowledge")
        log("-" * 70)
        ralph_knowledge = load_ralph_knowledge()
        
        # Get current positions
        log("\n" + "-" * 70)
        log("PHASE 1: GETTING CURRENT POSITIONS")
        log("-" * 70)
        
        positions = []
        if hasattr(optimus, 'self_healing_engine') and optimus.self_healing_engine:
            if hasattr(optimus.self_healing_engine, 'tradier_adapter'):
                tradier_adapter = optimus.self_healing_engine.tradier_adapter
                if tradier_adapter:
                    positions = tradier_adapter.get_positions()
                    log(f"Found {len(positions)} positions from Tradier")
        
        # Also check Optimus's internal position tracking
        optimus_positions = {}
        if hasattr(optimus, 'open_positions_dict') and optimus.open_positions_dict:
            optimus_positions = optimus.open_positions_dict
            log(f"Optimus tracking {len(optimus_positions)} positions internally")
        
        if not positions and not optimus_positions:
            log("No open positions found", "WARNING")
            return
        
        # Evaluate each position
        log("\n" + "-" * 70)
        log("PHASE 2: INTELLIGENT SELL EVALUATION")
        log("-" * 70)
        
        sell_decisions = []
        
        # Process Tradier positions
        for pos in positions:
            symbol = pos.get('symbol') or (pos.get('symbol_description', '').split()[0] if pos.get('symbol_description') else '')
            if not symbol:
                continue
                
            quantity = float(pos.get('quantity', 0))
            if quantity <= 0:
                continue
            
            # Skip short positions (negative quantity) - we only sell what we own
            if quantity < 0:
                log(f"Skipping {symbol}: Short position (quantity: {quantity})")
                continue
            
            log(f"\nEvaluating {symbol} ({quantity} shares)...")
            
            # Build position dict for evaluation
            position_data = {
                'entry_price': float(pos.get('cost_basis', pos.get('average_price', 0))),
                'quantity': quantity,
                'entry_time': pos.get('date_acquired', datetime.datetime.now().isoformat()),
                'current_price': float(pos.get('last', 0)),
                'unrealized_pnl': float(pos.get('unrealized_pl', 0))
            }
            
            # Merge with Optimus's internal tracking if available
            if symbol in optimus_positions:
                optimus_pos = optimus_positions[symbol]
                position_data.update({
                    'entry_price': optimus_pos.get('entry_price', position_data['entry_price']),
                    'entry_time': optimus_pos.get('entry_time', position_data['entry_time']),
                    'unrealized_pnl': optimus_pos.get('unrealized_pnl', position_data['unrealized_pnl'])
                })
            
            # Get intelligent sell decision
            decision = optimus.intelligent_sell_decision(symbol, position_data, ralph_knowledge)
            
            log(f"  Current Price: ${position_data['current_price']:.2f}")
            log(f"  Entry Price: ${position_data['entry_price']:.2f}")
            log(f"  P&L: {decision.get('current_pnl_pct', 0):.1%} (${decision.get('current_pnl', 0):.2f})")
            log(f"  Holding: {decision.get('holding_days', 0)} days ({decision.get('holding_period_hours', 0):.1f} hours)")
            log(f"  Decision: {'SELL' if decision['should_sell'] else 'HOLD'}")
            log(f"  Confidence: {decision['confidence']:.0%}")
            log(f"  Urgency: {decision['urgency'].upper()}")
            log(f"  Reason: {decision['reason']}")
            
            if decision['should_sell']:
                log(f"  → Sell Quantity: {decision['sell_quantity']} shares")
                sell_decisions.append({
                    'symbol': symbol,
                    'decision': decision,
                    'position_data': position_data
                })
        
        # Process Optimus internal positions not in Tradier
        for symbol, pos in optimus_positions.items():
            # Skip if already processed
            if any(d['symbol'] == symbol for d in sell_decisions):
                continue
            
            quantity = pos.get('quantity', 0)
            if quantity <= 0:
                continue
            
            log(f"\nEvaluating {symbol} ({quantity} shares) from Optimus tracking...")
            
            position_data = {
                'entry_price': pos.get('entry_price', 0),
                'quantity': quantity,
                'entry_time': pos.get('entry_time', datetime.datetime.now().isoformat()),
                'current_price': pos.get('current_price', pos.get('entry_price', 0)),
                'unrealized_pnl': pos.get('unrealized_pnl', 0)
            }
            
            decision = optimus.intelligent_sell_decision(symbol, position_data, ralph_knowledge)
            
            log(f"  Decision: {'SELL' if decision['should_sell'] else 'HOLD'}")
            log(f"  Confidence: {decision['confidence']:.0%}")
            log(f"  Reason: {decision['reason']}")
            
            if decision['should_sell']:
                sell_decisions.append({
                    'symbol': symbol,
                    'decision': decision,
                    'position_data': position_data
                })
        
        # Execute sell decisions
        log("\n" + "-" * 70)
        log("PHASE 3: EXECUTING SELL DECISIONS")
        log("-" * 70)
        
        if not sell_decisions:
            log("No sell decisions - all positions should be held")
            return
        
        executed_trades = []
        for sell_decision in sell_decisions:
            symbol = sell_decision['symbol']
            decision = sell_decision['decision']
            position_data = sell_decision['position_data']
            
            sell_quantity = decision['sell_quantity']
            urgency = decision['urgency']
            confidence = decision['confidence']
            
            log(f"\n{'🚨' if urgency == 'critical' else '⚠️' if urgency == 'high' else '📊'} Executing SELL for {symbol}...")
            log(f"  Quantity: {sell_quantity} shares")
            log(f"  Confidence: {confidence:.0%}")
            log(f"  Urgency: {urgency.upper()}")
            log(f"  Reason: {decision['reason']}")
            
            try:
                execution_details = {
                    "symbol": symbol,
                    "side": "sell",
                    "quantity": sell_quantity,
                    "order_type": "market",
                    "asset_type": "equity",
                    "strategy_id": "intelligent_sell",
                    "strategy_name": "Intelligent Sell Decision",
                    "reason": decision['reason'],
                    "exit_confidence": confidence,
                    "exit_urgency": urgency,
                    "override_timing": True,  # Bypass entry timing check for exits
                    "force_execute": urgency in ["high", "critical"]  # Force execute for urgent exits
                }
                
                result = optimus.execute_trade(execution_details)
                
                if result and result.get('status') in ['filled', 'executed', 'submitted']:
                    log(f"  ✅ Sell order executed successfully!")
                    log(f"  Order ID: {result.get('order_id', 'N/A')}")
                    executed_trades.append({
                        'symbol': symbol,
                        'quantity': sell_quantity,
                        'order_id': result.get('order_id'),
                        'status': result.get('status')
                    })
                else:
                    log(f"  ❌ Sell order failed: {result.get('reason', 'Unknown') if result else 'No result'}", "ERROR")
            except Exception as e:
                log(f"  ❌ Error executing sell: {e}", "ERROR")
                import traceback
                traceback.print_exc()
        
        # Summary
        log("\n" + "=" * 70)
        log("SUMMARY")
        log("=" * 70)
        log(f"Positions evaluated: {len(positions) + len(optimus_positions)}")
        log(f"Sell decisions made: {len(sell_decisions)}")
        log(f"Trades executed: {len(executed_trades)}")
        
        if executed_trades:
            log("\nExecuted Trades:")
            for trade in executed_trades:
                log(f"  ✅ {trade['symbol']}: {trade['quantity']} shares - Order {trade['order_id']}")
        
        log("\n" + "=" * 70)
        log("INTELLIGENT SELL EVALUATION COMPLETE")
        log("=" * 70)
        
    except Exception as e:
        log(f"Error: {e}", "ERROR")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

