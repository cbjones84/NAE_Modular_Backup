#!/usr/bin/env python3
"""
Batch Strategy Execution - Automated Multi-Strategy Execution

This script:
1. Generates strategies from Ralph (or uses predefined strategies)
2. Validates through Donnie
3. Executes multiple strategies through Optimus
4. Monitors execution and tracks progress toward $5M goal

ALIGNED WITH:
- 3 Core Goals
- Long-Term Plan (Phase-aware strategy execution)
- PDT Prevention (all positions hold overnight)
"""

import sys
import os
import datetime
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

def create_diverse_strategies(phase: str = "Phase 1") -> list:
    """
    Create diverse strategies aligned with current phase
    """
    strategies = []
    
    if phase == "Phase 1" or phase.startswith("Phase 1"):
        # Phase 1: Tier 1 - Wheel Strategy (Cash-Secured Puts)
        symbols = ["SPY", "AAPL", "MSFT", "QQQ", "TSLA"]
        
        for symbol in symbols:
            strategies.append({
                "name": f"Wheel Strategy - CSP {symbol}",
                "symbol": symbol,
                "strategy_type": "wheel_cash_secured_put",
                "tier": 1,
                "phase": "Phase 1",
                "action": "sell",
                "option_type": "put",
                "trust_score": 75.0,
                "backtest_score": 65.0,
                "expected_return": 0.15,
                "stop_loss_pct": 0.02,
                "parameters": {
                    "description": f"Cash-secured put on {symbol}",
                    "dte": 30,
                    "delta": 0.20,
                    "profit_target": 0.50
                },
                "aggregated_details": {
                    "strategy_category": "Income",
                    "pdt_compliant": True
                }
            })
    
    elif phase == "Phase 2" or phase.startswith("Phase 2"):
        # Phase 2: Tier 1 + Tier 2
        # Add momentum strategies
        momentum_symbols = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA"]
        
        for symbol in momentum_symbols:
            strategies.append({
                "name": f"Momentum Long Call {symbol}",
                "symbol": symbol,
                "strategy_type": "momentum_long_call",
                "tier": 2,
                "phase": "Phase 2",
                "action": "buy",
                "option_type": "call",
                "trust_score": 65.0,
                "backtest_score": 55.0,
                "expected_return": 0.30,
                "stop_loss_pct": 0.30,
                "parameters": {
                    "description": f"Momentum long call on {symbol}",
                    "dte": 14,
                    "delta": 0.15,
                    "profit_target": 0.75
                },
                "aggregated_details": {
                    "strategy_category": "Directional",
                    "pdt_compliant": True
                }
            })
    
    # Always include some wheel strategies
    wheel_symbols = ["SPY", "AAPL", "MSFT"]
    for symbol in wheel_symbols:
        strategies.append({
            "name": f"Wheel Strategy - CSP {symbol}",
            "symbol": symbol,
            "strategy_type": "wheel_cash_secured_put",
            "tier": 1,
            "phase": phase,
            "action": "sell",
            "option_type": "put",
            "trust_score": 75.0,
            "backtest_score": 65.0,
            "expected_return": 0.15,
            "stop_loss_pct": 0.02,
            "parameters": {
                "description": f"Cash-secured put on {symbol}",
                "dte": 30,
                "delta": 0.20,
                "profit_target": 0.50
            },
            "aggregated_details": {
                "strategy_category": "Income",
                "pdt_compliant": True
            }
        })
    
    return strategies

def batch_execute():
    """Execute multiple strategies in batch"""
    
    print("=" * 80)
    print("BATCH STRATEGY EXECUTION - Aligned with Long-Term Plan")
    print("=" * 80)
    print()
    
    # Initialize agents
    print("Initializing Agents...")
    try:
        from agents.optimus import OptimusAgent
        from agents.donnie import DonnieAgent
        from agents.ralph import RalphAgent
        
        optimus = OptimusAgent(sandbox=False)
        donnie = DonnieAgent()
        ralph = RalphAgent()
        
        print(f"✅ Optimus: {optimus.trading_mode.value} mode")
        print(f"✅ Donnie: Initialized")
        print(f"✅ Ralph: Initialized")
        print(f"   Current NAV: ${optimus.nav:.2f}")
        print(f"   Current Phase: {optimus.current_phase}")
        print()
        
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return False
    
    # Determine current phase
    current_phase = optimus.current_phase
    print(f"Current Phase: {current_phase}")
    print()
    
    # Create strategies aligned with phase
    print("Creating Strategies...")
    strategies = create_diverse_strategies(current_phase)
    print(f"✅ Created {len(strategies)} strategies")
    print()
    
    # Try to get additional strategies from Ralph
    print("Attempting to get strategies from Ralph...")
    try:
        ralph_strategies = ralph.generate_strategies()
        if ralph_strategies:
            print(f"✅ Ralph generated {len(ralph_strategies)} additional strategies")
            # Add Ralph's strategies if they meet criteria
            for rs in ralph_strategies:
                if rs.get("trust_score", 0) >= 55 and rs.get("backtest_score", 0) >= 50:
                    strategies.append(rs)
        else:
            print("ℹ️  Ralph generated no new strategies (using predefined)")
    except Exception as e:
        print(f"⚠️  Ralph strategy generation: {e}")
        print("   Using predefined strategies")
    
    print(f"Total Strategies Available: {len(strategies)}")
    print()
    
    # Validate strategies
    print("Validating Strategies...")
    validated = []
    for strategy in strategies:
        if donnie.validate_strategy(strategy):
            validated.append(strategy)
            print(f"  ✅ {strategy.get('name', 'Unknown')}")
        else:
            print(f"  ❌ {strategy.get('name', 'Unknown')} (rejected)")
    
    print(f"Validated: {len(validated)} of {len(strategies)}")
    print()
    
    if not validated:
        print("❌ No strategies validated. Exiting.")
        return False
    
    # Execute strategies (limit to avoid over-trading)
    max_executions = min(10, len(validated))
    print(f"Executing Top {max_executions} Strategies...")
    print("⚠️  All trades enforce PDT prevention (overnight hold minimum)")
    print()
    
    results = []
    for i, strategy in enumerate(validated[:max_executions], 1):
        print(f"[{i}/{max_executions}] {strategy.get('name', 'Unknown')}")
        
        # Prepare execution
        # For Wheel Strategy (selling puts), we simulate by buying stock (representing assignment)
        # In production, this would be actual options orders
        strategy_action = strategy.get("action", "buy")
        
        # For Wheel Strategy "sell" actions (selling puts), convert to "buy" (simulating assignment)
        # This represents the cash-secured put being assigned and receiving stock
        if strategy.get("strategy_type") == "wheel_cash_secured_put" and strategy_action == "sell":
            # Wheel Strategy: Selling puts results in stock assignment if ITM
            # We simulate by buying stock (representing assignment at strike)
            actual_action = "buy"  # Simulating assignment
            print(f"   Wheel Strategy: Simulating put assignment (selling put → buying stock)")
        else:
            actual_action = strategy_action
        
        exec_details = {
            "symbol": strategy.get("symbol", "SPY"),
            "side": actual_action,  # Use converted action
            "order_type": "market",
            "time_in_force": "day",
            "strategy_name": strategy.get("name", "Unknown"),
            "trust_score": strategy.get("trust_score", 55),
            "backtest_score": strategy.get("backtest_score", 50),
            "expected_return": strategy.get("expected_return", 0.10),
            "stop_loss_pct": strategy.get("stop_loss_pct", 0.02),
            "parameters": strategy.get("parameters", {}),
            "tier": strategy.get("tier", 1),
            "phase": strategy.get("phase", "Phase 1"),
            "pdt_compliant": True,
            "strategy_type": strategy.get("strategy_type", "unknown")
        }
        
        # Calculate position size based on tier
        if strategy.get("tier") == 1:
            position_pct = 0.05  # 5% for Tier 1
        elif strategy.get("tier") == 2:
            position_pct = 0.03  # 3% for Tier 2
        else:
            position_pct = 0.02  # 2% default
        
        # Get current price for better quantity calculation
        estimated_price = 100.0  # Default
        if optimus.polygon_client:
            try:
                real_price = optimus.polygon_client.get_real_time_price(exec_details["symbol"])
                if real_price and real_price > 0:
                    estimated_price = real_price
            except:
                pass
        
        exec_details["quantity"] = max(1, int((optimus.nav * position_pct) / estimated_price))
        exec_details["price"] = 0  # Market order
        
        # Execute
        try:
            result = optimus.execute_trade(exec_details)
            results.append({
                "strategy": strategy.get("name"),
                "status": result.get("status"),
                "order_id": result.get("order_id"),
                "reason": result.get("reason")
            })
            
            if result.get("status") == "filled":
                print(f"   ✅ FILLED - Order ID: {result.get('order_id', 'N/A')}")
            else:
                print(f"   ❌ {result.get('status', 'unknown').upper()}: {result.get('reason', 'N/A')}")
        
        except Exception as e:
            print(f"   ❌ Error: {e}")
            results.append({
                "strategy": strategy.get("name"),
                "status": "error",
                "error": str(e)
            })
        
        # Small delay between executions
        time.sleep(1)
        print()
    
    # Summary
    print("=" * 80)
    print("BATCH EXECUTION SUMMARY")
    print("=" * 80)
    print()
    
    filled = sum(1 for r in results if r.get("status") == "filled")
    rejected = sum(1 for r in results if r.get("status") == "rejected")
    
    print(f"Total Executed: {len(results)}")
    print(f"  ✅ Filled: {filled}")
    print(f"  ❌ Rejected: {rejected}")
    print()
    
    # Final status
    try:
        status = optimus.get_trading_status()
        print(f"Final Status:")
        print(f"  NAV: ${status.get('nav', 0):.2f}")
        print(f"  Total Value: ${status.get('total_value', 0):.2f}")
        print(f"  Daily P&L: ${status.get('daily_pnl', 0):.2f}")
        print(f"  Open Positions: {status.get('open_positions', 0)}")
        print(f"  Goal Progress: {(status.get('nav', 0) / optimus.target_goal) * 100:.4f}% toward $5M")
    except:
        pass
    
    print()
    print("=" * 80)
    
    return True


if __name__ == "__main__":
    success = batch_execute()
    sys.exit(0 if success else 1)

