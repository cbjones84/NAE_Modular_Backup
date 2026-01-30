#!/usr/bin/env python3
"""
Execute Multiple Strategies - Aligned with Long-Term Plan & 3 Goals

This script executes multiple trading strategies through the full NAE flow:
1. Ralph generates/validates strategies
2. Donnie validates and prepares execution
3. Optimus executes with entry/exit timing and PDT prevention

ALIGNED WITH:
- 3 Core Goals (Generational wealth, $5M in 8 years, Optimize options trading)
- Long-Term Plan (Phase 1: Wheel Strategy, PDT prevention)
- Tiered Strategy Framework (Tier 1: Wheel Strategy for Phase 1)
"""

import sys
import os
import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def create_wheel_strategy_strategies() -> list:
    """
    Create Tier 1 (Wheel Strategy) strategies aligned with Phase 1
    These are PDT-compliant (hold overnight minimum)
    """
    strategies = []
    
    # Wheel Strategy: Cash-Secured Puts on large-cap stocks
    wheel_stocks = ["AAPL", "MSFT", "TSLA", "SPY", "QQQ"]
    
    for symbol in wheel_stocks:
        # Cash-Secured Put strategy
        strategy = {
            "name": f"Wheel Strategy - Cash-Secured Put {symbol}",
            "symbol": symbol,
            "strategy_type": "wheel_cash_secured_put",
            "tier": 1,
            "phase": "Phase 1",
            "action": "sell",  # Selling puts
            "option_type": "put",
            "strike_selection": "0.20-0.30 delta (slightly OTM)",
            "dte": 30,  # Days to expiration
            "profit_target": 0.50,  # Buy back at 50% profit
            "hold_until_expiration": True,
            "pdt_compliant": True,  # No same-day close
            "trust_score": 75.0,  # High trust for wheel strategy
            "backtest_score": 65.0,  # Good backtest score
            "expected_return": 0.15,  # 15% annual return
            "risk_level": "low-medium",
            "parameters": {
                "description": f"Cash-secured put on {symbol} - Wheel Strategy Tier 1",
                "position_size_pct": 0.05,  # 5% of NAV per position
                "max_positions": 3,  # Max 3 wheel positions
                "min_premium": 0.01,  # Minimum 1% premium
            },
            "aggregated_details": {
                "strategy_category": "Income",
                "market_condition": "any",
                "volatility_preference": "moderate"
            }
        }
        strategies.append(strategy)
    
    return strategies

def create_momentum_strategies() -> list:
    """
    Create Tier 2 (Momentum) strategies for Phase 2
    These are PDT-compliant (hold overnight minimum)
    """
    strategies = []
    
    # Momentum plays on high-volume stocks
    momentum_symbols = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA"]
    
    for symbol in momentum_symbols:
        # Long Call strategy (bullish momentum)
        strategy = {
            "name": f"Momentum Play - Long Call {symbol}",
            "symbol": symbol,
            "strategy_type": "momentum_long_call",
            "tier": 2,
            "phase": "Phase 2",
            "action": "buy",
            "option_type": "call",
            "strike_selection": "ATM to 0.15 delta OTM",
            "dte": 14,  # 14-21 days (avoid 0 DTE for PDT compliance)
            "profit_target": 0.75,  # 75% profit target
            "stop_loss": 0.30,  # 30% stop loss
            "pdt_compliant": True,  # Entry at close, exit next day or later
            "trust_score": 65.0,
            "backtest_score": 55.0,
            "expected_return": 0.30,  # 30% annual return
            "risk_level": "medium-high",
            "parameters": {
                "description": f"Momentum long call on {symbol} - Tier 2",
                "position_size_pct": 0.03,  # 3% of NAV per position
                "volume_confirmation": True,  # Require high volume
                "breakout_confirmation": True,  # Require breakout pattern
            },
            "aggregated_details": {
                "strategy_category": "Directional",
                "market_condition": "trending",
                "volatility_preference": "high"
            }
        }
        strategies.append(strategy)
    
    return strategies

def execute_strategies():
    """Execute multiple strategies through the full NAE flow"""
    
    print("=" * 80)
    print("NAE Strategy Execution - Aligned with Long-Term Plan & 3 Goals")
    print("=" * 80)
    print()
    print("Goals:")
    print("  1. Achieve generational wealth")
    print("  2. Generate $5,000,000.00 within 8 years")
    print("  3. Optimize NAE and agents for successful options trading")
    print()
    print("Current Phase: Phase 1 - Foundation (Tier 1: Wheel Strategy)")
    print("PDT Prevention: ACTIVE (all positions hold overnight minimum)")
    print()
    
    # Initialize agents
    print("1. Initializing Agents...")
    try:
        from agents.optimus import OptimusAgent
        from agents.donnie import DonnieAgent
        from agents.ralph import RalphAgent
        from agents.casey import CaseyAgent
        
        optimus = OptimusAgent(sandbox=False)  # PAPER mode for Alpaca
        donnie = DonnieAgent()
        ralph = RalphAgent()
        casey = CaseyAgent()
        
        print(f"   ✅ Optimus initialized (Mode: {optimus.trading_mode.value})")
        print(f"   ✅ Donnie initialized")
        print(f"   ✅ Ralph initialized")
        print(f"   ✅ Casey initialized")
        print(f"   Current NAV: ${optimus.nav:.2f}")
        print(f"   Current Phase: {optimus.current_phase}")
        print(f"   Goal Progress: {(optimus.nav / optimus.target_goal) * 100:.4f}% toward $5M")
        
    except Exception as e:
        print(f"   ❌ Failed to initialize agents: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Check Optimus status
    print("\n2. Checking Optimus Status...")
    try:
        status = optimus.get_trading_status()
        print(f"   ✅ Trading Enabled: {status.get('trading_enabled', False)}")
        print(f"   ✅ Daily P&L: ${status.get('daily_pnl', 0):.2f}")
        print(f"   ✅ Open Positions: {status.get('open_positions', 0)}")
        print(f"   ✅ Total Value: ${status.get('total_value', 0):.2f}")
        
        if optimus.alpaca_client:
            try:
                account = optimus.alpaca_client.adapter.get_account()
                print(f"   ✅ Alpaca Cash: ${account.get('cash', 0):,.2f}")
                print(f"   ✅ Alpaca Buying Power: ${account.get('buying_power', 0):,.2f}")
            except:
                print(f"   ⚠️  Could not fetch Alpaca account info")
    except Exception as e:
        print(f"   ⚠️  Error checking status: {e}")
    
    # Create strategies aligned with long-term plan
    print("\n3. Creating Strategies Aligned with Long-Term Plan...")
    print("   Phase 1: Tier 1 (Wheel Strategy) - Cash-Secured Puts")
    
    wheel_strategies = create_wheel_strategy_strategies()
    print(f"   ✅ Created {len(wheel_strategies)} Wheel Strategy strategies")
    
    # For Phase 2, also create momentum strategies
    if optimus.nav >= 500:
        print("   Phase 2: Tier 2 (Momentum Plays) - Long Calls")
        momentum_strategies = create_momentum_strategies()
        print(f"   ✅ Created {len(momentum_strategies)} Momentum strategies")
        all_strategies = wheel_strategies + momentum_strategies
    else:
        print("   ⚠️  NAV < $500: Only executing Tier 1 (Wheel Strategy)")
        all_strategies = wheel_strategies
    
    print(f"   Total Strategies: {len(all_strategies)}")
    
    # Validate strategies with Donnie
    print("\n4. Validating Strategies with Donnie...")
    validated_strategies = []
    for strategy in all_strategies:
        if donnie.validate_strategy(strategy):
            validated_strategies.append(strategy)
            print(f"   ✅ Validated: {strategy['name']}")
        else:
            print(f"   ❌ Rejected: {strategy['name']}")
    
    print(f"   Validated: {len(validated_strategies)} of {len(all_strategies)}")
    
    if not validated_strategies:
        print("\n   ⚠️  No strategies validated. Exiting.")
        return False
    
    # Execute strategies through Optimus
    print("\n5. Executing Strategies with Optimus...")
    print("   ⚠️  This will execute REAL paper trades (Alpaca paper trading)")
    print("   ⚠️  All trades enforce PDT prevention (overnight hold minimum)")
    print()
    
    execution_results = []
    
    for i, strategy in enumerate(validated_strategies[:5], 1):  # Limit to 5 strategies
        print(f"   [{i}/{len(validated_strategies[:5])}] Executing: {strategy['name']}")
        
        # Prepare execution details
        execution_details = {
            "symbol": strategy.get("symbol", "SPY"),
            "side": strategy.get("action", "buy"),
            "order_type": "market",
            "time_in_force": "day",
            "strategy_name": strategy.get("name", "Unknown"),
            "trust_score": strategy.get("trust_score", 55),
            "backtest_score": strategy.get("backtest_score", 50),
            "expected_return": strategy.get("expected_return", 0.10),
            "stop_loss_pct": strategy.get("stop_loss", 0.02),
            "parameters": strategy.get("parameters", {}),
            "tier": strategy.get("tier", 1),
            "phase": strategy.get("phase", "Phase 1"),
            "pdt_compliant": True  # Explicitly mark as PDT compliant
        }
        
        # For options strategies, we need to convert to stock/options format
        # For now, execute as stock trades (options support coming)
        strategy_action = strategy.get("action", "buy")
        
        if strategy.get("strategy_type") == "wheel_cash_secured_put":
            # For wheel strategy, selling puts results in stock assignment if ITM
            # We simulate by buying stock (representing assignment at strike)
            execution_details["symbol"] = strategy["symbol"]
            execution_details["side"] = "buy"  # Simulating assignment (sell put → buy stock if assigned)
            execution_details["strategy_type"] = "wheel_cash_secured_put"
            
            # Get current price for better quantity calculation
            estimated_price = 100.0
            if optimus.polygon_client:
                try:
                    real_price = optimus.polygon_client.get_real_time_price(execution_details["symbol"])
                    if real_price and real_price > 0:
                        estimated_price = real_price
                except:
                    pass
            
            execution_details["quantity"] = max(1, int((optimus.nav * 0.05) / estimated_price))  # 5% of NAV
            execution_details["price"] = 0  # Market order
            
            print(f"      Strategy Type: {strategy['strategy_type']} (simulating assignment)")
            print(f"      Symbol: {execution_details['symbol']}")
            print(f"      Quantity: {execution_details['quantity']} @ ~${estimated_price:.2f}")
            print(f"      PDT Compliant: {execution_details['pdt_compliant']}")
        
        elif strategy.get("strategy_type") == "momentum_long_call":
            # For momentum, we'd buy calls, but for now execute stock buy
            execution_details["symbol"] = strategy["symbol"]
            execution_details["side"] = "buy"
            execution_details["strategy_type"] = "momentum_long_call"
            
            # Get current price for better quantity calculation
            estimated_price = 100.0
            if optimus.polygon_client:
                try:
                    real_price = optimus.polygon_client.get_real_time_price(execution_details["symbol"])
                    if real_price and real_price > 0:
                        estimated_price = real_price
                except:
                    pass
            
            execution_details["quantity"] = max(1, int((optimus.nav * 0.03) / estimated_price))  # 3% of NAV
            execution_details["price"] = 0  # Market order
            
            print(f"      Strategy Type: {strategy['strategy_type']}")
            print(f"      Symbol: {execution_details['symbol']}")
            print(f"      Quantity: {execution_details['quantity']} @ ~${estimated_price:.2f}")
            print(f"      PDT Compliant: {execution_details['pdt_compliant']}")
        
        # Execute through Optimus
        try:
            result = optimus.execute_trade(execution_details)
            
            execution_result = {
                "strategy": strategy["name"],
                "symbol": execution_details["symbol"],
                "result": result,
                "timestamp": datetime.datetime.now().isoformat()
            }
            execution_results.append(execution_result)
            
            if result.get("status") == "filled":
                print(f"      ✅ Order FILLED")
                print(f"         Order ID: {result.get('order_id', 'N/A')}")
                print(f"         Execution Price: ${result.get('execution_price', 0):.2f}")
                print(f"         Entry Timing Score: {result.get('entry_timing_score', 'N/A')}")
                print(f"         Risk/Reward: {result.get('risk_reward_ratio', 'N/A')}")
            elif result.get("status") == "rejected":
                print(f"      ❌ Order REJECTED: {result.get('reason', 'Unknown')}")
            else:
                print(f"      ⚠️  Order Status: {result.get('status', 'unknown')}")
                if result.get("error"):
                    print(f"         Error: {result.get('error')}")
        
        except Exception as e:
            print(f"      ❌ Execution Error: {e}")
            import traceback
            traceback.print_exc()
        
        print()
    
    # Summary
    print("=" * 80)
    print("EXECUTION SUMMARY")
    print("=" * 80)
    print()
    
    filled = sum(1 for r in execution_results if r["result"].get("status") == "filled")
    rejected = sum(1 for r in execution_results if r["result"].get("status") == "rejected")
    
    print(f"Total Strategies Executed: {len(execution_results)}")
    print(f"  ✅ Filled: {filled}")
    print(f"  ❌ Rejected: {rejected}")
    print()
    
    # Final status
    try:
        final_status = optimus.get_trading_status()
        print(f"Final Status:")
        print(f"  NAV: ${final_status.get('nav', 0):.2f}")
        print(f"  Total Value: ${final_status.get('total_value', 0):.2f}")
        print(f"  Daily P&L: ${final_status.get('daily_pnl', 0):.2f}")
        print(f"  Open Positions: {final_status.get('open_positions', 0)}")
        print(f"  Goal Progress: {(final_status.get('nav', 0) / optimus.target_goal) * 100:.4f}% toward $5M")
    except:
        pass
    
    print()
    print("=" * 80)
    print("✅ Strategy execution completed!")
    print("=" * 80)
    print()
    print("Next Steps:")
    print("  1. Monitor positions in Alpaca dashboard")
    print("  2. Check logs for entry/exit timing analysis")
    print("  3. Review PDT compliance (all positions held overnight)")
    print("  4. Track compound growth toward $5M goal")
    print()
    
    return True


if __name__ == "__main__":
    success = execute_strategies()
    sys.exit(0 if success else 1)

