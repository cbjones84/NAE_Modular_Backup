#!/usr/bin/env python3
"""
Test All Strategies - Complete Workflow
Ralph → Donnie → Optimus → Alpaca

Generates strategies, validates them, and executes approved ones
"""

import sys
from pathlib import Path
import time
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_all_strategies():
    """Test complete strategy workflow"""
    
    print("=" * 60)
    print("Complete Strategy Testing Workflow")
    print("Ralph → Donnie → Optimus → Alpaca")
    print("=" * 60)
    print()
    
    # Step 1: Initialize all agents
    print("Step 1: Initializing Agents...")
    print("-" * 60)
    
    try:
        from agents.ralph import RalphAgent
        from agents.donnie import DonnieAgent
        from agents.optimus import OptimusAgent
        
        ralph = RalphAgent()
        donnie = DonnieAgent()
        optimus = OptimusAgent(sandbox=False)  # PAPER mode with Alpaca
        
        print("✅ Ralph initialized (Strategy Generation)")
        print("✅ Donnie initialized (Strategy Validation)")
        print("✅ Optimus initialized (Trade Execution)")
        print(f"   Trading Mode: {optimus.trading_mode.value}")
        print(f"   Broker: Alpaca Paper Trading")
        
        if not optimus.alpaca_client:
            print("❌ Alpaca not configured - cannot execute trades")
            return False
        
        print("✅ Alpaca configured and ready")
        print()
    except Exception as e:
        print(f"❌ Failed to initialize agents: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 2: Generate strategies with Ralph
    print("Step 2: Generating Strategies with Ralph...")
    print("-" * 60)
    
    try:
        print("Running Ralph strategy generation cycle...")
        strategies = ralph.generate_strategies()
        
        if not strategies:
            print("⚠️  No strategies generated")
            return False
        
        print(f"✅ Generated {len(strategies)} strategies")
        print()
        
        # Display generated strategies
        for i, strategy in enumerate(strategies, 1):
            print(f"Strategy {i}:")
            print(f"  Name: {strategy.get('name', 'Unknown')}")
            print(f"  Trust Score: {strategy.get('trust_score', 0):.2f}")
            print(f"  Backtest Score: {strategy.get('backtest_score', 0):.2f}")
            print(f"  Max Drawdown: {strategy.get('max_drawdown', 0):.3f}")
            print(f"  Performance: {strategy.get('perf', 0):.3f}")
            print(f"  Sharpe Ratio: {strategy.get('sharpe', 0):.2f}")
            print()
    except Exception as e:
        print(f"❌ Strategy generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 3: Validate strategies with Donnie
    print("Step 3: Validating Strategies with Donnie...")
    print("-" * 60)
    
    validated_strategies = []
    for i, strategy in enumerate(strategies, 1):
        try:
            is_valid = donnie.validate_strategy(strategy)
            if is_valid:
                validated_strategies.append(strategy)
                print(f"✅ Strategy {i}: '{strategy.get('name', 'Unknown')}' - VALIDATED")
                print(f"   Trust Score: {strategy.get('trust_score', 0):.2f}")
                print(f"   Meta Confidence: {strategy.get('meta_confidence', 0):.2%}")
            else:
                print(f"❌ Strategy {i}: '{strategy.get('name', 'Unknown')}' - REJECTED")
                print(f"   Reason: Low trust/backtest score or validation failed")
        except Exception as e:
            print(f"⚠️  Strategy {i} validation error: {e}")
    
    print()
    print(f"Validated: {len(validated_strategies)} of {len(strategies)} strategies")
    print()
    
    if not validated_strategies:
        print("⚠️  No strategies passed validation")
        return False
    
    # Step 4: Check account status
    print("Step 4: Checking Account Status...")
    print("-" * 60)
    
    try:
        account = optimus.alpaca_client.adapter.get_account()
        positions = optimus.alpaca_client.adapter.get_positions()
        
        print(f"Cash: ${account.get('cash', 0):,.2f}")
        print(f"Buying Power: ${account.get('buying_power', 0):,.2f}")
        print(f"Portfolio Value: ${account.get('portfolio_value', 0):,.2f}")
        print(f"Current Positions: {len(positions)}")
        
        if positions:
            print("\nCurrent Positions:")
            for pos in positions:
                print(f"  {pos['symbol']}: {pos['quantity']} @ ${pos['avg_price']:.2f}")
        
        print()
    except Exception as e:
        print(f"⚠️  Error getting account status: {e}")
        print()
    
    # Step 5: Execute validated strategies
    print("Step 5: Executing Validated Strategies...")
    print("-" * 60)
    print("⚠️  This will execute REAL paper trades")
    print()
    
    execution_results = []
    
    for i, strategy in enumerate(validated_strategies, 1):
        try:
            # Determine symbol from strategy
            symbol = strategy.get('symbol', 'SPY')  # Default to SPY if not specified
            name = strategy.get('name', f'Strategy {i}')
            
            # Extract symbol from strategy name if possible
            if not symbol or symbol == 'SPY':
                # Try to extract from name
                name_lower = name.lower()
                common_symbols = ['spy', 'qqq', 'aapl', 'msft', 'googl', 'amzn', 'tsla', 'nvda']
                for sym in common_symbols:
                    if sym in name_lower:
                        symbol = sym.upper()
                        break
            
            # Create execution details
            execution_details = {
                "symbol": symbol,
                "action": "buy",
                "quantity": 1,  # Small position for testing
                "order_type": "market",
                "time_in_force": "day",
                "strategy_name": name,
                "trust_score": strategy.get('trust_score', 0),
                "backtest_score": strategy.get('backtest_score', 0)
            }
            
            print(f"Executing Strategy {i}: {name}")
            print(f"  Symbol: {symbol}")
            print(f"  Action: BUY 1 share")
            print(f"  Trust Score: {strategy.get('trust_score', 0):.2f}")
            
            # Execute trade
            result = optimus.execute_trade(execution_details)
            
            execution_results.append({
                "strategy": name,
                "symbol": symbol,
                "result": result
            })
            
            if result.get('status') not in ['rejected', 'failed']:
                print(f"  ✅ Order ID: {result.get('order_id', 'N/A')}")
                print(f"  ✅ Status: {result.get('status', 'unknown')}")
                print(f"  ✅ Broker: {result.get('broker', 'unknown')}")
            else:
                print(f"  ❌ Status: {result.get('status', 'unknown')}")
                if result.get('error'):
                    print(f"  ❌ Error: {result.get('error')}")
            
            print()
            
            # Small delay between orders
            time.sleep(1)
            
        except Exception as e:
            print(f"  ❌ Execution failed: {e}")
            execution_results.append({
                "strategy": strategy.get('name', f'Strategy {i}'),
                "symbol": symbol,
                "result": {"error": str(e)}
            })
            print()
    
    # Step 6: Summary
    print("=" * 60)
    print("Execution Summary")
    print("=" * 60)
    print()
    
    successful = sum(1 for r in execution_results if r['result'].get('status') not in ['rejected', 'failed'])
    failed = len(execution_results) - successful
    
    print(f"Strategies Generated: {len(strategies)}")
    print(f"Strategies Validated: {len(validated_strategies)}")
    print(f"Strategies Executed: {len(execution_results)}")
    print(f"  ✅ Successful: {successful}")
    print(f"  ❌ Failed: {failed}")
    print()
    
    # Check final positions
    print("Final Account Status:")
    print("-" * 60)
    try:
        account = optimus.alpaca_client.adapter.get_account()
        positions = optimus.alpaca_client.adapter.get_positions()
        
        print(f"Cash: ${account.get('cash', 0):,.2f}")
        print(f"Buying Power: ${account.get('buying_power', 0):,.2f}")
        print(f"Portfolio Value: ${account.get('portfolio_value', 0):,.2f}")
        print(f"Positions: {len(positions)}")
        
        if positions:
            print("\nAll Positions:")
            total_value = 0
            total_pnl = 0
            for pos in positions:
                print(f"  {pos['symbol']}: {pos['quantity']} @ ${pos['avg_price']:.2f}")
                print(f"    Value: ${pos['market_value']:,.2f}")
                print(f"    P&L: ${pos['unrealized_pl']:,.2f}")
                total_value += pos['market_value']
                total_pnl += pos['unrealized_pl']
            
            print(f"\nTotal Portfolio Value: ${total_value:,.2f}")
            print(f"Total Unrealized P&L: ${total_pnl:,.2f}")
    except Exception as e:
        print(f"Error getting final status: {e}")
    
    # Save results
    print()
    print("Saving results...")
    try:
        results_file = f"logs/strategy_test_results_{int(time.time())}.json"
        with open(results_file, 'w') as f:
            json.dump({
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "strategies_generated": len(strategies),
                "strategies_validated": len(validated_strategies),
                "strategies_executed": len(execution_results),
                "successful": successful,
                "failed": failed,
                "execution_results": execution_results
            }, f, indent=2)
        print(f"✅ Results saved to: {results_file}")
    except Exception as e:
        print(f"⚠️  Could not save results: {e}")
    
    print()
    print("=" * 60)
    print("✅ Strategy Testing Complete!")
    print("=" * 60)
    print()
    print("Next Steps:")
    print("  1. Monitor positions in Alpaca dashboard: https://app.alpaca.markets")
    print("  2. Review execution results in logs/")
    print("  3. Analyze strategy performance")
    print("  4. Adjust strategy parameters as needed")
    print()
    
    return True


if __name__ == "__main__":
    success = test_all_strategies()
    sys.exit(0 if success else 1)

