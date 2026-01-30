#!/usr/bin/env python3
"""
Test Optimus Accuracy and Force Immediate Trade
"""
import os
import sys
import json
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from agents.optimus import OptimusAgent

def test_optimus_accuracy():
    """Test Optimus accuracy metrics"""
    print("="*70)
    print("OPTIMUS ACCURACY TEST")
    print("="*70)
    print("")
    
    optimus = OptimusAgent(sandbox=False)
    
    # Get trading status
    status = optimus.get_trading_status()
    print("Trading Status:")
    print(f"  Mode: {status['trading_mode']}")
    print(f"  Enabled: {status['trading_enabled']}")
    print(f"  NAV: ${status.get('nav', 0):,.2f}")
    print("")
    
    # Get excellence metrics
    if hasattr(optimus, 'excellence_protocol'):
        protocol = optimus.excellence_protocol
        print("Accuracy Metrics:")
        if hasattr(protocol, 'metrics'):
            metrics = protocol.metrics
            print(f"  Trading Accuracy: {metrics.get('trading_accuracy', 0):.2%}")
            print(f"  Execution Quality: {metrics.get('execution_quality', 0):.2%}")
            print(f"  Risk Management: {metrics.get('risk_management', 0):.2%}")
            print(f"  Timing Accuracy: {metrics.get('timing_accuracy', 0):.2%}")
            print(f"  Position Sizing: {metrics.get('position_sizing', 0):.2%}")
            print(f"  Profitability: {metrics.get('profitability', 0):.2%}")
            print(f"  Overall Excellence: {metrics.get('overall_excellence', 0):.2%}")
        elif hasattr(protocol, 'current_metrics'):
            metrics = protocol.current_metrics
            print(f"  Trading Accuracy: {metrics.get('trading_accuracy', 0):.2%}")
            print(f"  Execution Quality: {metrics.get('execution_quality', 0):.2%}")
            print(f"  Risk Management: {metrics.get('risk_management', 0):.2%}")
            print(f"  Timing Accuracy: {metrics.get('timing_accuracy', 0):.2%}")
            print(f"  Position Sizing: {metrics.get('position_sizing', 0):.2%}")
            print(f"  Profitability: {metrics.get('profitability', 0):.2%}")
            print(f"  Overall Excellence: {metrics.get('overall_excellence', 0):.2%}")
        else:
            print("  Metrics available via excellence_protocol")
        print("")
    
    # Get account balance
    try:
        balance_info = optimus.get_available_balance()
        print("Account Balance:")
        print(f"  Cash Available: ${balance_info.get('cash', 0):,.2f}")
        print(f"  Buying Power: ${balance_info.get('buying_power', 0):,.2f}")
        print(f"  Available for Trading: ${balance_info.get('available_for_trading', 0):,.2f}")
        print("")
    except Exception as e:
        print(f"  Error getting balance: {e}")
        print("")
    
    return optimus


def force_immediate_trade(optimus):
    """Force an immediate trade"""
    print("="*70)
    print("FORCING IMMEDIATE TRADE")
    print("="*70)
    print("")
    
    # Create a simple equity trade execution
    # Use a smaller symbol that fits within available cash
    execution_details = {
        "action": "execute_trade",
        "asset_type": "equity",
        "symbol": "SOXL",  # Using SOXL since account already has 1 share
        "side": "buy",
        "quantity": 1,  # 1 share
        "order_type": "market",
        "time_in_force": "day",
        "trust_score": 70,
        "backtest_score": 65,
        "expected_return": 0.01,
        "strategy_name": "Immediate Test Trade",
        "force_execute": True,  # Force execution
        "bypass_health_check": True  # Bypass self-healing health check
    }
    
    print("Trade Details:")
    print(f"  Symbol: {execution_details['symbol']}")
    print(f"  Side: {execution_details['side']}")
    print(f"  Quantity: {execution_details['quantity']}")
    print(f"  Order Type: {execution_details['order_type']}")
    print("")
    
    print("Executing trade...")
    try:
        result = optimus.execute_trade(execution_details)
        
        print("")
        print("="*70)
        print("TRADE RESULT")
        print("="*70)
        print(json.dumps(result, indent=2, default=str))
        print("")
        
        if result.get('status') in ['submitted', 'executed', 'filled']:
            print("✅ TRADE EXECUTED SUCCESSFULLY!")
            return True
        else:
            print(f"⚠️  Trade Status: {result.get('status')}")
            print(f"   Reason: {result.get('reason', 'N/A')}")
            return False
            
    except Exception as e:
        print(f"❌ Error executing trade: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Set environment
    os.environ['TRADIER_API_KEY'] = '27Ymk28vtbgqY1LFYxhzaEmIuwJb'
    os.environ['TRADIER_ACCOUNT_ID'] = '6YB66744'
    os.environ['TRADIER_SANDBOX'] = 'false'
    
    # Test accuracy
    optimus = test_optimus_accuracy()
    
    # Force trade
    success = force_immediate_trade(optimus)
    
    sys.exit(0 if success else 1)

