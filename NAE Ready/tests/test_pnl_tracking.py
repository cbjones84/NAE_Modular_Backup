#!/usr/bin/env python3
"""
Test script to verify P&L tracking in sandbox mode
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from agents.optimus import OptimusAgent
import json
import time

def print_status(optimus, title="Current Status"):
    """Print current trading status"""
    print(f"\n{'='*70}")
    print(f"{title}")
    print(f"{'='*70}")
    status = optimus.get_trading_status()
    
    print(f"Trading Mode: {status['trading_mode']}")
    print(f"Daily P&L: ${status['daily_pnl']:.2f}")
    print(f"  - Realized P&L: ${status['realized_pnl']:.2f}")
    print(f"  - Unrealized P&L: ${status['unrealized_pnl']:.2f}")
    print(f"NAV: ${status['nav']:.2f}")
    print(f"Total Value: ${status['total_value']:.2f}")
    print(f"Open Positions: {status['open_positions']}")
    
    if status['open_positions_detail']:
        print("\nOpen Positions Detail:")
        for symbol, pos in status['open_positions_detail'].items():
            print(f"  {symbol}:")
            print(f"    Quantity: {pos['quantity']}")
            print(f"    Entry Price: ${pos['entry_price']:.2f}")
            print(f"    Current Price: ${pos['current_price']:.2f}")
            print(f"    Unrealized P&L: ${pos['unrealized_pnl']:.2f}")

def test_pnl_tracking():
    """Test P&L tracking functionality"""
    print("="*70)
    print("P&L Tracking Test - Sandbox Mode")
    print("="*70)
    
    # Initialize Optimus in sandbox mode
    optimus = OptimusAgent(sandbox=True)
    
    print_status(optimus, "Initial Status")
    
    # Test 1: Open a long position
    print("\n" + "="*70)
    print("Test 1: Opening a long position (BUY)")
    print("="*70)
    
    buy_order = {
        'symbol': 'SPY',
        'side': 'buy',
        'quantity': 10,
        'price': 450.0  # Will use market price if Polygon client is available
    }
    
    result = optimus.execute_trade(buy_order)
    print(f"Trade Result: {json.dumps(result, indent=2)}")
    print_status(optimus, "After Opening Position")
    
    # Test 2: Add to existing position
    print("\n" + "="*70)
    print("Test 2: Adding to existing position (BUY more)")
    print("="*70)
    
    buy_more_order = {
        'symbol': 'SPY',
        'side': 'buy',
        'quantity': 5,
        'price': 452.0
    }
    
    result = optimus.execute_trade(buy_more_order)
    print(f"Trade Result: {json.dumps(result, indent=2)}")
    print_status(optimus, "After Adding to Position")
    
    # Test 3: Partial close - realize some P&L
    print("\n" + "="*70)
    print("Test 3: Partially closing position (SELL)")
    print("="*70)
    
    sell_order = {
        'symbol': 'SPY',
        'side': 'sell',
        'quantity': 8,
        'price': 455.0  # Selling at higher price - should realize profit
    }
    
    result = optimus.execute_trade(sell_order)
    print(f"Trade Result: {json.dumps(result, indent=2)}")
    print_status(optimus, "After Partial Close")
    
    # Test 4: Fully close remaining position
    print("\n" + "="*70)
    print("Test 4: Fully closing remaining position (SELL)")
    print("="*70)
    
    sell_remaining_order = {
        'symbol': 'SPY',
        'side': 'sell',
        'quantity': 7,  # Should close remaining 7 shares
        'price': 458.0
    }
    
    result = optimus.execute_trade(sell_remaining_order)
    print(f"Trade Result: {json.dumps(result, indent=2)}")
    print_status(optimus, "After Full Close")
    
    # Test 5: Open another position
    print("\n" + "="*70)
    print("Test 5: Opening new position in different symbol")
    print("="*70)
    
    buy_aapl_order = {
        'symbol': 'AAPL',
        'side': 'buy',
        'quantity': 20,
        'price': 150.0
    }
    
    result = optimus.execute_trade(buy_aapl_order)
    print(f"Trade Result: {json.dumps(result, indent=2)}")
    print_status(optimus, "After Opening New Position")
    
    # Test 6: Mark to market update
    print("\n" + "="*70)
    print("Test 6: Mark-to-Market Update (Simulating price change)")
    print("="*70)
    
    # Force mark to market
    optimus._mark_to_market()
    print_status(optimus, "After Mark-to-Market")
    
    print("\n" + "="*70)
    print("P&L Tracking Test Complete!")
    print("="*70)
    print("\nSummary:")
    print(f"  Total Realized P&L: ${optimus.realized_pnl:.2f}")
    print(f"  Current Unrealized P&L: ${optimus.unrealized_pnl:.2f}")
    print(f"  Daily P&L: ${optimus.daily_pnl:.2f}")
    print(f"  Open Positions: {optimus.open_positions}")

if __name__ == "__main__":
    test_pnl_tracking()

