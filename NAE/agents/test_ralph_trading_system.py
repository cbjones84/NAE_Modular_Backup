#!/usr/bin/env python3
"""
Test script for Ralph Trading System
Simulates the complete trading workflow with detailed logging
"""

import time
import logging
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch
import sys
import os

# Add parent paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../NAE Ready'))

from ralph_github_continuous import (
    TradierClient,
    TradingSafetyManager,
    TradingState,
    fractional_kelly,
    TradierError
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MockTradierClient:
    """Mock Tradier client for testing"""
    
    def __init__(self):
        self.account_id = "TEST_ACCOUNT"
        self.equity = 10000.0
        self.buying_power = 8000.0
        self.cash = 5000.0
        self.error_count = 0
        
    def get_balances(self):
        """Return mock balances"""
        return {
            "equity": self.equity,
            "buying_power": self.buying_power,
            "cash": self.cash,
            "settled_cash": self.cash
        }
    
    def get_positions(self):
        """Return mock positions"""
        return []
    
    def submit_order(self, symbol, side, quantity, **kwargs):
        """Mock order submission"""
        logger.info(f"📤 Mock order submitted: {side} {quantity} {symbol}")
        return {
            "order": {
                "id": f"ORDER_{int(time.time())}",
                "status": "filled"
            }
        }


def test_fractional_kelly():
    """Test fractional Kelly position sizing"""
    print("\n" + "="*70)
    print("TEST 1: Fractional Kelly Position Sizing")
    print("="*70)
    
    # Test case 1: High win rate, good risk/reward
    win_rate = 0.65  # 65% win rate
    avg_win = 200.0   # Average win $200
    avg_loss = 100.0  # Average loss $100
    
    pct = fractional_kelly(win_rate, avg_win, avg_loss, fraction=0.90, max_pct=0.25)
    print(f"\n📊 Test Parameters:")
    print(f"   Win Rate: {win_rate*100:.1f}%")
    print(f"   Avg Win: ${avg_win:.2f}")
    print(f"   Avg Loss: ${avg_loss:.2f}")
    print(f"   Kelly Fraction: 90%")
    print(f"   Max Position: 25%")
    
    print(f"\n✅ Result:")
    print(f"   Position Size: {pct*100:.2f}% of equity")
    
    equity = 10000.0
    price = 150.0
    notional = equity * pct
    quantity = int(notional / price)
    
    print(f"\n💰 Position Details (Equity: ${equity:,.2f}, Price: ${price:.2f}):")
    print(f"   Notional Value: ${notional:,.2f}")
    print(f"   Quantity: {quantity} shares")
    
    return pct, notional, quantity


def test_pre_trade_checks():
    """Test pre-trade safety checks"""
    print("\n" + "="*70)
    print("TEST 2: Pre-Trade Safety Checks")
    print("="*70)
    
    # Create mock client
    mock_client = MockTradierClient()
    
    # Create safety manager
    safety_manager = TradingSafetyManager(mock_client)
    
    # Test 1: Normal check
    print("\n🔍 Test 2.1: Normal Pre-Trade Check")
    allowed, reason = safety_manager.pre_trade_check("AAPL", "buy")
    print(f"   Result: {'✅ ALLOWED' if allowed else '❌ BLOCKED'}")
    print(f"   Reason: {reason}")
    
    # Test 2: Low buying power
    print("\n🔍 Test 2.2: Low Buying Power Check")
    mock_client.buying_power = 20.0  # Below $25 minimum
    allowed, reason = safety_manager.pre_trade_check("AAPL", "buy")
    print(f"   Result: {'✅ ALLOWED' if allowed else '❌ BLOCKED'}")
    print(f"   Reason: {reason}")
    mock_client.buying_power = 8000.0  # Reset
    
    # Test 3: Daily loss limit
    print("\n🔍 Test 2.3: Daily Loss Limit Check")
    safety_manager.state.initial_equity = 10000.0
    safety_manager.state.current_equity = 6000.0  # 40% loss
    allowed, reason = safety_manager.pre_trade_check("AAPL", "buy")
    print(f"   Result: {'✅ ALLOWED' if allowed else '❌ BLOCKED'}")
    print(f"   Reason: {reason}")
    safety_manager.state.current_equity = 10000.0  # Reset
    
    return safety_manager


def test_position_sizing():
    """Test position size calculation"""
    print("\n" + "="*70)
    print("TEST 3: Position Size Calculation")
    print("="*70)
    
    mock_client = MockTradierClient()
    safety_manager = TradingSafetyManager(mock_client)
    
    equity = 10000.0
    win_rate = 0.60
    avg_win = 300.0
    avg_loss = 150.0
    price = 100.0
    
    quantity, notional = safety_manager.calculate_position_size(
        equity, win_rate, avg_win, avg_loss, price
    )
    
    print(f"\n📊 Input Parameters:")
    print(f"   Equity: ${equity:,.2f}")
    print(f"   Win Rate: {win_rate*100:.1f}%")
    print(f"   Avg Win: ${avg_win:.2f}")
    print(f"   Avg Loss: ${avg_loss:.2f}")
    print(f"   Stock Price: ${price:.2f}")
    
    print(f"\n✅ Calculated Position:")
    print(f"   Quantity: {quantity} shares")
    print(f"   Notional Value: ${notional:,.2f}")
    print(f"   Position %: {(notional/equity)*100:.2f}% of equity")
    
    return quantity, notional


def test_circuit_breaker():
    """Test circuit breaker functionality"""
    print("\n" + "="*70)
    print("TEST 4: Circuit Breaker System")
    print("="*70)
    
    mock_client = MockTradierClient()
    safety_manager = TradingSafetyManager(mock_client)
    
    # Test 1: Normal operation
    print("\n🔍 Test 4.1: Normal Operation (No Errors)")
    allowed, reason = safety_manager._check_circuit_breaker()
    print(f"   Result: {'✅ OK' if allowed else '❌ TRIGGERED'}")
    print(f"   Consecutive Errors: {safety_manager.state.consecutive_errors}")
    
    # Test 2: Simulate errors
    print("\n🔍 Test 4.2: Simulating Errors")
    for i in range(8):
        safety_manager.record_error(TradierError(f"Test error {i+1}"))
        allowed, reason = safety_manager._check_circuit_breaker()
        print(f"   Error {i+1}: {'✅ OK' if allowed else '❌ CIRCUIT BREAKER TRIGGERED'}")
        if not allowed:
            print(f"   Reason: {reason}")
            break
    
    # Test 3: Reset on success
    print("\n🔍 Test 4.3: Reset on Success")
    safety_manager.record_success()
    allowed, reason = safety_manager._check_circuit_breaker()
    print(f"   Result: {'✅ OK' if allowed else '❌ TRIGGERED'}")
    print(f"   Consecutive Errors: {safety_manager.state.consecutive_errors}")
    
    # Test 4: Drawdown check
    print("\n🔍 Test 4.4: Drawdown Check")
    safety_manager.state.initial_equity = 10000.0
    safety_manager.state.current_equity = 4000.0  # 60% drawdown
    allowed, reason = safety_manager._check_circuit_breaker()
    print(f"   Drawdown: {(1 - safety_manager.state.current_equity/safety_manager.state.initial_equity)*100:.1f}%")
    print(f"   Result: {'✅ OK' if allowed else '❌ CIRCUIT BREAKER TRIGGERED'}")
    if not allowed:
        print(f"   Reason: {reason}")


def test_time_filters():
    """Test time-of-day filters"""
    print("\n" + "="*70)
    print("TEST 5: Time-of-Day Filters")
    print("="*70)
    
    mock_client = MockTradierClient()
    safety_manager = TradingSafetyManager(mock_client)
    
    # Test different times
    test_times = [
        (9, 25, "Before market open"),
        (9, 35, "First 10 minutes (filtered)"),
        (10, 0, "Normal trading hours"),
        (15, 45, "Last 20 minutes (filtered)"),
        (16, 5, "After market close"),
    ]
    
    for hour, minute, description in test_times:
        with patch('ralph_github_continuous.datetime') as mock_dt:
            mock_now = datetime(2024, 1, 15, hour, minute)  # Monday
            mock_dt.now.return_value = mock_now
            mock_dt.combine = datetime.combine
            mock_dt.timedelta = timedelta
            
            # Need to patch inside the method
            from ralph_github_continuous import datetime as dt_module
            original_now = dt_module.now
            
            def mock_now_func():
                return mock_now
            
            dt_module.now = mock_now_func
            
            try:
                allowed, reason = safety_manager._is_market_hours()
                print(f"\n🕐 {description} ({hour:02d}:{minute:02d}):")
                print(f"   Result: {'✅ ALLOWED' if allowed else '❌ BLOCKED'}")
                print(f"   Reason: {reason}")
            finally:
                dt_module.now = original_now


def test_full_cycle():
    """Test a complete trading cycle"""
    print("\n" + "="*70)
    print("TEST 6: Complete Trading Cycle Simulation")
    print("="*70)
    
    mock_client = MockTradierClient()
    safety_manager = TradingSafetyManager(mock_client)
    
    # Simulate a trading cycle
    print("\n📈 Simulating Trading Cycle:")
    print("   Step 1: Check account balances")
    balances = mock_client.get_balances()
    print(f"      Equity: ${balances['equity']:,.2f}")
    print(f"      Buying Power: ${balances['buying_power']:,.2f}")
    
    print("\n   Step 2: Pre-trade checks")
    symbol = "TSLA"
    side = "buy"
    allowed, reason = safety_manager.pre_trade_check(symbol, side)
    print(f"      Result: {'✅ PASSED' if allowed else '❌ FAILED'}")
    print(f"      Reason: {reason}")
    
    if allowed:
        print("\n   Step 3: Calculate position size")
        equity = balances['equity']
        win_rate = 0.65
        avg_win = 250.0
        avg_loss = 120.0
        price = 200.0
        
        quantity, notional = safety_manager.calculate_position_size(
            equity, win_rate, avg_win, avg_loss, price
        )
        print(f"      Quantity: {quantity} shares")
        print(f"      Notional: ${notional:,.2f}")
        
        print("\n   Step 4: Submit order")
        order_result = mock_client.submit_order(symbol, side, quantity)
        print(f"      Order ID: {order_result['order']['id']}")
        print(f"      Status: {order_result['order']['status']}")
        
        print("\n   Step 5: Record success")
        safety_manager.record_success()
        print(f"      Errors reset: {safety_manager.state.consecutive_errors == 0}")


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*70)
    print("NAE TRADING SYSTEM - COMPREHENSIVE TEST SUITE")
    print("="*70)
    print(f"Test Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Run all tests
        test_fractional_kelly()
        test_pre_trade_checks()
        test_position_sizing()
        test_circuit_breaker()
        test_time_filters()
        test_full_cycle()
        
        print("\n" + "="*70)
        print("✅ ALL TESTS COMPLETED SUCCESSFULLY")
        print("="*70)
        print(f"Test Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except Exception as e:
        print("\n" + "="*70)
        print(f"❌ TEST FAILED: {e}")
        print("="*70)
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    run_all_tests()

