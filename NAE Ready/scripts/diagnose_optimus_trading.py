#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diagnose why Optimus isn't making trades

This script checks:
1. Trading enabled status
2. Market status
3. Opportunity detection
4. Score thresholds
5. Execution details creation
6. Trade validation failures
"""

import os
import sys
from datetime import datetime

# Fix encoding for Windows
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Add NAE to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from agents.optimus import OptimusAgent
from scripts.trigger_optimus_option_trade import (
    get_market_status,
    find_best_option_opportunity,
    create_option_execution_details
)

def diagnose_optimus():
    print("="*80)
    print("OPTIMUS TRADING DIAGNOSTIC")
    print("="*80)
    print()
    
    # 1. Initialize Optimus
    print("1. Initializing Optimus...")
    try:
        optimus = OptimusAgent(sandbox=False)  # LIVE MODE
        print(f"   ✅ Optimus initialized (sandbox=True)")
        print(f"   Trading enabled: {optimus.trading_enabled}")
        
        if not optimus.trading_enabled:
            print("   ⚠️  WARNING: Trading is DISABLED")
            if hasattr(optimus, 'deactivate_kill_switch'):
                print("   Attempting to enable trading...")
                optimus.deactivate_kill_switch("Diagnostic check")
                print(f"   Trading enabled after deactivate_kill_switch: {optimus.trading_enabled}")
        else:
            print("   ✅ Trading is ENABLED")
    except Exception as e:
        print(f"   ❌ Failed to initialize Optimus: {e}")
        import traceback
        traceback.print_exc()
        return
    print()
    
    # 2. Check market status
    print("2. Checking market status...")
    try:
        market_status = get_market_status()
        print(f"   Market open: {market_status.get('is_open')}")
        print(f"   Can trade: {market_status.get('can_trade')}")
        print(f"   Current time: {market_status.get('current_time')}")
        print(f"   Phase: {market_status.get('phase')} - {market_status.get('phase_description')}")
        print(f"   Minutes until close: {market_status.get('minutes_until_close')}")
        
        if not market_status.get('is_open'):
            print("   ⚠️  Market is CLOSED - no trades will execute")
    except Exception as e:
        print(f"   ❌ Error checking market status: {e}")
    print()
    
    # 3. Check Tradier health
    print("3. Checking Tradier health...")
    try:
        if hasattr(optimus, 'self_healing_engine') and optimus.self_healing_engine:
            engine = optimus.self_healing_engine
            health = engine.self_healing_engine.get_health_score()
            print(f"   Health score: {health:.2f}")
            
            if health < 0.7:
                print(f"   ⚠️  Health score below 0.7 threshold - trades may be blocked")
            
            status = engine.self_healing_engine.get_status()
            issues = status.get('issues', [])
            if issues:
                print(f"   Issues found: {len(issues)}")
                for issue in issues[:3]:
                    print(f"     - {issue.get('severity', 'unknown')}: {issue.get('message', 'No message')}")
        else:
            print("   ⚠️  Self-healing engine not available")
    except Exception as e:
        print(f"   ⚠️  Error checking health: {e}")
    print()
    
    # 4. Test opportunity detection
    print("4. Testing opportunity detection...")
    try:
        symbols = ["SPY", "QQQ", "AAPL"]
        print(f"   Scanning {len(symbols)} symbols...")
        
        opportunity = find_best_option_opportunity(optimus, symbols)
        
        if opportunity:
            score = opportunity.get("score", 0)
            symbol = opportunity.get("symbol", "UNKNOWN")
            print(f"   ✅ Opportunity found: {symbol} (score: {score:.2f})")
            print(f"   IV Edge: {opportunity.get('iv_edge', 'N/A')}")
            print(f"   Spread %: {opportunity.get('spread_pct', 'N/A')}")
            
            if score <= 30:
                print(f"   ⚠️  Score ({score:.2f}) is below threshold (30) - trade won't execute")
            
            # Check if we can create execution details
            phase = market_status.get("phase", "general")
            print(f"   Creating execution details for phase: {phase}...")
            execution_details = create_option_execution_details(opportunity, phase=phase)
            
            if execution_details:
                print(f"   ✅ Execution details created")
                print(f"   Strategy: {execution_details.get('strategy_name')}")
                print(f"   Symbol: {execution_details.get('symbol')}")
                print(f"   Side: {execution_details.get('side')}")
                print(f"   Quantity: {execution_details.get('quantity')}")
                print(f"   Strike: {execution_details.get('strike')}")
                print(f"   Trust score: {execution_details.get('trust_score')}")
                
                # Check for price
                price = execution_details.get("price", 0)
                if not price or price == 0:
                    print(f"   ⚠️  WARNING: No 'price' in execution details!")
                    print(f"   This will cause execute_trade to skip validation")
                
                # Try to execute (dry run)
                if score > 30 and execution_details:
                    print(f"\n   Testing trade execution (dry run)...")
                    # Don't actually execute, just check what would happen
                    print(f"   Would execute trade: {execution_details.get('symbol')}")
            else:
                print(f"   ❌ Failed to create execution details")
        else:
            print(f"   ⚠️  No opportunities found")
            print(f"   This could mean:")
            print(f"     - No option signals generated")
            print(f"     - IV edge not favorable")
            print(f"     - Poor liquidity (spread too wide)")
            print(f"     - Market conditions not suitable")
    except Exception as e:
        print(f"   ❌ Error detecting opportunities: {e}")
        import traceback
        traceback.print_exc()
    print()
    
    # 5. Check risk system
    print("5. Checking risk system...")
    try:
        if hasattr(optimus, 'risk_system') and optimus.risk_system:
            can_trade, reason = optimus.risk_system.can_execute_trade("Optimus")
            print(f"   Risk system allows trading: {can_trade}")
            if not can_trade:
                print(f"   ⚠️  Blocked by risk system: {reason}")
        else:
            print("   ⚠️  Risk system not available")
    except Exception as e:
        print(f"   ⚠️  Error checking risk system: {e}")
    print()
    
    # 6. Check NAV and balance
    print("6. Checking account status...")
    try:
        nav = optimus.nav
        print(f"   NAV: ${nav:,.2f}")
        
        balance_info = optimus.get_available_balance()
        print(f"   Available cash: ${balance_info.get('cash', 0):,.2f}")
        print(f"   Available for trading: ${balance_info.get('available_for_trading', 0):,.2f}")
        print(f"   Buying power: ${balance_info.get('buying_power', 0):,.2f}")
        
        if balance_info.get('available_for_trading', 0) < 100:
            print(f"   ⚠️  Low buying power - may prevent trades")
    except Exception as e:
        print(f"   ⚠️  Error checking balance: {e}")
    print()
    
    # 7. Summary
    print("="*80)
    print("DIAGNOSTIC SUMMARY")
    print("="*80)
    print()
    print("Potential issues to check:")
    print("1. Is trading enabled?")
    print("2. Is market open?")
    print("3. Are opportunities being found (score > 30)?")
    print("4. Is 'price' included in execution details?")
    print("5. Are risk checks passing?")
    print("6. Is there sufficient buying power?")
    print("7. Is Tradier health score > 0.7?")
    print()
    print("="*80)


if __name__ == "__main__":
    diagnose_optimus()

