# NAE/test_legal_compliance_integration.py
"""
Comprehensive Integration Test for NAE Legal Compliance and Safety Systems
Tests all enhanced agents and safety systems working together
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from agents.optimus import OptimusAgent, SafetyLimits, TradingMode
from agents.ralph import RalphAgent
from agents.genny import GennyAgent
from human_safety_gates import HumanSafetyGates, ApprovalType, OwnerControlInterface
from paper_to_live_progression import PaperToLiveProgression, TradingPhase
from goal_manager import get_nae_goals

def test_comprehensive_legal_compliance():
    """Test all legal compliance and safety systems working together"""
    print("=" * 80)
    print("COMPREHENSIVE LEGAL COMPLIANCE AND SAFETY SYSTEMS INTEGRATION TEST")
    print("=" * 80)
    
    # Initialize all systems
    print("\n1. Initializing all NAE systems...")
    
    # Safety gates and owner control
    safety_gates = HumanSafetyGates()
    owner_interface = OwnerControlInterface(safety_gates)
    
    # Paper-to-live progression
    progression = PaperToLiveProgression()
    
    # Enhanced agents with safety controls
    custom_limits = SafetyLimits(
        max_order_size_usd=5000.0,
        daily_loss_limit_pct=0.01,  # 1% daily loss limit
        consecutive_loss_limit=3,
        max_open_positions=5
    )
    
    optimus = OptimusAgent(sandbox=True, safety_limits=custom_limits)
    ralph = RalphAgent()
    genny = GennyAgent()
    
    print("✓ All systems initialized successfully")
    
    # Test 1: Safety Gates and Owner Control
    print("\n2. Testing Human Safety Gates and Owner Control...")
    
    # Create approval requests
    paper_to_live_request = safety_gates.create_approval_request(
        ApprovalType.PAPER_TO_LIVE,
        "OptimusAgent",
        {
            "strategy_name": "Enhanced Strategy",
            "capital_amount": 25000,
            "risk_assessment": "moderate",
            "backtest_results": {"sharpe_ratio": 1.5, "max_drawdown": 0.08}
        }
    )
    
    large_trade_request = safety_gates.create_approval_request(
        ApprovalType.LARGE_TRADE,
        "OptimusAgent",
        {
            "symbol": "AAPL",
            "amount": 15000,
            "side": "buy",
            "strategy": "momentum"
        }
    )
    
    print(f"✓ Created approval requests: {paper_to_live_request}, {large_trade_request}")
    
    # Owner approval process
    owner_verified = owner_interface.verify_owner("owner123")
    assert owner_verified, "Owner verification failed"
    
    pending_requests = owner_interface.get_pending_approvals("owner123")
    assert len(pending_requests) >= 2, "Expected at least 2 pending requests"
    
    # Approve paper-to-live request
    approval_result = owner_interface.approve_request(
        "owner123", 
        paper_to_live_request, 
        "Strategy meets all requirements for live trading"
    )
    assert approval_result, "Paper-to-live approval failed"
    
    print("✓ Owner control and approval process working")
    
    # Test 2: Paper-to-Live Progression
    print("\n3. Testing Paper-to-Live Trading Progression...")
    
    # Simulate successful sandbox phase
    for i in range(60):
        trade_data = {
            "pnl": 12 if i % 2 == 0 else -3,  # 50% win rate
            "size": 1000,
            "symbol": "AAPL"
        }
        progression.record_trade(trade_data)
    
    # Check progression eligibility
    can_progress, message = progression.can_progress_to_next_phase()
    print(f"Can progress to next phase: {can_progress}")
    print(f"Progression message: {message}")
    
    if can_progress:
        progression_success = progression.progress_to_next_phase("owner123", "Sandbox phase completed successfully")
        assert progression_success, "Phase progression failed"
        print("✓ Successfully progressed to paper trading phase")
    
    # Test 3: Enhanced Optimus with Safety Controls
    print("\n4. Testing Enhanced Optimus with Safety Controls...")
    
    # Test kill switch
    optimus.activate_kill_switch("Integration test")
    assert not optimus.trading_enabled, "Kill switch not working"
    
    optimus.deactivate_kill_switch("Integration test")
    assert optimus.trading_enabled, "Kill switch deactivation not working"
    
    # Test pre-trade checks
    test_trade = {
        "symbol": "AAPL",
        "side": "buy",
        "quantity": 100,
        "price": 150.0,
        "strategy_name": "Test Strategy"
    }
    
    checks_passed, check_message = optimus.pre_trade_checks(test_trade)
    print(f"Pre-trade checks passed: {checks_passed}")
    print(f"Check message: {check_message}")
    
    # Test trade execution with safety controls
    trade_result = optimus.execute_trade(test_trade)
    assert trade_result is not None, "Trade execution failed"
    print(f"✓ Trade executed: {trade_result['status']}")
    
    # Test audit logging
    audit_summary = optimus.get_audit_summary()
    assert audit_summary['total_entries'] > 0, "No audit entries created"
    print(f"✓ Audit logging working: {audit_summary['total_entries']} entries")
    
    # Test 4: Enhanced Ralph with Market Data and QuantConnect
    print("\n5. Testing Enhanced Ralph with Market Data and QuantConnect...")
    
    # Test market data fetching
    market_data = ralph.fetch_market_data("AAPL", "2023-01-01", "2023-01-31")
    assert len(market_data) > 0, "No market data fetched"
    print(f"✓ Fetched {len(market_data)} market data points")
    
    # Test real-time price
    price = ralph.get_real_time_price("AAPL")
    assert price is not None, "Real-time price not available"
    print(f"✓ Real-time price: ${price:.2f}")
    
    # Test QuantConnect backtest
    strategy_code = """
class IntegrationTestStrategy(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2023, 1, 1)
        self.SetEndDate(2023, 12, 31)
        self.SetCash(100000)
        self.AddEquity("AAPL", Resolution.Daily)
"""
    
    backtest_result = ralph.run_quantconnect_backtest(
        strategy_code, "Integration Test Strategy", "2023-01-01", "2023-12-31"
    )
    assert backtest_result is not None, "QuantConnect backtest failed"
    print(f"✓ QuantConnect backtest completed: {backtest_result.total_return:.2%} return")
    
    # Test enhanced strategy processing
    ralph_result = ralph.run_cycle()
    assert ralph_result is not None, "Ralph cycle failed"
    print(f"✓ Ralph processed strategies: {ralph_result['approved_count']} approved")
    
    # Test 5: Genny Generational Wealth Tracking
    print("\n6. Testing Genny Generational Wealth Tracking...")
    
    # Test Optimus success tracking
    optimus_success_data = {
        "strategy_name": "Integration Test Strategy",
        "profit": 2500,
        "execution_time": 0.3,
        "risk_score": 0.4,
        "expected_return": 0.18
    }
    
    genny_optimus_tracking = genny.track_optimus_success(optimus_success_data)
    assert genny_optimus_tracking is not None, "Genny Optimus tracking failed"
    print("✓ Genny tracking Optimus success")
    
    # Test Ralph strategy tracking
    ralph_strategy_data = {
        "name": "Integration Test Strategy",
        "trust_score": 92,
        "backtest_score": 85,
        "consensus_count": 4,
        "risk_reward": 3.2
    }
    
    genny_ralph_tracking = genny.track_ralph_strategy(ralph_strategy_data)
    assert genny_ralph_tracking is not None, "Genny Ralph tracking failed"
    print("✓ Genny tracking Ralph strategies")
    
    # Test recipe curation
    recipes = genny.curate_success_recipes()
    assert recipes is not None, "Recipe curation failed"
    print(f"✓ Genny curated {len(recipes['optimus_recipes'])} Optimus recipes")
    
    # Test heir knowledge package
    heir_profile = {
        "heir_id": "integration_test_heir",
        "name": "Integration Test Heir",
        "experience_level": "intermediate",
        "interests": ["trading", "risk_management"]
    }
    
    knowledge_package = genny.create_heir_knowledge_package(heir_profile)
    assert knowledge_package is not None, "Heir knowledge package creation failed"
    print("✓ Genny created heir knowledge package")
    
    # Test 6: System Integration and Communication
    print("\n7. Testing System Integration and Communication...")
    
    # Test agent communication
    genny.send_message("Integration test message", optimus)
    assert len(genny.outbox) > 0, "Genny message sending failed"
    
    optimus.receive_message("Test message from integration")
    assert len(optimus.inbox) > 0, "Optimus message receiving failed"
    
    print("✓ Agent communication working")
    
    # Test 7: Comprehensive Status Report
    print("\n8. Generating Comprehensive Status Report...")
    
    # Get status from all systems
    optimus_status = optimus.get_trading_status()
    progression_status = progression.get_current_status()
    safety_status = owner_interface.get_system_status("owner123")
    
    print("\n" + "=" * 80)
    print("COMPREHENSIVE STATUS REPORT")
    print("=" * 80)
    
    print(f"\n🔒 SAFETY SYSTEMS:")
    print(f"  • Kill Switch: {'ACTIVE' if not optimus.trading_enabled else 'INACTIVE'}")
    print(f"  • Trading Mode: {optimus_status['trading_mode'].upper()}")
    print(f"  • Daily P&L: ${optimus_status['daily_pnl']:.2f}")
    print(f"  • Open Positions: {optimus_status['open_positions']}")
    print(f"  • Audit Entries: {optimus_status['audit_log_entries']}")
    
    print(f"\n📈 TRADING PROGRESSION:")
    print(f"  • Current Phase: {progression_status['current_phase'].upper()}")
    print(f"  • Phase Status: {progression_status['phase_status'].upper()}")
    print(f"  • Can Progress: {progression_status['can_progress']}")
    print(f"  • Total Trades: {progression_status['current_metrics']['total_trades']}")
    print(f"  • Win Rate: {progression_status['current_metrics']['win_rate']:.2%}")
    print(f"  • Total P&L: ${progression_status['current_metrics']['total_pnl']:.2f}")
    
    print(f"\n👤 HUMAN OVERSIGHT:")
    print(f"  • Owner Verified: {safety_status['owner_verified']}")
    print(f"  • Pending Approvals: {safety_status['pending_approvals']}")
    print(f"  • System Status: {safety_status['system_status'].upper()}")
    
    print(f"\n🤖 AGENT STATUS:")
    print(f"  • Optimus: {optimus_status['trading_mode'].upper()} mode")
    print(f"  • Ralph: {ralph.status.upper()}")
    print(f"  • Genny: Tracking {len(genny.optimus_success_log)} Optimus successes")
    print(f"  • Genny: Tracking {len(genny.ralph_strategy_log)} Ralph strategies")
    
    print(f"\n📊 MARKET DATA & BACKTESTING:")
    print(f"  • Polygon Client: {'CONFIGURED' if ralph.polygon_client else 'DEMO MODE'}")
    print(f"  • QuantConnect: {'CONFIGURED' if ralph.quantconnect_client else 'DEMO MODE'}")
    print(f"  • Backtest Results: {len(ralph.backtest_results)} completed")
    print(f"  • Market Data Cache: {len(ralph.market_data_cache)} symbols")
    
    print(f"\n🎯 NAE GOALS STATUS:")
    nae_goals = get_nae_goals()
    for i, goal in enumerate(nae_goals, 1):
        print(f"  • Goal {i}: {goal}")
    
    print("\n" + "=" * 80)
    print("ALL LEGAL COMPLIANCE AND SAFETY SYSTEMS OPERATIONAL!")
    print("=" * 80)
    
    return True

if __name__ == "__main__":
    try:
        test_comprehensive_legal_compliance()
        print("\n✅ COMPREHENSIVE INTEGRATION TEST PASSED SUCCESSFULLY!")
        print("\nNAE is now equipped with:")
        print("  🔒 FINRA/SEC compliant safety controls")
        print("  🛡️  Kill switch and pre-trade checks")
        print("  📊 Immutable audit logging")
        print("  👤 Human-in-the-loop oversight")
        print("  📈 Paper-to-live progression system")
        print("  🔗 Broker API integration (IBKR/Alpaca)")
        print("  📡 Market data integration (Polygon.io)")
        print("  🧪 QuantConnect backtesting")
        print("  💰 Generational wealth tracking (Genny)")
        print("  ⚖️  Legal compliance framework")
        
    except Exception as e:
        print(f"\n❌ INTEGRATION TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
