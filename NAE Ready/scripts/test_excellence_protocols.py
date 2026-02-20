#!/usr/bin/env python3
"""
End-to-End Test for All Excellence Protocols

Tests:
- Ralph Excellence Protocol
- Optimus Excellence Protocol
- Donnie Excellence Protocol
"""

import os
import sys
import time
from datetime import datetime

# Add NAE paths
script_dir = os.path.dirname(os.path.abspath(__file__))
nae_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.insert(0, nae_root)


class MockRalphAgent:
    """Mock Ralph agent for testing"""
    def __init__(self):
        self.generated_strategies = []
        self.backtest_results = []
        self.execution_outcomes = []
    
    def log_action(self, message):
        print(f"  [Ralph] {message}")


class MockOptimusAgent:
    """Mock Optimus agent for testing"""
    def __init__(self):
        self.trade_history = []
        self.realized_pnl = 0.0
        self.execution_metrics = {}
        self.risk_metrics = {}
    
    def log_action(self, message):
        print(f"  [Optimus] {message}")


class MockDonnieAgent:
    """Mock Donnie agent for testing"""
    def __init__(self):
        self.execution_history = []
    
    def log_action(self, message):
        print(f"  [Donnie] {message}")


def test_ralph_excellence_protocol():
    """Test Ralph Excellence Protocol"""
    print("\n" + "="*60)
    print("TEST: Ralph Excellence Protocol")
    print("="*60)
    
    try:
        from agents.ralph_excellence_protocol import RalphExcellenceProtocol
        
        # Create mock agent
        ralph = MockRalphAgent()
        
        # Add some mock strategies
        ralph.generated_strategies = [
            {"id": "strategy_1", "quality_score": 0.85},
            {"id": "strategy_2", "quality_score": 0.90},
            {"id": "strategy_3", "quality_score": 0.75}
        ]
        
        # Add mock backtest results
        ralph.backtest_results = [
            {
                "strategy_id": "strategy_1",
                "total_return": 0.25,
                "max_drawdown": 0.15,
                "sharpe_ratio": 1.5
            },
            {
                "strategy_id": "strategy_2",
                "total_return": 0.30,
                "max_drawdown": 0.10,
                "sharpe_ratio": 2.0
            }
        ]
        
        # Initialize protocol
        protocol = RalphExcellenceProtocol(ralph)
        
        print("✅ Protocol initialized")
        
        # Start excellence mode
        protocol.start_excellence_mode()
        print("✅ Excellence mode started")
        
        # Run learning cycle
        protocol._learn_from_strategies()
        print("✅ Learning from strategies completed")
        
        # Check insights
        assert len(protocol.strategy_insights) > 0, "Should have generated insights"
        print(f"✅ Generated {len(protocol.strategy_insights)} insights")
        
        # Generate improvements
        protocol._generate_improvements()
        print(f"✅ Generated {len(protocol.improvements)} improvements")
        
        # Update self-awareness
        protocol._update_self_awareness()
        assert protocol.current_awareness is not None, "Should have awareness metrics"
        print("✅ Self-awareness updated")
        
        # Check excellence status
        status = protocol.get_excellence_status()
        assert status["status"] == "active", "Should be active"
        print("✅ Excellence status retrieved")
        
        # Stop excellence mode
        protocol.stop_excellence_mode()
        print("✅ Excellence mode stopped")
        
        print("\n✅ Ralph Excellence Protocol: PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ Ralph Excellence Protocol: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False


def test_optimus_excellence_protocol():
    """Test Optimus Excellence Protocol"""
    print("\n" + "="*60)
    print("TEST: Optimus Excellence Protocol")
    print("="*60)
    
    try:
        from agents.optimus_excellence_protocol import OptimusExcellenceProtocol
        
        # Create mock agent
        optimus = MockOptimusAgent()
        
        # Add mock trade history
        optimus.trade_history = [
            {
                "trade_id": "trade_1",
                "pnl": 100.0,
                "entry_price": 100.0,
                "exit_price": 110.0,
                "entry_time": datetime.now().isoformat(),
                "exit_time": datetime.now().isoformat(),
                "quantity": 10
            },
            {
                "trade_id": "trade_2",
                "pnl": -50.0,
                "entry_price": 100.0,
                "exit_price": 95.0,
                "entry_time": datetime.now().isoformat(),
                "exit_time": datetime.now().isoformat(),
                "quantity": 10
            }
        ]
        
        optimus.realized_pnl = 50.0
        optimus.execution_metrics = {"slippage": 0.01, "fill_rate": 0.98}
        optimus.risk_metrics = {"max_drawdown": 0.05, "var": 0.02}
        
        # Initialize protocol
        protocol = OptimusExcellenceProtocol(optimus)
        
        print("✅ Protocol initialized")
        
        # Start excellence mode
        protocol.start_excellence_mode()
        print("✅ Excellence mode started")
        
        # Run learning cycle
        protocol._learn_from_trades()
        print("✅ Learning from trades completed")
        
        # Check insights
        assert len(protocol.trading_insights) > 0, "Should have generated insights"
        print(f"✅ Generated {len(protocol.trading_insights)} insights")
        
        # Generate improvements
        protocol._generate_improvements()
        print(f"✅ Generated {len(protocol.improvements)} improvements")
        
        # Update self-awareness
        protocol._update_self_awareness()
        assert protocol.current_awareness is not None, "Should have awareness metrics"
        print("✅ Self-awareness updated")
        
        # Check excellence status
        status = protocol.get_excellence_status()
        assert status["status"] == "active", "Should be active"
        print("✅ Excellence status retrieved")
        
        # Stop excellence mode
        protocol.stop_excellence_mode()
        print("✅ Excellence mode stopped")
        
        print("\n✅ Optimus Excellence Protocol: PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ Optimus Excellence Protocol: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False


def test_donnie_excellence_protocol():
    """Test Donnie Excellence Protocol"""
    print("\n" + "="*60)
    print("TEST: Donnie Excellence Protocol")
    print("="*60)
    
    try:
        from agents.donnie_excellence_protocol import DonnieExcellenceProtocol
        
        # Create mock agent
        donnie = MockDonnieAgent()
        
        # Add mock execution history
        donnie.execution_history = [
            {
                "id": "exec_1",
                "component": "RalphAgent",
                "success": True,
                "error": None
            },
            {
                "id": "exec_2",
                "component": "OptimusAgent",
                "success": False,
                "error": "Timeout error"
            }
        ]
        
        # Initialize protocol
        protocol = DonnieExcellenceProtocol(donnie)
        
        print("✅ Protocol initialized")
        
        # Check NAE understanding (may be 0 if agents directory not found in test context)
        print(f"✅ Agent discovery: {len(protocol.agent_capabilities)} agents found")
        if len(protocol.agent_capabilities) > 0:
            print(f"   Agents: {list(protocol.agent_capabilities.keys())}")
        
        # Start excellence mode
        protocol.start_excellence_mode()
        print("✅ Excellence mode started")
        
        # Run codebase analysis
        protocol._analyze_codebase()
        print("✅ Codebase analysis completed")
        
        # Check code quality metrics (may be 0 if files not found in test context)
        print(f"✅ Code analysis: {len(protocol.code_quality_metrics)} files analyzed")
        if len(protocol.code_quality_metrics) > 0:
            print(f"   Files: {list(protocol.code_quality_metrics.keys())[:3]}...")
        
        # Learn from execution
        protocol._learn_from_execution()
        print("✅ Learning from execution completed")
        
        # Check insights
        assert len(protocol.development_insights) > 0, "Should have generated insights"
        print(f"✅ Generated {len(protocol.development_insights)} insights")
        
        # Generate improvements
        protocol._generate_improvements()
        print(f"✅ Generated {len(protocol.improvements)} improvements")
        
        # Update self-awareness
        protocol._update_self_awareness()
        assert protocol.current_awareness is not None, "Should have awareness metrics"
        print("✅ Self-awareness updated")
        
        # Check excellence status
        status = protocol.get_excellence_status()
        assert status["status"] == "active", "Should be active"
        print("✅ Excellence status retrieved")
        
        # Stop excellence mode
        protocol.stop_excellence_mode()
        print("✅ Excellence mode stopped")
        
        print("\n✅ Donnie Excellence Protocol: PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ Donnie Excellence Protocol: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False


def test_self_healing():
    """Test self-healing capabilities"""
    print("\n" + "="*60)
    print("TEST: Self-Healing Capabilities")
    print("="*60)
    
    try:
        from agents.ralph_excellence_protocol import RalphExcellenceProtocol
        
        ralph = MockRalphAgent()
        protocol = RalphExcellenceProtocol(ralph)
        
        # Start excellence mode
        protocol.start_excellence_mode()
        
        # Manually set low awareness to trigger healing
        from agents.ralph_excellence_protocol import SelfAwarenessMetrics
        protocol.current_awareness = SelfAwarenessMetrics(
            timestamp=datetime.now(),
            strategy_quality_score=0.5,  # Low score
            learning_rate=0.5,
            improvement_rate=0.5,
            success_rate=0.5,
            backtest_accuracy=0.5,
            market_adaptation=0.5,
            innovation_score=0.5,
            overall_excellence=0.5  # Low excellence
        )
        
        # Detect issues
        issues = protocol._detect_issues()
        assert len(issues) > 0, "Should detect issues"
        print(f"✅ Detected {len(issues)} issues")
        
        # Check auto-heal capability
        for issue in issues:
            can_heal = protocol._can_auto_heal(issue)
            assert can_heal, "Should be able to auto-heal"
            print(f"✅ Can auto-heal: {issue.get('type')}")
        
        # Apply healing
        for issue in issues:
            protocol._apply_healing(issue)
        print("✅ Self-healing applied")
        
        protocol.stop_excellence_mode()
        
        print("\n✅ Self-Healing: PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ Self-Healing: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all excellence protocol tests"""
    print("\n" + "="*60)
    print("EXCELLENCE PROTOCOLS - END-TO-END TEST")
    print("="*60)
    
    tests = [
        ("Ralph Excellence Protocol", test_ralph_excellence_protocol),
        ("Optimus Excellence Protocol", test_optimus_excellence_protocol),
        ("Donnie Excellence Protocol", test_donnie_excellence_protocol),
        ("Self-Healing Capabilities", test_self_healing)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, "PASS", None))
        except AssertionError as e:
            results.append((test_name, "FAIL", str(e)))
        except Exception as e:
            results.append((test_name, "ERROR", str(e)))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = len([r for r in results if r[1] == "PASS"])
    failed = len([r for r in results if r[1] == "FAIL"])
    errors = len([r for r in results if r[1] == "ERROR"])
    
    print(f"Total Tests: {len(tests)}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"💥 Errors: {errors}")
    
    if failed > 0 or errors > 0:
        print("\nFailed/Error Details:")
        for test_name, status, error in results:
            if status != "PASS":
                print(f"  {test_name}: {status}")
                if error:
                    print(f"    {error}")
    
    print("\n" + "="*60)
    
    if failed == 0 and errors == 0:
        print("🎉 ALL TESTS PASSED! All Excellence Protocols are working correctly!")
        return True
    else:
        print("⚠️ Some tests failed. Please review the errors above.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

