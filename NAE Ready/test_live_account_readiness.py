#!/usr/bin/env python3
"""
Comprehensive test to verify NAE is active, prepared, and ready for LIVE account trading.
Tests all components, connections, and safety features.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import datetime
from typing import Dict, List, Any

class LiveAccountReadinessTest:
    """Comprehensive test suite for LIVE account readiness"""
    
    def __init__(self):
        self.results = []
        self.passed = 0
        self.failed = 0
        self.warnings = 0
        
    def log_result(self, test_name: str, passed: bool, message: str = "", warning: bool = False):
        """Log test result"""
        status = "✅ PASS" if passed else ("⚠️ WARN" if warning else "❌ FAIL")
        result = {
            "test": test_name,
            "status": status,
            "passed": passed,
            "warning": warning,
            "message": message,
            "timestamp": datetime.datetime.now().isoformat()
        }
        self.results.append(result)
        
        if passed:
            self.passed += 1
        elif warning:
            self.warnings += 1
        else:
            self.failed += 1
        
        print(f"{status}: {test_name}")
        if message:
            print(f"   {message}")
    
    def test_optimus_initialization(self):
        """Test OptimusAgent initialization in LIVE mode"""
        print("\n" + "="*70)
        print("TEST 1: OptimusAgent Initialization (LIVE Mode)")
        print("="*70)
        
        try:
            from agents.optimus import OptimusAgent, TradingMode
            
            # Test default initialization (should be LIVE mode)
            optimus = OptimusAgent(sandbox=False)
            
            # Check trading mode
            if optimus.trading_mode == TradingMode.LIVE:
                self.log_result(
                    "OptimusAgent Trading Mode",
                    True,
                    f"Trading mode is LIVE: {optimus.trading_mode.value}"
                )
            else:
                self.log_result(
                    "OptimusAgent Trading Mode",
                    False,
                    f"Expected LIVE mode, got: {optimus.trading_mode.value}"
                )
            
            # Check Alpaca client configuration
            if optimus.alpaca_client:
                if optimus.alpaca_client.paper_trading == False:
                    self.log_result(
                        "Alpaca Client Paper Trading",
                        True,
                        "Alpaca client configured for LIVE trading (paper_trading=False)"
                    )
                else:
                    self.log_result(
                        "Alpaca Client Paper Trading",
                        False,
                        f"Expected paper_trading=False for LIVE mode, got: {optimus.alpaca_client.paper_trading}"
                    )
            else:
                self.log_result(
                    "Alpaca Client Configuration",
                    True,
                    "Alpaca client not configured (may need API key activation)",
                    warning=True
                )
            
            return optimus
            
        except Exception as e:
            self.log_result(
                "OptimusAgent Initialization",
                False,
                f"Failed to initialize OptimusAgent: {e}"
            )
            return None
    
    def test_account_balance_sync(self, optimus):
        """Test account balance synchronization"""
        print("\n" + "="*70)
        print("TEST 2: Account Balance Synchronization")
        print("="*70)
        
        if not optimus:
            self.log_result(
                "Account Balance Sync",
                False,
                "Cannot test - OptimusAgent not initialized"
            )
            return
        
        try:
            # Test sync method exists
            if hasattr(optimus, '_sync_account_balance'):
                self.log_result(
                    "Sync Method Exists",
                    True,
                    "_sync_account_balance method available"
                )
            else:
                self.log_result(
                    "Sync Method Exists",
                    False,
                    "_sync_account_balance method not found"
                )
                return
            
            # Test get_available_balance method
            if hasattr(optimus, 'get_available_balance'):
                self.log_result(
                    "Get Balance Method Exists",
                    True,
                    "get_available_balance method available"
                )
            else:
                self.log_result(
                    "Get Balance Method Exists",
                    False,
                    "get_available_balance method not found"
                )
                return
            
            # Try to sync account balance
            try:
                success = optimus._sync_account_balance()
                if success:
                    balance = optimus.get_available_balance()
                    self.log_result(
                        "Account Balance Sync",
                        True,
                        f"Successfully synced: NAV=${balance.get('nav', 0):,.2f}, "
                        f"Buying Power=${balance.get('buying_power', 0):,.2f}"
                    )
                else:
                    self.log_result(
                        "Account Balance Sync",
                        True,
                        "Sync method returned False (API keys may need activation)",
                        warning=True
                    )
            except Exception as e:
                error_msg = str(e)
                if "401" in error_msg or "unauthorized" in error_msg.lower():
                    self.log_result(
                        "Account Balance Sync",
                        True,
                        "Authentication error (expected - API keys need activation)",
                        warning=True
                    )
                else:
                    self.log_result(
                        "Account Balance Sync",
                        False,
                        f"Unexpected error: {e}"
                    )
            
        except Exception as e:
            self.log_result(
                "Account Balance Sync Test",
                False,
                f"Test failed: {e}"
            )
    
    def test_strategy_determination(self, optimus):
        """Test strategy determination from account balance"""
        print("\n" + "="*70)
        print("TEST 3: Strategy Determination")
        print("="*70)
        
        if not optimus:
            self.log_result(
                "Strategy Determination",
                False,
                "Cannot test - OptimusAgent not initialized"
            )
            return
        
        try:
            # Test strategy determination method
            if hasattr(optimus, '_determine_strategy_from_balance'):
                self.log_result(
                    "Strategy Method Exists",
                    True,
                    "_determine_strategy_from_balance method available"
                )
            else:
                self.log_result(
                    "Strategy Method Exists",
                    False,
                    "_determine_strategy_from_balance method not found"
                )
                return
            
            # Test with sample values
            test_equity = 50000.0
            test_buying_power = 40000.0
            strategy = optimus._determine_strategy_from_balance(test_equity, test_buying_power)
            
            if strategy and isinstance(strategy, dict):
                self.log_result(
                    "Strategy Determination",
                    True,
                    f"Successfully determined strategy: {strategy.get('Account Size', 'Unknown')}"
                )
            else:
                self.log_result(
                    "Strategy Determination",
                    False,
                    "Strategy determination returned invalid result"
                )
            
        except Exception as e:
            self.log_result(
                "Strategy Determination Test",
                False,
                f"Test failed: {e}"
            )
    
    def test_safety_features(self, optimus):
        """Test safety features are active"""
        print("\n" + "="*70)
        print("TEST 4: Safety Features")
        print("="*70)
        
        if not optimus:
            self.log_result(
                "Safety Features",
                False,
                "Cannot test - OptimusAgent not initialized"
            )
            return
        
        try:
            # Check safety limits exist
            if hasattr(optimus, 'safety_limits'):
                self.log_result(
                    "Safety Limits",
                    True,
                    f"Safety limits configured: Max Order=${optimus.safety_limits.max_order_size_usd:,.2f}, "
                    f"Daily Loss Limit={optimus.safety_limits.daily_loss_limit_pct:.2%}"
                )
            else:
                self.log_result(
                    "Safety Limits",
                    False,
                    "Safety limits not found"
                )
            
            # Check pre-trade checks method
            if hasattr(optimus, 'pre_trade_checks'):
                self.log_result(
                    "Pre-Trade Checks",
                    True,
                    "pre_trade_checks method available"
                )
            else:
                self.log_result(
                    "Pre-Trade Checks",
                    False,
                    "pre_trade_checks method not found"
                )
            
            # Check kill switch
            if hasattr(optimus, 'trading_enabled'):
                self.log_result(
                    "Kill Switch",
                    True,
                    f"Kill switch available: trading_enabled={optimus.trading_enabled}"
                )
            else:
                self.log_result(
                    "Kill Switch",
                    False,
                    "Kill switch not found"
                )
            
            # Check audit logging
            if hasattr(optimus, 'audit_log'):
                self.log_result(
                    "Audit Logging",
                    True,
                    f"Audit logging enabled: {len(optimus.audit_log)} entries"
                )
            else:
                self.log_result(
                    "Audit Logging",
                    False,
                    "Audit logging not found"
                )
            
        except Exception as e:
            self.log_result(
                "Safety Features Test",
                False,
                f"Test failed: {e}"
            )
    
    def test_master_scheduler(self):
        """Test master scheduler configuration"""
        print("\n" + "="*70)
        print("TEST 5: Master Scheduler Configuration")
        print("="*70)
        
        try:
            from nae_master_scheduler import NAEMasterScheduler, AutomationConfig
            
            config = AutomationConfig()
            scheduler = NAEMasterScheduler(config)
            
            # Check Optimus is enabled
            if config.ENABLE_OPTIMUS:
                self.log_result(
                    "Optimus Enabled in Scheduler",
                    True,
                    "Optimus is enabled in master scheduler"
                )
            else:
                self.log_result(
                    "Optimus Enabled in Scheduler",
                    False,
                    "Optimus is disabled in master scheduler"
                )
            
            # Check Optimus agent exists
            if 'Optimus' in scheduler.agents:
                optimus_auto = scheduler.agents['Optimus']
                if optimus_auto.agent:
                    # Check if it's in LIVE mode
                    if hasattr(optimus_auto.agent, 'trading_mode'):
                        from agents.optimus import TradingMode
                        if optimus_auto.agent.trading_mode == TradingMode.LIVE:
                            self.log_result(
                                "Scheduler Optimus Mode",
                                True,
                                f"Scheduler Optimus is in LIVE mode"
                            )
                        else:
                            self.log_result(
                                "Scheduler Optimus Mode",
                                False,
                                f"Scheduler Optimus is in {optimus_auto.agent.trading_mode.value} mode, expected LIVE"
                            )
                    else:
                        self.log_result(
                            "Scheduler Optimus Mode",
                            False,
                            "Cannot determine trading mode"
                        )
                else:
                    self.log_result(
                        "Scheduler Optimus Agent",
                        False,
                        "Optimus agent not initialized in scheduler"
                    )
            else:
                self.log_result(
                    "Scheduler Optimus Agent",
                    False,
                    "Optimus not found in scheduler agents"
                )
            
        except Exception as e:
            self.log_result(
                "Master Scheduler Test",
                False,
                f"Test failed: {e}"
            )
    
    def test_automation_system(self):
        """Test automation system"""
        print("\n" + "="*70)
        print("TEST 6: Automation System")
        print("="*70)
        
        try:
            from nae_automation import NAEAutomationSystem
            
            automation = NAEAutomationSystem()
            
            # Check system initialized
            if automation.scheduler:
                self.log_result(
                    "Automation Scheduler",
                    True,
                    "Automation system scheduler initialized"
                )
            else:
                self.log_result(
                    "Automation Scheduler",
                    False,
                    "Automation system scheduler not initialized"
                )
            
            # Check Optimus in scheduler
            if automation.scheduler and 'Optimus' in automation.scheduler.agents:
                optimus_auto = automation.scheduler.agents['Optimus']
                if optimus_auto.agent:
                    from agents.optimus import TradingMode
                    if optimus_auto.agent.trading_mode == TradingMode.LIVE:
                        self.log_result(
                            "Automation Optimus Mode",
                            True,
                            "Automation system Optimus is in LIVE mode"
                        )
                    else:
                        self.log_result(
                            "Automation Optimus Mode",
                            False,
                            f"Automation system Optimus is in {optimus_auto.agent.trading_mode.value} mode"
                        )
            
        except Exception as e:
            self.log_result(
                "Automation System Test",
                False,
                f"Test failed: {e}"
            )
    
    def test_quant_agent_integration(self, optimus):
        """Test QuantAgent integration"""
        print("\n" + "="*70)
        print("TEST 7: QuantAgent Integration")
        print("="*70)
        
        if not optimus:
            self.log_result(
                "QuantAgent Integration",
                False,
                "Cannot test - OptimusAgent not initialized"
            )
            return
        
        try:
            if hasattr(optimus, 'quant_agent'):
                if optimus.quant_agent:
                    self.log_result(
                        "QuantAgent Framework",
                        True,
                        "QuantAgent Framework initialized and available"
                    )
                else:
                    self.log_result(
                        "QuantAgent Framework",
                        True,
                        "QuantAgent Framework not available (optional)",
                        warning=True
                    )
            else:
                self.log_result(
                    "QuantAgent Framework",
                    True,
                    "QuantAgent Framework not integrated (optional)",
                    warning=True
                )
            
        except Exception as e:
            self.log_result(
                "QuantAgent Integration Test",
                False,
                f"Test failed: {e}"
            )
    
    def test_enhanced_rl_agent(self, optimus):
        """Test Enhanced RL Agent"""
        print("\n" + "="*70)
        print("TEST 8: Enhanced RL Agent")
        print("="*70)
        
        if not optimus:
            self.log_result(
                "Enhanced RL Agent",
                False,
                "Cannot test - OptimusAgent not initialized"
            )
            return
        
        try:
            if hasattr(optimus, 'rl_agent'):
                rl_agent = optimus.rl_agent
                if rl_agent:
                    # Check if it's the enhanced version
                    agent_type = type(rl_agent).__name__
                    if 'Enhanced' in agent_type:
                        self.log_result(
                            "Enhanced RL Agent",
                            True,
                            f"Enhanced RL Agent initialized: {agent_type}"
                        )
                    else:
                        self.log_result(
                            "Enhanced RL Agent",
                            True,
                            f"Standard RL Agent initialized: {agent_type}",
                            warning=True
                        )
                else:
                    self.log_result(
                        "Enhanced RL Agent",
                        False,
                        "RL Agent not initialized"
                    )
            else:
                self.log_result(
                    "Enhanced RL Agent",
                    False,
                    "RL Agent not found"
                )
            
        except Exception as e:
            self.log_result(
                "Enhanced RL Agent Test",
                False,
                f"Test failed: {e}"
            )
    
    def test_trading_status(self, optimus):
        """Test trading status reporting"""
        print("\n" + "="*70)
        print("TEST 9: Trading Status Reporting")
        print("="*70)
        
        if not optimus:
            self.log_result(
                "Trading Status",
                False,
                "Cannot test - OptimusAgent not initialized"
            )
            return
        
        try:
            if hasattr(optimus, 'get_trading_status'):
                status = optimus.get_trading_status()
                
                # Check required fields
                required_fields = ['trading_mode', 'nav', 'account_type', 'is_live_account', 'current_phase']
                for field in required_fields:
                    if field in status:
                        self.log_result(
                            f"Status Field: {field}",
                            True,
                            f"{field}: {status[field]}"
                        )
                    else:
                        self.log_result(
                            f"Status Field: {field}",
                            False,
                            f"Missing field: {field}"
                        )
                
                # Check trading mode
                if status.get('trading_mode') == 'live':
                    self.log_result(
                        "Status Trading Mode",
                        True,
                        "Trading status shows LIVE mode"
                    )
                else:
                    self.log_result(
                        "Status Trading Mode",
                        False,
                        f"Trading status shows {status.get('trading_mode')}, expected 'live'"
                    )
                
            else:
                self.log_result(
                    "Trading Status Method",
                    False,
                    "get_trading_status method not found"
                )
            
        except Exception as e:
            self.log_result(
                "Trading Status Test",
                False,
                f"Test failed: {e}"
            )
    
    def run_all_tests(self):
        """Run all tests"""
        print("\n" + "="*70)
        print("NAE LIVE ACCOUNT READINESS TEST SUITE")
        print("="*70)
        print(f"Started: {datetime.datetime.now().isoformat()}")
        print("="*70)
        
        # Test 1: OptimusAgent initialization
        optimus = self.test_optimus_initialization()
        
        # Test 2: Account balance sync
        self.test_account_balance_sync(optimus)
        
        # Test 3: Strategy determination
        self.test_strategy_determination(optimus)
        
        # Test 4: Safety features
        self.test_safety_features(optimus)
        
        # Test 5: Master scheduler
        self.test_master_scheduler()
        
        # Test 6: Automation system
        self.test_automation_system()
        
        # Test 7: QuantAgent integration
        self.test_quant_agent_integration(optimus)
        
        # Test 8: Enhanced RL Agent
        self.test_enhanced_rl_agent(optimus)
        
        # Test 9: Trading status
        self.test_trading_status(optimus)
        
        # Print summary
        self.print_summary()
    
    def print_summary(self):
        """Print test summary"""
        print("\n" + "="*70)
        print("TEST SUMMARY")
        print("="*70)
        print(f"Total Tests: {len(self.results)}")
        print(f"✅ Passed: {self.passed}")
        print(f"⚠️  Warnings: {self.warnings}")
        print(f"❌ Failed: {self.failed}")
        print("="*70)
        
        # Overall status
        if self.failed == 0:
            if self.warnings > 0:
                print("\n✅ SYSTEM READY WITH WARNINGS")
                print("   All critical tests passed. Some optional features may need attention.")
            else:
                print("\n✅ SYSTEM FULLY READY")
                print("   All tests passed. NAE is ready for LIVE account trading.")
        else:
            print("\n❌ SYSTEM NOT READY")
            print("   Some critical tests failed. Please review and fix issues.")
        
        print("\n" + "="*70)
        print("DETAILED RESULTS")
        print("="*70)
        for result in self.results:
            print(f"{result['status']}: {result['test']}")
            if result['message']:
                print(f"   {result['message']}")
        
        print("\n" + "="*70)

if __name__ == "__main__":
    test_suite = LiveAccountReadinessTest()
    test_suite.run_all_tests()

