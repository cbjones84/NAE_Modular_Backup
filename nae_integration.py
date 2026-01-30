# NAE/nae_integration.py
"""
NAE Integration Module
Integrates all new systems: vault, environment, models, testing, execution, planning
"""

import os
import sys
from typing import Dict, Any, Optional

# Import all new systems
try:
    from secure_vault import get_vault, SecureVault
    from environment_manager import get_env_manager, EnvironmentManager, Environment
    from model_config import get_model_manager, ModelAssignmentManager
    from autotest_framework import AutoTestFramework
    from command_executor import get_executor, SafeCommandExecutor
    from multi_step_planner import get_planner, MultiStepPlanner
    from debug_tools import get_debug_tools, DebugTools
except ImportError as e:
    print(f"Warning: Some modules not available: {e}")

class NAEIntegration:
    """Main integration class for all NAE systems"""
    
    def __init__(self):
        self.vault: Optional[SecureVault] = None
        self.env_manager: Optional[EnvironmentManager] = None
        self.model_manager: Optional[ModelAssignmentManager] = None
        self.test_framework: Optional[AutoTestFramework] = None
        self.executor: Optional[SafeCommandExecutor] = None
        self.planner: Optional[MultiStepPlanner] = None
        self.debug_tools: Optional[DebugTools] = None
        
        self._initialize_systems()
    
    def _initialize_systems(self):
        """Initialize all integrated systems"""
        try:
            self.vault = get_vault()
            print("✅ Secure Vault initialized")
        except Exception as e:
            print(f"⚠️  Vault initialization failed: {e}")
        
        try:
            self.env_manager = get_env_manager()
            print(f"✅ Environment Manager initialized: {self.env_manager.current_environment}")
        except Exception as e:
            print(f"⚠️  Environment Manager initialization failed: {e}")
        
        try:
            self.model_manager = get_model_manager()
            print(f"✅ Model Assignment Manager initialized: {len(self.model_manager.assignments)} agents configured")
        except Exception as e:
            print(f"⚠️  Model Manager initialization failed: {e}")
        
        try:
            self.test_framework = AutoTestFramework()
            print("✅ AutoTest Framework initialized")
        except Exception as e:
            print(f"⚠️  Test Framework initialization failed: {e}")
        
        try:
            self.executor = get_executor()
            print("✅ Command Executor initialized")
        except Exception as e:
            print(f"⚠️  Command Executor initialization failed: {e}")
        
        try:
            self.planner = get_planner()
            print("✅ Multi-Step Planner initialized")
        except Exception as e:
            print(f"⚠️  Multi-Step Planner initialization failed: {e}")
        
        try:
            self.debug_tools = get_debug_tools()
            print("✅ Debug Tools initialized")
        except Exception as e:
            print(f"⚠️  Debug Tools initialization failed: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get status of all integrated systems"""
        status = {
            "vault": self.vault is not None,
            "environment": {
                "initialized": self.env_manager is not None,
                "current": self.env_manager.current_environment.value if self.env_manager else None
            },
            "model_manager": self.model_manager is not None,
            "test_framework": self.test_framework is not None,
            "executor": self.executor is not None,
            "planner": self.planner is not None,
            "debug_tools": self.debug_tools is not None
        }
        return status
    
    def run_full_test_suite(self) -> Dict[str, Any]:
        """Run full test suite"""
        if not self.test_framework:
            return {"error": "Test framework not initialized"}
        
        print("\n" + "=" * 60)
        print("RUNNING FULL TEST SUITE")
        print("=" * 60)
        
        results = self.test_framework.run_all_agent_tests()
        self.test_framework.print_summary()
        report = self.test_framework.generate_report()
        
        return report
    
    def create_autogen_workflow(self, goal: str) -> Optional[str]:
        """Create AutoGen-style workflow"""
        if not self.planner:
            print("Error: Planner not initialized")
            return None
        
        plan = self.planner.create_autogen_workflow(goal)
        print(f"✅ Created workflow plan: {plan.plan_id}")
        print(f"   Goal: {plan.goal}")
        print(f"   Steps: {len(plan.steps)}")
        
        return plan.plan_id
    
    def execute_workflow(self, plan_id: str) -> Dict[str, Any]:
        """Execute a workflow plan"""
        if not self.planner:
            return {"error": "Planner not initialized"}
        
        print(f"\nExecuting workflow: {plan_id}")
        result = self.planner.execute_plan(plan_id)
        
        if result.get("status") == "completed":
            print("✅ Workflow completed successfully")
        else:
            print(f"❌ Workflow failed: {result.get('error')}")
        
        return result
    
    def migrate_api_keys_to_vault(self) -> int:
        """Migrate API keys from JSON to secure vault"""
        if not self.vault:
            print("Error: Vault not initialized")
            return 0
        
        json_file = "config/api_keys.json"
        migrated = self.vault.migrate_from_json(json_file)
        
        if migrated > 0:
            print(f"✅ Migrated {migrated} secrets to vault")
        else:
            print("⚠️  No secrets migrated (check if JSON file exists and has real keys)")
        
        return migrated
    
    def switch_environment(self, env_name: str) -> bool:
        """Switch environment"""
        if not self.env_manager:
            print("Error: Environment manager not initialized")
            return False
        
        try:
            env = Environment(env_name.lower())
            success = self.env_manager.set_environment(env)
            if success:
                print(f"✅ Switched to {env_name} environment")
            return success
        except ValueError:
            print(f"Error: Invalid environment: {env_name}")
            return False


# Global integration instance
_nae_integration = None

def get_nae_integration() -> NAEIntegration:
    """Get global NAE integration instance"""
    global _nae_integration
    if _nae_integration is None:
        _nae_integration = NAEIntegration()
    return _nae_integration


if __name__ == "__main__":
    print("🚀 NAE Integration System")
    print("=" * 60)
    
    # Initialize integration
    nae = get_nae_integration()
    
    # Show status
    status = nae.get_system_status()
    print("\nSystem Status:")
    for key, value in status.items():
        print(f"  {key}: {value}")
    
    # Run tests
    print("\n" + "=" * 60)
    response = input("Run full test suite? (y/n): ")
    if response.lower() == 'y':
        nae.run_full_test_suite()
    
    # Migrate keys
    print("\n" + "=" * 60)
    response = input("Migrate API keys to vault? (y/n): ")
    if response.lower() == 'y':
        nae.migrate_api_keys_to_vault()


