# NAE/autotest_framework.py
"""
Comprehensive AutoTest Framework for NAE
Provides automated testing, debugging, and verification
"""

import os
import sys
import subprocess
import json
import time
import traceback
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import importlib
import inspect

class TestStatus(Enum):
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"

@dataclass
class TestResult:
    """Test result data"""
    name: str
    status: TestStatus
    duration: float
    message: str
    error: Optional[str] = None
    output: Optional[str] = None

class AutoTestFramework:
    """Comprehensive testing framework for NAE"""
    
    def __init__(self, output_dir: str = "logs/tests"):
        self.output_dir = output_dir
        self.results: List[TestResult] = []
        os.makedirs(self.output_dir, exist_ok=True)
    
    def run_agent_tests(self, agent_name: str) -> List[TestResult]:
        """Run all tests for a specific agent"""
        results = []
        
        # Test agent import
        results.append(self._test_import(agent_name))
        
        # Test agent initialization
        results.append(self._test_initialization(agent_name))
        
        # Test agent methods
        results.append(self._test_methods(agent_name))
        
        # Test agent goals integration
        results.append(self._test_goals_integration(agent_name))
        
        self.results.extend(results)
        return results
    
    def _test_import(self, agent_name: str) -> TestResult:
        """Test agent import"""
        start_time = time.time()
        try:
            module_name = f"agents.{agent_name.lower()}"
            module = importlib.import_module(module_name)
            class_name = f"{agent_name}Agent"
            agent_class = getattr(module, class_name)
            
            duration = time.time() - start_time
            return TestResult(
                name=f"{agent_name}.import",
                status=TestStatus.PASSED,
                duration=duration,
                message=f"Successfully imported {class_name}"
            )
        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                name=f"{agent_name}.import",
                status=TestStatus.FAILED,
                duration=duration,
                message=f"Failed to import {agent_name}",
                error=str(e)
            )
    
    def _test_initialization(self, agent_name: str) -> TestResult:
        """Test agent initialization"""
        start_time = time.time()
        try:
            module_name = f"agents.{agent_name.lower()}"
            module = importlib.import_module(module_name)
            class_name = f"{agent_name}Agent"
            agent_class = getattr(module, class_name)
            
            # Try to initialize
            if agent_name.lower() == "optimus":
                agent = agent_class(sandbox=True)
            else:
                agent = agent_class()
            
            duration = time.time() - start_time
            return TestResult(
                name=f"{agent_name}.initialization",
                status=TestStatus.PASSED,
                duration=duration,
                message=f"Successfully initialized {class_name}"
            )
        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                name=f"{agent_name}.initialization",
                status=TestStatus.FAILED,
                duration=duration,
                message=f"Failed to initialize {agent_name}",
                error=str(e)
            )
    
    def _test_methods(self, agent_name: str) -> TestResult:
        """Test agent methods exist"""
        start_time = time.time()
        try:
            module_name = f"agents.{agent_name.lower()}"
            module = importlib.import_module(module_name)
            class_name = f"{agent_name}Agent"
            agent_class = getattr(module, class_name)
            
            # Check for required methods
            required_methods = ["log_action"]
            if agent_name.lower() == "optimus":
                agent = agent_class(sandbox=True)
            else:
                agent = agent_class()
            
            missing_methods = []
            for method in required_methods:
                if not hasattr(agent, method):
                    missing_methods.append(method)
            
            duration = time.time() - start_time
            if missing_methods:
                return TestResult(
                    name=f"{agent_name}.methods",
                    status=TestStatus.FAILED,
                    duration=duration,
                    message=f"Missing methods: {missing_methods}",
                    error=f"Required methods not found: {missing_methods}"
                )
            else:
                return TestResult(
                    name=f"{agent_name}.methods",
                    status=TestStatus.PASSED,
                    duration=duration,
                    message="All required methods present"
                )
        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                name=f"{agent_name}.methods",
                status=TestStatus.FAILED,
                duration=duration,
                message=f"Error testing methods: {str(e)}",
                error=str(e)
            )
    
    def _test_goals_integration(self, agent_name: str) -> TestResult:
        """Test agent goals integration"""
        start_time = time.time()
        try:
            module_name = f"agents.{agent_name.lower()}"
            module = importlib.import_module(module_name)
            class_name = f"{agent_name}Agent"
            agent_class = getattr(module, class_name)
            
            if agent_name.lower() == "optimus":
                agent = agent_class(sandbox=True)
            else:
                agent = agent_class()
            
            # Check if agent has goals attribute
            if not hasattr(agent, "goals"):
                duration = time.time() - start_time
                return TestResult(
                    name=f"{agent_name}.goals",
                    status=TestStatus.FAILED,
                    duration=duration,
                    message="Agent missing goals attribute",
                    error="Goals not integrated"
                )
            
            goals = agent.goals
            if not isinstance(goals, list) or len(goals) == 0:
                duration = time.time() - start_time
                return TestResult(
                    name=f"{agent_name}.goals",
                    status=TestStatus.FAILED,
                    duration=duration,
                    message="Goals not properly loaded",
                    error="Goals list is empty or invalid"
                )
            
            duration = time.time() - start_time
            return TestResult(
                name=f"{agent_name}.goals",
                status=TestStatus.PASSED,
                duration=duration,
                message=f"Goals integrated: {len(goals)} goals found"
            )
        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                name=f"{agent_name}.goals",
                status=TestStatus.FAILED,
                duration=duration,
                message=f"Error testing goals: {str(e)}",
                error=str(e)
            )
    
    def run_integration_test(self, test_name: str, test_func: Callable) -> TestResult:
        """Run an integration test"""
        start_time = time.time()
        try:
            result = test_func()
            duration = time.time() - start_time
            
            if result:
                return TestResult(
                    name=test_name,
                    status=TestStatus.PASSED,
                    duration=duration,
                    message="Integration test passed"
                )
            else:
                return TestResult(
                    name=test_name,
                    status=TestStatus.FAILED,
                    duration=duration,
                    message="Integration test failed"
                )
        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                name=test_name,
                status=TestStatus.ERROR,
                duration=duration,
                message=f"Integration test error: {str(e)}",
                error=str(e) + "\n" + traceback.format_exc()
            )
    
    def run_all_agent_tests(self) -> Dict[str, List[TestResult]]:
        """Run tests for all agents"""
        agents = ["Ralph", "Casey", "Donnie", "Optimus", "Splinter", 
                 "Bebop", "Phisher", "Genny", "Rocksteady", "Shredder",
                 "Mikey", "Leo", "April"]
        
        all_results = {}
        for agent in agents:
            print(f"Testing {agent}...")
            results = self.run_agent_tests(agent)
            all_results[agent] = results
        
        return all_results
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate test report"""
        total = len(self.results)
        passed = sum(1 for r in self.results if r.status == TestStatus.PASSED)
        failed = sum(1 for r in self.results if r.status == TestStatus.FAILED)
        errors = sum(1 for r in self.results if r.status == TestStatus.ERROR)
        
        report = {
            "timestamp": time.time(),
            "summary": {
                "total": total,
                "passed": passed,
                "failed": failed,
                "errors": errors,
                "pass_rate": passed / total if total > 0 else 0
            },
            "results": [
                {
                    "name": r.name,
                    "status": r.status.value,
                    "duration": r.duration,
                    "message": r.message,
                    "error": r.error
                }
                for r in self.results
            ]
        }
        
        # Save report
        report_file = os.path.join(self.output_dir, f"test_report_{int(time.time())}.json")
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report
    
    def print_summary(self):
        """Print test summary"""
        total = len(self.results)
        passed = sum(1 for r in self.results if r.status == TestStatus.PASSED)
        failed = sum(1 for r in self.results if r.status == TestStatus.FAILED)
        errors = sum(1 for r in self.results if r.status == TestStatus.ERROR)
        
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        print(f"Total Tests: {total}")
        print(f"Passed: {passed} ✅")
        print(f"Failed: {failed} ❌")
        print(f"Errors: {errors} ⚠️")
        print(f"Pass Rate: {passed/total*100:.1f}%")
        print("=" * 60)
        
        if failed > 0 or errors > 0:
            print("\nFailed Tests:")
            for r in self.results:
                if r.status in [TestStatus.FAILED, TestStatus.ERROR]:
                    print(f"  - {r.name}: {r.message}")
                    if r.error:
                        print(f"    Error: {r.error[:200]}")


if __name__ == "__main__":
    # Run all tests
    framework = AutoTestFramework()
    
    print("Running NAE Agent Tests...")
    framework.run_all_agent_tests()
    
    framework.print_summary()
    report = framework.generate_report()
    print(f"\nReport saved to: logs/tests/test_report_{int(time.time())}.json")


