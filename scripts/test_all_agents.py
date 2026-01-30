#!/usr/bin/env python3
"""
NAE Comprehensive Agent Test Suite
Tests all agents in test/sandbox mode and reports on each
"""

import sys
import os
import traceback
from datetime import datetime
from typing import Dict, List, Any, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    MAGENTA = '\033[95m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def print_header(text):
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{text.center(80)}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.RESET}\n")

def print_section(text):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'─'*80}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'─'*80}{Colors.RESET}\n")

class AgentTester:
    """Test all NAE agents"""
    
    def __init__(self):
        self.results = {}
        self.test_mode = True
        
    def test_agent(self, name: str, agent_class, init_args: Dict = None, test_methods: List[str] = None):
        """Test a single agent"""
        print_section(f"Testing {name} Agent")
        
        result = {
            "name": name,
            "status": "unknown",
            "initialized": False,
            "capabilities": [],
            "errors": [],
            "warnings": []
        }
        
        try:
            # Initialize agent
            print(f"{Colors.BLUE}Initializing {name}...{Colors.RESET}")
            init_args = init_args or {}
            
            # Common test mode settings
            if 'sandbox' not in init_args and hasattr(agent_class, '__init__'):
                # Try to set sandbox/test mode
                if 'sandbox' in agent_class.__init__.__code__.co_varnames:
                    init_args['sandbox'] = True
            
            agent = agent_class(**init_args)
            result["initialized"] = True
            print(f"{Colors.GREEN}✅ {name} initialized successfully{Colors.RESET}")
            
            # Test basic attributes
            if hasattr(agent, 'name'):
                print(f"   Name: {agent.name}")
            if hasattr(agent, 'goals'):
                print(f"   Goals: {len(agent.goals) if isinstance(agent.goals, list) else 'Configured'}")
            
            # Test capabilities
            capabilities = []
            if hasattr(agent, 'get_capabilities_summary'):
                try:
                    caps = agent.get_capabilities_summary()
                    if isinstance(caps, dict):
                        capabilities = caps.get('capabilities', [])
                        print(f"{Colors.CYAN}   Capabilities: {', '.join(capabilities[:5])}...{Colors.RESET}")
                except Exception as e:
                    result["warnings"].append(f"Could not get capabilities: {e}")
            
            result["capabilities"] = capabilities
            
            # Test specific methods if provided
            if test_methods:
                for method_name in test_methods:
                    if hasattr(agent, method_name):
                        try:
                            method = getattr(agent, method_name)
                            print(f"{Colors.BLUE}   Testing {method_name}(){Colors.RESET}")
                            # Call with minimal args if it's a method
                            if callable(method):
                                # Try to call with safe defaults
                                try:
                                    method_result = method()
                                    print(f"{Colors.GREEN}      ✅ {method_name}() executed{Colors.RESET}")
                                except TypeError:
                                    # Method requires arguments, skip for now
                                    print(f"{Colors.YELLOW}      ⚠️  {method_name}() requires arguments{Colors.RESET}")
                        except Exception as e:
                            result["errors"].append(f"{method_name}(): {str(e)}")
            
            result["status"] = "working"
            
        except ImportError as e:
            result["status"] = "not_available"
            result["errors"].append(f"Import error: {str(e)}")
            print(f"{Colors.YELLOW}⚠️  {name} not available: {e}{Colors.RESET}")
            # Continue testing other agents
            self.results[name] = result
            return result
            
        except Exception as e:
            result["status"] = "error"
            result["errors"].append(str(e))
            print(f"{Colors.RED}❌ {name} failed: {e}{Colors.RESET}")
            traceback.print_exc()
        
        self.results[name] = result
        return result
    
    def test_ralph(self):
        """Test Ralph agent"""
        return self.test_agent(
            "Ralph",
            __import__('agents.ralph', fromlist=['RalphAgent']).RalphAgent,
            {},
            ['fetch_market_data', 'get_real_time_price']
        )
    
    def test_donnie(self):
        """Test Donnie agent"""
        return self.test_agent(
            "Donnie",
            __import__('agents.donnie', fromlist=['DonnieAgent']).DonnieAgent,
            {}
        )
    
    def test_optimus(self):
        """Test Optimus agent"""
        return self.test_agent(
            "Optimus",
            __import__('agents.optimus', fromlist=['OptimusAgent']).OptimusAgent,
            {'sandbox': True}  # Use sandbox mode
        )
    
    def test_casey(self):
        """Test Casey agent"""
        return self.test_agent(
            "Casey",
            __import__('agents.casey', fromlist=['CaseyAgent']).CaseyAgent,
            {}
        )
    
    def test_splinter(self):
        """Test Splinter agent"""
        return self.test_agent(
            "Splinter",
            __import__('agents.splinter', fromlist=['SplinterAgent']).SplinterAgent,
            {}
        )
    
    def test_rocksteady(self):
        """Test Rocksteady agent"""
        return self.test_agent(
            "Rocksteady",
            __import__('agents.rocksteady', fromlist=['RocksteadyAgent']).RocksteadyAgent,
            {}
        )
    
    def test_bebop(self):
        """Test Bebop agent"""
        return self.test_agent(
            "Bebop",
            __import__('agents.bebop', fromlist=['BebopAgent']).BebopAgent,
            {}
        )
    
    def test_phisher(self):
        """Test Phisher agent"""
        return self.test_agent(
            "Phisher",
            __import__('agents.phisher', fromlist=['PhisherAgent']).PhisherAgent,
            {}
        )
    
    def test_genny(self):
        """Test Genny agent"""
        return self.test_agent(
            "Genny",
            __import__('agents.genny', fromlist=['GennyAgent']).GennyAgent,
            {}
        )
    
    def test_april(self):
        """Test April agent (optional)"""
        try:
            return self.test_agent(
                "April",
                __import__('agents.april', fromlist=['AprilAgent']).AprilAgent,
                {}
            )
        except ImportError:
            return {"name": "April", "status": "not_available", "errors": ["Module not found"]}
    
    def test_mikey(self):
        """Test Mikey agent (optional)"""
        try:
            return self.test_agent(
                "Mikey",
                __import__('agents.mikey', fromlist=['MikeyAgent']).MikeyAgent,
                {}
            )
        except ImportError:
            return {"name": "Mikey", "status": "not_available", "errors": ["Module not found"]}
    
    def test_leo(self):
        """Test Leo agent (optional)"""
        try:
            return self.test_agent(
                "Leo",
                __import__('agents.leo', fromlist=['LeoAgent']).LeoAgent,
                {}
            )
        except ImportError:
            return {"name": "Leo", "status": "not_available", "errors": ["Module not found"]}
    
    def test_shredder(self):
        """Test Shredder agent (optional)"""
        try:
            return self.test_agent(
                "Shredder",
                __import__('agents.shredder', fromlist=['ShredderAgent']).ShredderAgent,
                {}
            )
        except ImportError:
            return {"name": "Shredder", "status": "not_available", "errors": ["Module not found"]}
    
    def run_all_tests(self):
        """Run tests for all agents"""
        print_header("NAE COMPREHENSIVE AGENT TEST SUITE")
        print(f"{Colors.CYAN}Test Mode: ENABLED{Colors.RESET}")
        print(f"{Colors.CYAN}Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Colors.RESET}")
        
        # Test all agents
        print_section("CORE TRADING AGENTS")
        self.test_ralph()
        self.test_donnie()
        self.test_optimus()
        
        print_section("ORCHESTRATION AGENTS")
        self.test_casey()
        self.test_splinter()
        
        print_section("SECURITY AGENTS")
        self.test_rocksteady()
        self.test_phisher()
        
        print_section("MONITORING AGENTS")
        self.test_bebop()
        
        print_section("SPECIALIZED AGENTS")
        self.test_genny()
        
        print_section("OPTIONAL AGENTS")
        self.test_april()
        self.test_mikey()
        self.test_leo()
        self.test_shredder()
        
        # Generate report
        self.generate_report()
    
    def generate_report(self):
        """Generate comprehensive test report"""
        print_header("TEST RESULTS SUMMARY")
        
        # Categorize results
        working = []
        errors = []
        not_available = []
        
        for name, result in self.results.items():
            if result.get("status") == "working":
                working.append(result)
            elif result.get("status") == "not_available":
                not_available.append(result)
            else:
                errors.append(result)
        
        # Print summary
        print(f"\n{Colors.GREEN}✅ WORKING AGENTS: {len(working)}{Colors.RESET}")
        for result in working:
            status_icon = "✅" if result["initialized"] else "⚠️"
            print(f"   {status_icon} {result['name']}")
            if result.get("capabilities"):
                print(f"      Capabilities: {len(result['capabilities'])}")
            if result.get("errors"):
                print(f"      {Colors.YELLOW}Warnings: {len(result['errors'])}{Colors.RESET}")
        
        if errors:
            print(f"\n{Colors.RED}❌ AGENTS WITH ERRORS: {len(errors)}{Colors.RESET}")
            for result in errors:
                print(f"   ❌ {result['name']}")
                for error in result.get("errors", [])[:3]:
                    print(f"      - {error[:80]}")
        
        if not_available:
            print(f"\n{Colors.YELLOW}⚠️  UNAVAILABLE AGENTS: {len(not_available)}{Colors.RESET}")
            for result in not_available:
                print(f"   ⚠️  {result['name']}")
        
        # Detailed report
        print_header("DETAILED AGENT REPORTS")
        
        for name, result in sorted(self.results.items()):
            print(f"\n{Colors.BOLD}{name} Agent{Colors.RESET}")
            print(f"   Status: {result['status']}")
            print(f"   Initialized: {'Yes' if result['initialized'] else 'No'}")
            
            if result.get("capabilities"):
                print(f"   Capabilities: {len(result['capabilities'])} found")
            
            if result.get("errors"):
                print(f"   {Colors.RED}Errors ({len(result['errors'])}):{Colors.RESET}")
                for error in result['errors'][:5]:
                    print(f"      - {error}")
            
            if result.get("warnings"):
                print(f"   {Colors.YELLOW}Warnings ({len(result['warnings'])}):{Colors.RESET}")
                for warning in result['warnings'][:3]:
                    print(f"      - {warning}")
        
        # Final summary
        total = len(self.results)
        working_count = len(working)
        success_rate = (working_count / total * 100) if total > 0 else 0
        
        print_header("FINAL SUMMARY")
        print(f"Total Agents Tested: {total}")
        print(f"{Colors.GREEN}Working: {working_count}{Colors.RESET}")
        print(f"{Colors.RED}Errors: {len(errors)}{Colors.RESET}")
        print(f"{Colors.YELLOW}Unavailable: {len(not_available)}{Colors.RESET}")
        print(f"Success Rate: {success_rate:.1f}%")
        
        if working_count == total:
            print(f"\n{Colors.GREEN}{Colors.BOLD}✅ ALL AGENTS WORKING!{Colors.RESET}")
        elif working_count > total / 2:
            print(f"\n{Colors.YELLOW}{Colors.BOLD}⚠️  MOST AGENTS WORKING{Colors.RESET}")
        else:
            print(f"\n{Colors.RED}{Colors.BOLD}❌ MULTIPLE AGENT ISSUES{Colors.RESET}")

def main():
    """Main test execution"""
    try:
        tester = AgentTester()
        tester.run_all_tests()
        return 0
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}Test interrupted by user{Colors.RESET}")
        return 1
    except Exception as e:
        print(f"\n{Colors.RED}Test suite failed: {e}{Colors.RESET}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())

