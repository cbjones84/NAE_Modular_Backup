#!/usr/bin/env python3
"""
NAE Comprehensive System Test
Tests all components and provides detailed summary
"""

import os
import sys
import json
from pathlib import Path

sys.path.insert(0, '.')

def test_config_files():
    """Test configuration files exist"""
    config_files = [
        'config/api_keys.json',
        'config/.vault.encrypted',
        'config/.master.key',
        'config/environment_profiles.json',
        'config/model_assignments.json'
    ]
    results = {}
    for f in config_files:
        exists = os.path.exists(f)
        results[f] = exists
    return results

def test_agent_imports():
    """Test all agents can be imported"""
    agents = ['Ralph', 'Casey', 'Donnie', 'Optimus', 'Splinter', 'Bebop', 
              'Phisher', 'Genny', 'Rocksteady', 'Shredder', 'Mikey', 'Leo', 'April']
    results = {}
    failed = []
    
    for agent in agents:
        try:
            mod = __import__(f'agents.{agent.lower()}', fromlist=[agent])
            cls = getattr(mod, f'{agent}Agent')
            results[agent] = True
        except Exception as e:
            results[agent] = False
            failed.append((agent, str(e)))
    
    return results, failed

def test_goals_integration():
    """Test goals integration"""
    try:
        from goal_manager import get_nae_goals
        goals = get_nae_goals()
        return True, len(goals), goals
    except Exception as e:
        return False, 0, []

def test_core_agents():
    """Test core agent initialization"""
    results = {}
    try:
        from agents.ralph import RalphAgent
        r = RalphAgent()
        results['Ralph'] = {'status': r.status, 'goals': len(r.goals) if hasattr(r, 'goals') else 0}
    except Exception as e:
        results['Ralph'] = {'error': str(e)}
    
    try:
        from agents.casey import CaseyAgent
        c = CaseyAgent()
        results['Casey'] = {'status': 'initialized', 'goals': len(c.goals)}
    except Exception as e:
        results['Casey'] = {'error': str(e)}
    
    try:
        from agents.optimus import OptimusAgent
        o = OptimusAgent(sandbox=True)
        results['Optimus'] = {'status': o.trading_mode.value, 'goals': len(o.goals)}
    except Exception as e:
        results['Optimus'] = {'error': str(e)}
    
    try:
        from agents.donnie import DonnieAgent
        d = DonnieAgent()
        results['Donnie'] = {'status': 'initialized', 'goals': len(d.goals)}
    except Exception as e:
        results['Donnie'] = {'error': str(e)}
    
    return results

def test_integration_systems():
    """Test integration systems"""
    results = {}
    
    try:
        from nae_integration import get_nae_integration
        nae = get_nae_integration()
        status = nae.get_system_status()
        results['integration'] = status
    except Exception as e:
        results['integration'] = {'error': str(e)}
    
    try:
        from secure_vault import get_vault
        vault = get_vault()
        secrets = vault.list_secrets()
        results['vault'] = {'initialized': True, 'paths': len(secrets)}
    except Exception as e:
        results['vault'] = {'error': str(e)}
    
    try:
        from environment_manager import get_env_manager
        env = get_env_manager()
        results['environment'] = {'initialized': True, 'current': env.current_environment.value}
    except Exception as e:
        results['environment'] = {'error': str(e)}
    
    try:
        from model_config import get_model_manager
        models = get_model_manager()
        results['models'] = {'initialized': True, 'count': len(models.assignments)}
    except Exception as e:
        results['models'] = {'error': str(e)}
    
    return results

def test_command_executor():
    """Test command execution"""
    try:
        from command_executor import get_executor
        executor = get_executor()
        result = executor.execute_python_code('print("Test"); x = 10 + 20')
        return {'status': result.status.value, 'success': result.status.value == 'success'}
    except Exception as e:
        return {'error': str(e), 'success': False}

def test_planner():
    """Test multi-step planner"""
    try:
        from multi_step_planner import get_planner
        planner = get_planner()
        plan = planner.create_plan('Test', [{
            'name': 'Test Step',
            'agent': 'Ralph',
            'action': 'execute_method',
            'parameters': {},
            'dependencies': []
        }])
        return {'status': 'success', 'plan_id': plan.plan_id}
    except Exception as e:
        return {'error': str(e), 'success': False}

def check_environment_variables():
    """Check environment variables"""
    env_vars = {
        'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY'),
        'ANTHROPIC_API_KEY': os.getenv('ANTHROPIC_API_KEY'),
        'NAE_ENVIRONMENT': os.getenv('NAE_ENVIRONMENT', 'sandbox (default)'),
        'NAE_VAULT_PASSWORD': os.getenv('NAE_VAULT_PASSWORD')
    }
    return env_vars

def main():
    """Run all tests and generate summary"""
    print("=" * 70)
    print("NAE COMPREHENSIVE SYSTEM TEST")
    print("=" * 70)
    print()
    
    results = {}
    
    # Test 1: Configuration Files
    print("1. Testing Configuration Files...")
    config_results = test_config_files()
    results['config_files'] = config_results
    config_passed = sum(1 for v in config_results.values() if v)
    print(f"   ✅ {config_passed}/{len(config_results)} files exist")
    print()
    
    # Test 2: Agent Imports
    print("2. Testing Agent Imports...")
    agent_results, failed = test_agent_imports()
    results['agent_imports'] = agent_results
    import_passed = sum(1 for v in agent_results.values() if v)
    print(f"   ✅ {import_passed}/{len(agent_results)} agents importable")
    if failed:
        print(f"   ❌ Failed: {[f[0] for f in failed]}")
    print()
    
    # Test 3: Goals Integration
    print("3. Testing Goals Integration...")
    goals_ok, goals_count, goals = test_goals_integration()
    results['goals'] = {'ok': goals_ok, 'count': goals_count}
    if goals_ok:
        print(f"   ✅ Goals loaded: {goals_count} goals")
    else:
        print("   ❌ Goals integration failed")
    print()
    
    # Test 4: Core Agents
    print("4. Testing Core Agent Initialization...")
    core_results = test_core_agents()
    results['core_agents'] = core_results
    for agent, status in core_results.items():
        if 'error' in status:
            print(f"   ❌ {agent}: {status['error']}")
        else:
            print(f"   ✅ {agent}: {status.get('status', 'OK')}")
    print()
    
    # Test 5: Integration Systems
    print("5. Testing Integration Systems...")
    integration_results = test_integration_systems()
    results['integration_systems'] = integration_results
    for system, status in integration_results.items():
        if 'error' in status:
            print(f"   ❌ {system}: {status['error']}")
        else:
            print(f"   ✅ {system}: Initialized")
    print()
    
    # Test 6: Command Executor
    print("6. Testing Command Executor...")
    executor_result = test_command_executor()
    results['executor'] = executor_result
    if executor_result.get('success'):
        print(f"   ✅ Command Executor: {executor_result['status']}")
    else:
        print(f"   ❌ Command Executor: {executor_result.get('error', 'Failed')}")
    print()
    
    # Test 7: Multi-Step Planner
    print("7. Testing Multi-Step Planner...")
    planner_result = test_planner()
    results['planner'] = planner_result
    if planner_result.get('success'):
        print(f"   ✅ Multi-Step Planner: {planner_result['status']}")
    else:
        print(f"   ❌ Multi-Step Planner: {planner_result.get('error', 'Failed')}")
    print()
    
    # Test 8: Environment Variables
    print("8. Checking Environment Variables...")
    env_vars = check_environment_variables()
    results['environment_vars'] = env_vars
    for var, value in env_vars.items():
        if value and value != 'sandbox (default)' and value != 'your-vault-password':
            print(f"   ✅ {var}: Set")
        else:
            print(f"   ⚠️  {var}: Not set (using default)")
    print()
    
    # Summary
    print("=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    total_tests = 0
    passed_tests = 0
    
    # Count passed/failed
    config_passed = sum(1 for v in config_results.values() if v)
    total_tests += len(config_results)
    passed_tests += config_passed
    
    import_passed = sum(1 for v in agent_results.values() if v)
    total_tests += len(agent_results)
    passed_tests += import_passed
    
    goals_test = 1 if goals_ok else 0
    total_tests += 1
    passed_tests += goals_test
    
    core_passed = sum(1 for v in core_results.values() if 'error' not in v)
    total_tests += len(core_results)
    passed_tests += core_passed
    
    integration_passed = sum(1 for v in integration_results.values() if 'error' not in v)
    total_tests += len(integration_results)
    passed_tests += integration_passed
    
    executor_test = 1 if executor_result.get('success') else 0
    total_tests += 1
    passed_tests += executor_test
    
    planner_test = 1 if planner_result.get('success') else 0
    total_tests += 1
    passed_tests += planner_test
    
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests} ✅")
    print(f"Failed: {total_tests - passed_tests} ❌")
    print(f"Pass Rate: {passed_tests/total_tests*100:.1f}%")
    print()
    
    # Save results
    results['summary'] = {
        'total': total_tests,
        'passed': passed_tests,
        'failed': total_tests - passed_tests,
        'pass_rate': passed_tests/total_tests*100
    }
    
    output_file = 'logs/system_test_results.json'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Detailed results saved to: {output_file}")
    print("=" * 70)
    
    return results

if __name__ == "__main__":
    main()


