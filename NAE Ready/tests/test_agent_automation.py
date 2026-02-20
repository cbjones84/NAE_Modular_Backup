#!/usr/bin/env python3
"""
NAE Agent Automation Test Suite
Tests that all agents are properly automated and functioning
"""

import os
import sys
import time
import json
import datetime
from typing import Dict, List, Any

# Add parent directory to path
sys.path.append(os.path.dirname(__file__))

# Import scheduler
from nae_master_scheduler import NAEMasterScheduler, AutomationConfig

# Import agents to test directly
from agents.ralph import RalphAgent
from agents.donnie import DonnieAgent
from agents.optimus import OptimusAgent
from agents.casey import CaseyAgent
from agents.splinter import SplinterAgent
from agents.rocksteady import RocksteadyAgent
from agents.bebop import BebopAgent
from agents.phisher import PhisherAgent

# Try to import optional agents
try:
    from agents.genny import GennyAgent
    GENNY_AVAILABLE = True
except ImportError:
    GENNY_AVAILABLE = False

# ----------------------
# Test Functions
# ----------------------
def test_agent_initialization():
    """Test that all agents can be initialized"""
    print("\n" + "="*70)
    print("Test 1: Agent Initialization")
    print("="*70)
    
    results = {}
    
    try:
        ralph = RalphAgent()
        results['Ralph'] = {'status': 'success', 'has_run_cycle': hasattr(ralph, 'run_cycle')}
    except Exception as e:
        results['Ralph'] = {'status': 'error', 'error': str(e)}
    
    try:
        donnie = DonnieAgent()
        results['Donnie'] = {'status': 'success', 'has_run_cycle': hasattr(donnie, 'run_cycle')}
    except Exception as e:
        results['Donnie'] = {'status': 'error', 'error': str(e)}
    
    try:
        optimus = OptimusAgent(sandbox=True)
        results['Optimus'] = {'status': 'success', 'has_run_cycle': hasattr(optimus, 'run_cycle')}
    except Exception as e:
        results['Optimus'] = {'status': 'error', 'error': str(e)}
    
    try:
        casey = CaseyAgent()
        results['Casey'] = {'status': 'success', 'has_run': hasattr(casey, 'run')}
    except Exception as e:
        results['Casey'] = {'status': 'error', 'error': str(e)}
    
    try:
        splinter = SplinterAgent()
        results['Splinter'] = {'status': 'success', 'has_run': hasattr(splinter, 'run')}
    except Exception as e:
        results['Splinter'] = {'status': 'error', 'error': str(e)}
    
    try:
        rocksteady = RocksteadyAgent()
        results['Rocksteady'] = {'status': 'success', 'has_run': hasattr(rocksteady, 'run')}
    except Exception as e:
        results['Rocksteady'] = {'status': 'error', 'error': str(e)}
    
    try:
        bebop = BebopAgent()
        results['Bebop'] = {'status': 'success', 'has_run': hasattr(bebop, 'run')}
    except Exception as e:
        results['Bebop'] = {'status': 'error', 'error': str(e)}
    
    try:
        phisher = PhisherAgent()
        results['Phisher'] = {'status': 'success', 'has_run': hasattr(phisher, 'run')}
    except Exception as e:
        results['Phisher'] = {'status': 'error', 'error': str(e)}
    
    if GENNY_AVAILABLE:
        try:
            genny = GennyAgent()
            results['Genny'] = {'status': 'success', 'has_run_cycle': hasattr(genny, 'run_cycle')}
        except Exception as e:
            results['Genny'] = {'status': 'error', 'error': str(e)}
    
    # Print results
    for agent_name, result in results.items():
        if result['status'] == 'success':
            print(f"  ✓ {agent_name}: Initialized successfully")
        else:
            print(f"  ✗ {agent_name}: Failed - {result.get('error', 'Unknown error')}")
    
    return results

def test_scheduler_initialization():
    """Test that scheduler can be initialized with all agents"""
    print("\n" + "="*70)
    print("Test 2: Scheduler Initialization")
    print("="*70)
    
    try:
        scheduler = NAEMasterScheduler()
        agents_count = len(scheduler.agents)
        print(f"  ✓ Scheduler initialized with {agents_count} agents")
        
        # Check each agent
        for name, agent_auto in scheduler.agents.items():
            status = agent_auto.get_status()
            print(f"    - {name}: {'ENABLED' if status['enabled'] else 'DISABLED'}")
        
        return {'status': 'success', 'agents_count': agents_count}
    except Exception as e:
        print(f"  ✗ Scheduler initialization failed: {e}")
        return {'status': 'error', 'error': str(e)}

def test_agent_cycles():
    """Test that agent cycles can be run"""
    print("\n" + "="*70)
    print("Test 3: Agent Cycle Execution")
    print("="*70)
    
    results = {}
    
    # Test Ralph
    try:
        ralph = RalphAgent()
        result = ralph.run_cycle()
        results['Ralph'] = {
            'status': 'success',
            'approved_count': result.get('approved_count', 0)
        }
        print(f"  ✓ Ralph: Generated {result.get('approved_count', 0)} strategies")
    except Exception as e:
        results['Ralph'] = {'status': 'error', 'error': str(e)}
        print(f"  ✗ Ralph: Failed - {e}")
    
    # Test Donnie (with strategies from Ralph)
    try:
        donnie = DonnieAgent()
        if 'Ralph' in results and results['Ralph']['status'] == 'success':
            strategies = ralph.top_strategies(3)
            donnie.receive_strategies(strategies)
        
        donnie.run_cycle(sandbox=True)
        results['Donnie'] = {
            'status': 'success',
            'executed': len(donnie.execution_history)
        }
        print(f"  ✓ Donnie: Processed {len(donnie.execution_history)} strategies")
    except Exception as e:
        results['Donnie'] = {'status': 'error', 'error': str(e)}
        print(f"  ✗ Donnie: Failed - {e}")
    
    # Test Optimus
    try:
        optimus = OptimusAgent(sandbox=True)
        # Create a simple execution batch
        execution_batch = [{
            'symbol': 'SPY',
            'side': 'buy',
            'quantity': 10,
            'price': 450.0
        }]
        optimus.run_cycle(execution_batch)
        results['Optimus'] = {
            'status': 'success',
            'executed': len(optimus.execution_history)
        }
        print(f"  ✓ Optimus: Executed {len(optimus.execution_history)} trades")
    except Exception as e:
        results['Optimus'] = {'status': 'error', 'error': str(e)}
        print(f"  ✗ Optimus: Failed - {e}")
    
    # Test Casey
    try:
        casey = CaseyAgent()
        casey.run()
        results['Casey'] = {'status': 'success'}
        print(f"  ✓ Casey: Ran successfully")
    except Exception as e:
        results['Casey'] = {'status': 'error', 'error': str(e)}
        print(f"  ✗ Casey: Failed - {e}")
    
    # Test Splinter
    try:
        splinter = SplinterAgent()
        splinter.run()
        results['Splinter'] = {'status': 'success'}
        print(f"  ✓ Splinter: Ran successfully")
    except Exception as e:
        results['Splinter'] = {'status': 'error', 'error': str(e)}
        print(f"  ✗ Splinter: Failed - {e}")
    
    # Test Rocksteady
    try:
        rocksteady = RocksteadyAgent()
        rocksteady.run()
        results['Rocksteady'] = {'status': 'success'}
        print(f"  ✓ Rocksteady: Ran successfully")
    except Exception as e:
        results['Rocksteady'] = {'status': 'error', 'error': str(e)}
        print(f"  ✗ Rocksteady: Failed - {e}")
    
    # Test Bebop
    try:
        bebop = BebopAgent()
        bebop.run()
        results['Bebop'] = {'status': 'success'}
        print(f"  ✓ Bebop: Ran successfully")
    except Exception as e:
        results['Bebop'] = {'status': 'error', 'error': str(e)}
        print(f"  ✗ Bebop: Failed - {e}")
    
    # Test Phisher
    try:
        phisher = PhisherAgent()
        phisher.run()
        results['Phisher'] = {'status': 'success'}
        print(f"  ✓ Phisher: Ran successfully")
    except Exception as e:
        results['Phisher'] = {'status': 'error', 'error': str(e)}
        print(f"  ✗ Phisher: Failed - {e}")
    
    return results

def test_scheduler_execution():
    """Test that scheduler can execute agent cycles"""
    print("\n" + "="*70)
    print("Test 4: Scheduler Execution")
    print("="*70)
    
    try:
        scheduler = NAEMasterScheduler()
        
        # Run a few cycles manually
        print("  Running Ralph cycle...")
        if 'Ralph' in scheduler.agents:
            result = scheduler.agents['Ralph'].run()
            print(f"    Result: {result.get('status')}")
        
        print("  Running Donnie cycle...")
        if 'Donnie' in scheduler.agents:
            result = scheduler.agents['Donnie'].run(sandbox=True)
            print(f"    Result: {result.get('status')}")
        
        print("  Running Optimus cycle...")
        if 'Optimus' in scheduler.agents:
            result = scheduler.agents['Optimus'].run([])
            print(f"    Result: {result.get('status')}")
        
        # Get status
        status = scheduler.get_status()
        print(f"\n  Scheduler Status:")
        print(f"    Running: {status['running']}")
        print(f"    Agents: {len(status['agents'])}")
        print(f"    Scheduled Jobs: {len(status['schedules'])}")
        
        return {'status': 'success', 'scheduler_status': status}
    except Exception as e:
        print(f"  ✗ Scheduler execution failed: {e}")
        return {'status': 'error', 'error': str(e)}

def test_agent_communication():
    """Test that agents can communicate with each other"""
    print("\n" + "="*70)
    print("Test 5: Agent Communication")
    print("="*70)
    
    try:
        # Test Ralph -> Donnie flow
        ralph = RalphAgent()
        ralph.run_cycle()
        strategies = ralph.top_strategies(3)
        
        donnie = DonnieAgent()
        donnie.receive_strategies(strategies)
        
        print(f"  ✓ Ralph generated {len(strategies)} strategies")
        print(f"  ✓ Donnie received {len(donnie.candidate_strategies)} strategies")
        
        # Test Donnie -> Optimus flow
        optimus = OptimusAgent(sandbox=True)
        donnie.run_cycle(sandbox=True, optimus_agent=optimus)
        
        print(f"  ✓ Donnie executed {len(donnie.execution_history)} strategies")
        print(f"  ✓ Optimus received {len(optimus.inbox)} messages")
        
        return {'status': 'success'}
    except Exception as e:
        print(f"  ✗ Agent communication failed: {e}")
        return {'status': 'error', 'error': str(e)}

def test_schedule_configuration():
    """Test that schedules are properly configured"""
    print("\n" + "="*70)
    print("Test 6: Schedule Configuration")
    print("="*70)
    
    try:
        scheduler = NAEMasterScheduler()
        status = scheduler.get_status()
        
        print(f"  Total scheduled jobs: {len(status['schedules'])}")
        
        for schedule_info in status['schedules']:
            if 'job' in schedule_info:
                job_desc = schedule_info['job']
                next_run = schedule_info.get('next_run', 'Not scheduled')
                print(f"    - {job_desc}")
                print(f"      Next run: {next_run}")
            elif 'agent' in schedule_info:
                agent_name = schedule_info['agent']
                interval = schedule_info['interval']
                last_run = schedule_info.get('last_run', 0)
                print(f"    - {agent_name}: Interval {interval}s, Last run: {last_run}")
        
        return {'status': 'success', 'job_count': len(status['schedules'])}
    except Exception as e:
        print(f"  ✗ Schedule configuration test failed: {e}")
        import traceback
        traceback.print_exc()
        return {'status': 'error', 'error': str(e)}

# ----------------------
# Main Test Runner
# ----------------------
def run_all_tests():
    """Run all automation tests"""
    print("="*70)
    print("NAE Agent Automation Test Suite")
    print("="*70)
    
    results = {}
    
    # Run all tests
    results['initialization'] = test_agent_initialization()
    results['scheduler_init'] = test_scheduler_initialization()
    results['cycles'] = test_agent_cycles()
    results['scheduler_exec'] = test_scheduler_execution()
    results['communication'] = test_agent_communication()
    results['schedules'] = test_schedule_configuration()
    
    # Summary
    print("\n" + "="*70)
    print("Test Summary")
    print("="*70)
    
    total_tests = len(results)
    passed_tests = sum(1 for r in results.values() if r.get('status') == 'success')
    
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print(f"Success Rate: {(passed_tests/total_tests*100):.1f}%")
    
    # Save results
    results_file = f"logs/automation_test_results_{int(time.time())}.json"
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {results_file}")
    
    return results

if __name__ == "__main__":
    run_all_tests()

