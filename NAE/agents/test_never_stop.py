#!/usr/bin/env python3
"""
Test script to verify all agents never stop functionality
"""
import sys
import os
import time
import signal
import subprocess
from pathlib import Path

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../NAE Ready/agents'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_agent_imports():
    """Test that all agents can be imported"""
    print("=" * 70)
    print("Test 1: Agent Imports")
    print("=" * 70)
    
    agents = [
        ('ralph_github_continuous', 'run_forever_with_restart'),
        ('optimus', 'optimus_main_loop'),
        ('donnie', 'donnie_main_loop'),
        ('splinter', 'splinter_main_loop'),
        ('genny', 'genny_main_loop'),
        ('casey', 'casey_main_loop'),
        ('ralph', 'ralph_main_loop'),
    ]
    
    results = []
    for module_name, func_name in agents:
        try:
            module = __import__(module_name)
            if hasattr(module, func_name):
                func = getattr(module, func_name)
                if callable(func):
                    results.append((module_name, func_name, True, None))
                    print(f"  ✅ {module_name}.{func_name}")
                else:
                    results.append((module_name, func_name, False, "Not callable"))
                    print(f"  ❌ {module_name}.{func_name} - Not callable")
            else:
                results.append((module_name, func_name, False, "Function not found"))
                print(f"  ❌ {module_name}.{func_name} - Function not found")
        except Exception as e:
            results.append((module_name, func_name, False, str(e)))
            print(f"  ❌ {module_name}.{func_name} - {e}")
    
    passed = sum(1 for r in results if r[2])
    total = len(results)
    print(f"\n✅ Passed: {passed}/{total}")
    return passed == total


def test_infinite_loops():
    """Test that all agents have infinite loops"""
    print("\n" + "=" * 70)
    print("Test 2: Infinite Loop Structure")
    print("=" * 70)
    
    files = [
        ('NAE/agents/ralph_github_continuous.py', 'run_forever_with_restart'),
        ('NAE Ready/agents/optimus.py', 'optimus_main_loop'),
        ('NAE Ready/agents/donnie.py', 'donnie_main_loop'),
        ('NAE Ready/agents/splinter.py', 'splinter_main_loop'),
        ('NAE Ready/agents/genny.py', 'genny_main_loop'),
        ('NAE Ready/agents/casey.py', 'casey_main_loop'),
        ('NAE Ready/agents/ralph.py', 'ralph_main_loop'),
    ]
    
    results = []
    for file_path, func_name in files:
        try:
            full_path = Path(__file__).parent.parent.parent / file_path
            if not full_path.exists():
                results.append((file_path, False, "File not found"))
                print(f"  ❌ {file_path} - File not found")
                continue
            
            content = full_path.read_text()
            
            # Check for function definition
            if f'def {func_name}(' not in content:
                results.append((file_path, False, "Function not found"))
                print(f"  ❌ {file_path} - Function not found")
                continue
            
            # Check for infinite loop
            if 'while True:' not in content:
                results.append((file_path, False, "No infinite loop"))
                print(f"  ❌ {file_path} - No infinite loop")
                continue
            
            # Check for restart logic
            if 'restart_count' not in content and 'RESTARTING' not in content:
                results.append((file_path, False, "No restart logic"))
                print(f"  ⚠️  {file_path} - No restart logic")
            
            # Check for error handling
            if 'KeyboardInterrupt' not in content:
                results.append((file_path, False, "No KeyboardInterrupt handling"))
                print(f"  ⚠️  {file_path} - No KeyboardInterrupt handling")
            
            results.append((file_path, True, None))
            print(f"  ✅ {file_path} - Has infinite loop structure")
            
        except Exception as e:
            results.append((file_path, False, str(e)))
            print(f"  ❌ {file_path} - {e}")
    
    passed = sum(1 for r in results if r[1])
    total = len(results)
    print(f"\n✅ Passed: {passed}/{total}")
    return passed == total


def test_nae_structure():
    """Test NAE specific structure"""
    print("\n" + "=" * 70)
    print("Test 3: NAE Main Loop Structure")
    print("=" * 70)
    
    try:
        from ralph_github_continuous import run_forever_with_restart, continuous_research_loop
        import inspect
        
        source = inspect.getsource(run_forever_with_restart)
        
        checks = [
            ('while True:', 'Infinite loop'),
            ('continuous_research_loop()', 'Calls main loop'),
            ('restart_count', 'Restart counter'),
            ('KeyboardInterrupt', 'Keyboard interrupt handling'),
            ('SystemExit', 'System exit handling'),
            ('NEVER EXIT', 'Never exit comment'),
        ]
        
        all_pass = True
        for check_str, check_name in checks:
            if check_str in source:
                print(f"  ✅ {check_name}")
            else:
                print(f"  ❌ {check_name} - Missing")
                all_pass = False
        
        print(f"\n{'✅' if all_pass else '❌'} NAE structure check")
        return all_pass
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("NAE Agents Never-Stop Test Suite")
    print("=" * 70)
    
    test1 = test_agent_imports()
    test2 = test_infinite_loops()
    test3 = test_nae_structure()
    
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    print(f"Test 1 (Imports): {'✅ PASSED' if test1 else '❌ FAILED'}")
    print(f"Test 2 (Infinite Loops): {'✅ PASSED' if test2 else '❌ FAILED'}")
    print(f"Test 3 (NAE Structure): {'✅ PASSED' if test3 else '❌ FAILED'}")
    print("=" * 70)
    
    if test1 and test2 and test3:
        print("\n✅ ALL TESTS PASSED - Agents configured to never stop!")
        return 0
    else:
        print("\n❌ SOME TESTS FAILED - Review agent implementations")
        return 1


if __name__ == "__main__":
    sys.exit(main())

