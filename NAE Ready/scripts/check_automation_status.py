#!/usr/bin/env python3
"""
Check NAE Continuous Automation Status
"""

import sys
import os
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

def check_status():
    """Check if continuous automation is running"""
    
    print("=" * 80)
    print("NAE Continuous Automation Status Check")
    print("=" * 80)
    print()
    
    # Check if process is running
    try:
        result = subprocess.run(
            ["ps", "aux"], 
            capture_output=True, 
            text=True
        )
        
        if "nae_continuous_automation.py" in result.stdout:
            print("✅ Continuous Automation Daemon: RUNNING")
            for line in result.stdout.split('\n'):
                if "nae_continuous_automation.py" in line and "grep" not in line:
                    parts = line.split()
                    if len(parts) > 1:
                        print(f"   Process ID: {parts[1]}")
                        print(f"   CPU: {parts[2]}%")
                        print(f"   Memory: {parts[3]}%")
                    break
        else:
            print("❌ Continuous Automation Daemon: NOT RUNNING")
            print("   Start with: python3 nae_continuous_automation.py")
    except Exception as e:
        print(f"⚠️  Could not check process status: {e}")
    
    print()
    
    # Check Optimus status
    print("Optimus Agent Status:")
    try:
        from agents.optimus import OptimusAgent
        
        optimus = OptimusAgent(sandbox=False)
        status = optimus.get_trading_status()
        
        nav = status.get("nav", 0)
        daily_pnl = status.get("daily_pnl", 0)
        open_positions = status.get("open_positions", 0)
        goal_progress = (nav / optimus.target_goal) * 100
        
        print(f"   ✅ NAV: ${nav:.2f}")
        print(f"   ✅ Daily P&L: ${daily_pnl:.2f}")
        print(f"   ✅ Open Positions: {open_positions}")
        print(f"   ✅ Trading Enabled: {status.get('trading_enabled', False)}")
        print(f"   ✅ Goal Progress: {goal_progress:.4f}% toward $5M")
        print(f"   ✅ Current Phase: {optimus.current_phase}")
    except Exception as e:
        print(f"   ❌ Error checking Optimus: {e}")
    
    print()
    
    # Check feedback loop data
    print("Feedback Loop Status:")
    feedback_dir = Path("data/feedback_loop")
    if feedback_dir.exists():
        performance_files = list(feedback_dir.glob("performance_*.json"))
        recommendation_files = list(feedback_dir.glob("recommendations_*.json"))
        
        print(f"   ✅ Data directory exists")
        print(f"   ✅ Performance files: {len(performance_files)}")
        print(f"   ✅ Recommendation files: {len(recommendation_files)}")
        
        if performance_files:
            latest = sorted(performance_files)[-1]
            print(f"   ✅ Latest performance data: {latest.name}")
        
        if recommendation_files:
            latest = sorted(recommendation_files)[-1]
            print(f"   ✅ Latest recommendations: {latest.name}")
    else:
        print(f"   ⚠️  Data directory not created yet (will be created on first feedback cycle)")
    
    print()
    
    # Check recent logs
    print("Recent Activity (last 5 log entries):")
    try:
        log_files = [
            "logs/optimus.log",
            "logs/casey.log",
            "logs/splinter.log"
        ]
        
        for log_file in log_files:
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                    if lines:
                        print(f"\n   {log_file}:")
                        for line in lines[-3:]:
                            print(f"      {line.strip()}")
    except Exception as e:
        print(f"   ⚠️  Could not read logs: {e}")
    
    print()
    print("=" * 80)
    print("Status Check Complete")
    print("=" * 80)
    print()
    print("To start automation: python3 nae_continuous_automation.py")
    print("To stop automation: Press Ctrl+C or kill the process")


if __name__ == "__main__":
    check_status()

