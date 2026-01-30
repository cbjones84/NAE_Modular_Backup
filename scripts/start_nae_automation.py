#!/usr/bin/env python3
"""
Start NAE Master Automation Scheduler
Starts all agents with their scheduled intervals
"""

import sys
import os
import subprocess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def start_nae_automation(background=False):
    """Start NAE automation scheduler"""
    
    print("="*80)
    print("STARTING NAE AUTOMATION")
    print("="*80)
    
    # Check if scheduler exists
    scheduler_file = "nae_master_scheduler.py"
    if not os.path.exists(scheduler_file):
        print(f"❌ Error: {scheduler_file} not found")
        return False
    
    # Check if already running
    try:
        ps_result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        if 'nae_master_scheduler' in ps_result.stdout:
            print("⚠️  NAE scheduler appears to already be running")
            print("   Check with: ps aux | grep nae_master_scheduler")
            return False
    except:
        pass
    
    # Install schedule module if not available
    print("\n📦 Checking dependencies...")
    try:
        import schedule
        print("   ✅ schedule module available")
    except ImportError:
        print("   ⚠️  schedule module not installed")
        print("   Installing schedule module...")
        try:
            subprocess.run([sys.executable, '-m', 'pip', 'install', 'schedule'], check=True)
            print("   ✅ schedule module installed")
        except Exception as e:
            print(f"   ⚠️  Could not install schedule: {e}")
            print("   Will use fallback scheduling")
    
    # Start scheduler
    print("\n🚀 Starting NAE Master Scheduler...")
    
    if background:
        # Run in background
        log_file = "logs/scheduler.log"
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        with open(log_file, 'w') as f:
            f.write(f"NAE Master Scheduler started at {__import__('datetime').datetime.now()}\n")
        
        cmd = f"nohup {sys.executable} {scheduler_file} >> {log_file} 2>&1 &"
        print(f"   Running: {cmd}")
        subprocess.Popen(cmd, shell=True)
        
        print(f"\n✅ NAE automation started in background!")
        print(f"   Log file: {log_file}")
        print(f"   Check status: ps aux | grep nae_master_scheduler")
        print(f"   View logs: tail -f {log_file}")
    else:
        # Run in foreground
        print("\n✅ Starting scheduler in foreground...")
        print("   Press Ctrl+C to stop")
        print("="*80 + "\n")
        
        # Import and run
        from nae_master_scheduler import NAEMasterScheduler
        scheduler = NAEMasterScheduler()
        scheduler.start()
    
    return True

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Start NAE automation scheduler')
    parser.add_argument('--background', '-b', action='store_true', help='Run in background')
    args = parser.parse_args()
    
    try:
        start_nae_automation(background=args.background)
    except KeyboardInterrupt:
        print("\n\n❌ Startup cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error starting automation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

