#!/usr/bin/env python3
"""
Schedule NAE Master Scheduler restart at a specific time
"""

import os
import sys
import time
import subprocess
import datetime
from datetime import timedelta

# Try to import pytz, fallback to manual EST offset if not available
try:
    import pytz
    USE_PYTZ = True
except ImportError:
    USE_PYTZ = False

def get_est_time():
    """Get current EST time"""
    if USE_PYTZ:
        est = pytz.timezone('US/Eastern')
        return datetime.datetime.now(est)
    else:
        # Manual EST offset (UTC-5 for EST, UTC-4 for EDT)
        # For simplicity, assume EST (UTC-5)
        utc_now = datetime.datetime.utcnow()
        est_offset = timedelta(hours=-5)
        return utc_now + est_offset

def restart_scheduler_at_time(target_hour: int, target_minute: int, timezone_str: str = 'US/Eastern'):
    """
    Schedule a restart of the NAE master scheduler at a specific time
    
    Args:
        target_hour: Hour (24-hour format)
        target_minute: Minute
        timezone_str: Timezone string (default: US/Eastern)
    """
    print("="*80)
    print("NAE SCHEDULER RESTART SCHEDULER")
    print("="*80)
    
    # Calculate target time in EST
    if USE_PYTZ:
        tz = pytz.timezone(timezone_str)
        now = datetime.datetime.now(tz)
        target_time = now.replace(hour=target_hour, minute=target_minute, second=0, microsecond=0)
    else:
        now = get_est_time()
        target_time = now.replace(hour=target_hour, minute=target_minute, second=0, microsecond=0)
    
    # If target time has already passed today, schedule for tomorrow
    if now.time() >= target_time.time():
        target_time = target_time + datetime.timedelta(days=1)
        print(f"⚠️  Target time already passed today. Scheduling for tomorrow.")
    
    print(f"\n📅 SCHEDULE:")
    print(f"   Current time (EST): {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Restart time (EST): {target_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Calculate wait time
    wait_seconds = (target_time - now).total_seconds()
    wait_hours = int(wait_seconds // 3600)
    wait_minutes = int((wait_seconds % 3600) // 60)
    
    print(f"   Wait time: {wait_hours} hours, {wait_minutes} minutes")
    print(f"   Total seconds: {int(wait_seconds)}")
    
    # Stop current scheduler
    print("\n🛑 STOPPING CURRENT SCHEDULER...")
    try:
        result = subprocess.run(['pkill', '-f', 'nae_master_scheduler'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("   ✅ Scheduler stopped")
        else:
            print("   ⚠️  No running scheduler found (this is OK)")
    except Exception as e:
        print(f"   ⚠️  Error stopping scheduler: {e}")
    
    # Wait until target time
    print(f"\n⏳ WAITING UNTIL {target_time.strftime('%H:%M:%S %Z')}...")
    print("   (Press Ctrl+C to cancel)")
    
    try:
        time.sleep(wait_seconds)
    except KeyboardInterrupt:
        print("\n\n❌ Restart cancelled by user")
        return False
    
    # Restart scheduler
    print("\n🚀 RESTARTING SCHEDULER...")
    
    nae_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    scheduler_script = os.path.join(nae_dir, "nae_master_scheduler.py")
    log_file = os.path.join(nae_dir, "logs", "scheduler.log")
    
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    command = f"nohup python3 {scheduler_script} > {log_file} 2>&1 &"
    
    try:
        subprocess.run(command, shell=True, check=True, cwd=nae_dir)
        print(f"   ✅ Scheduler restarted!")
        print(f"   📋 Logs: {log_file}")
        print(f"   💡 Monitor with: tail -f {log_file}")
        
        return True
    except Exception as e:
        print(f"   ❌ Error restarting scheduler: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) >= 3:
        hour = int(sys.argv[1])
        minute = int(sys.argv[2])
        timezone = sys.argv[3] if len(sys.argv) > 3 else 'US/Eastern'
        
        restart_scheduler_at_time(hour, minute, timezone)
    else:
        print("Usage: python3 schedule_restart.py <hour> <minute> [timezone]")
        print("Example: python3 schedule_restart.py 13 45 US/Eastern")
        print("\nRestarting at 1:45pm EST...")
        restart_scheduler_at_time(13, 45, 'US/Eastern')

