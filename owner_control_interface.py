# owner_control_interface.py
"""
Owner Control Interface for NAE Goal Management

This interface allows the owner to control goal #2 and manage the system.
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))
from goal_manager import (
    set_owner, verify_owner, stop_goal_2, restart_goal_2, 
    get_goal_status, get_stop_command_history
)

def main():
    print("🔐 NAE Owner Control Interface")
    print("=" * 50)
    
    # Check current status
    status = get_goal_status()
    print(f"\n📊 Current Status:")
    print(f"Goal #2 Active: {status['goal_2_active']}")
    print(f"Owner Set: {status['owner_set']}")
    print(f"Stop Commands: {status['stop_commands_count']}")
    
    if not status['owner_set']:
        print(f"\n⚠️  No owner set yet!")
        owner_id = input("Enter your owner identifier (this will be permanent): ").strip()
        if owner_id:
            if set_owner(owner_id):
                print("✅ Owner set successfully!")
            else:
                print("❌ Failed to set owner")
                return
        else:
            print("❌ No owner identifier provided")
            return
    
    # Owner verification
    print(f"\n🔑 Owner Verification Required:")
    owner_id = input("Enter your owner identifier: ").strip()
    
    if not verify_owner(owner_id):
        print("❌ Owner verification failed!")
        return
    
    print("✅ Owner verified!")
    
    # Show current goals
    print(f"\n📋 Current Goals:")
    goals = get_goal_status()['goals']
    for i, goal in enumerate(goals, 1):
        print(f"  {i}. {goal}")
    
    # Control menu
    while True:
        print(f"\n🎛️  Owner Control Menu:")
        print("1. Stop Goal #2")
        print("2. Restart Goal #2")
        print("3. View Command History")
        print("4. View Current Status")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == "1":
            reason = input("Enter reason for stopping Goal #2: ").strip()
            if stop_goal_2(owner_id, reason):
                print("✅ Goal #2 stopped successfully!")
            else:
                print("❌ Failed to stop Goal #2")
        
        elif choice == "2":
            reason = input("Enter reason for restarting Goal #2: ").strip()
            if restart_goal_2(owner_id, reason):
                print("✅ Goal #2 restarted successfully!")
            else:
                print("❌ Failed to restart Goal #2")
        
        elif choice == "3":
            history = get_stop_command_history(owner_id)
            if history:
                print(f"\n📜 Command History:")
                for i, cmd in enumerate(history, 1):
                    print(f"  {i}. {cmd['timestamp']} - {cmd.get('reason', 'No reason')}")
            else:
                print("No command history available")
        
        elif choice == "4":
            status = get_goal_status()
            print(f"\n📊 Current Status:")
            print(f"Goal #2 Active: {status['goal_2_active']}")
            print(f"Owner Set: {status['owner_set']}")
            print(f"Stop Commands: {status['stop_commands_count']}")
            print(f"Last Updated: {status['last_updated']}")
        
        elif choice == "5":
            print("👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice")

if __name__ == "__main__":
    main()


