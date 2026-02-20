# update_all_agents_goals.py
"""
Update all agents to use the new goal system with owner controls
"""

import os
import glob

def update_agent_file(file_path):
    """Update a single agent file to use the new goal system"""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Check if already updated
        if 'from goal_manager import get_nae_goals' in content:
            print(f"  {file_path} - Already updated")
            return True
        
        # Find the old GOALS definition
        old_goals_patterns = [
            'GOALS = [\n    "Achieve generational wealth",\n    "Generate $5,000,000 EVERY 8 years",\n    "Optimize NAE and agents for successful options trading"\n]',
            'GOALS = [\n    "Achieve generational wealth",\n    "Generate $5,000,000 EVERY 8 years",\n    "Optimize NAE and agents for successful options trading"\n]',
            'GOALS = ["Achieve generational wealth", "Generate $5,000,000 EVERY 8 years", "Optimize NAE and agents for successful options trading"]'
        ]
        
        updated = False
        for pattern in old_goals_patterns:
            if pattern in content:
                # Replace with new goal system
                new_content = content.replace(pattern, '''# Goals managed by GoalManager
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from goal_manager import get_nae_goals
GOALS = get_nae_goals()''')
                
                with open(file_path, 'w') as f:
                    f.write(new_content)
                
                print(f"  {file_path} - Updated successfully")
                updated = True
                break
        
        if not updated:
            print(f"  {file_path} - No old goals pattern found")
        
        return updated
        
    except Exception as e:
        print(f"  {file_path} - Error: {e}")
        return False

def main():
    print("🔄 Updating all agents to use new goal system...")
    
    # Find all agent files
    agent_files = []
    
    # Main agents directory
    main_agents = glob.glob("agents/*.py")
    agent_files.extend(main_agents)
    
    # Generated scripts directory
    generated_agents = glob.glob("agents/generated_scripts/*.py")
    agent_files.extend(generated_agents)
    
    print(f"Found {len(agent_files)} agent files to update")
    
    updated_count = 0
    for file_path in agent_files:
        if update_agent_file(file_path):
            updated_count += 1
    
    print(f"\n✅ Update complete!")
    print(f"Updated {updated_count} out of {len(agent_files)} files")
    
    # Test the goal system
    print("\n🧪 Testing goal system...")
    try:
        from goal_manager import get_nae_goals, get_goal_status
        goals = get_nae_goals()
        status = get_goal_status()
        
        print("Current goals:")
        for i, goal in enumerate(goals, 1):
            print(f"  {i}. {goal}")
        
        print(f"\nGoal #2 active: {status['goal_2_active']}")
        print(f"Owner set: {status['owner_set']}")
        
    except Exception as e:
        print(f"Error testing goal system: {e}")

if __name__ == "__main__":
    main()


