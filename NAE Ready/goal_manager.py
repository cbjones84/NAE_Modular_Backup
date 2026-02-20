# goal_manager.py
"""
NAE Goal Management System with Owner Controls

This module manages goals across all agents and provides owner-only controls
for stopping the $5M generation goal.

GROWTH MILESTONES (from nae_mission_control.py):
Year 1:  $9,411      (+$6,911 returns)
Year 2:  $44,110     (+$39,210 returns)
Year 3:  $152,834    (+$145,534 returns)
Year 4:  $388,657    (+$378,957 returns)
Year 5:  $982,500    (+$970,400 returns)
Year 6:  $2,477,897  (+$2,463,397 returns)
Year 7:  $6,243,561  [TARGET EXCEEDED!]
Year 8:  $15,726,144 [FINAL: $15.7M]

TRADING MODE: LIVE (Tradier ONLY)
AGGRESSIVENESS: VERY_AGGRESSIVE (1.5x position sizing)
"""

import os
import json
import hashlib
import datetime
from typing import List, Dict, Optional

# Growth Milestones - THE OFFICIAL TARGETS
GROWTH_MILESTONES = {
    1: 9_411,
    2: 44_110,
    3: 152_834,
    4: 388_657,
    5: 982_500,
    6: 2_477_897,
    7: 6_243_561,
    8: 15_726_144
}

# Returns per year
GROWTH_RETURNS = {
    1: 6_911,
    2: 39_210,
    3: 145_534,
    4: 378_957,
    5: 970_400,
    6: 2_463_397,
    7: 6_243_561,  # Exceeds original $5M target
    8: 15_706_844  # Final stretch goal
}

# Primary targets
STARTING_CAPITAL = 100.0
BIWEEKLY_DEPOSIT = 100.0
TARGET_GOAL = 5_000_000.0  # Original target (exceeded in Year 7)
STRETCH_GOAL = 15_726_144.0  # Final 8-year goal
TARGET_YEARS = 8


class GoalManager:
    """Centralized goal management with owner controls and growth milestones"""
    
    def __init__(self):
        self.config_file = "config/goal_manager.json"
        self.owner_id = None
        self.goals = [
            "Achieve generational wealth through VERY_AGGRESSIVE LIVE trading",
            f"Generate ${TARGET_GOAL:,.2f} within {TARGET_YEARS} years (stretch goal: ${STRETCH_GOAL:,.2f}), every 8 years consistently until commanded to stop by owner, and only the owner",
            "Optimize NAE and agents for successful options trading via Tradier"
        ]
        self.milestones = GROWTH_MILESTONES.copy()
        self.returns = GROWTH_RETURNS.copy()
        self.goal_2_active = True  # ACTIVE by default
        self.goal_2_stop_commands = []
        self._load_config()
    
    def _load_config(self):
        """Load configuration from file"""
        try:
            if os.path.exists(self.config_file):
                os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    self.owner_id = config.get('owner_id')
                    self.goals = config.get('goals', self.goals)
                    self.milestones = config.get('milestones', self.milestones)
                    self.returns = config.get('returns', self.returns)
                    self.goal_2_active = config.get('goal_2_active', True)
                    self.goal_2_stop_commands = config.get('goal_2_stop_commands', [])
        except Exception as e:
            print(f"[GoalManager] Error loading config: {e}")
            self._save_config()
    
    def _save_config(self):
        """Save configuration to file"""
        try:
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            config = {
                'owner_id': self.owner_id,
                'goals': self.goals,
                'milestones': self.milestones,
                'returns': self.returns,
                'target_goal': TARGET_GOAL,
                'stretch_goal': STRETCH_GOAL,
                'starting_capital': STARTING_CAPITAL,
                'biweekly_deposit': BIWEEKLY_DEPOSIT,
                'target_years': TARGET_YEARS,
                'goal_2_active': self.goal_2_active,
                'goal_2_stop_commands': self.goal_2_stop_commands,
                'trading_mode': 'LIVE',
                'broker': 'tradier',
                'aggressiveness': 'VERY_AGGRESSIVE',
                'last_updated': datetime.datetime.now().isoformat()
            }
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            print(f"[GoalManager] Error saving config: {e}")
    
    def get_milestone(self, year: int) -> float:
        """Get milestone target for a specific year"""
        return self.milestones.get(year, 0)
    
    def get_return(self, year: int) -> float:
        """Get expected return for a specific year"""
        return self.returns.get(year, 0)
    
    def get_all_milestones(self) -> Dict[int, float]:
        """Get all growth milestones"""
        return self.milestones.copy()
    
    def get_milestone_status(self, nav: float, year: int) -> Dict:
        """Get milestone status for current year"""
        milestone = self.get_milestone(year)
        progress = (nav / milestone * 100) if milestone > 0 else 0
        on_track = progress >= 80  # Consider on track if at least 80%
        
        return {
            'year': year,
            'nav': nav,
            'milestone': milestone,
            'progress_pct': progress,
            'on_track': on_track,
            'gap': milestone - nav,
            'target_goal': TARGET_GOAL,
            'stretch_goal': STRETCH_GOAL
        }
    
    def set_owner(self, owner_identifier: str) -> bool:
        """Set the owner identifier (can only be done once)"""
        if self.owner_id is None:
            self.owner_id = self._hash_identifier(owner_identifier)
            self._save_config()
            print(f"[GoalManager] Owner set successfully")
            return True
        else:
            print(f"[GoalManager] Owner already set. Cannot change owner.")
            return False
    
    def verify_owner(self, owner_identifier: str) -> bool:
        """Verify if the provided identifier matches the owner"""
        if self.owner_id is None:
            print(f"[GoalManager] No owner set yet")
            return False
        
        provided_hash = self._hash_identifier(owner_identifier)
        is_owner = provided_hash == self.owner_id
        
        if is_owner:
            print(f"[GoalManager] Owner verification successful")
        else:
            print(f"[GoalManager] Owner verification failed")
        
        return is_owner
    
    def _hash_identifier(self, identifier: str) -> str:
        """Create a secure hash of the owner identifier"""
        return hashlib.sha256(identifier.encode()).hexdigest()
    
    def get_goals(self) -> List[str]:
        """Get current goals (with goal 2 status)"""
        goals = self.goals.copy()
        if not self.goal_2_active:
            goals[1] = f"Generate ${TARGET_GOAL:,.2f} within {TARGET_YEARS} years - STOPPED BY OWNER"
        return goals
    
    def stop_goal_2(self, owner_identifier: str, reason: str = "") -> bool:
        """Stop goal #2 (only owner can do this)"""
        if not self.verify_owner(owner_identifier):
            print(f"[GoalManager] UNAUTHORIZED: Only owner can stop goal #2")
            return False
        
        if not self.goal_2_active:
            print(f"[GoalManager] Goal #2 is already stopped")
            return True
        
        self.goal_2_active = False
        stop_command = {
            "timestamp": datetime.datetime.now().isoformat(),
            "reason": reason,
            "owner_verified": True
        }
        self.goal_2_stop_commands.append(stop_command)
        self._save_config()
        
        print(f"[GoalManager] Goal #2 STOPPED by owner at {stop_command['timestamp']}")
        print(f"[GoalManager] Reason: {reason if reason else 'No reason provided'}")
        return True
    
    def restart_goal_2(self, owner_identifier: str, reason: str = "") -> bool:
        """Restart goal #2 (only owner can do this)"""
        if not self.verify_owner(owner_identifier):
            print(f"[GoalManager] UNAUTHORIZED: Only owner can restart goal #2")
            return False
        
        if self.goal_2_active:
            print(f"[GoalManager] Goal #2 is already active")
            return True
        
        self.goal_2_active = True
        restart_command = {
            "timestamp": datetime.datetime.now().isoformat(),
            "reason": reason,
            "owner_verified": True
        }
        self.goal_2_stop_commands.append(restart_command)
        self._save_config()
        
        print(f"[GoalManager] Goal #2 RESTARTED by owner at {restart_command['timestamp']}")
        print(f"[GoalManager] Reason: {reason if reason else 'No reason provided'}")
        return True
    
    def get_goal_status(self) -> Dict:
        """Get current goal status"""
        return {
            "goals": self.get_goals(),
            "milestones": self.milestones,
            "target_goal": TARGET_GOAL,
            "stretch_goal": STRETCH_GOAL,
            "goal_2_active": self.goal_2_active,
            "owner_set": self.owner_id is not None,
            "stop_commands_count": len(self.goal_2_stop_commands),
            "trading_mode": "LIVE",
            "broker": "tradier",
            "aggressiveness": "VERY_AGGRESSIVE",
            "last_updated": datetime.datetime.now().isoformat()
        }
    
    def get_stop_command_history(self, owner_identifier: str) -> List[Dict]:
        """Get history of stop/restart commands (owner only)"""
        if not self.verify_owner(owner_identifier):
            print(f"[GoalManager] UNAUTHORIZED: Only owner can view command history")
            return []
        
        return self.goal_2_stop_commands.copy()


# Global goal manager instance
goal_manager = GoalManager()

def get_nae_goals() -> List[str]:
    """Get current NAE goals"""
    return goal_manager.get_goals()

def get_growth_milestones() -> Dict[int, float]:
    """Get all growth milestones"""
    return goal_manager.get_all_milestones()

def get_milestone(year: int) -> float:
    """Get milestone for specific year"""
    return goal_manager.get_milestone(year)

def get_milestone_status(nav: float, year: int) -> Dict:
    """Get milestone status"""
    return goal_manager.get_milestone_status(nav, year)

def set_owner(owner_identifier: str) -> bool:
    """Set the owner (can only be done once)"""
    return goal_manager.set_owner(owner_identifier)

def verify_owner(owner_identifier: str) -> bool:
    """Verify owner identity"""
    return goal_manager.verify_owner(owner_identifier)

def stop_goal_2(owner_identifier: str, reason: str = "") -> bool:
    """Stop goal #2 (owner only)"""
    return goal_manager.stop_goal_2(owner_identifier, reason)

def restart_goal_2(owner_identifier: str, reason: str = "") -> bool:
    """Restart goal #2 (owner only)"""
    return goal_manager.restart_goal_2(owner_identifier, reason)

def get_goal_status() -> Dict:
    """Get current goal status"""
    return goal_manager.get_goal_status()

def get_stop_command_history(owner_identifier: str) -> List[Dict]:
    """Get history of stop/restart commands (owner only)"""
    return goal_manager.get_stop_command_history(owner_identifier)
