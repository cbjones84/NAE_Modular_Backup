#!/usr/bin/env python3
"""
Growth Milestones Tracker - Track NAE progress toward $5M / $15.7M goals

Provides milestone tracking, progress monitoring, and phase detection
for NAE's 8-year growth plan from $100 to $15.7M.

GROWTH MILESTONES:
Year 1:  $9,411      (+$6,911 returns)
Year 2:  $44,110     (+$39,210 returns)
Year 3:  $152,834    (+$145,534 returns)
Year 4:  $388,657    (+$378,957 returns)
Year 5:  $982,500    (+$970,400 returns)
Year 6:  $2,477,897  (+$2,463,397 returns)
Year 7:  $6,243,561  [TARGET EXCEEDED!]
Year 8:  $15,726,144 [FINAL: $15.7M]
"""

import os
import json
import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path


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
    7: 6_243_561,
    8: 15_706_844
}

# Primary targets
STARTING_CAPITAL = 100.0
TARGET_GOAL = 5_000_000.0
STRETCH_GOAL = 15_726_144.0
TARGET_YEARS = 8


@dataclass
class MilestoneStatus:
    """Current milestone status"""
    current_nav: float
    current_year: int
    year_target: float
    progress_pct: float
    on_track: bool
    gap_to_target: float
    final_goal: float
    overall_progress_pct: float
    days_elapsed: int
    estimated_completion_year: Optional[float] = None


class GrowthMilestonesTracker:
    """
    Track NAE's progress toward growth milestones
    
    Monitors current NAV against yearly targets and provides
    status, phase detection, and projection capabilities.
    """
    
    def __init__(self, start_date: Optional[datetime.datetime] = None):
        """
        Initialize milestone tracker
        
        Args:
            start_date: When NAE started trading (defaults to Jan 6, 2026)
        """
        # Default start date: January 6, 2026 (when LIVE trading began)
        if start_date is None:
            self.start_date = datetime.datetime(2026, 1, 6)
        else:
            self.start_date = start_date
        
        self.milestones = GROWTH_MILESTONES.copy()
        self.returns = GROWTH_RETURNS.copy()
        self.starting_capital = STARTING_CAPITAL
        self.target_goal = TARGET_GOAL
        self.stretch_goal = STRETCH_GOAL
        self.target_years = TARGET_YEARS
        
        # Progress tracking file
        self.tracking_file = "logs/milestone_progress.json"
        os.makedirs(os.path.dirname(self.tracking_file), exist_ok=True)
    
    def get_current_year(self) -> int:
        """
        Calculate which year of the 8-year plan we're currently in
        
        Returns:
            Year number (1-8)
        """
        days_elapsed = (datetime.datetime.now() - self.start_date).days
        years_elapsed = days_elapsed / 365.25
        current_year = min(8, max(1, int(years_elapsed) + 1))
        return current_year
    
    def get_days_elapsed(self) -> int:
        """Get days since start"""
        return (datetime.datetime.now() - self.start_date).days
    
    def get_milestone_for_year(self, year: int) -> float:
        """Get milestone target for specific year"""
        return self.milestones.get(year, 0)
    
    def get_status(self, current_nav: float) -> MilestoneStatus:
        """
        Get current milestone status
        
        Args:
            current_nav: Current account NAV
        
        Returns:
            MilestoneStatus with progress info
        """
        current_year = self.get_current_year()
        year_target = self.get_milestone_for_year(current_year)
        
        # Progress toward current year's target
        progress_pct = (current_nav / year_target * 100) if year_target > 0 else 0
        
        # On track if >= 80% of year target or ahead of schedule
        on_track = progress_pct >= 80.0 or current_nav >= year_target
        
        # Gap to target (negative if ahead)
        gap_to_target = year_target - current_nav
        
        # Overall progress toward final goal
        overall_progress_pct = (current_nav / self.stretch_goal * 100) if self.stretch_goal > 0 else 0
        
        # Estimate completion year based on current growth rate
        days_elapsed = self.get_days_elapsed()
        estimated_year = None
        if days_elapsed > 0 and current_nav > self.starting_capital:
            # Calculate compound annual growth rate
            years_elapsed = days_elapsed / 365.25
            if years_elapsed > 0:
                cagr = (current_nav / self.starting_capital) ** (1 / years_elapsed) - 1
                if cagr > 0:
                    # Years needed to reach stretch goal at current rate
                    years_to_goal = (self.stretch_goal / current_nav) ** (1 / (1 + cagr)) - 1
                    estimated_year = years_elapsed + years_to_goal
        
        return MilestoneStatus(
            current_nav=current_nav,
            current_year=current_year,
            year_target=year_target,
            progress_pct=progress_pct,
            on_track=on_track,
            gap_to_target=gap_to_target,
            final_goal=self.stretch_goal,
            overall_progress_pct=overall_progress_pct,
            days_elapsed=days_elapsed,
            estimated_completion_year=estimated_year
        )
    
    def log_progress(self, current_nav: float, agent_name: str = "system") -> Dict:
        """
        Log progress to tracking file
        
        Args:
            current_nav: Current account NAV
            agent_name: Which agent is logging
        
        Returns:
            Progress data that was logged
        """
        status = self.get_status(current_nav)
        
        progress_data = {
            "timestamp": datetime.datetime.now().isoformat(),
            "agent": agent_name,
            "nav": current_nav,
            "year": status.current_year,
            "year_target": status.year_target,
            "progress_pct": status.progress_pct,
            "on_track": status.on_track,
            "gap": status.gap_to_target,
            "overall_progress_pct": status.overall_progress_pct,
            "days_elapsed": status.days_elapsed
        }
        
        try:
            # Append to tracking file
            with open(self.tracking_file, 'a') as f:
                f.write(json.dumps(progress_data) + '\n')
        except Exception as e:
            print(f"[GrowthMilestonesTracker] Error logging progress: {e}")
        
        return progress_data
    
    def get_phase(self, current_nav: float) -> str:
        """
        Get current growth phase based on NAV
        
        Args:
            current_nav: Current account NAV
        
        Returns:
            Phase name (Micro, Small, Growth, or Scale)
        """
        if current_nav < 1_000:
            return "Micro"
        elif current_nav < 10_000:
            return "Small"
        elif current_nav < 100_000:
            return "Growth"
        else:
            return "Scale"
    
    def is_on_track(self, current_nav: float, tolerance_pct: float = 0.80) -> Tuple[bool, str]:
        """
        Check if we're on track to hit milestones
        
        Args:
            current_nav: Current account NAV
            tolerance_pct: Tolerance for being "on track" (default 80% = 0.80)
        
        Returns:
            (on_track, message)
        """
        status = self.get_status(current_nav)
        
        if status.on_track:
            return True, f"On track: {status.progress_pct:.1f}% of Year {status.current_year} target"
        else:
            return False, f"Behind: {status.progress_pct:.1f}% of Year {status.current_year} target (need {tolerance_pct*100:.0f}%)"
    
    def get_all_milestones(self) -> Dict[int, float]:
        """Get all yearly milestones"""
        return self.milestones.copy()
    
    def get_summary(self, current_nav: float) -> Dict:
        """
        Get comprehensive summary of milestone progress
        
        Args:
            current_nav: Current account NAV
        
        Returns:
            Summary dict with all milestone info
        """
        status = self.get_status(current_nav)
        phase = self.get_phase(current_nav)
        on_track, track_message = self.is_on_track(current_nav)
        
        return {
            "nav": current_nav,
            "phase": phase,
            "year": status.current_year,
            "year_target": status.year_target,
            "progress_to_year": status.progress_pct,
            "on_track": on_track,
            "track_message": track_message,
            "gap_to_year_target": status.gap_to_target,
            "overall_progress": status.overall_progress_pct,
            "final_goal": status.final_goal,
            "days_elapsed": status.days_elapsed,
            "estimated_completion_year": status.estimated_completion_year,
            "all_milestones": self.milestones,
            "start_date": self.start_date.isoformat()
        }


# Convenience functions
def get_tracker() -> GrowthMilestonesTracker:
    """Get a global milestone tracker instance"""
    global _tracker
    if '_tracker' not in globals():
        _tracker = GrowthMilestonesTracker()
    return _tracker


def check_milestone_status(current_nav: float) -> MilestoneStatus:
    """Check milestone status for current NAV"""
    tracker = get_tracker()
    return tracker.get_status(current_nav)


def is_on_track(current_nav: float) -> bool:
    """Quick check if on track"""
    tracker = get_tracker()
    on_track, _ = tracker.is_on_track(current_nav)
    return on_track


# Module-level instance
_tracker: Optional[GrowthMilestonesTracker] = None
