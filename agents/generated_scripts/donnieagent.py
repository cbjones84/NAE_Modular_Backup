# Auto-generated/refined agent by Casey
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from goal_manager import get_nae_goals
GOALS = get_nae_goals()

class DonnieAgent:
    def __init__(self, goals=None):
        self.goals = goals if goals else GOALS
        self.log_file = "logs/donnieagent.log"
        import os, datetime
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        self.log_action("DonnieAgent initialized.")

    def log_action(self, message):
        ts = datetime.datetime.now().isoformat()
        with open(self.log_file, "a") as f:
            f.write(f"[{ts}] {message}\n")
        print(f"[DonnieAgent LOG] {message}")

    def run(self):
        self.log_action("DonnieAgent running...")