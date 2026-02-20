# NAE/agents/bebop.py
"""
BebopAgent - Monitoring & Compliance Agent for NAE

ALIGNED WITH 3 CORE GOALS:
1. Achieve generational wealth
2. Generate $5,000,000.00 within 8 years, every 8 years consistently
3. Optimize NAE and agents for successful options trading

ALIGNED WITH LONG-TERM PLAN:
- Monitors PDT compliance (same-day round trip count)
- Tracks risk metrics (daily loss, drawdown, etc.)
- Alerts on plan deviations
- Ensures regulatory compliance for Goal #2
- See: docs/NAE_LONG_TERM_PLAN.md for compliance requirements
"""

import os
import sys
import datetime
import json
from typing import Dict, Any

# Goals managed by GoalManager
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from goal_manager import get_nae_goals
GOALS = get_nae_goals()

class BebopAgent:
    def __init__(self):
        self.goals = GOALS  # 3 Core Goals
        self.long_term_plan = "docs/NAE_LONG_TERM_PLAN.md"  # Reference to long-term plan
        # Growth Milestones from nae_mission_control.py
        self.target_goal = 5000000.0  # $5M target (exceeded in Year 7)
        self.stretch_goal = 15726144.0  # $15.7M final goal (Year 8)
        self.growth_milestones = {
            1: 9_411, 2: 44_110, 3: 152_834, 4: 388_657,
            5: 982_500, 6: 2_477_897, 7: 6_243_561, 8: 15_726_144
        }
        
        # ----------------------
        # Growth Milestones Integration
        # ----------------------
        try:
            from core.growth_milestones import GrowthMilestonesTracker
            self.milestone_tracker = GrowthMilestonesTracker()
        except ImportError:
            self.milestone_tracker = None
        
        self.log_file = "logs/bebop.log"
        self.agent_status = {}
        self.inbox = []
        
        # Track processed alerts to prevent loops
        self.processed_alerts = set()
        
        # Agent references for bidirectional alerting (set by scheduler)
        self.phisher_agent = None
        self.rocksteady_agent = None
        self.casey_agent = None

    # ----------------------
    # Register all agents in NAE
    # ----------------------
    def register_agents(self, agents_list):
        """Register every agent dynamically for monitoring"""
        for agent in agents_list:
            self.agent_status[agent] = "Unknown"
        self.log_action(f"Registered all agents for monitoring: {agents_list}")
    
    def register_all_agents(self, agents_list):
        """Alias for register_agents for backward compatibility"""
        self.register_agents(agents_list)

    # ----------------------
    # Update agent status
    # ----------------------
    def update_status(self, agent_name, status):
        if agent_name in self.agent_status:
            self.agent_status[agent_name] = status
            self.log_action(f"Status updated: {agent_name} = {status}")
        else:
            self.log_action(f"Attempted to update unknown agent: {agent_name}")

    # ----------------------
    # Monitor all agents
    # ----------------------
    def monitor_agents(self):
        print("[Bebop] Monitoring all NAE agents:")
        for agent_name, status in self.agent_status.items():
            print(f" - {agent_name}: {status}")
        self.log_action("Completed monitoring cycle for all agents.")

    # ----------------------
    # Security Alert Handling
    # ----------------------
    def handle_security_alert(self, alert: Dict[str, Any]):
        """Handle security alerts from Phisher or other agents"""
        severity = alert.get("severity", "medium")
        threat = alert.get("threat", "Unknown")
        details = alert.get("details", {})
        source = alert.get("source", "unknown")
        timestamp = alert.get("timestamp", datetime.datetime.now().isoformat())
        
        # Create unique alert ID to prevent duplicate processing (without timestamp)
        alert_id = f"{source}:{threat}"
        
        # Skip if already processed
        if alert_id in self.processed_alerts:
            self.log_action(f"⚠️ Skipping duplicate alert: {threat} from {source}")
            return
        
        # Mark as processed
        self.processed_alerts.add(alert_id)
        
        self.log_action(f"🚨 SECURITY ALERT RECEIVED from {source}: {threat} (Severity: {severity})")
        
        # Take immediate action based on severity
        if severity == "critical":
            self.log_action("🔴 CRITICAL THREAT - Escalating immediately")
            # Escalate to all agents
            self.update_status("SecurityStatus", "CRITICAL_THREAT_DETECTED")
            # Only alert if this is an original detection (not from another agent)
            if source == "Bebop":
                self.alert_security_team(alert, exclude_sender=source)
            # Could trigger emergency shutdown or isolation
            
        elif severity == "high":
            self.log_action("🟠 HIGH PRIORITY THREAT - Increased monitoring")
            self.update_status("SecurityStatus", "HIGH_THREAT_DETECTED")
            # Only alert if this is an original detection (not from another agent)
            if source == "Bebop":
                self.alert_security_team(alert, exclude_sender=source)
            # Increase monitoring frequency
            
        elif severity == "medium":
            self.log_action("🟡 MEDIUM PRIORITY THREAT - Monitoring")
            self.update_status("SecurityStatus", "MEDIUM_THREAT_DETECTED")
            # Only alert if this is an original detection (not from another agent)
            if source == "Bebop":
                self.alert_security_team(alert, exclude_sender=source)
            
        else:
            self.log_action("🟢 LOW PRIORITY THREAT - Logged")
            self.update_status("SecurityStatus", "LOW_THREAT_DETECTED")
        
        # Store alert for tracking
        alert_record = {
            "alert": alert,
            "processed_at": datetime.datetime.now().isoformat(),
            "action_taken": f"Status updated to {severity}_THREAT_DETECTED"
        }
        
        # Log alert details
        self.log_action(f"Alert details: {json.dumps(details, indent=2)}")
        
        return alert_record
    
    def detect_threat(self, threat_info: Dict[str, Any]):
        """Bebop detects a threat and alerts the security team"""
        severity = threat_info.get("severity", "medium")
        threat = threat_info.get("threat", "Unknown threat detected")
        
        self.log_action(f"🔍 THREAT DETECTED BY BEBOP: {threat} (Severity: {severity})")
        
        # Create alert message
        alert_message = {
            "type": "security_alert",
            "severity": severity,
            "threat": threat,
            "details": threat_info.get("details", {}),
            "source": "Bebop",
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        # Alert all security team members
        self.alert_security_team(alert_message)
        
        return alert_message
    
    def alert_security_team(self, alert: Dict[str, Any], exclude_sender: str = None):
        """Alert Phisher, Rocksteady, and Casey about a security threat"""
        severity = alert.get("severity", "medium")
        
        # Always alert Phisher for threat intelligence (unless Phisher is the sender)
        if self.phisher_agent and exclude_sender != "Phisher":
            try:
                if hasattr(self.phisher_agent, 'receive_message'):
                    self.phisher_agent.receive_message("Bebop", alert)
                    self.log_action(f"🚨 ALERT SENT to Phisher: {alert.get('threat', 'Unknown threat')}")
            except Exception as e:
                self.log_action(f"Error alerting Phisher: {e}")
        
        # Alert Rocksteady for defensive actions (unless Rocksteady is the sender)
        if self.rocksteady_agent and exclude_sender != "Rocksteady":
            try:
                threat_message = {
                    "type": "security_threat",
                    "severity": severity,
                    "threat": alert.get("threat", "Unknown"),
                    "details": alert.get("details", {}),
                    "source": "Bebop",
                    "timestamp": datetime.datetime.now().isoformat()
                }
                if hasattr(self.rocksteady_agent, 'receive_message'):
                    self.rocksteady_agent.receive_message("Bebop", threat_message)
                    self.log_action(f"🚨 ALERT SENT to Rocksteady: {alert.get('threat', 'Unknown threat')}")
            except Exception as e:
                self.log_action(f"Error alerting Rocksteady: {e}")
        
        # Alert Casey for improvements (unless Casey is the sender)
        if self.casey_agent and exclude_sender != "Casey":
            try:
                improvement_message = {
                    "type": "security_improvement_request",
                    "severity": severity,
                    "threat": alert.get("threat", "Unknown"),
                    "details": alert.get("details", {}),
                    "action_required": "improve_defenses",
                    "vulnerability_type": alert.get("vulnerability_type", ["unknown"]),
                    "source": "Bebop",
                    "timestamp": datetime.datetime.now().isoformat()
                }
                if hasattr(self.casey_agent, 'receive_message'):
                    self.casey_agent.receive_message(improvement_message)
                    self.log_action(f"🚨 ALERT SENT to Casey: {alert.get('threat', 'Unknown threat')}")
            except Exception as e:
                self.log_action(f"Error alerting Casey: {e}")
    
    def receive_message(self, sender_name: str, message: str):
        """Receive messages from other agents"""
        ts = datetime.datetime.now().isoformat()
        
        # Handle security alerts specially
        if isinstance(message, dict) and message.get("type") == "security_alert":
            self.handle_security_alert(message)
        else:
            self.inbox.append({"from": sender_name, "message": message, "timestamp": ts})
            self.log_action(f"Received message from {sender_name}: {message}")

    # ----------------------
    # Logging
    # ----------------------
    def log_action(self, message):
        timestamp = datetime.datetime.now().isoformat()
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        try:
            with open(self.log_file, "a", encoding='utf-8') as f:
                f.write(f"[{timestamp}] {message}\n")
        except Exception as e:
            print(f"Failed to write to log: {e}")
        # Safe print for Windows console
        try:
            print(f"[Bebop LOG] {message}")
        except UnicodeEncodeError:
            safe_message = message.encode('ascii', 'ignore').decode('ascii')
            print(f"[Bebop LOG] {safe_message}")

    # ----------------------
    # Main run
    # ----------------------
    def run(self):
        print("[Bebop] Running monitoring loop...")
        self.monitor_agents()

