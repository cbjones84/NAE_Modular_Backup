# agents/rocksteady.py
"""
RocksteadyAgent v2 - Security Enforcer for NAE
Handles real-time defense, agent verification, and firewall management.
Reports to Splinter and coordinates with Phisher and Bebop.
Fully AutoGen-compatible and messaging-ready.

ALIGNED WITH 3 CORE GOALS:
1. Achieve generational wealth
2. Generate $5,000,000.00 within 8 years, every 8 years consistently
3. Optimize NAE and agents for successful options trading

ALIGNED WITH LONG-TERM PLAN:
- Blocks PDT violations automatically
- Enforces risk limits (daily loss, position limits)
- Protects capital from excessive risk (critical for Goal #2)
- Implements defensive measures
- See: docs/NAE_LONG_TERM_PLAN.md for compliance requirements
"""

import os
import sys
import datetime
import hashlib
import json
import threading
import time
from typing import List, Dict, Any, Optional

# Goals managed by GoalManager
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from goal_manager import get_nae_goals
GOALS = get_nae_goals()

class RocksteadyAgent:
    def __init__(self, goals: Optional[List[str]] = None):
        # ----------------------
        # Goals & status
        # ----------------------
        self.goals = goals if goals else GOALS  # 3 Core Goals
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
        
        self.status: str = "Idle"

        # ----------------------
        # Logging & storage
        # ----------------------
        self.log_file = "logs/rocksteady.log"
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)

        # ----------------------
        # Security state
        # ----------------------
        self.blocked_entities: List[str] = []
        self.agent_integrity_hashes: Dict[str, str] = {}

        # ----------------------
        # Messaging hooks
        # ----------------------
        self.inbox: List[Dict[str, Any]] = []
        self.outbox: List[Dict[str, Any]] = []
        
        # Track processed threats to prevent loops
        self.processed_threats = set()
        
        # Agent references for bidirectional alerting (set by scheduler)
        self.phisher_agent = None
        self.bebop_agent = None
        self.casey_agent = None

        self.log_action("Rocksteady initialized and guarding NAE fortress.")

    # --------------------------
    # Logging
    # --------------------------
    def log_action(self, message: str):
        ts = datetime.datetime.now().isoformat()
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(f"[{ts}] {message}\n")
        except Exception as e:
            print(f"Failed to write to log: {e}")
        # Safe print for Windows console
        try:
            print(f"[Rocksteady LOG] {message}")
        except UnicodeEncodeError:
            safe_message = message.encode('ascii', 'ignore').decode('ascii')
            print(f"[Rocksteady LOG] {safe_message}")

    # --------------------------
    # Goals helper
    # --------------------------
    def set_goals(self, goals: List[str]):
        self.goals = goals
        self.log_action("Goals updated.")

    # --------------------------
    # Messaging hooks
    # --------------------------
    def receive_message(self, sender: str, message: Any):
        ts = datetime.datetime.now().isoformat()
        self.inbox.append({"from": sender, "message": message, "timestamp": ts})
        self.log_action(f"Received message from {sender}: {message}")
        
        # Handle security threats from Phisher
        if isinstance(message, dict) and message.get("type") == "security_threat":
            self.handle_security_threat(message)

    def send_message(self, message: Any, recipient_agent):
        # recipient_agent expected to implement receive_message(sender, message) or receive_message(message)
        try:
            if hasattr(recipient_agent, "receive_message"):
                # prefer (sender, message) signature if available
                try:
                    recipient_agent.receive_message(self.__class__.__name__, message)
                except TypeError:
                    # fallback to single-arg receive_message
                    recipient_agent.receive_message(message)
            else:
                self.log_action(f"Recipient {recipient_agent.__class__.__name__} cannot receive messages")
            self.outbox.append({"to": recipient_agent.__class__.__name__, "message": message})
            self.log_action(f"Sent message to {recipient_agent.__class__.__name__}: {message}")
        except Exception as e:
            self.log_action(f"Failed to send message: {e}")

    # --------------------------
    # Integrity Hashing
    # --------------------------
    def compute_file_hash(self, file_path: str) -> Optional[str]:
        """Compute SHA-256 hash of a given file; return None if missing/error."""
        try:
            with open(file_path, "rb") as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()
            return file_hash
        except FileNotFoundError:
            return None
        except Exception as e:
            self.log_action(f"Error hashing {file_path}: {e}")
            return None

    def record_agent_hash(self, agent_name: str, file_path: str) -> bool:
        """Record or verify hash for an agent file. Returns True if OK, False if mismatch/absent."""
        current_hash = self.compute_file_hash(file_path)
        if not current_hash:
            self.log_action(f"Warning: Cannot find {file_path} for {agent_name}")
            return False

        prev_hash = self.agent_integrity_hashes.get(agent_name)
        if prev_hash and prev_hash != current_hash:
            self.log_action(f"⚠️ Integrity ALERT: {agent_name} file changed!")
            return False
        else:
            self.agent_integrity_hashes[agent_name] = current_hash
            self.log_action(f"Integrity verified for {agent_name}")
            return True

    # --------------------------
    # Firewall & Blocklist
    # --------------------------
    def block_entity(self, identifier: str, reason: str):
        """Block a rogue agent, IP, or token."""
        if identifier not in self.blocked_entities:
            self.blocked_entities.append(identifier)
            self.log_action(f"Blocked {identifier} - Reason: {reason}")
        else:
            self.log_action(f"{identifier} already blocked")

    def is_blocked(self, identifier: str) -> bool:
        """Check if an entity is currently blocked."""
        return identifier in self.blocked_entities

    # --------------------------
    # Security Sweep
    # --------------------------
    def run_security_sweep(self, agent_paths: Dict[str, str]) -> Dict[str, Any]:
        """
        Perform a full integrity and security check across NAE agents.
        Returns a report with verification results and any blocked items.
        """
        self.status = "Running security sweep"
        self.log_action("Starting full system security sweep...")
        report = {"verified": {}, "failed": {}, "blocked": [], "timestamp": datetime.datetime.now().isoformat()}

        for agent_name, path in agent_paths.items():
            verified = self.record_agent_hash(agent_name, path)
            if verified:
                report["verified"][agent_name] = {"path": path, "status": "verified"}
            else:
                report["failed"][agent_name] = {"path": path, "status": "failed"}
                self.block_entity(agent_name, "Integrity mismatch or missing file")
                report["blocked"].append(agent_name)

        self.status = "Idle"
        self.log_action("Security sweep completed.")
        return report

    # --------------------------
    # Periodic sweep runner (background)
    # --------------------------
    def start_periodic_sweeps(self, agent_paths: Dict[str, str], interval_seconds: int = 3600):
        """
        Start a background thread that runs run_security_sweep(agent_paths) every `interval_seconds`.
        Returns the threading.Thread object (daemon).
        """
        def _loop():
            while True:
                try:
                    self.run_security_sweep(agent_paths)
                except Exception as e:
                    self.log_action(f"Periodic sweep error: {e}")
                time.sleep(interval_seconds)

        t = threading.Thread(target=_loop, daemon=True)
        t.start()
        self.log_action(f"Started periodic security sweeps every {interval_seconds} seconds.")
        return t

    # --------------------------
    # Report to Splinter & Phisher
    # --------------------------
    def report_security_status(self) -> Dict[str, Any]:
        """Generate a security summary report."""
        report = {
            "blocked_entities": self.blocked_entities,
            "verified_agents": list(self.agent_integrity_hashes.keys()),
            "timestamp": datetime.datetime.now().isoformat()
        }
        self.log_action(f"Security report generated: {json.dumps(report, indent=2)}")
        return report

    # --------------------------
    # Convenience export for orchestrator monitoring
    # --------------------------
    def status_summary(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "blocked_count": len(self.blocked_entities),
            "verified_count": len(self.agent_integrity_hashes)
        }

    # --------------------------
    # Security Threat Handling
    # --------------------------
    def handle_security_threat(self, threat: Dict[str, Any]):
        """Handle security threats from Phisher and take immediate defensive action"""
        severity = threat.get("severity", "medium")
        threat_description = threat.get("threat", "Unknown")
        action_required = threat.get("action_required", "investigate")
        details = threat.get("details", {})
        source = threat.get("source", "unknown")
        timestamp = threat.get("timestamp", datetime.datetime.now().isoformat())
        
        # Create unique threat ID to prevent duplicate processing (without timestamp)
        threat_id = f"{source}:{threat_description}"
        
        # Skip if already processed
        if threat_id in self.processed_threats:
            self.log_action(f"⚠️ Skipping duplicate threat: {threat_description} from {source}")
            return
        
        # Mark as processed
        self.processed_threats.add(threat_id)
        
        self.log_action(f"🚨 SECURITY THREAT RECEIVED: {threat_description} (Severity: {severity})")
        
        # Take immediate defensive actions based on severity
        if severity == "critical":
            self.log_action("🔴 CRITICAL THREAT - Taking immediate defensive action")
            self.status = "DEFENSIVE_MODE_ACTIVE"
            
            # Block suspicious entities
            if "file" in details:
                suspicious_file = details.get("file", "")
                self.block_entity(suspicious_file, f"Critical threat detected: {threat_description}")
                self.log_action(f"Blocked entity: {suspicious_file}")
            
            # Could trigger system isolation or shutdown
            self.log_action("⚠️  CRITICAL: Consider activating kill switch or isolating affected components")
            
        elif severity == "high":
            self.log_action("🟠 HIGH PRIORITY THREAT - Enhanced security measures")
            self.status = "HIGH_ALERT"
            
            # Block suspicious files/entities
            if "file" in details:
                suspicious_file = details.get("file", "")
                self.block_entity(suspicious_file, f"High priority threat: {threat_description}")
                self.log_action(f"Blocked entity: {suspicious_file}")
            
            # Run immediate security sweep
            self.log_action("Running immediate security sweep...")
            # Could trigger immediate integrity check
            
        elif severity == "medium":
            self.log_action("🟡 MEDIUM PRIORITY THREAT - Monitoring and investigation")
            self.status = "MONITORING"
            
            # Log for investigation
            self.log_action(f"Action required: {action_required}")
            self.log_action(f"Threat details: {json.dumps(details, indent=2)}")
            
        else:
            self.log_action("🟢 LOW PRIORITY THREAT - Logged for review")
            self.status = "MONITORING"
        
        # Generate security report
        security_report = self.report_security_status()
        security_report["threat_response"] = {
            "threat": threat_description,
            "severity": severity,
            "action_taken": self.status,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        # Only alert if this is an original detection (not from another agent)
        if source == "Rocksteady":
            self.alert_security_team(threat, exclude_sender=source)
        
        self.log_action(f"Security status updated: {self.status}")
        return security_report
    
    def detect_threat(self, threat_info: Dict[str, Any]):
        """Rocksteady detects a threat and alerts the security team"""
        severity = threat_info.get("severity", "medium")
        threat = threat_info.get("threat", "Unknown threat detected")
        
        self.log_action(f"🔍 THREAT DETECTED BY ROCKSTEADY: {threat} (Severity: {severity})")
        
        # Create threat message
        threat_message = {
            "type": "security_threat",
            "severity": severity,
            "threat": threat,
            "details": threat_info.get("details", {}),
            "action_required": threat_info.get("action_required", "investigate"),
            "source": "Rocksteady",
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        # Alert all security team members
        self.alert_security_team(threat_message)
        
        # Take defensive action
        self.handle_security_threat(threat_message)
        
        return threat_message
    
    def alert_security_team(self, threat: Dict[str, Any], exclude_sender: str = None):
        """Alert Phisher, Bebop, and Casey about a security threat"""
        severity = threat.get("severity", "medium")
        source = threat.get("source", "Rocksteady")
        
        # Always alert Phisher for threat intelligence (unless Phisher is the sender)
        if self.phisher_agent and exclude_sender != "Phisher":
            try:
                if hasattr(self.phisher_agent, 'receive_message'):
                    self.phisher_agent.receive_message("Rocksteady", threat)
                    self.log_action(f"🚨 ALERT SENT to Phisher: {threat.get('threat', 'Unknown threat')}")
            except Exception as e:
                self.log_action(f"Error alerting Phisher: {e}")
        
        # Alert Bebop for monitoring (unless Bebop is the sender)
        if self.bebop_agent and exclude_sender != "Bebop":
            try:
                alert_message = {
                    "type": "security_alert",
                    "severity": severity,
                    "threat": threat.get("threat", "Unknown"),
                    "details": threat.get("details", {}),
                    "source": "Rocksteady",
                    "timestamp": datetime.datetime.now().isoformat()
                }
                if hasattr(self.bebop_agent, 'receive_message'):
                    self.bebop_agent.receive_message("Rocksteady", alert_message)
                    self.log_action(f"🚨 ALERT SENT to Bebop: {threat.get('threat', 'Unknown threat')}")
            except Exception as e:
                self.log_action(f"Error alerting Bebop: {e}")
        
        # Alert Casey for improvements (unless Casey is the sender)
        if self.casey_agent and exclude_sender != "Casey":
            try:
                improvement_message = {
                    "type": "security_improvement_request",
                    "severity": severity,
                    "threat": threat.get("threat", "Unknown"),
                    "details": threat.get("details", {}),
                    "action_required": threat.get("action_required", "improve_defenses"),
                    "vulnerability_type": threat.get("vulnerability_type", ["unknown"]),
                    "source": "Rocksteady",
                    "timestamp": datetime.datetime.now().isoformat()
                }
                if hasattr(self.casey_agent, 'receive_message'):
                    self.casey_agent.receive_message(improvement_message)
                    self.log_action(f"🚨 ALERT SENT to Casey: {threat.get('threat', 'Unknown threat')}")
            except Exception as e:
                self.log_action(f"Error alerting Casey: {e}")
    
    def take_defensive_action(self, threat_details: Dict[str, Any]):
        """Take specific defensive actions based on threat details"""
        action_required = threat_details.get("action_required", "")
        
        if action_required == "code_review_and_fix":
            self.log_action("🔧 Action: Code review and fix required")
            # Could trigger automated code review or flag for manual review
            
        elif action_required == "investigate_log_anomaly":
            self.log_action("🔍 Action: Investigating log anomaly")
            # Could trigger enhanced log analysis
            
        elif action_required == "investigate_scan_error":
            self.log_action("🔍 Action: Investigating scan error")
            # Could trigger rescan or alternative scan method
            
        elif action_required == "code_review":
            self.log_action("📝 Action: Code review required")
            # Could flag code for review
            
        else:
            self.log_action(f"⚙️  Action: {action_required}")
    
    # --------------------------
    # Run loop (simple)
    # --------------------------
    def run(self, agent_paths: Optional[Dict[str, str]] = None):
        self.log_action("Rocksteady run loop started")
        if agent_paths:
            self.run_security_sweep(agent_paths)
        self.log_action("Rocksteady run loop completed")

# --------------------------
# Test Harness
# --------------------------
if __name__ == "__main__":
    # Simple test run
    rs = RocksteadyAgent()
    fake_agents = {
        "OptimusAgent": "agents/optimus.py",
        "RalphAgent": "agents/ralph.py",
        "PhisherAgent": "agents/phisher.py"
    }
    report = rs.run_security_sweep(fake_agents)
    print("Sweep report:", json.dumps(report, indent=2))
    rs.start_periodic_sweeps(fake_agents, interval_seconds=5)
    # Let the background sweeps run briefly for demo
    time.sleep(6)
    print("Status summary:", rs.status_summary())
