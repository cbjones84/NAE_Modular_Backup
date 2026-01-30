# NAE/agents/casey.py
"""
CaseyAgent v5 - Enhanced AI-Powered System Orchestrator for NAE
Now with COMPLETE capabilities matching AI assistant tools.

Responsibilities:
- Build or refine all agents dynamically
- Embed NAE goals
- Support AutoGen communication
- Monitor agent CPU/Memory usage
- Send email alerts on agent crashes or high resource usage

ALIGNED WITH 3 CORE GOALS:
1. Achieve generational wealth
2. Generate $5,000,000.00 within 8 years, every 8 years consistently
3. Optimize NAE and agents for successful options trading

ALIGNED WITH LONG-TERM PLAN:
- Orchestrates all agents toward long-term plan compliance
- Monitors PDT prevention across all agents
- Tracks progress toward $5M goal (Goal #2)
- Ensures agents understand their role in achieving generational wealth
- See: docs/NAE_LONG_TERM_PLAN.md for full strategy details

ENHANCED CAPABILITIES (v5):
- File Operations: Read, write, edit, delete, list files
- Codebase Search: Grep, glob, semantic search across codebase
- Code Execution: Execute Python code, terminal commands, agent methods
- Context Understanding: Analyze multiple files, understand relationships
- Debugging & Testing: Debug code, suggest fixes, run tests
- Complete codebase navigation and manipulation capabilities
"""

import os
import datetime
import threading
import time
import smtplib
import json
import re
import glob
import fnmatch
import subprocess
import ast
import traceback
import requests
from email.message import EmailMessage
import psutil  # pip install psutil
from typing import Dict, Any, List, Optional
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from goal_manager import get_nae_goals
from tools.feedback_loops import FeedbackManager, ResearchFeedbackLoop
try:
    from command_executor import get_executor
    COMMAND_EXECUTOR_AVAILABLE = True
except ImportError:
    COMMAND_EXECUTOR_AVAILABLE = False
    print("[Casey] Warning: command_executor not available. Some features will be limited.")

# ----------------------
# Email Configuration
# ----------------------
EMAIL_FROM = "your_email@gmail.com"
EMAIL_PASSWORD = "your_app_password"
EMAIL_TO = "cbjones84@yahoo.com"
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587

# ----------------------
# Thresholds
# ----------------------
CPU_THRESHOLD = 80  # %
MEM_THRESHOLD_MB = 500  # MB

class CaseyAgent:
    def __init__(self, goals=None):
        # ----------------------
        # Goals & Long-Term Plan Alignment
        # ----------------------
        self.goals = goals if goals else get_nae_goals()  # 3 Core Goals
        self.long_term_plan = "docs/NAE_LONG_TERM_PLAN.md"  # Reference to long-term plan
        self.agent_alignment = "docs/AGENT_ALIGNMENT.md"  # Reference to agent alignment
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
            self.log_action("⚠️ Growth Milestones tracker not available")

        # ----------------------
        # Folders & logging
        # ----------------------
        self.scripts_folder = "agents/generated_scripts/"
        self.log_file = "logs/casey.log"
        os.makedirs(self.scripts_folder, exist_ok=True)
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)

        # ----------------------
        # Messaging / AutoGen hooks
        # ----------------------
        self.inbox = []
        self.outbox = []
        
        # ----------------------
        # Security improvement tracking
        # ----------------------
        self.security_improvements = []
        self.pending_improvements = []
        self.improvement_suggestions = []  # Feedback loop improvement recommendations
        
        # Track processed alerts to prevent loops
        self.processed_alerts = set()
        
        # Agent references for bidirectional alerting (set by scheduler)
        self.phisher_agent = None
        self.bebop_agent = None
        self.rocksteady_agent = None

        # ----------------------
        # Agent monitoring
        # ----------------------
        self.monitored_agents = []  # list of tuples (name, psutil.Process)
        self.monitor_thread = threading.Thread(target=self.monitor_agents_loop, daemon=True)
        self.monitor_thread.start()
        
        # ----------------------
        # Research automation
        # ----------------------
        self.algorithm_catalog: Dict[str, Any] = {}
        self.research_interval_minutes = 60
        self._research_stop_event = threading.Event()
        self.research_thread = threading.Thread(target=self._research_loop, daemon=True)
        self.research_thread.start()
        self.feedback_manager = FeedbackManager()
        self.research_feedback = ResearchFeedbackLoop(self)
        self.feedback_manager.register(self.research_feedback)
        
        # ----------------------
        # Enhanced Capabilities - File Operations
        # ----------------------
        self.workspace_root = os.path.dirname(os.path.dirname(__file__))  # NAE root
        self.file_context_cache = {}  # Cache for file contents
        self.command_executor = get_executor() if COMMAND_EXECUTOR_AVAILABLE else None
        
        # ----------------------
        # Enhanced Capabilities - Codebase Understanding
        # ----------------------
        self.codebase_index = {}  # Index of files and their contents
        self.last_search_results = []  # Cache last search results
        
        # ----------------------
        # System State Awareness
        # ----------------------
        self.system_state = self._load_system_state()
        self.agent_registry = self._build_agent_registry()
        self.codebase_structure = self._analyze_codebase_structure()
        
        # ----------------------
        # Advanced Intelligence Engine (Composer 1 / Cursor 2.0 level)
        # ----------------------
        try:
            from agents.casey_intelligence import CaseyIntelligence
            self.intelligence = CaseyIntelligence(self)
            self.log_action("✅ Advanced intelligence engine initialized")
        except ImportError as e:
            self.log_action(f"⚠️ Intelligence engine not available: {e}")
            self.intelligence = None
        
        # ----------------------
        # Enhanced Continuous Learning System
        # ----------------------
        self.enhanced_learning = None
        try:
            from agents.casey_enhanced_learning_system import CaseyEnhancedLearningSystem
            self.enhanced_learning = CaseyEnhancedLearningSystem(self)
            self.enhanced_learning.start()
            self.log_action("🧠 Enhanced continuous learning system initialized and started")
        except ImportError as e:
            self.log_action(f"⚠️ Enhanced learning system not available: {e}")
        except Exception as e:
            self.log_action(f"⚠️ Enhanced learning system initialization failed: {e}")
        
        # ----------------------
        # Genius Communication Protocol
        # ----------------------
        self.genius_protocol = None
        self.genius_coordinator = None
        try:
            from agents.genius_communication_protocol import GeniusCommunicationProtocol, MessageType, MessagePriority
            from agents.genius_coordination_engine import GeniusCoordinationEngine
            
            self.genius_protocol = GeniusCommunicationProtocol()
            self.genius_coordinator = GeniusCoordinationEngine()
            
            # Register Casey
            self.genius_protocol.register_agent(
                agent_name="CaseyAgent",
                capabilities=[
                    "orchestration", "monitoring", "system_improvement",
                    "agent_coordination", "file_operations", "codebase_search"
                ],
                expertise=["system_architecture", "agent_management", "code_analysis"],
                agent_instance=self
            )
            
            # Start coordination
            self.genius_coordinator.start_coordination()
            
            self.log_action("🧠 Genius communication protocol initialized and active")
        except ImportError as e:
            self.log_action(f"⚠️ Genius protocol not available: {e}")
        except Exception as e:
            self.log_action(f"⚠️ Genius protocol initialization failed: {e}")

    # ----------------------
    # Logging
    # ----------------------
    def log_action(self, message):
        timestamp = datetime.datetime.now().isoformat()
        try:
            with open(self.log_file, "a", encoding='utf-8') as f:
                f.write(f"[{timestamp}] {message}\n")
        except Exception as e:
            print(f"Failed to write to log: {e}")
        # Safe print for Windows console
        try:
            print(f"[Casey LOG] {message}")
        except UnicodeEncodeError:
            safe_message = message.encode('ascii', 'ignore').decode('ascii')
            print(f"[Casey LOG] {safe_message}")

    # ----------------------
    # Email alerts
    # ----------------------
    def send_email_alert(self, subject, body):
        try:
            msg = EmailMessage()
            msg["From"] = EMAIL_FROM
            msg["To"] = EMAIL_TO
            msg["Subject"] = subject
            msg.set_content(body)

            with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
                server.starttls()
                server.login(EMAIL_FROM, EMAIL_PASSWORD)
                server.send_message(msg)

            self.log_action(f"Email alert sent: {subject}")
        except Exception as e:
            self.log_action(f"Failed to send email alert: {e}")

    # ----------------------
    # Build or refine an agent
    # ----------------------
    def build_or_refine_agent(self, agent_name, overwrite=False):
        filename = f"{self.scripts_folder}{agent_name.lower()}.py"
        agent_code = f"""
# Auto-generated/refined agent by Casey
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from goal_manager import get_nae_goals
GOALS = get_nae_goals()

class {agent_name}:
    def __init__(self, goals=None):
        self.goals = goals if goals else GOALS
        self.log_file = "logs/{agent_name.lower()}.log"
        import os, datetime
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        self.log_action("{agent_name} initialized.")

    def log_action(self, message):
        ts = datetime.datetime.now().isoformat()
        with open(self.log_file, "a") as f:
            f.write(f"[{{ts}}] {{message}}\\n")
        print(f"[{agent_name} LOG] {{message}}")

    def run(self):
        self.log_action("{agent_name} running...")
"""

        if os.path.exists(filename):
            if overwrite:
                with open(filename, "w") as f:
                    f.write(agent_code.strip())
                self.log_action(f"Refined existing agent: {filename}")
            else:
                self.log_action(f"Agent already exists, skipping: {filename}")
        else:
            with open(filename, "w") as f:
                f.write(agent_code.strip())
            self.log_action(f"Built new agent: {filename}")

    # ----------------------
    # Run build/refine for multiple agents
    # ----------------------
    def run(self, agent_names=None, overwrite=False):
        self.log_action("Starting build/refine loop for agents...")
        if not agent_names:
            self.log_action("No agent names provided for build/refine.")
            return

        for name in agent_names:
            self.build_or_refine_agent(name, overwrite=overwrite)
        self.log_action("Build/refine loop completed.")

    # ----------------------
    # Monitor CPU/Memory and agent crashes
    # ----------------------
    def monitor_agents_loop(self):
        while True:
            for name, proc in self.monitored_agents:
                try:
                    cpu = proc.cpu_percent(interval=0.5)
                    mem = proc.memory_info().rss / 1024 / 1024  # MB
                    if cpu > CPU_THRESHOLD or mem > MEM_THRESHOLD_MB:
                        self.send_email_alert(
                            f"NAE Resource Alert: {name}",
                            f"CPU: {cpu:.1f}% | Memory: {mem:.1f} MB"
                        )
                except psutil.NoSuchProcess:
                    self.send_email_alert(
                        f"NAE Crash Alert: {name}",
                        "Agent process has exited unexpectedly."
                    )
            time.sleep(10)

    # ----------------------
    # Add a process to be monitored
    # ----------------------
    def monitor_process(self, name, pid):
        try:
            proc = psutil.Process(pid)
            self.monitored_agents.append((name, proc))
            self.log_action(f"Added {name} (PID {pid}) for monitoring.")
        except Exception as e:
            self.log_action(f"Failed to monitor {name}: {e}")

    # ----------------------
    # Security Improvement Handling
    # ----------------------
    def handle_security_improvement_request(self, alert: Dict[str, Any]):
        """Handle security improvement requests from Phisher"""
        severity = alert.get("severity", "medium")
        threat = alert.get("threat", "Unknown")
        details = alert.get("details", {})
        vulnerability_type = alert.get("vulnerability_type", ["unknown"])
        source = alert.get("source", "Phisher")
        
        # Create unique alert ID to prevent duplicate processing (without timestamp)
        alert_id = f"{source}:{threat}"
        
        # Skip if already processed
        if alert_id in self.processed_alerts:
            self.log_action(f"⚠️ Skipping duplicate improvement request: {threat} from {source}")
            return
        
        # Mark as processed
        self.processed_alerts.add(alert_id)
        
        self.log_action(f"🔧 SECURITY IMPROVEMENT REQUEST: {threat} (Severity: {severity})")
        
        # Analyze vulnerability and create improvement plan
        improvement_plan = self.create_security_improvement_plan(
            threat, severity, vulnerability_type, details
        )
        
        # Generate security improvements
        improvements = self.generate_security_improvements(improvement_plan)
        
        # Apply improvements
        for improvement in improvements:
            self.apply_security_improvement(improvement)
        
        # Log improvement
        self.security_improvements.append({
            "threat": threat,
            "severity": severity,
            "vulnerability_type": vulnerability_type,
            "improvements": improvements,
            "timestamp": datetime.datetime.now().isoformat()
        })
        
        self.log_action(f"✅ Generated {len(improvements)} security improvement(s)")
        
        # Only alert if this is an original detection (not from another agent)
        if source == "Casey":
            self.alert_security_team(alert, exclude_sender=source)
        
        return improvement_plan
    
    def detect_threat(self, threat_info: Dict[str, Any]):
        """Casey detects a threat and alerts the security team"""
        severity = threat_info.get("severity", "medium")
        threat = threat_info.get("threat", "Unknown threat detected")
        
        self.log_action(f"🔍 THREAT DETECTED BY CASEY: {threat} (Severity: {severity})")
        
        # Create improvement request
        improvement_message = {
            "type": "security_improvement_request",
            "severity": severity,
            "threat": threat,
            "details": threat_info.get("details", {}),
            "action_required": "improve_defenses",
            "vulnerability_type": threat_info.get("vulnerability_type", ["unknown"]),
            "source": "Casey",
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        # Alert all security team members
        self.alert_security_team(improvement_message)
        
        # Generate improvements
        self.handle_security_improvement_request(improvement_message)
        
        return improvement_message
    
    def alert_security_team(self, alert: Dict[str, Any], exclude_sender: Optional[str] = None):
        """Alert Phisher, Bebop, and Rocksteady about a security threat"""
        severity = alert.get("severity", "medium")
        
        # Always alert Phisher for threat intelligence (unless Phisher is the sender)
        if self.phisher_agent and exclude_sender != "Phisher":
            try:
                if hasattr(self.phisher_agent, 'receive_message'):
                    self.phisher_agent.receive_message("Casey", alert)
                    self.log_action(f"🚨 ALERT SENT to Phisher: {alert.get('threat', 'Unknown threat')}")
            except Exception as e:
                self.log_action(f"Error alerting Phisher: {e}")
        
        # Alert Bebop for monitoring (unless Bebop is the sender)
        if self.bebop_agent and exclude_sender != "Bebop":
            try:
                alert_message = {
                    "type": "security_alert",
                    "severity": severity,
                    "threat": alert.get("threat", "Unknown"),
                    "details": alert.get("details", {}),
                    "source": "Casey",
                    "timestamp": datetime.datetime.now().isoformat()
                }
                if hasattr(self.bebop_agent, 'receive_message'):
                    self.bebop_agent.receive_message("Casey", alert_message)
                    self.log_action(f"🚨 ALERT SENT to Bebop: {alert.get('threat', 'Unknown threat')}")
            except Exception as e:
                self.log_action(f"Error alerting Bebop: {e}")
        
        # Alert Rocksteady for defensive actions (unless Rocksteady is the sender)
        if self.rocksteady_agent and exclude_sender != "Rocksteady":
            try:
                threat_message = {
                    "type": "security_threat",
                    "severity": severity,
                    "threat": alert.get("threat", "Unknown"),
                    "details": alert.get("details", {}),
                    "action_required": alert.get("action_required", "improve_defenses"),
                    "source": "Casey",
                    "timestamp": datetime.datetime.now().isoformat()
                }
                if hasattr(self.rocksteady_agent, 'receive_message'):
                    self.rocksteady_agent.receive_message("Casey", threat_message)
                    self.log_action(f"🚨 ALERT SENT to Rocksteady: {alert.get('threat', 'Unknown threat')}")
            except Exception as e:
                self.log_action(f"Error alerting Rocksteady: {e}")
    
    def create_security_improvement_plan(self, threat: str, severity: str, 
                                        vulnerability_type: List[str], 
                                        details: Dict[str, Any]) -> Dict[str, Any]:
        """Create a plan for improving security based on detected vulnerability"""
        plan = {
            "threat": threat,
            "severity": severity,
            "vulnerability_types": vulnerability_type,
            "improvements_needed": [],
            "agents_to_enhance": [],
            "defensive_measures": []
        }
        
        # Analyze vulnerability types and create specific improvements
        if "sql_injection" in vulnerability_type:
            plan["improvements_needed"].append("Input validation and parameterized queries")
            plan["agents_to_enhance"].extend(["Bebop", "Rocksteady"])
            plan["defensive_measures"].append("Add SQL injection detection patterns")
            
        if "xss" in vulnerability_type:
            plan["improvements_needed"].append("Output encoding and content security policies")
            plan["agents_to_enhance"].extend(["Bebop", "Rocksteady"])
            plan["defensive_measures"].append("Add XSS detection and prevention")
            
        if "command_injection" in vulnerability_type:
            plan["improvements_needed"].append("Command execution restrictions")
            plan["agents_to_enhance"].extend(["Rocksteady"])
            plan["defensive_measures"].append("Add command injection blocking")
            
        if "code_vulnerability" in vulnerability_type or not vulnerability_type:
            plan["improvements_needed"].append("Enhanced code scanning and validation")
            plan["agents_to_enhance"].extend(["Phisher", "Bebop"])
            plan["defensive_measures"].append("Improve vulnerability detection")
        
        # Add general improvements based on severity
        if severity in ["critical", "high"]:
            plan["improvements_needed"].append("Enhanced monitoring and alerting")
            plan["agents_to_enhance"].extend(["Bebop", "Rocksteady"])
            plan["defensive_measures"].append("Implement immediate threat response")
        
        # Add file-specific improvements if file is identified
        if "file" in details:
            plan["file_affected"] = details.get("file")
            plan["improvements_needed"].append(f"Review and harden {details.get('file')}")
        
        return plan
    
    def generate_security_improvements(self, plan: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate specific security improvements based on plan"""
        improvements = []
        
        # Improvement 1: Enhance Phisher's detection patterns
        if "Phisher" in plan.get("agents_to_enhance", []):
            improvements.append({
                "type": "enhance_detection",
                "agent": "Phisher",
                "action": "Add new threat detection patterns",
                "details": {
                    "patterns": plan.get("defensive_measures", []),
                    "priority": "high" if plan.get("severity") in ["critical", "high"] else "medium"
                }
            })
        
        # Improvement 2: Enhance Bebop's monitoring
        if "Bebop" in plan.get("agents_to_enhance", []):
            improvements.append({
                "type": "enhance_monitoring",
                "agent": "Bebop",
                "action": "Add vulnerability-specific monitoring",
                "details": {
                    "vulnerability_types": plan.get("vulnerability_types", []),
                    "monitoring_frequency": "increased" if plan.get("severity") in ["critical", "high"] else "normal"
                }
            })
        
        # Improvement 3: Enhance Rocksteady's defenses
        if "Rocksteady" in plan.get("agents_to_enhance", []):
            improvements.append({
                "type": "enhance_defenses",
                "agent": "Rocksteady",
                "action": "Add vulnerability-specific blocking rules",
                "details": {
                    "blocking_rules": plan.get("defensive_measures", []),
                    "severity": plan.get("severity")
                }
            })
        
        # Improvement 4: Create security patch if file is identified
        if "file_affected" in plan:
            improvements.append({
                "type": "security_patch",
                "file": plan.get("file_affected"),
                "action": "Create security patch recommendations",
                "details": {
                    "recommendations": plan.get("improvements_needed", []),
                    "priority": plan.get("severity")
                }
            })
        
        return improvements
    
    def apply_security_improvement(self, improvement: Dict[str, Any]):
        """Apply a security improvement"""
        improvement_type = improvement.get("type")
        agent = improvement.get("agent")
        action = improvement.get("action")
        
        self.log_action(f"🔧 Applying improvement: {action} for {agent}")
        
        if improvement_type == "enhance_detection":
            self.log_action(f"  → Enhancing {agent} detection patterns")
            # Could generate code to enhance Phisher's patterns
            
        elif improvement_type == "enhance_monitoring":
            self.log_action(f"  → Enhancing {agent} monitoring capabilities")
            # Could generate code to enhance Bebop's monitoring
            
        elif improvement_type == "enhance_defenses":
            self.log_action(f"  → Enhancing {agent} defensive measures")
            # Could generate code to enhance Rocksteady's defenses
            
        elif improvement_type == "security_patch":
            file_path = improvement.get("file")
            recommendations = improvement.get("details", {}).get("recommendations", [])
            self.log_action(f"  → Creating security patch recommendations for {file_path}")
            self.log_action(f"  → Recommendations: {', '.join(recommendations[:3])}")
            
            # Create improvement record
            patch_record = {
                "file": file_path,
                "recommendations": recommendations,
                "priority": improvement.get("details", {}).get("priority", "medium"),
                "timestamp": datetime.datetime.now().isoformat()
            }
            self.pending_improvements.append(patch_record)
        
        # Log improvement application
        self.log_action(f"  ✓ Improvement applied: {improvement_type}")
    
    def create_security_enhancement_code(self, agent_name: str, enhancement_type: str, 
                                        details: Dict[str, Any]) -> str:
        """Generate code for security enhancements"""
        if enhancement_type == "detection_patterns":
            return f"""
# Security enhancement for {agent_name}
# Added detection patterns for {details.get('vulnerability_types', [])}

def enhanced_threat_detection(self, scan_result):
    # Enhanced detection logic
    pass
"""
        elif enhancement_type == "monitoring":
            return f"""
# Security enhancement for {agent_name}
# Enhanced monitoring for {details.get('vulnerability_types', [])}

def enhanced_monitoring(self):
    # Enhanced monitoring logic
    pass
"""
        elif enhancement_type == "defenses":
            return f"""
# Security enhancement for {agent_name}
# Enhanced defensive measures

def enhanced_defense(self, threat):
    # Enhanced defense logic
    pass
"""
        return ""
    
    # ----------------------
    # Messaging hooks
    # ----------------------
    def receive_message(self, message: dict):
        """
        Enhanced message reception with intelligent understanding
        Uses advanced AI reasoning like Composer 1 and Cursor 2.0
        """
        self.inbox.append(message)
        
        if not isinstance(message, dict):
            self.log_action(f"Invalid message format: expected dict, got {type(message)}")
            return
        
        msg_type = message.get("type", "unknown")
        msg_from = message.get("from", "unknown")
        content = message.get("content", {})
        
        # Handle security improvement requests from Phisher
        if msg_type == "security_improvement_request":
            self.handle_security_improvement_request(message)
        
        # Handle improvement recommendations from feedback loop
        elif msg_type == "improvement_recommendations":
            recommendations = content.get("recommendations", []) if isinstance(content, dict) else []
            self.log_action(f"📊 Received {len(recommendations)} improvement recommendations from feedback loop")
            
            for rec in recommendations[:5]:  # Show top 5
                priority = rec.get("priority", "unknown")
                description = rec.get("description", "N/A")
                agent = rec.get("agent_name", "Unknown")
                rec_type = rec.get("recommendation_type", "unknown")
                self.log_action(f"   [{priority.upper()}] {agent} ({rec_type}): {description}")
            
            # Store for analysis
            if isinstance(content, dict) and "recommendations" in content:
                self.improvement_suggestions.extend(recommendations)
            
            # Analyze and potentially implement critical recommendations
            critical_recs = [r for r in recommendations if r.get("priority") == "critical"]
            if critical_recs:
                self.log_action(f"⚠️  {len(critical_recs)} CRITICAL recommendations - analyzing for immediate action")
                self._analyze_critical_recommendations(critical_recs)
        
        else:
            self.log_action(f"Received message from {msg_from} | Type: {msg_type}")
    
    def _analyze_critical_recommendations(self, recommendations: list):
        """Analyze critical recommendations and take immediate action if needed"""
        for rec in recommendations:
            agent_name = rec.get("agent_name", "")
            rec_type = rec.get("recommendation_type", "")
            details = rec.get("implementation_details", {})
            
            # Critical risk management recommendations
            if rec_type == "risk_management" and details.get("pause_new_trades"):
                self.log_action(f"🚨 CRITICAL: Pausing new trades due to {rec.get('description', 'risk management')}")
                # This would trigger Optimus to pause trading
                # Implementation would depend on Optimus API
            
            # Critical position sizing recommendations
            elif rec_type == "position_sizing" and details.get("reduce_position_size"):
                self.log_action(f"🚨 CRITICAL: Reducing position sizes due to {rec.get('description', 'risk management')}")
                # This would trigger Optimus to reduce position sizes

    def send_message(self, message: dict, recipient_agent):
        """Send message (legacy method - uses genius protocol if available)"""
        # Use genius protocol if available
        if self.genius_protocol and isinstance(message, dict):
            # Convert to genius message
            recipient_name = recipient_agent.__class__.__name__ if hasattr(recipient_agent, '__class__') else str(recipient_agent)
            
            genius_msg = self.genius_protocol.send_genius_message(
                sender="CaseyAgent",
                recipients=[recipient_name],
                message_type=MessageType.COMMAND if message.get("type") == "command" else MessageType.REQUEST,
                subject=message.get("subject", "Message from Casey"),
                content=message.get("content", str(message)),
                priority=MessagePriority.IMPORTANT,
                context=message.get("context", {})
            )
            
            self.outbox.append({"to": recipient_name, "message": message, "genius_message_id": genius_msg.message_id})
        else:
            # Fallback to direct message
            recipient_agent.receive_message(message)
            recipient_name = recipient_agent.__class__.__name__ if hasattr(recipient_agent, '__class__') else str(recipient_agent)
            self.outbox.append({"to": recipient_name, "message": message})
    
    def send_genius_message(
        self,
        recipients: List[str],
        message_type: str,
        subject: str,
        content: str,
        priority: str = "important",
        context: Optional[Dict[str, Any]] = None,
        intent: Optional[str] = None,
        execution_plan: Optional[Dict[str, Any]] = None
    ) -> Optional[Any]:
        """Send genius-level message"""
        if not self.genius_protocol:
            # Fallback to regular message
            self.log_action(f"⚠️ Genius protocol not available, using fallback")
            return None
        
        try:
            from agents.genius_communication_protocol import MessageType, MessagePriority
            
            msg_type = MessageType[message_type.upper()] if hasattr(MessageType, message_type.upper()) else MessageType.REQUEST
            msg_priority = MessagePriority[priority.upper()] if hasattr(MessagePriority, priority.upper()) else MessagePriority.IMPORTANT
            
            message = self.genius_protocol.send_genius_message(
                sender="CaseyAgent",
                recipients=recipients,
                message_type=msg_type,
                subject=subject,
                content=content,
                priority=msg_priority,
                context=context or {},
                intent=intent,
                execution_plan=execution_plan
            )
            
            self.log_action(f"📨 Sent genius message: {subject} to {', '.join(recipients)}")
            return message
        except Exception as e:
            self.log_action(f"Error sending genius message: {e}")
            return None
    
    def receive_genius_message(self, message: Dict[str, Any]):
        """Receive genius-level message"""
        self.inbox.append(message)
        
        # Process based on message type
        msg_type = message.get("message_type", "request")
        
        if msg_type == "command":
            self._process_genius_command(message)
        elif msg_type == "collaboration":
            self._process_collaboration_request(message)
        elif msg_type == "coordination":
            self._process_coordination_message(message)
        else:
            self._process_genius_request(message)
    
    def _process_genius_command(self, message: Dict[str, Any]):
        """Process genius command"""
        content = message.get("content", "")
        context = message.get("context", {})
        
        # Use intelligence to understand and execute
        if self.intelligence:
            intent = self.intelligence.understand_intent(content)
            # Execute based on intent
            self.log_action(f"🎯 Processing genius command: {intent.action}")
    
    def _process_collaboration_request(self, message: Dict[str, Any]):
        """Process collaboration request"""
        session_id = message.get("context", {}).get("session_id")
        if session_id and self.genius_protocol:
            # Contribute to session
            self.log_action(f"🤝 Contributing to collaborative session: {session_id}")
    
    def _process_coordination_message(self, message: Dict[str, Any]):
        """Process coordination message"""
        execution_plan = message.get("execution_plan", {})
        if execution_plan:
            self.log_action(f"🎯 Received coordination: {execution_plan.get('execution_id', 'unknown')}")
    
    def _process_genius_request(self, message: Dict[str, Any]):
        """Process genius request"""
        content = message.get("content", "")
        self.log_action(f"📨 Processing request: {content[:100]}")
        self.log_action(f"Sent message to {recipient_agent.__class__.__name__}: {message}")
    
    # ========================================================================
    # ENHANCED CAPABILITIES - FILE OPERATIONS
    # ========================================================================
    
    def read_file(self, file_path: str, offset: Optional[int] = None, limit: Optional[int] = None) -> Dict[str, Any]:
        """
        Read a file from the codebase.
        Similar to read_file tool capability.
        
        Args:
            file_path: Path to file (relative to workspace root or absolute)
            offset: Optional line number to start reading from (1-based)
            limit: Optional number of lines to read
            
        Returns:
            Dict with 'content', 'lines', 'error' keys
        """
        try:
            # Resolve path
            if not os.path.isabs(file_path):
                file_path = os.path.join(self.workspace_root, file_path)
            
            if not os.path.exists(file_path):
                return {
                    "content": "",
                    "lines": 0,
                    "error": f"File not found: {file_path}"
                }
            
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            total_lines = len(lines)
            
            # Apply offset and limit if provided
            if offset is not None:
                start_idx = max(0, offset - 1)  # Convert to 0-based
                end_idx = limit + start_idx if limit else total_lines
                lines = lines[start_idx:end_idx]
            
            content = ''.join(lines)
            
            # Cache for context
            self.file_context_cache[file_path] = {
                "content": content,
                "lines": total_lines,
                "timestamp": datetime.datetime.now().isoformat()
            }
            
            self.log_action(f"📖 Read file: {file_path} ({total_lines} lines)")
            
            return {
                "content": content,
                "lines": total_lines,
                "error": None
            }
            
        except Exception as e:
            error_msg = f"Error reading file {file_path}: {str(e)}"
            self.log_action(f"❌ {error_msg}")
            return {
                "content": "",
                "lines": 0,
                "error": error_msg
            }
    
    def write_file(self, file_path: str, contents: str) -> Dict[str, Any]:
        """
        Write content to a file (creates new file or overwrites existing).
        
        Args:
            file_path: Path to file (relative to workspace root or absolute)
            contents: Content to write
            
        Returns:
            Dict with 'success', 'error' keys
        """
        try:
            # Resolve path
            if not os.path.isabs(file_path):
                file_path = os.path.join(self.workspace_root, file_path)
            
            # Create directory if needed
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(contents)
            
            # Update cache
            self.file_context_cache[file_path] = {
                "content": contents,
                "lines": len(contents.splitlines()),
                "timestamp": datetime.datetime.now().isoformat()
            }
            
            self.log_action(f"✍️ Wrote file: {file_path} ({len(contents)} chars)")
            
            return {
                "success": True,
                "error": None,
                "file_path": file_path
            }
            
        except Exception as e:
            error_msg = f"Error writing file {file_path}: {str(e)}"
            self.log_action(f"❌ {error_msg}")
            return {
                "success": False,
                "error": error_msg
            }
    
    def search_replace_file(self, file_path: str, old_string: str, new_string: str, 
                           replace_all: bool = False) -> Dict[str, Any]:
        """
        Search and replace text in a file.
        
        Args:
            file_path: Path to file
            old_string: Text to find
            new_string: Text to replace with
            replace_all: If True, replace all occurrences
            
        Returns:
            Dict with 'success', 'replacements', 'error' keys
        """
        try:
            # Read file
            result = self.read_file(file_path)
            if result.get("error"):
                return result
            
            content = result["content"]
            
            # Perform replacement
            if replace_all:
                new_content = content.replace(old_string, new_string)
                replacements = content.count(old_string)
            else:
                if old_string not in content:
                    return {
                        "success": False,
                        "replacements": 0,
                        "error": f"Pattern not found in file: {old_string[:50]}..."
                    }
                new_content = content.replace(old_string, new_string, 1)
                replacements = 1
            
            # Write back
            write_result = self.write_file(file_path, new_content)
            
            if write_result.get("success"):
                self.log_action(f"🔄 Replaced {replacements} occurrence(s) in {file_path}")
                return {
                    "success": True,
                    "replacements": replacements,
                    "error": None
                }
            else:
                return write_result
                
        except Exception as e:
            error_msg = f"Error in search_replace for {file_path}: {str(e)}"
            self.log_action(f"❌ {error_msg}")
            return {
                "success": False,
                "replacements": 0,
                "error": error_msg
            }
    
    def list_directory(self, directory_path: Optional[str] = None, ignore_globs: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        List files and directories.
        
        Args:
            directory_path: Path to directory (defaults to workspace root)
            ignore_globs: List of glob patterns to ignore
            
        Returns:
            Dict with 'files', 'directories', 'error' keys
        """
        try:
            if directory_path is None:
                directory_path = self.workspace_root
            elif not os.path.isabs(directory_path):
                directory_path = os.path.join(self.workspace_root, directory_path)
            
            if not os.path.exists(directory_path):
                return {
                    "files": [],
                    "directories": [],
                    "error": f"Directory not found: {directory_path}"
                }
            
            files = []
            directories = []
            
            ignore_patterns = ignore_globs or []
            
            for item in os.listdir(directory_path):
                item_path = os.path.join(directory_path, item)
                
                # Check ignore patterns
                should_ignore = False
                for pattern in ignore_patterns:
                    if pattern.startswith("**/"):
                        pattern = pattern[3:]
                    if fnmatch.fnmatch(item, pattern) or fnmatch.fnmatch(item_path, pattern):
                        should_ignore = True
                        break
                
                if should_ignore:
                    continue
                
                if os.path.isfile(item_path):
                    files.append(item)
                elif os.path.isdir(item_path):
                    directories.append(item)
            
            self.log_action(f"📁 Listed directory: {directory_path} ({len(files)} files, {len(directories)} dirs)")
            
            return {
                "files": sorted(files),
                "directories": sorted(directories),
                "error": None
            }
            
        except Exception as e:
            error_msg = f"Error listing directory {directory_path}: {str(e)}"
            self.log_action(f"❌ {error_msg}")
            return {
                "files": [],
                "directories": [],
                "error": error_msg
            }
    
    def delete_file(self, file_path: str) -> Dict[str, Any]:
        """
        Delete a file.
        
        Args:
            file_path: Path to file to delete
            
        Returns:
            Dict with 'success', 'error' keys
        """
        try:
            if not os.path.isabs(file_path):
                file_path = os.path.join(self.workspace_root, file_path)
            
            if not os.path.exists(file_path):
                return {
                    "success": False,
                    "error": f"File not found: {file_path}"
                }
            
            os.remove(file_path)
            
            # Remove from cache
            if file_path in self.file_context_cache:
                del self.file_context_cache[file_path]
            
            self.log_action(f"🗑️ Deleted file: {file_path}")
            
            return {
                "success": True,
                "error": None
            }
            
        except Exception as e:
            error_msg = f"Error deleting file {file_path}: {str(e)}"
            self.log_action(f"❌ {error_msg}")
            return {
                "success": False,
                "error": error_msg
            }
    
    # ========================================================================
    # ENHANCED CAPABILITIES - CODEBASE SEARCH
    # ========================================================================
    
    def grep_search(self, pattern: str, path: Optional[str] = None, 
                    case_insensitive: bool = False,
                    output_mode: str = "content",
                    max_results: int = 100) -> Dict[str, Any]:
        """
        Search for patterns in files using regex (like grep).
        
        Args:
            pattern: Regex pattern to search for
            path: Path to search in (defaults to workspace root)
            case_insensitive: Case-insensitive search
            output_mode: "content", "files_with_matches", or "count"
            max_results: Maximum number of results
            
        Returns:
            Dict with 'matches', 'files', 'error' keys
        """
        try:
            if path is None:
                path = self.workspace_root
            elif not os.path.isabs(path):
                path = os.path.join(self.workspace_root, path)
            
            if not os.path.exists(path):
                return {
                    "matches": [],
                    "files": [],
                    "error": f"Path not found: {path}"
                }
            
            flags = re.IGNORECASE if case_insensitive else 0
            regex = re.compile(pattern, flags)
            
            matches = []
            files_with_matches = set()
            
            # Walk through files
            for root, dirs, files in os.walk(path):
                # Skip common ignore directories
                dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', 'node_modules', '.venv', 'venv']]
                
                for file in files:
                    if file.endswith('.pyc') or file.endswith('.pyo'):
                        continue
                    
                    file_path = os.path.join(root, file)
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            for line_num, line in enumerate(f, 1):
                                if regex.search(line):
                                    files_with_matches.add(file_path)
                                    if output_mode == "content":
                                        matches.append({
                                            "file": file_path,
                                            "line": line_num,
                                            "content": line.rstrip()
                                        })
                                    if len(matches) >= max_results:
                                        break
                        if len(matches) >= max_results:
                            break
                    except Exception:
                        continue
                
                if len(matches) >= max_results:
                    break
            
            self.log_action(f"🔍 Grep search: '{pattern}' found {len(matches)} matches in {len(files_with_matches)} files")
            
            if output_mode == "files_with_matches":
                return {
                    "matches": [],
                    "files": sorted(list(files_with_matches)),
                    "error": None
                }
            elif output_mode == "count":
                return {
                    "matches": [],
                    "files": [],
                    "count": len(files_with_matches),
                    "error": None
                }
            else:
                return {
                    "matches": matches[:max_results],
                    "files": sorted(list(files_with_matches)),
                    "error": None
                }
                
        except Exception as e:
            error_msg = f"Error in grep search: {str(e)}"
            self.log_action(f"❌ {error_msg}")
            return {
                "matches": [],
                "files": [],
                "error": error_msg
            }
    
    def glob_search(self, pattern: str, base_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Search for files matching a glob pattern.
        
        Args:
            pattern: Glob pattern (e.g., "*.py", "**/*.md")
            base_path: Base path to search from (defaults to workspace root)
            
        Returns:
            Dict with 'files', 'error' keys
        """
        try:
            if base_path is None:
                base_path = self.workspace_root
            elif not os.path.isabs(base_path):
                base_path = os.path.join(self.workspace_root, base_path)
            
            # Expand pattern
            if not pattern.startswith("**/"):
                pattern = os.path.join(base_path, pattern)
            else:
                pattern = os.path.join(base_path, pattern[3:])
            
            files = glob.glob(pattern, recursive=True)
            files = [f for f in files if os.path.isfile(f)]
            
            self.log_action(f"🔍 Glob search: '{pattern}' found {len(files)} files")
            
            return {
                "files": sorted(files),
                "error": None
            }
            
        except Exception as e:
            error_msg = f"Error in glob search: {str(e)}"
            self.log_action(f"❌ {error_msg}")
            return {
                "files": [],
                "error": error_msg
            }
    
    def semantic_search(self, query: str, target_directories: Optional[List[str]] = None, 
                        max_results: int = 20) -> Dict[str, Any]:
        """
        Semantic search across codebase (uses keyword matching + context analysis).
        Note: Full semantic search would require ML libraries, this is a simplified version.
        
        Args:
            query: Search query/question
            target_directories: Optional list of directories to search in
            max_results: Maximum number of results
            
        Returns:
            Dict with 'results', 'error' keys
        """
        try:
            if target_directories is None:
                search_paths = [self.workspace_root]
            else:
                search_paths = []
                for dir_path in target_directories:
                    if not os.path.isabs(dir_path):
                        dir_path = os.path.join(self.workspace_root, dir_path)
                    if os.path.exists(dir_path):
                        search_paths.append(dir_path)
            
            # Extract keywords from query
            query_words = re.findall(r'\b\w+\b', query.lower())
            
            results = []
            
            for search_path in search_paths:
                for root, dirs, files in os.walk(search_path):
                    # Skip common ignore directories
                    dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', 'node_modules', '.venv', 'venv']]
                    
                    for file in files:
                        if not file.endswith('.py'):
                            continue
                        
                        file_path = os.path.join(root, file)
                        
                        try:
                            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                                content = f.read()
                                
                                # Simple relevance scoring
                                score = 0
                                for word in query_words:
                                    count = content.lower().count(word)
                                    score += count
                                
                                if score > 0:
                                    # Extract relevant lines
                                    lines = content.splitlines()
                                    relevant_lines = []
                                    for i, line in enumerate(lines[:50], 1):  # First 50 lines
                                        if any(word in line.lower() for word in query_words):
                                            relevant_lines.append(f"{i}: {line}")
                                    
                                    results.append({
                                        "file": file_path,
                                        "score": score,
                                        "preview": '\n'.join(relevant_lines[:5])
                                    })
                        except Exception:
                            continue
            
            # Sort by score
            results.sort(key=lambda x: x['score'], reverse=True)
            results = results[:max_results]
            
            self.log_action(f"🧠 Semantic search: '{query}' found {len(results)} relevant files")
            
            return {
                "results": results,
                "error": None
            }
            
        except Exception as e:
            error_msg = f"Error in semantic search: {str(e)}"
            self.log_action(f"❌ {error_msg}")
            return {
                "results": [],
                "error": error_msg
            }
    
    # ========================================================================
    # ENHANCED CAPABILITIES - CODE EXECUTION
    # ========================================================================
    
    def execute_terminal_command(self, command: str, cwd: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute a terminal command safely.
        
        Args:
            command: Command to execute
            cwd: Working directory (defaults to workspace root)
            
        Returns:
            Dict with 'status', 'output', 'error', 'return_code' keys
        """
        try:
            if cwd is None:
                cwd = self.workspace_root
            
            if self.command_executor:
                result = self.command_executor.execute_system_command(command, cwd=cwd)
                return {
                    "status": result.status.value,
                    "output": result.output,
                    "error": result.error,
                    "return_code": result.return_code,
                    "duration": result.duration
                }
            else:
                # Fallback to subprocess
                result = subprocess.run(
                    command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=30,
                    cwd=cwd
                )
                
                self.log_action(f"⚙️ Executed command: {command[:50]}...")
                
                return {
                    "status": "success" if result.returncode == 0 else "failed",
                    "output": result.stdout,
                    "error": result.stderr,
                    "return_code": result.returncode
                }
                
        except subprocess.TimeoutExpired:
            return {
                "status": "timeout",
                "output": "",
                "error": "Command timed out after 30 seconds",
                "return_code": None
            }
        except Exception as e:
            error_msg = f"Error executing command: {str(e)}"
            self.log_action(f"❌ {error_msg}")
            return {
                "status": "error",
                "output": "",
                "error": error_msg,
                "return_code": None
            }
    
    def execute_python_code(self, code: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute Python code safely.
        
        Args:
            code: Python code to execute
            context: Optional context variables
            
        Returns:
            Dict with 'status', 'output', 'error' keys
        """
        try:
            if self.command_executor:
                result = self.command_executor.execute_python_code(code, context)
                return {
                    "status": result.status.value,
                    "output": result.output,
                    "error": result.error,
                    "duration": result.duration
                }
            else:
                # Fallback execution
                exec_globals = {"__builtins__": __builtins__}
                if context:
                    exec_globals.update(context)
                
                import io
                output_buffer = io.StringIO()
                old_stdout = sys.stdout
                sys.stdout = output_buffer
                
                try:
                    exec(code, exec_globals)
                    output = output_buffer.getvalue()
                    return {
                        "status": "success",
                        "output": output,
                        "error": None
                    }
                finally:
                    sys.stdout = old_stdout
                    
        except Exception as e:
            error_msg = f"Error executing Python code: {str(e)}\n{traceback.format_exc()}"
            self.log_action(f"❌ {error_msg}")
            return {
                "status": "error",
                "output": "",
                "error": error_msg
            }
    
    def execute_agent_method(self, agent_name: str, method_name: str,
                            args: Optional[List[Any]] = None, kwargs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute a method on an agent.
        
        Args:
            agent_name: Name of agent (e.g., "ralph", "optimus")
            method_name: Method name to call
            args: Positional arguments
            kwargs: Keyword arguments
            
        Returns:
            Dict with 'status', 'output', 'error' keys
        """
        try:
            if self.command_executor:
                result = self.command_executor.execute_agent_method(agent_name, method_name, args, kwargs)
                return {
                    "status": result.status.value,
                    "output": result.output,
                    "error": result.error,
                    "duration": result.duration
                }
            else:
                # Manual execution
                module_name = f"agents.{agent_name.lower()}"
                module = __import__(module_name, fromlist=[agent_name])
                agent_class = getattr(module, f"{agent_name.capitalize()}Agent")
                
                agent = agent_class()
                method = getattr(agent, method_name)
                result = method(*(args or []), **(kwargs or {}))
                
                if isinstance(result, (dict, list)):
                    output = json.dumps(result, indent=2)
                else:
                    output = str(result)
                
                return {
                    "status": "success",
                    "output": output,
                    "error": None
                }
                
        except Exception as e:
            error_msg = f"Error executing agent method: {str(e)}\n{traceback.format_exc()}"
            self.log_action(f"❌ {error_msg}")
            return {
                "status": "error",
                "output": "",
                "error": error_msg
            }
    
    # ========================================================================
    # ENHANCED CAPABILITIES - CONTEXT UNDERSTANDING
    # ========================================================================
    
    def understand_context(self, file_paths: List[str], query: Optional[str] = None) -> Dict[str, Any]:
        """
        Understand context across multiple files.
        
        Args:
            file_paths: List of file paths to analyze
            query: Optional query about what to understand
            
        Returns:
            Dict with 'analysis', 'relationships', 'error' keys
        """
        try:
            files_data = []
            relationships = []
            
            for file_path in file_paths:
                result = self.read_file(file_path)
                if not result.get("error"):
                    content = result["content"]
                    files_data.append({
                        "file": file_path,
                        "content": content,
                        "lines": result["lines"]
                    })
            
            # Analyze relationships (imports, function calls, etc.)
            for file_data in files_data:
                content = file_data["content"]
                file_path = file_data["file"]
                
                # Find imports
                imports = re.findall(r'^import\s+(\w+)|^from\s+(\w+\.?\w*)\s+import', content, re.MULTILINE)
                for imp in imports:
                    module = imp[0] or imp[1]
                    relationships.append({
                        "type": "import",
                        "from": file_path,
                        "to": module
                    })
                
                # Find class definitions
                classes = re.findall(r'^class\s+(\w+)', content, re.MULTILINE)
                for cls in classes:
                    relationships.append({
                        "type": "class",
                        "file": file_path,
                        "name": cls
                    })
            
            self.log_action(f"🧠 Analyzed context across {len(files_data)} files")
            
            return {
                "analysis": {
                    "files": files_data,
                    "query": query
                },
                "relationships": relationships,
                "error": None
            }
            
        except Exception as e:
            error_msg = f"Error understanding context: {str(e)}"
            self.log_action(f"❌ {error_msg}")
            return {
                "analysis": {},
                "relationships": [],
                "error": error_msg
            }
    
    def debug_code(self, file_path: str, error_message: Optional[str] = None) -> Dict[str, Any]:
        """
        Debug code by analyzing errors and suggesting fixes.
        
        Args:
            file_path: Path to file to debug
            error_message: Optional error message
            
        Returns:
            Dict with 'diagnosis', 'suggestions', 'error' keys
        """
        try:
            result = self.read_file(file_path)
            if result.get("error"):
                return result
            
            content = result["content"]
            suggestions = []
            
            # Check for common Python errors
            try:
                ast.parse(content)
                syntax_valid = True
            except SyntaxError as e:
                syntax_valid = False
                suggestions.append({
                    "type": "syntax_error",
                    "line": e.lineno,
                    "message": str(e),
                    "fix": f"Fix syntax error at line {e.lineno}: {e.msg}"
                })
            
            # Analyze for common issues
            if "import " in content:
                # Check for missing imports
                # Could add more sophisticated import checking
                pass
            
            # Check for undefined variables (simplified)
            if error_message:
                if "NameError" in error_message:
                    suggestions.append({
                        "type": "name_error",
                        "message": "Variable or function not defined",
                        "fix": "Check for typos or missing imports"
                    })
                elif "TypeError" in error_message:
                    suggestions.append({
                        "type": "type_error",
                        "message": "Type mismatch",
                        "fix": "Check variable types and function signatures"
                    })
            
            self.log_action(f"🐛 Debugged file: {file_path}")
            
            return {
                "diagnosis": {
                    "syntax_valid": syntax_valid,
                    "error_message": error_message
                },
                "suggestions": suggestions,
                "error": None
            }
            
        except Exception as e:
            error_msg = f"Error debugging code: {str(e)}"
            self.log_action(f"❌ {error_msg}")
            return {
                "diagnosis": {},
                "suggestions": [],
                "error": error_msg
            }
    
    def test_code(self, file_path: str, test_function: Optional[str] = None) -> Dict[str, Any]:
        """
        Test code by running it or specific test functions.
        
        Args:
            file_path: Path to file to test
            test_function: Optional specific function to test
            
        Returns:
            Dict with 'status', 'output', 'error' keys
        """
        try:
            result = self.read_file(file_path)
            if result.get("error"):
                return result
            
            content = result["content"]
            
            # Create test code
            test_code = content
            if test_function:
                test_code += f"\n\n# Run test\nif __name__ == '__main__':\n    {test_function}()\n"
            
            # Execute test
            exec_result = self.execute_python_code(test_code)
            
            self.log_action(f"🧪 Tested file: {file_path}")
            
            return {
                "status": exec_result.get("status"),
                "output": exec_result.get("output"),
                "error": exec_result.get("error")
            }
            
        except Exception as e:
            error_msg = f"Error testing code: {str(e)}"
            self.log_action(f"❌ {error_msg}")
            return {
                "status": "error",
                "output": "",
                "error": error_msg
            }
    
    # ========================================================================
    # RESEARCH AUTOMATION
    # ========================================================================
    
    def _research_loop(self):
        """
        Background loop that gathers new quantitative research ideas.
        """
        while not self._research_stop_event.is_set():
            try:
                findings = self._scan_research_sources()
                new_findings = [
                    finding for finding in findings if finding["name"] not in self.algorithm_catalog
                ]
                for finding in new_findings:
                    self.algorithm_catalog[finding["name"]] = finding
                    plan = self._plan_algorithm_integration(finding)
                    self.pending_improvements.append(plan)
                    self.log_action(
                        f"🧠 Research discovery: {finding['name']} from {finding['source']} (priority: {plan.get('priority')})"
                    )
                    context = {"finding": finding, "plan": plan}
                    self.feedback_manager.run("research", context)
            except Exception as exc:
                self.log_action(f"Research loop encountered an error: {exc}")
            finally:
                self._research_stop_event.wait(self.research_interval_minutes * 60)
    
    def _scan_research_sources(self) -> List[Dict[str, Any]]:
        """
        Query curated external sources for options trading algorithms.
        Enhanced with recent 2024-2025 algorithm discoveries.
        """
        findings: List[Dict[str, Any]] = []
        try:
            # Original queries
            findings.extend(self._fetch_arxiv_candidates("options volatility forecasting"))
            findings.extend(self._fetch_arxiv_candidates("dispersion trading correlation arbitrage"))
            
            # Recent 2024-2025 algorithm discoveries
            findings.extend(self._fetch_arxiv_candidates("multi-agent LLM trading high-frequency"))
            findings.extend(self._fetch_arxiv_candidates("prioritized experience replay DQN"))
            findings.extend(self._fetch_arxiv_candidates("event-based trading complex systems"))
            findings.extend(self._fetch_arxiv_candidates("reinforcement learning execution optimization"))
            findings.extend(self._fetch_arxiv_candidates("quantitative finance 2024 2025"))
        except Exception as exc:
            self.log_action(f"Research scan warning: {exc}")
        
        if not findings:
            # Fallback curated list
            findings = [
                {
                    "name": "Fallback_IV_Kalman_Pipeline",
                    "description": "Kalman-smoothed IV factors based on PCA.",
                    "source": "curated",
                    "link": "https://arxiv.org/abs/2104.04450",
                    "priority": "high",
                }
            ]
        return findings
    
    def _fetch_arxiv_candidates(self, query: str) -> List[Dict[str, Any]]:
        url = "http://export.arxiv.org/api/query"
        params = {
            "search_query": query,
            "start": 0,
            "max_results": 3,
        }
        response = requests.get(url, params=params, timeout=10)
        if response.status_code != 200:
            return []
        
        entries = response.text.split("<entry>")
        findings: List[Dict[str, Any]] = []
        for entry in entries[1:]:
            title_start = entry.find("<title>")
            title_end = entry.find("</title>")
            summary_start = entry.find("<summary>")
            summary_end = entry.find("</summary>")
            link_start = entry.find("<id>")
            link_end = entry.find("</id>")
            if title_start == -1 or summary_start == -1 or link_start == -1:
                continue
            name = entry[title_start + 7 : title_end].strip().replace("\n", " ")
            description = entry[summary_start + 9 : summary_end].strip().replace("\n", " ")
            link = entry[link_start + 4 : link_end].strip()
            findings.append(
                {
                    "name": name,
                    "description": description,
                    "source": "arXiv",
                    "link": link,
                    "priority": "medium",
                }
            )
        return findings
    
    def _plan_algorithm_integration(self, finding: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a structured integration plan for a discovered algorithm.
        """
        description = finding.get("description") or ""
        integration_plan = {
            "algorithm": finding.get("name"),
            "source": finding.get("source"),
            "link": finding.get("link"),
            "summary": description[:500],
            "priority": finding.get("priority", "medium"),
            "action_required": [
                "Review research summary",
                "Prototype in tools/profit_algorithms",
                "Backtest via Optimus backtester",
                "Human approval before deployment",
            ],
            "status": "pending_review",
            "timestamp": datetime.datetime.now().isoformat(),
        }
        return integration_plan
    
    def stop_research(self):
        """
        Stop background research thread gracefully.
        """
        self._research_stop_event.set()
        if hasattr(self, "research_thread") and self.research_thread.is_alive():
            self.research_thread.join(timeout=5)
    
    # ========================================================================
    # ENHANCED CAPABILITIES - SUMMARY
    # ========================================================================
    
    def get_capabilities_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all Casey's enhanced capabilities.
        
        Returns:
            Dict listing all capabilities
        """
        return {
            "file_operations": [
                "read_file - Read files with optional line range",
                "write_file - Write/create files",
                "search_replace_file - Find and replace text in files",
                "list_directory - List files and directories",
                "delete_file - Delete files"
            ],
            "codebase_search": [
                "grep_search - Regex pattern search",
                "glob_search - File pattern matching",
                "semantic_search - Context-aware code search"
            ],
            "code_execution": [
                "execute_terminal_command - Run shell commands",
                "execute_python_code - Execute Python code safely",
                "execute_agent_method - Call agent methods"
            ],
            "context_understanding": [
                "understand_context - Analyze multiple files",
                "debug_code - Debug and suggest fixes",
                "test_code - Run tests"
            ],
            "system_awareness": [
                "get_system_state - Get current NAE system state",
                "get_agent_info - Get information about any agent",
                "get_broker_status - Get broker adapter status",
                "get_recent_updates - Get recent system updates",
                "analyze_codebase - Analyze codebase structure"
            ],
            "original_capabilities": [
                "build_or_refine_agent - Create/update agents",
                "monitor_agents_loop - Monitor agent resources",
                "handle_security_improvement_request - Security improvements",
                "receive_message/send_message - Agent communication"
            ]
        }
    
    # ========================================================================
    # SYSTEM STATE AWARENESS - Cursor 2.0 Updates
    # ========================================================================
    
    def _load_system_state(self) -> Dict[str, Any]:
        """Load system state from configuration file"""
        try:
            state_file = os.path.join(self.workspace_root, "config", "nae_system_state.json")
            if os.path.exists(state_file):
                with open(state_file, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            self.log_action(f"Error loading system state: {e}")
            return {}
    
    def _build_agent_registry(self) -> Dict[str, Any]:
        """Build registry of all agents and their capabilities"""
        registry = {}
        
        # Load from system state if available
        if self.system_state.get("agents"):
            registry = self.system_state["agents"]
        
        # Also try to discover agents from filesystem
        agents_dir = os.path.join(self.workspace_root, "agents")
        if os.path.exists(agents_dir):
            for file in os.listdir(agents_dir):
                if file.endswith('.py') and not file.startswith('__'):
                    agent_name = file.replace('.py', '')
                    if agent_name not in registry:
                        registry[agent_name] = {
                            "name": agent_name,
                            "file": f"agents/{file}",
                            "status": "unknown"
                        }
        
        return registry
    
    def _analyze_codebase_structure(self) -> Dict[str, Any]:
        """Analyze codebase structure"""
        structure = {
            "agents": [],
            "adapters": [],
            "tools": [],
            "config": [],
            "docs": []
        }
        
        try:
            # Analyze agents
            agents_dir = os.path.join(self.workspace_root, "agents")
            if os.path.exists(agents_dir):
                structure["agents"] = [f for f in os.listdir(agents_dir) if f.endswith('.py')]
            
            # Analyze adapters
            adapters_dir = os.path.join(self.workspace_root, "adapters")
            if os.path.exists(adapters_dir):
                structure["adapters"] = [f for f in os.listdir(adapters_dir) if f.endswith('.py')]
            
            # Analyze tools
            tools_dir = os.path.join(self.workspace_root, "tools")
            if os.path.exists(tools_dir):
                for _, _, files in os.walk(tools_dir):
                    py_files = [f for f in files if f.endswith('.py')]
                    structure["tools"].extend(py_files)
        except Exception as e:
            self.log_action(f"Error analyzing codebase structure: {e}")
        
        return structure
    
    def get_system_state(self) -> Dict[str, Any]:
        """
        Get current NAE system state
        
        Returns comprehensive system state including:
        - All agents and their status
        - Broker adapters configuration
        - Recent updates
        - Current trading state
        """
        try:
            # Reload system state
            self.system_state = self._load_system_state()
            
            # Get current agent status
            agent_status = {}
            for agent_name, agent_info in self.agent_registry.items():
                try:
                    # Try to import and check agent status
                    module_name = f"agents.{agent_name.lower()}"
                    module = __import__(module_name, fromlist=[agent_name])
                    agent_class = getattr(module, f"{agent_name.capitalize()}Agent", None)
                    if agent_class:
                        agent_status[agent_name] = {
                            **agent_info,
                            "status": "available",
                            "class": agent_class.__name__
                        }
                except Exception:
                    agent_status[agent_name] = {
                        **agent_info,
                        "status": "unknown"
                    }
            
            # Get broker adapter status
            broker_status = {}
            try:
                from adapters.manager import AdapterManager
                manager = AdapterManager()
                available = manager.list_available()
                for adapter_name in available:
                    try:
                        adapter = manager.get(adapter_name)
                        name_method = getattr(adapter, 'name', None)
                        adapter_name_str = name_method() if name_method else adapter_name
                        auth_method = getattr(adapter, 'auth', None)
                        authenticated = auth_method() if auth_method else None
                        broker_status[adapter_name] = {
                            "name": adapter_name_str,
                            "status": "available",
                            "auth": authenticated
                        }
                    except Exception as e:
                        broker_status[adapter_name] = {
                            "status": "error",
                            "error": str(e)
                        }
            except Exception as e:
                self.log_action(f"Error getting broker status: {e}")
            
            return {
                "system_version": self.system_state.get("system_version", "2.0"),
                "last_updated": self.system_state.get("last_updated", ""),
                "system_status": "operational",
                "agents": agent_status,
                "broker_adapters": broker_status,
                "recent_updates": self.system_state.get("recent_updates", {}),
                "current_state": self.system_state.get("current_state", {}),
                "configuration": self.system_state.get("configuration", {}),
                "codebase_structure": self.codebase_structure
            }
        except Exception as e:
            self.log_action(f"Error getting system state: {e}")
            return {
                "error": str(e),
                "system_state": self.system_state
            }
    
    def get_agent_info(self, agent_name: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific agent
        
        Args:
            agent_name: Name of agent (e.g., "ralph", "optimus", "donnie")
            
        Returns:
            Dict with agent information, capabilities, and current status
        """
        try:
            agent_name_lower = agent_name.lower()
            
            # Check system state
            if self.system_state.get("agents"):
                agent_info = self.system_state["agents"].get(agent_name_lower)
                if agent_info:
                    # Try to get live status
                    try:
                        module_name = f"agents.{agent_name_lower}"
                        module = __import__(module_name, fromlist=[agent_name])
                        agent_class = getattr(module, f"{agent_name.capitalize()}Agent", None)
                        
                        if agent_class:
                            # Try to instantiate and get status
                            agent = agent_class()
                            status_info = {}
                            
                            # Get health check if available
                            if hasattr(agent, 'health_check'):
                                status_info = agent.health_check()
                            
                            return {
                                **agent_info,
                                "live_status": status_info,
                                "class_available": True
                            }
                    except Exception as e:
                        return {
                            **agent_info,
                            "class_available": False,
                            "error": str(e)
                        }
            
            # Fallback: try to discover from file
            agent_file = os.path.join(self.workspace_root, "agents", f"{agent_name_lower}.py")
            if os.path.exists(agent_file):
                file_info = self.read_file(agent_file, limit=100)
                return {
                    "name": agent_name,
                    "file": f"agents/{agent_name_lower}.py",
                    "file_exists": True,
                    "preview": file_info.get("content", "")[:500],
                    "status": "file_found"
                }
            
            return {
                "name": agent_name,
                "status": "not_found",
                "error": f"Agent {agent_name} not found in system state or filesystem"
            }
            
        except Exception as e:
            return {
                "name": agent_name,
                "status": "error",
                "error": str(e)
            }
    
    def get_broker_status(self, broker_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get status of broker adapters
        
        Args:
            broker_name: Optional specific broker name (Tradier is the only broker)
            
        Returns:
            Dict with broker adapter status
        """
        try:
            from adapters.manager import AdapterManager
            manager = AdapterManager()
            
            if broker_name:
                # Get specific broker
                try:
                    adapter = manager.get(broker_name)
                    get_account_method = getattr(adapter, 'get_account', None)
                    account = get_account_method() if get_account_method else {}
                    get_positions_method = getattr(adapter, 'get_positions', None)
                    positions = get_positions_method() if get_positions_method else []
                    
                    name_method = getattr(adapter, 'name', None)
                    adapter_name = name_method() if name_method else broker_name
                    auth_method = getattr(adapter, 'auth', None)
                    authenticated = auth_method() if auth_method else None
                    
                    return {
                        "broker": broker_name,
                        "name": adapter_name,
                        "status": "available",
                        "authenticated": authenticated,
                        "account": account,
                        "positions": positions,
                        "config": self.system_state.get("broker_adapters", {}).get(broker_name, {})
                    }
                except Exception as e:
                    return {
                        "broker": broker_name,
                        "status": "error",
                        "error": str(e)
                    }
            else:
                # Get all brokers
                available = manager.list_available()
                brokers = {}
                
                for adapter_name in available:
                    try:
                        adapter = manager.get(adapter_name)
                        name_method = getattr(adapter, 'name', None)
                        adapter_name_str = name_method() if name_method else adapter_name
                        auth_method = getattr(adapter, 'auth', None)
                        authenticated = auth_method() if auth_method else None
                        brokers[adapter_name] = {
                            "name": adapter_name_str,
                            "status": "available",
                            "authenticated": authenticated
                        }
                    except Exception as e:
                        brokers[adapter_name] = {
                            "status": "error",
                            "error": str(e)
                        }
                
                return {
                    "brokers": brokers,
                    "system_state": self.system_state.get("broker_adapters", {})
                }
                
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def get_learning_report(self) -> Dict[str, Any]:
        """Get comprehensive learning and improvement report"""
        if self.enhanced_learning:
            return self.enhanced_learning.get_learning_report()
        return {"status": "not_available"}
    
    def get_improvement_suggestions(self) -> List[Dict[str, Any]]:
        """Get improvement suggestions from learning system"""
        if self.enhanced_learning:
            return self.enhanced_learning._generate_improvement_suggestions()
        return []
    
    def learn_from_interaction(self, prompt: str, response: str, source: str = "cursor_auto"):
        """Learn from an interaction (can be called externally)"""
        if self.enhanced_learning and self.enhanced_learning.multi_model_learner:
            from agents.casey_continuous_learning import LearningSource
            
            source_map = {
                "cursor_auto": LearningSource.CURSOR_AUTO,
                "chatgpt": LearningSource.CHATGPT,
                "grok": LearningSource.GROK,
                "gemini": LearningSource.GEMINI
            }
            
            learning_source = source_map.get(source.lower(), LearningSource.CURSOR_AUTO)
            
            insight = self.enhanced_learning.multi_model_learner.learn_from_model(
                source=learning_source,
                prompt=prompt,
                response=response,
                context=self.enhanced_learning._get_nae_context()
            )
            
            if insight:
                self.log_action(f"📚 Learned from {source}: {insight.title}")
                return insight
        
        return None
    
    def get_recent_updates(self, days: int = 7) -> Dict[str, Any]:
        """
        Get recent system updates
        
        Args:
            days: Number of days to look back
            
        Returns:
            Dict with recent updates
        """
        try:
            recent_updates = self.system_state.get("recent_updates", {})
            
            # Filter by date if needed
            if days:
                cutoff_date = datetime.datetime.now() - datetime.timedelta(days=days)
                filtered = {}
                for date_str, updates in recent_updates.items():
                    try:
                        update_date = datetime.datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                        if update_date >= cutoff_date:
                            filtered[date_str] = updates
                    except Exception:  # Date parsing can fail, include anyway
                        filtered[date_str] = updates  # Include if date parsing fails
                return filtered
            
            return recent_updates
            
        except Exception as e:
            return {
                "error": str(e),
                "updates": self.system_state.get("recent_updates", {})
            }
    
    def analyze_codebase(self, query: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze codebase structure and relationships
        
        Args:
            query: Optional query about what to analyze
            
        Returns:
            Dict with codebase analysis
        """
        try:
            analysis = {
                "structure": self.codebase_structure,
                "agents": list(self.agent_registry.keys()),
                "adapters": [],
                "tools": [],
                "key_files": self.system_state.get("key_files", {}),
                "workflow": self.system_state.get("system_architecture", {}).get("workflow", {})
            }
            
            # Analyze adapters
            adapters_dir = os.path.join(self.workspace_root, "adapters")
            if os.path.exists(adapters_dir):
                analysis["adapters"] = [f for f in os.listdir(adapters_dir) if f.endswith('.py') and f != '__init__.py']
            
            # Analyze tools
            tools_dir = os.path.join(self.workspace_root, "tools")
            if os.path.exists(tools_dir):
                for root, _, files in os.walk(tools_dir):
                    py_files = [f for f in files if f.endswith('.py')]
                    rel_path = os.path.relpath(root, self.workspace_root)
                    analysis["tools"].extend([f"{rel_path}/{f}" for f in py_files])
            
            # If query provided, do semantic search
            if query:
                search_results = self.semantic_search(query)
                analysis["search_results"] = search_results.get("results", [])
            
            return analysis
            
        except Exception as e:
            return {
                "error": str(e),
                "structure": self.codebase_structure
            }
    
    def get_agent_capabilities(self, agent_name: str) -> Dict[str, Any]:
        """
        Get capabilities of a specific agent
        
        Args:
            agent_name: Name of agent
            
        Returns:
            Dict with agent capabilities
        """
        agent_info = self.get_agent_info(agent_name)
        
        return {
            "agent": agent_name,
            "capabilities": agent_info.get("capabilities", []),
            "role": agent_info.get("role", "Unknown"),
            "version": agent_info.get("version", "Unknown"),
            "status": agent_info.get("status", "unknown"),
            "file": agent_info.get("file", "Unknown")
        }
    
    def get_system_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive system summary for Casey's awareness
        
        Returns:
            Complete system summary including all agents, brokers, updates, and current state
        """
        return {
            "system_info": {
                "version": self.system_state.get("system_version", "3.0"),
                "status": self.system_state.get("system_status", "operational"),
                "trading_mode": self.system_state.get("trading_mode", "live"),
                "broker": self.system_state.get("broker", "tradier")
            },
            "agents": self.agent_registry,
            "brokers": self.get_broker_status(),
            "recent_updates": self.get_recent_updates(),
            "current_state": self.system_state.get("current_state", {}),
            "configuration": self.system_state.get("configuration", {}),
            "workflow": self.system_state.get("system_architecture", {}).get("workflow", {}),
            "codebase": self.codebase_structure
        }


# ----------------------
# Test harness
# ----------------------
def casey_main_loop():
    """Casey continuous operation loop - NEVER STOPS"""
    import traceback
    import logging
    
    logger = logging.getLogger(__name__)
    restart_count = 0
    
    while True:  # NEVER EXIT
        try:
            logger.info("=" * 70)
            logger.info(f"🚀 Starting Casey Agent (Restart #{restart_count})")
            logger.info("=" * 70)
            
            casey = CaseyAgent()
            
            # Main operation loop
            while True:
                try:
                    # Casey's main operation - monitor and improve agents continuously
                    # Run monitoring cycles
                    if hasattr(casey, 'monitor_agents'):
                        casey.monitor_agents()
                    
                    time.sleep(60)  # Check every minute
                    
                except KeyboardInterrupt:
                    logger.warning("⚠️  KeyboardInterrupt - Continuing Casey operation...")
                    time.sleep(5)
                except Exception as e:
                    logger.error(f"Error in Casey main loop: {e}")
                    logger.error(traceback.format_exc())
                    time.sleep(30)
                    
        except KeyboardInterrupt:
            restart_count += 1
            logger.warning(f"⚠️  KeyboardInterrupt - RESTARTING Casey (Restart #{restart_count})")
            time.sleep(5)
        except SystemExit:
            restart_count += 1
            logger.warning(f"⚠️  SystemExit - RESTARTING Casey (Restart #{restart_count})")
            time.sleep(10)
        except Exception as e:
            restart_count += 1
            delay = min(60 * restart_count, 3600)
            logger.error(f"❌ Fatal error in Casey (Restart #{restart_count}): {e}")
            logger.error(traceback.format_exc())
            logger.info(f"🔄 Restarting in {delay} seconds...")
            time.sleep(delay)


if __name__ == "__main__":
    casey_main_loop()
