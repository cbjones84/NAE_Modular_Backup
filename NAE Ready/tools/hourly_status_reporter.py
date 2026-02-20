#!/usr/bin/env python3
"""
Hourly Agent Status Reporter
Sends email status updates on every agent at the top of each hour.
"""

import os
import sys
import smtplib
import time
import threading
import datetime
from email.message import EmailMessage
from typing import Dict, Any, List, Optional
from pathlib import Path

# Add NAE to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Email Configuration (from config)
try:
    from config.enhanced_casey_config import EMAIL_TO, EMAIL_FROM, EMAIL_PASSWORD, SMTP_SERVER, SMTP_PORT
except ImportError:
    try:
        # Try loading from agents.casey module
        from agents.casey import EMAIL_TO, EMAIL_FROM, EMAIL_PASSWORD, SMTP_SERVER, SMTP_PORT
    except ImportError:
        # Fallback to environment variables or defaults
        EMAIL_TO = os.getenv("NAE_STATUS_EMAIL", "cbjones84@yahoo.com")
        EMAIL_FROM = os.getenv("NAE_EMAIL_FROM", "your_email@gmail.com")
        EMAIL_PASSWORD = os.getenv("NAE_EMAIL_PASSWORD", "your_app_password")
        SMTP_SERVER = os.getenv("NAE_SMTP_SERVER", "smtp.gmail.com")
        SMTP_PORT = int(os.getenv("NAE_SMTP_PORT", "587"))


class HourlyStatusReporter:
    """Sends hourly status reports on all agents via email"""
    
    def __init__(self, agent_instances: Dict[str, Any]):
        """
        Initialize status reporter
        
        Args:
            agent_instances: Dictionary of agent_name -> agent_instance
        """
        self.agent_instances = agent_instances
        self.email_to = EMAIL_TO
        self.email_from = EMAIL_FROM
        self.email_password = EMAIL_PASSWORD
        self.smtp_server = SMTP_SERVER
        self.smtp_port = SMTP_PORT
        self.running = False
        self.report_thread = None
        
    def get_agent_status(self, agent_name: str, agent: Any) -> Dict[str, Any]:
        """Get status for a single agent"""
        status = {
            "name": agent_name,
            "status": "UNKNOWN",
            "running": False,
            "issues": [],
            "details": {}
        }
        
        try:
            # Check if agent exists
            if agent is None:
                status["status"] = "NOT_INITIALIZED"
                status["issues"].append("Agent instance is None")
                return status
            
            # Basic agent info
            status["running"] = True
            
            # Agent-specific status checks
            if agent_name == "Optimus":
                status["status"] = "RUNNING"
                status["details"]["trading_enabled"] = getattr(agent, "trading_enabled", False)
                status["details"]["trading_mode"] = getattr(agent, "trading_mode", None)
                if hasattr(agent, "trading_mode"):
                    status["details"]["trading_mode_value"] = agent.trading_mode.value if hasattr(agent.trading_mode, "value") else str(agent.trading_mode)
                
                # Check if LIVE mode
                if status["details"]["trading_mode_value"] != "live":
                    status["issues"].append(f"⚠️ NOT IN LIVE MODE - Current: {status['details']['trading_mode_value']}")
                
                # Tradier check
                if hasattr(agent, "self_healing_engine") and agent.self_healing_engine:
                    tradier_adapter = getattr(agent.self_healing_engine, "tradier_adapter", None)
                    if tradier_adapter:
                        sandbox_status = getattr(tradier_adapter, "sandbox", True)
                        status["details"]["tradier_sandbox"] = sandbox_status
                        if sandbox_status:
                            status["issues"].append("⚠️ Tradier is in SANDBOX mode - should be LIVE")
                        else:
                            status["details"]["tradier_status"] = "LIVE ✅"
                
                status["details"]["nav"] = getattr(agent, "nav", 0)
                status["details"]["open_positions"] = getattr(agent, "open_positions", 0)
                status["details"]["daily_pnl"] = getattr(agent, "daily_pnl", 0)
                
                # Check milestone progress
                if hasattr(agent, "milestone_tracker") and agent.milestone_tracker:
                    nav = status["details"]["nav"]
                    milestone_status = agent.milestone_tracker.get_milestone_status(nav)
                    status["details"]["milestone_progress"] = f"{milestone_status.progress_pct:.1f}%"
                    status["details"]["milestone_year"] = milestone_status.current_year
                    status["details"]["aggressiveness"] = milestone_status.aggressiveness
            
            elif agent_name == "Ralph":
                status["status"] = "RUNNING"
                if hasattr(agent, "strategy_database"):
                    status["details"]["strategies_generated"] = len(agent.strategy_database) if agent.strategy_database else 0
                status["details"]["github_enabled"] = hasattr(agent, "github_client")
            
            elif agent_name == "Donnie":
                status["status"] = "RUNNING"
                if hasattr(agent, "pending_strategies"):
                    status["details"]["pending_strategies"] = len(agent.pending_strategies) if agent.pending_strategies else 0
            
            elif agent_name == "Casey":
                status["status"] = "RUNNING"
                if hasattr(agent, "research_dashboard"):
                    status["details"]["research_active"] = agent.research_dashboard is not None
            
            elif agent_name == "Splinter":
                status["status"] = "RUNNING"
                if hasattr(agent, "agent_instances"):
                    status["details"]["agents_monitored"] = len(agent.agent_instances) if agent.agent_instances else 0
            
            elif agent_name in ["Bebop", "Rocksteady", "Phisher", "Genny", "Shredder"]:
                status["status"] = "RUNNING"
            
            else:
                status["status"] = "RUNNING"
            
            # Check for common issues
            if hasattr(agent, "inbox") and agent.inbox and len(agent.inbox) > 100:
                status["issues"].append(f"Large inbox ({len(agent.inbox)} messages)")
            
            # Check log file exists
            if hasattr(agent, "log_file"):
                log_path = Path(agent.log_file)
                if log_path.exists():
                    status["details"]["log_size_kb"] = round(log_path.stat().st_size / 1024, 2)
                else:
                    status["issues"].append("Log file does not exist")
        
        except Exception as e:
            status["status"] = "ERROR"
            status["issues"].append(f"Error checking status: {str(e)}")
        
        return status
    
    def generate_status_report(self) -> str:
        """Generate comprehensive status report for all agents"""
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("NAE HOURLY STATUS REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        all_agents_ok = True
        critical_issues = []
        
        # Check each agent
        for agent_name, agent in self.agent_instances.items():
            status = self.get_agent_status(agent_name, agent)
            
            # Status indicator
            if status["status"] == "RUNNING" and not status["issues"]:
                indicator = "✅"
            elif status["status"] == "RUNNING" and status["issues"]:
                indicator = "⚠️"
                all_agents_ok = False
            elif status["status"] == "ERROR":
                indicator = "❌"
                all_agents_ok = False
            else:
                indicator = "❓"
                all_agents_ok = False
            
            report_lines.append(f"{indicator} {status['name']}: {status['status']}")
            
            # Add details
            if status["details"]:
                for key, value in status["details"].items():
                    report_lines.append(f"   • {key}: {value}")
            
            # Add issues
            if status["issues"]:
                for issue in status["issues"]:
                    report_lines.append(f"   ⚠️ {issue}")
                    if "LIVE" in issue or "SANDBOX" in issue:
                        critical_issues.append(f"{agent_name}: {issue}")
            
            report_lines.append("")
        
        # Summary
        report_lines.append("=" * 80)
        if all_agents_ok and not critical_issues:
            report_lines.append("✅ ALL AGENTS RUNNING SUCCESSFULLY")
        elif critical_issues:
            report_lines.append("⚠️ ACTION REQUIRED - CRITICAL ISSUES DETECTED:")
            for issue in critical_issues:
                report_lines.append(f"   • {issue}")
            report_lines.append("")
            report_lines.append("🔧 HOTFIX REQUIRED: Check configurations and restart if needed")
        else:
            report_lines.append("⚠️ SOME AGENTS HAVE ISSUES - Review details above")
        
        report_lines.append("=" * 80)
        
        return "\n".join(report_lines)
    
    def send_status_email(self, report: str):
        """Send status report via email"""
        try:
            msg = EmailMessage()
            msg["From"] = self.email_from
            msg["To"] = self.email_to
            msg["Subject"] = f"NAE Hourly Status Report - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}"
            msg.set_content(report)
            
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.email_from, self.email_password)
                server.send_message(msg)
            
            print(f"✅ Hourly status report sent to {self.email_to}")
            return True
        
        except Exception as e:
            print(f"❌ Failed to send hourly status email: {e}")
            return False
    
    def _hourly_report_loop(self):
        """Background thread that sends reports at the top of each hour"""
        while self.running:
            try:
                now = datetime.datetime.now()
                
                # Calculate seconds until next hour
                next_hour = (now.replace(minute=0, second=0, microsecond=0) + 
                            datetime.timedelta(hours=1))
                seconds_until_next_hour = (next_hour - now).total_seconds()
                
                # Wait until top of next hour (with a small buffer)
                if seconds_until_next_hour > 60:
                    # Wait until 1 minute before the hour
                    time.sleep(seconds_until_next_hour - 60)
                    # Then wait the final minute
                    time.sleep(60)
                else:
                    # If we're close to the hour, wait until next hour
                    time.sleep(seconds_until_next_hour + 1)
                
                # Generate and send report
                if self.running:
                    report = self.generate_status_report()
                    self.send_status_email(report)
                    
            except Exception as e:
                print(f"Error in hourly report loop: {e}")
                time.sleep(3600)  # Wait 1 hour before retrying
    
    def start(self):
        """Start the hourly reporting loop"""
        if self.running:
            return
        
        self.running = True
        self.report_thread = threading.Thread(
            target=self._hourly_report_loop,
            name="HourlyStatusReporter",
            daemon=True
        )
        self.report_thread.start()
        print(f"✅ Hourly status reporter started - reports will be sent to {self.email_to}")
    
    def stop(self):
        """Stop the hourly reporting loop"""
        self.running = False
        if self.report_thread:
            self.report_thread.join(timeout=5)



