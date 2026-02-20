# agents/phisher.py
"""
PhisherAgent - Security, code-audit, and simple penetration-testing hooks for NAE.

Features:
- Goal-aware (3 Goals embedded)
- Static code analysis using bandit if installed, otherwise heuristic checks
- Runtime log scanning (simple anomaly detection)
- Simulated penetration-test hook (placeholder for real tests)
- Logging and reporting for Bebop/Splinter/Casey/Donnie to consume

ALIGNED WITH 3 CORE GOALS:
1. Achieve generational wealth
2. Generate $5,000,000.00 within 8 years, every 8 years consistently
3. Optimize NAE and agents for successful options trading

ALIGNED WITH LONG-TERM PLAN:
- Monitors PDT compliance (same-day round trip detection)
- Detects compliance violations that could hinder Goal #2
- Protects system integrity for generational wealth building
- See: docs/NAE_LONG_TERM_PLAN.md for compliance requirements
"""

import os
import sys
import datetime
import json
import re
import subprocess
import time
import requests
from typing import List, Dict, Any, Optional
from bs4 import BeautifulSoup
import hashlib

# Goals managed by GoalManager
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from goal_manager import get_nae_goals
GOALS = get_nae_goals()

# Try to import web scraping dependencies
try:
    from bs4 import BeautifulSoup
    BEAUTIFULSOUP_AVAILABLE = True
except ImportError:
    BEAUTIFULSOUP_AVAILABLE = False
    print("[WARNING] BeautifulSoup not available. Install with: pip install beautifulsoup4")

class PhisherAgent:
    def __init__(self, bandit_path: str = "bandit"):
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
        
        self.log_file = "logs/phisher.log"
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        self.bandit_path = bandit_path
        self.heuristic_patterns = [
            r"\beval\(",                # use of eval
            r"\bexec\(",                # use of exec
            r"os\.system\(",            # shell exec
            r"subprocess\.Popen\(",     # subprocess usage
            r"socket\.socket\(",        # raw sockets
            r"requests\.get\(",         # network call (could be ok)
            r"open\([^,]+, *['\"]w",    # file write
            r"ssh",                     # ssh usage
            r"private_key",             # potential key leakage
        ]
        
        # Security alert thresholds
        self.threat_levels = {
            "critical": ["unauthorized", "breach", "malware", "exploit", "backdoor"],
            "high": ["vulnerability", "injection", "xss", "sql", "csrf"],
            "medium": ["error", "exception", "failed", "traceback"],
            "low": ["warning", "deprecated", "insecure"]
        }
        
        # Agent references for bidirectional alerting (set by scheduler)
        self.bebop_agent = None
        self.rocksteady_agent = None
        self.casey_agent = None
        
        # Inbox for tracking messages
        self.inbox = []
        
        # Track processed threats to prevent loops
        self.processed_threats = set()
        
        # Threat detection history
        self.detected_threats = []
        
        # Threat intelligence data
        self.threat_intelligence = {
            "known_threats": [],
            "threat_patterns": [],
            "scamming_tactics": [],
            "hacking_tactics": [],
            "exploit_techniques": [],
            "attack_methodologies": [],
            "mitre_attack_tactics": [],
            "owasp_top_10": [],
            "bug_bounty_reports": [],
            "exploit_database": [],
            "last_intelligence_update": None
        }
        
        # Pentesting knowledge base
        self.pentest_knowledge = {
            "attack_vectors": [],
            "exploit_payloads": [],
            "vulnerability_patterns": [],
            "authentication_bypass": [],
            "privilege_escalation": [],
            "post_exploitation": [],
            "network_attack": [],
            "web_attack": [],
            "api_attack": []
        }
        
        # Attack simulation capabilities
        self.attack_simulation = {
            "enabled": True,
            "safe_mode": True,  # Only simulate, don't execute dangerous operations
            "attack_history": [],
            "vulnerabilities_found": []
        }
        
        # System testing configuration
        self.test_config = {
            "test_entire_system": True,
            "test_frequency_hours": 1,
            "test_components": [
                "agents",
                "api_integrations",
                "secure_vault",
                "logging",
                "communication",
                "data_storage"
            ]
        }
        
        # Rate limiting for web scraping
        self.last_scrape_time = {}
        self.scrape_delay = 2.0  # seconds between scrapes

    # ----------------------
    # Utility: timestamped logging
    # ----------------------
    def log_action(self, message: str):
        timestamp = datetime.datetime.now().isoformat()
        try:
            with open(self.log_file, "a", encoding='utf-8') as f:
                f.write(f"[{timestamp}] {message}\n")
        except Exception as e:
            print(f"Failed to write to log: {e}")
        # Safe print for Windows console
        try:
            print(f"[Phisher LOG] {message}")
        except UnicodeEncodeError:
            safe_message = message.encode('ascii', 'ignore').decode('ascii')
            print(f"[Phisher LOG] {safe_message}")

    # ----------------------
    # Primary: run Bandit static scan (if available)
    # ----------------------
    def run_bandit_scan(self, file_path: str) -> Dict[str, Any]:
        """
        Run Bandit against a single file and return JSON-like report.
        If bandit not installed or fails, return {'error': '...'}.
        """
        try:
            # Call bandit with json output (-f json), single file
            result = subprocess.run(
                [self.bandit_path, "-f", "json", "-q", file_path],
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode != 0 and result.stdout.strip() == "":
                # bandit may write to stderr
                return {"error": result.stderr.strip() or "Bandit returned non-zero status."}
            # parse stdout JSON if present
            try:
                report = json.loads(result.stdout)
                self.log_action(f"Bandit scan completed for {file_path}: {len(report.get('results', []))} issues")
                return report
            except Exception:
                return {"error": "Bandit output not JSON or empty.", "raw": result.stdout, "stderr": result.stderr}
        except FileNotFoundError:
            return {"error": "bandit_not_found"}
        except Exception as e:
            return {"error": str(e)}

    # ----------------------
    # Fallback: heuristic static scan
    # ----------------------
    def heuristic_scan(self, file_path: str) -> Dict[str, Any]:
        """
        Basic regex-based scanning for risky patterns. Returns list of matches.
        """
        matches = []
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            for pat in self.heuristic_patterns:
                found = re.findall(pat, content)
                if found:
                    matches.append({"pattern": pat, "count": len(found)})
            self.log_action(f"Heuristic scan for {file_path} found {len(matches)} pattern groups")
            return {"matches": matches}
        except Exception as e:
            return {"error": str(e)}

    # ----------------------
    # Public: audit code file (Bandit preferred, heuristics fallback)
    # ----------------------
    def audit_code_file(self, file_path: str) -> Dict[str, Any]:
        """
        Audits a single code file. Returns a structured report.
        """
        if not os.path.exists(file_path):
            msg = f"File not found: {file_path}"
            self.log_action(msg)
            return {"error": msg}

        # Try Bandit first
        bandit_report = self.run_bandit_scan(file_path)
        if isinstance(bandit_report, dict) and bandit_report.get("error") == "bandit_not_found":
            self.log_action("Bandit not installed; using heuristic scan instead.")
            heuristic = self.heuristic_scan(file_path)
            report = {"engine": "heuristic", "details": heuristic}
            return report

        if isinstance(bandit_report, dict) and bandit_report.get("error"):
            # If bandit ran but errored, include fallback
            heuristic = self.heuristic_scan(file_path)
            report = {"engine": "bandit_error", "bandit": bandit_report, "heuristic": heuristic}
            return report

        # Successful bandit JSON
        return {"engine": "bandit", "report": bandit_report}

    # ----------------------
    # Scan multiple files
    # ----------------------
    def audit_codebase(self, file_paths: List[str]) -> Dict[str, Any]:
        results = {}
        for p in file_paths:
            results[p] = self.audit_code_file(p)
        self.log_action(f"Completed codebase audit: {len(file_paths)} files")
        return results

    # ----------------------
    # Advanced Penetration Testing Capabilities
    # ----------------------
    def simulated_pen_test(self, target_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Advanced simulated penetration test using learned hacking techniques:
        - Port scanning simulation
        - Vulnerability assessment
        - Attack vector identification
        - Exploit simulation (safe mode)
        """
        self.log_action(f"🔥 Running ADVANCED pentest against {target_info.get('name', 'unknown')}")
        
        target = target_info.get("name", "unknown")
        target_type = target_info.get("type", "application")
        
        report = {
            "target": target_info,
            "timestamp": datetime.datetime.now().isoformat(),
            "pentester": "Phisher",
            "methodology": "OWASP + MITRE ATT&CK",
            "issues_found": [],
            "attack_vectors": [],
            "vulnerabilities": [],
            "recommendations": [],
            "exploit_attempts": []
        }
        
        # Phase 1: Reconnaissance
        recon_results = self._pentest_reconnaissance(target_info)
        report["reconnaissance"] = recon_results
        
        # Phase 2: Vulnerability Scanning
        vuln_results = self._pentest_vulnerability_scan(target_info)
        report["vulnerabilities"].extend(vuln_results.get("vulnerabilities", []))
        
        # Phase 3: Attack Simulation (using learned techniques)
        attack_results = self._pentest_attack_simulation(target_info)
        report["attack_vectors"].extend(attack_results.get("attack_vectors", []))
        report["exploit_attempts"].extend(attack_results.get("exploit_attempts", []))
        
        # Phase 4: Post-Exploitation Assessment
        post_exploit = self._pentest_post_exploitation(target_info)
        report["post_exploitation"] = post_exploit
        
        # Phase 5: Generate comprehensive report
        report["summary"] = f"Pentest completed: {len(report['vulnerabilities'])} vulnerabilities found, {len(report['attack_vectors'])} attack vectors identified"
        report["recommendations"] = self._generate_pentest_recommendations(report)
        
        # Store in attack history
        self.attack_simulation["attack_history"].append({
            "target": target,
            "timestamp": datetime.datetime.now().isoformat(),
            "vulnerabilities_found": len(report["vulnerabilities"]),
            "attack_vectors": len(report["attack_vectors"])
        })
        
        self.log_action(f"✅ Pentest complete: {len(report['vulnerabilities'])} vulnerabilities, {len(report['attack_vectors'])} attack vectors")
        return report
    
    def _pentest_reconnaissance(self, target_info: Dict[str, Any]) -> Dict[str, Any]:
        """Phase 1: Information gathering and reconnaissance"""
        self.log_action("🔍 Phase 1: Reconnaissance")
        
        recon = {
            "target_discovery": [],
            "port_scan": [],
            "service_enumeration": [],
            "technology_stack": [],
            "attack_surface": []
        }
        
        # Simulate port scanning (safe mode)
        common_ports = [22, 80, 443, 3306, 5432, 6379, 8080, 8443]
        for port in common_ports:
            recon["port_scan"].append({
                "port": port,
                "status": "filtered" if port in [22, 443] else "open",
                "service": self._identify_service(port),
                "risk": "high" if port in [22, 3306, 5432] else "medium"
            })
        
        # Technology stack identification
        recon["technology_stack"] = [
            {"component": "Python", "version": "3.x", "risk": "medium"},
            {"component": "Redis", "version": "unknown", "risk": "medium"},
            {"component": "API", "type": "REST", "risk": "medium"}
        ]
        
        # Attack surface mapping
        recon["attack_surface"] = [
            "API endpoints",
            "File system access",
            "Database connections",
            "External service integrations",
            "Authentication mechanisms"
        ]
        
        return recon
    
    def _pentest_vulnerability_scan(self, target_info: Dict[str, Any]) -> Dict[str, Any]:
        """Phase 2: Vulnerability scanning using learned patterns"""
        self.log_action("🔍 Phase 2: Vulnerability Scanning")
        
        vulnerabilities = []
        
        # Use learned attack patterns to identify vulnerabilities
        learned_attacks = list(set(self.pentest_knowledge["attack_vectors"]))
        
        # Scan for common vulnerabilities based on learned techniques
        vuln_patterns = [
            {"name": "SQL Injection", "pattern": "sql.*injection", "severity": "critical"},
            {"name": "XSS", "pattern": "xss|cross.*site.*scripting", "severity": "high"},
            {"name": "Command Injection", "pattern": "command.*injection", "severity": "critical"},
            {"name": "Authentication Bypass", "pattern": "authentication.*bypass", "severity": "critical"},
            {"name": "Insecure Deserialization", "pattern": "insecure.*deserialization", "severity": "high"},
            {"name": "SSRF", "pattern": "server.*side.*request.*forgery", "severity": "high"},
            {"name": "XXE", "pattern": "xml.*external.*entity", "severity": "high"},
            {"name": "CSRF", "pattern": "csrf", "severity": "medium"}
        ]
        
        for vuln in vuln_patterns:
            # Check if learned attacks match this vulnerability pattern
            matching_attacks = [a for a in learned_attacks if re.search(vuln["pattern"], a, re.IGNORECASE)]
            if matching_attacks:
                vulnerabilities.append({
                    "vulnerability": vuln["name"],
                    "severity": vuln["severity"],
                    "learned_from": matching_attacks[:3],  # Show top 3 sources
                    "description": f"Potential {vuln['name']} vulnerability identified based on learned attack patterns",
                    "recommendation": self._get_vulnerability_recommendation(vuln["name"])
                })
        
        return {"vulnerabilities": vulnerabilities}
    
    def _pentest_attack_simulation(self, target_info: Dict[str, Any]) -> Dict[str, Any]:
        """Phase 3: Simulate attacks using learned techniques"""
        self.log_action("🔥 Phase 3: Attack Simulation (Safe Mode)")
        
        attack_vectors = []
        exploit_attempts = []
        
        # Get learned attack patterns
        learned_attacks = list(set(self.pentest_knowledge["attack_vectors"]))
        
        # Simulate attack vectors based on learned techniques
        attack_methods = [
            {
                "name": "SQL Injection Test",
                "method": "sqli",
                "payload": "' OR '1'='1",
                "target": "input_fields",
                "expected_result": "authentication_bypass"
            },
            {
                "name": "XSS Test",
                "method": "xss",
                "payload": "<script>alert('XSS')</script>",
                "target": "user_input",
                "expected_result": "code_execution"
            },
            {
                "name": "Command Injection Test",
                "method": "command_injection",
                "payload": "; ls -la",
                "target": "system_commands",
                "expected_result": "command_execution"
            },
            {
                "name": "Authentication Bypass Test",
                "method": "auth_bypass",
                "payload": "admin'--",
                "target": "login",
                "expected_result": "privilege_escalation"
            },
            {
                "name": "SSRF Test",
                "method": "ssrf",
                "payload": "http://127.0.0.1:6379",
                "target": "api_endpoints",
                "expected_result": "internal_access"
            }
        ]
        
        for attack in attack_methods:
            # Check if this attack method is in learned techniques
            learned_methods = [m for m in learned_attacks if attack["method"] in m.lower()]
            
            if learned_methods or self.attack_simulation["safe_mode"]:
                attack_vectors.append({
                    "attack_name": attack["name"],
                    "method": attack["method"],
                    "payload": attack["payload"],
                    "learned_from": learned_methods[:2] if learned_methods else ["phisher_knowledge_base"],
                    "simulated": True,
                    "safe_mode": True
                })
                
                exploit_attempts.append({
                    "exploit": attack["name"],
                    "status": "simulated",
                    "result": "Would test for " + attack["expected_result"],
                    "risk_level": "high" if attack["method"] in ["sqli", "command_injection"] else "medium"
                })
        
        return {
            "attack_vectors": attack_vectors,
            "exploit_attempts": exploit_attempts
        }
    
    def _pentest_post_exploitation(self, target_info: Dict[str, Any]) -> Dict[str, Any]:
        """Phase 4: Post-exploitation assessment"""
        self.log_action("🔍 Phase 4: Post-Exploitation Assessment")
        
        return {
            "privilege_escalation": {
                "possible": True,
                "methods": ["sudo exploitation", "kernel exploits", "service misconfiguration"],
                "risk": "high"
            },
            "persistence": {
                "possible": True,
                "methods": ["backdoor installation", "scheduled tasks", "service modification"],
                "risk": "critical"
            },
            "data_exfiltration": {
                "possible": True,
                "methods": ["encrypted channels", "DNS tunneling", "cloud storage"],
                "risk": "high"
            },
            "lateral_movement": {
                "possible": True,
                "methods": ["credential dumping", "pass-the-hash", "network scanning"],
                "risk": "high"
            }
        }
    
    def _identify_service(self, port: int) -> str:
        """Identify service running on port"""
        services = {
            22: "SSH",
            80: "HTTP",
            443: "HTTPS",
            3306: "MySQL",
            5432: "PostgreSQL",
            6379: "Redis",
            8080: "HTTP-Proxy",
            8443: "HTTPS-Alt"
        }
        return services.get(port, "Unknown")
    
    def _get_vulnerability_recommendation(self, vuln_name: str) -> str:
        """Get recommendation for fixing vulnerability"""
        recommendations = {
            "SQL Injection": "Use parameterized queries and input validation",
            "XSS": "Implement output encoding and Content Security Policy",
            "Command Injection": "Avoid shell execution, use safe APIs",
            "Authentication Bypass": "Implement proper authentication checks and session management",
            "Insecure Deserialization": "Avoid deserializing untrusted data",
            "SSRF": "Validate and sanitize all input URLs",
            "XXE": "Disable XML external entity processing",
            "CSRF": "Implement CSRF tokens and SameSite cookies"
        }
        return recommendations.get(vuln_name, "Review security best practices")
    
    def _generate_pentest_recommendations(self, report: Dict[str, Any]) -> List[str]:
        """Generate comprehensive recommendations based on pentest results"""
        recommendations = []
        
        if report.get("vulnerabilities"):
            recommendations.append(f"Immediately address {len(report['vulnerabilities'])} identified vulnerabilities")
            recommendations.append("Implement input validation and output encoding")
            recommendations.append("Review and harden authentication mechanisms")
        
        if report.get("attack_vectors"):
            recommendations.append(f"Mitigate {len(report['attack_vectors'])} identified attack vectors")
            recommendations.append("Implement Web Application Firewall (WAF)")
            recommendations.append("Conduct regular security assessments")
        
        recommendations.extend([
            "Implement security monitoring and logging",
            "Perform regular penetration testing",
            "Follow OWASP Top 10 guidelines",
            "Implement defense in depth strategy"
        ])
        
        return recommendations
    
    def _fetch_mitre_attack_tactics(self) -> List[Dict[str, Any]]:
        """Fetch MITRE ATT&CK framework tactics"""
        tactics = []
        
        # MITRE ATT&CK Tactics (simplified - in production would fetch from API)
        mitre_tactics = [
            "Initial Access", "Execution", "Persistence", "Privilege Escalation",
            "Defense Evasion", "Credential Access", "Discovery", "Lateral Movement",
            "Collection", "Command and Control", "Exfiltration", "Impact"
        ]
        
        for tactic in mitre_tactics:
            tactics.append({
                "tactic": tactic,
                "source": "MITRE ATT&CK",
                "source_url": "https://attack.mitre.org/",
                "source_type": "attack_framework",
                "description": f"MITRE ATT&CK tactic: {tactic}",
                "timestamp": datetime.datetime.now().isoformat()
            })
        
        # Store in knowledge base
        self.threat_intelligence["mitre_attack_tactics"] = tactics
        
        return tactics
    
    def _fetch_owasp_top_10(self) -> List[Dict[str, Any]]:
        """Fetch OWASP Top 10 vulnerabilities"""
        owasp_top10 = [
            "Broken Access Control",
            "Cryptographic Failures",
            "Injection",
            "Insecure Design",
            "Security Misconfiguration",
            "Vulnerable and Outdated Components",
            "Identification and Authentication Failures",
            "Software and Data Integrity Failures",
            "Security Logging and Monitoring Failures",
            "Server-Side Request Forgery (SSRF)"
        ]
        
        tactics = []
        for vuln in owasp_top10:
            tactics.append({
                "tactic": vuln,
                "source": "OWASP Top 10",
                "source_url": "https://owasp.org/www-project-top-ten/",
                "source_type": "vulnerability_list",
                "description": f"OWASP Top 10 vulnerability: {vuln}",
                "timestamp": datetime.datetime.now().isoformat()
            })
        
        # Store in knowledge base
        self.threat_intelligence["owasp_top_10"] = tactics
        
        return tactics

    # ----------------------
    # Simple runtime log scan (anomaly detection)
    # ----------------------
    def scan_runtime_logs(self, log_file: str, keywords: List[str] = None) -> Dict[str, Any]:
        """
        Scan runtime logs for suspicious keywords or error spikes.
        """
        if keywords is None:
            keywords = ["error", "traceback", "unauthorized", "failed", "exception", "panic", "segfault"]
        if not os.path.exists(log_file):
            return {"error": "log_file_not_found", "file": log_file}

        hits = []
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                for i, line in enumerate(f, 1):
                    lower = line.lower()
                    for kw in keywords:
                        if kw in lower:
                            hits.append({"line_no": i, "keyword": kw, "line": line.strip()})
            self.log_action(f"Runtime log scan on {log_file} found {len(hits)} hits")
            return {"file": log_file, "hits": hits}
        except Exception as e:
            return {"error": str(e)}

    # ----------------------
    # Report formatting for orchestrator consumption
    # ----------------------
    def make_report(self, title: str, details: Dict[str, Any]) -> Dict[str, Any]:
        report = {
            "title": title,
            "timestamp": datetime.datetime.now().isoformat(),
            "goals": self.goals,
            "details": details
        }
        # optionally write to disk
        out_path = f"logs/phisher_report_{int(datetime.datetime.now().timestamp())}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        self.log_action(f"Report written: {out_path}")
        return report

    # ----------------------
    # Threat Detection and Alerting
    # ----------------------
    def detect_threat_level(self, scan_result: Dict[str, Any]) -> str:
        """Determine threat level from scan result"""
        # Check for critical patterns
        result_str = json.dumps(scan_result).lower()
        
        for level, keywords in self.threat_levels.items():
            for keyword in keywords:
                if keyword in result_str:
                    return level
        
        return "low"
    
    def alert_bebop(self, threat_info: Dict[str, Any]):
        """Alert Bebop agent about detected threat"""
        if self.bebop_agent:
            try:
                alert_message = {
                    "type": "security_alert",
                    "severity": threat_info.get("severity", "medium"),
                    "threat": threat_info.get("threat", "Unknown"),
                    "details": threat_info.get("details", {}),
                    "timestamp": datetime.datetime.now().isoformat(),
                    "source": "Phisher"
                }
                
                if hasattr(self.bebop_agent, 'receive_message'):
                    self.bebop_agent.receive_message("Phisher", alert_message)
                    self.log_action(f"🚨 ALERT SENT to Bebop: {threat_info.get('threat', 'Unknown threat')}")
                else:
                    self.log_action("Warning: Bebop agent does not have receive_message method")
            except Exception as e:
                self.log_action(f"Error alerting Bebop: {e}")
        else:
            self.log_action("Warning: Bebop agent not available for alerting")
    
    def alert_rocksteady(self, threat_info: Dict[str, Any]):
        """Alert Rocksteady agent about detected threat"""
        if self.rocksteady_agent:
            try:
                alert_message = {
                    "type": "security_threat",
                    "severity": threat_info.get("severity", "medium"),
                    "threat": threat_info.get("threat", "Unknown"),
                    "details": threat_info.get("details", {}),
                    "action_required": threat_info.get("action_required", "investigate"),
                    "timestamp": datetime.datetime.now().isoformat(),
                    "source": "Phisher"
                }
                
                if hasattr(self.rocksteady_agent, 'receive_message'):
                    self.rocksteady_agent.receive_message("Phisher", alert_message)
                    self.log_action(f"🚨 ALERT SENT to Rocksteady: {threat_info.get('threat', 'Unknown threat')}")
                else:
                    self.log_action("Warning: Rocksteady agent does not have receive_message method")
            except Exception as e:
                self.log_action(f"Error alerting Rocksteady: {e}")
        else:
            self.log_action("Warning: Rocksteady agent not available for alerting")
    
    def alert_casey(self, threat_info: Dict[str, Any]):
        """Alert Casey agent about detected threat for system improvements"""
        if self.casey_agent:
            try:
                alert_message = {
                    "type": "security_improvement_request",
                    "severity": threat_info.get("severity", "medium"),
                    "threat": threat_info.get("threat", "Unknown"),
                    "details": threat_info.get("details", {}),
                    "action_required": threat_info.get("action_required", "improve_defenses"),
                    "vulnerability_type": threat_info.get("vulnerability_type", "unknown"),
                    "timestamp": datetime.datetime.now().isoformat(),
                    "source": "Phisher"
                }
                
                if hasattr(self.casey_agent, 'receive_message'):
                    self.casey_agent.receive_message(alert_message)
                    self.log_action(f"🚨 ALERT SENT to Casey: {threat_info.get('threat', 'Unknown threat')}")
                else:
                    self.log_action("Warning: Casey agent does not have receive_message method")
            except Exception as e:
                self.log_action(f"Error alerting Casey: {e}")
        else:
            self.log_action("Warning: Casey agent not available for alerting")
    
    def alert_security_team(self, threat_info: Dict[str, Any]):
        """Alert Bebop, Rocksteady, and Casey about security threats"""
        severity = threat_info.get("severity", "medium")
        
        # Always alert both agents for critical/high threats
        if severity in ["critical", "high"]:
            self.log_action(f"🚨 CRITICAL THREAT DETECTED: {threat_info.get('threat')}")
            self.alert_bebop(threat_info)
            self.alert_rocksteady(threat_info)
            self.alert_casey(threat_info)  # Also alert Casey for improvements
            
            # Record threat
            self.detected_threats.append({
                **threat_info,
                "alerted_at": datetime.datetime.now().isoformat()
            })
        else:
            # Alert Bebop for monitoring
            self.alert_bebop(threat_info)
            # Alert Casey for improvements even on medium/low threats
            self.alert_casey(threat_info)
    
    def analyze_scan_results(self, scan_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze scan results and identify threats"""
        threats = []
        
        for file_path, result in scan_results.items():
            if isinstance(result, dict):
                # Check for errors or issues
                if result.get("error"):
                    threats.append({
                        "threat": f"Scan error in {file_path}",
                        "severity": "medium",
                        "details": {"file": file_path, "error": result["error"]},
                        "action_required": "investigate_scan_error"
                    })
                
                # Check bandit results
                if result.get("engine") == "bandit" and result.get("report"):
                    bandit_report = result["report"]
                    issues = bandit_report.get("results", [])
                    if issues:
                        high_severity_count = sum(1 for issue in issues if issue.get("issue_severity") in ["HIGH", "MEDIUM"])
                        if high_severity_count > 0:
                            # Determine vulnerability type
                            vulnerability_types = set()
                            for issue in issues[:5]:
                                issue_type = issue.get("test_id", "unknown")
                                if "SQL" in issue_type or "sql" in str(issue.get("issue_text", "")).lower():
                                    vulnerability_types.add("sql_injection")
                                if "XSS" in issue_type or "xss" in str(issue.get("issue_text", "")).lower():
                                    vulnerability_types.add("xss")
                                if "shell" in issue_type.lower() or "command" in issue_type.lower():
                                    vulnerability_types.add("command_injection")
                            
                            threats.append({
                                "threat": f"Security vulnerabilities found in {file_path}",
                                "severity": "high" if high_severity_count > 3 else "medium",
                                "vulnerability_type": list(vulnerability_types) if vulnerability_types else ["code_vulnerability"],
                                "details": {
                                    "file": file_path,
                                    "issue_count": len(issues),
                                    "high_severity": high_severity_count,
                                    "issues": issues[:5]  # First 5 issues
                                },
                                "action_required": "code_review_and_fix"
                            })
                
                # Check heuristic results
                if result.get("engine") == "heuristic" and result.get("details"):
                    matches = result["details"].get("matches", [])
                    if matches:
                        threat_level = self.detect_threat_level(result)
                        threats.append({
                            "threat": f"Suspicious patterns detected in {file_path}",
                            "severity": threat_level,
                            "details": {
                                "file": file_path,
                                "pattern_count": len(matches),
                                "patterns": matches
                            },
                            "action_required": "code_review"
                        })
        
        return threats
    
    def analyze_log_scan(self, log_scan_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze log scan results for threats"""
        threats = []
        
        if isinstance(log_scan_results, dict):
            hits = log_scan_results.get("hits", [])
            if hits:
                # Count hits by keyword
                keyword_counts = {}
                for hit in hits:
                    keyword = hit.get("keyword", "unknown")
                    keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1
                
                # Determine severity based on keywords
                severity = "low"
                for keyword in keyword_counts.keys():
                    for level, keywords in self.threat_levels.items():
                        if keyword in keywords and level in ["critical", "high"]:
                            severity = level
                            break
                    if severity in ["critical", "high"]:
                        break
                
                if len(hits) > 10:  # High volume of errors
                    severity = "high" if severity != "critical" else severity
                
                if severity in ["critical", "high"] or len(hits) > 5:
                    threats.append({
                        "threat": f"Anomalous activity detected in {log_scan_results.get('file', 'unknown')}",
                        "severity": severity,
                        "details": {
                            "file": log_scan_results.get("file"),
                            "hit_count": len(hits),
                            "keyword_counts": keyword_counts,
                            "sample_hits": hits[:5]
                        },
                        "action_required": "investigate_log_anomaly"
                    })
        
        return threats
    
    def check_against_learned_threats(self, intelligence: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check if NAE system is vulnerable to learned threats"""
        threats = []
        
        # Check against CVE data
        for cve in intelligence.get("cve_alerts", []):
            if cve.get("cvss_score", 0) >= 7.0:
                # Check if NAE uses affected technologies
                description = cve.get("description", "").lower()
                if any(tech in description for tech in ["python", "http", "api", "sql", "encryption"]):
                    threats.append({
                        "threat": f"Potential vulnerability: {cve.get('cve_id')}",
                        "severity": cve.get("severity", "high"),
                        "details": {
                            "cve_id": cve.get("cve_id"),
                            "description": cve.get("description"),
                            "cvss_score": cve.get("cvss_score")
                        },
                        "action_required": "review_cve",
                        "source": "threat_intelligence"
                    })
        
        # Check against learned hacking tactics
        for tactic in intelligence.get("hacking_tactics", []):
            tactic_name = tactic.get("tactic", "").lower()
            # Check if codebase might be vulnerable to this tactic
            if any(keyword in tactic_name for keyword in ["injection", "xss", "csrf"]):
                # Scan codebase for potential vulnerabilities
                threats.append({
                    "threat": f"Potential exposure to {tactic_name} attack",
                    "severity": "high",
                    "details": {
                        "tactic": tactic_name,
                        "source": tactic.get("source")
                    },
                    "action_required": "enhance_defenses",
                    "source": "threat_intelligence"
                })
        
        return threats
    
    # ----------------------
    # Run loop with threat detection, intelligence gathering, and system testing
    # ----------------------
    def run(self):
        self.log_action("🔥 Phisher agent run loop started - BEST HACKING AGENT FOR NAE")
        threats_detected = []
        
        # Step 1: Gather threat intelligence from ALL sources (hackers, exploits, bug bounties)
        self.log_action("Step 1: Gathering threat intelligence from ALL sources (hacking agents, exploits, bug bounties)...")
        intelligence = self.scrape_threat_intelligence()
        
        # Step 1.5: Perform comprehensive pentest on NAE system
        self.log_action("Step 1.5: Running comprehensive pentest on NAE system...")
        nae_pentest = self.simulated_pen_test({
            "name": "NAE System",
            "type": "full_system",
            "components": ["agents", "api", "vault", "database", "network"]
        })
        self.log_action(f"✅ Pentest complete: {len(nae_pentest.get('vulnerabilities', []))} vulnerabilities found")
        
        # Step 2: Scan logs folder for anomalies
        self.log_action("Step 2: Scanning runtime logs...")
        logs_dir = "logs"
        if os.path.isdir(logs_dir):
            for fname in os.listdir(logs_dir):
                if fname.endswith(".log"):
                    path = os.path.join(logs_dir, fname)
                    log_result = self.scan_runtime_logs(path)
                    log_threats = self.analyze_log_scan(log_result)
                    threats_detected.extend(log_threats)
        
        # Step 3: Scan critical code files for vulnerabilities
        self.log_action("Step 3: Scanning critical code files...")
        critical_files = [
            "agents/optimus.py",
            "agents/ralph.py",
            "agents/donnie.py",
            "agents/casey.py",
            "agents/bebop.py",
            "agents/rocksteady.py",
            "agents/phisher.py",
            "agents/splinter.py",
            "nae_master_scheduler.py",
            "secure_vault.py"
        ]
        
        code_scan_results = {}
        for file_path in critical_files:
            if os.path.exists(file_path):
                code_scan_results[file_path] = self.audit_code_file(file_path)
        
        code_threats = self.analyze_scan_results(code_scan_results)
        threats_detected.extend(code_threats)
        
        # Step 4: Comprehensive system security testing
        self.log_action("Step 4: Running comprehensive system security test...")
        system_test_results = self.test_entire_nae_system()
        
        # Convert system test vulnerabilities to threats
        if system_test_results.get("vulnerabilities_found"):
            for vuln in system_test_results["vulnerabilities_found"]:
                threats_detected.append({
                    "threat": f"Security vulnerability in {vuln.get('component', 'unknown')}",
                    "severity": vuln.get("severity", "medium"),
                    "details": vuln,
                    "action_required": "fix_vulnerability",
                    "source": "system_test"
                })
        
        # Step 5: Check if NAE is vulnerable to learned threats
        self.log_action("Step 5: Checking NAE against learned threats...")
        learned_threats = self.check_against_learned_threats(intelligence)
        threats_detected.extend(learned_threats)
        
        # Step 6: Alert security team for all detected threats
        if threats_detected:
            self.log_action(f"🚨 DETECTED {len(threats_detected)} THREAT(S)")
            for threat in threats_detected:
                self.alert_security_team(threat)
        else:
            self.log_action("✅ No threats detected - system secure")
        
        # Step 7: Generate security report
        security_report = {
            "timestamp": datetime.datetime.now().isoformat(),
            "threats_detected": len(threats_detected),
            "threats": threats_detected,
            "threat_intelligence": intelligence,
            "system_test_results": system_test_results,
            "security_score": system_test_results.get("security_score", 100.0),
            "recommendations": system_test_results.get("recommendations", [])
        }
        
        # Save report
        report_path = f"logs/phisher_security_report_{int(time.time())}.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(security_report, f, indent=2, default=str)
        self.log_action(f"Security report saved: {report_path}")
        
        self.log_action("Phisher run loop completed")
        return {
            "threats_detected": len(threats_detected),
            "threats": threats_detected,
            "intelligence_gathered": intelligence,
            "system_test_results": system_test_results,
            "security_score": system_test_results.get("security_score", 100.0)
        }
    
    # ----------------------
    # Message Handling for Bidirectional Alerting
    # ----------------------
    def receive_message(self, sender: str, message: Any):
        """Receive messages from other agents (Bebop, Rocksteady, Casey)"""
        timestamp = datetime.datetime.now().isoformat()
        
        # Store message
        self.inbox.append({
            "from": sender,
            "message": message,
            "timestamp": timestamp
        })
        
        self.log_action(f"📨 Received message from {sender}")
        
        # Handle security alerts/threats from other agents
        if isinstance(message, dict):
            msg_type = message.get("type", "")
            threat = message.get("threat", "")
            
            # Create unique threat ID to prevent duplicate processing (without timestamp)
            threat_id = f"{sender}:{threat}"
            
            # Skip if we've already processed this threat
            if threat_id in self.processed_threats:
                self.log_action(f"⚠️ Skipping duplicate threat: {threat} from {sender}")
                return
            
            # Mark as processed
            self.processed_threats.add(threat_id)
            
            if msg_type == "security_alert":
                # Alert from Bebop or other monitoring agent
                self.log_action(f"🚨 SECURITY ALERT from {sender}: {threat}")
                # Update threat intelligence with this information
                self._update_threat_intelligence_from_alert(message)
                # Don't re-alert - just learn from it
                
            elif msg_type == "security_threat":
                # Threat from Rocksteady or other defensive agent
                self.log_action(f"🚨 SECURITY THREAT from {sender}: {threat}")
                # Update threat intelligence
                self._update_threat_intelligence_from_alert(message)
                # Don't re-alert - just learn from it
                
            elif msg_type == "security_improvement_request":
                # Improvement request from Casey
                self.log_action(f"🔧 SECURITY IMPROVEMENT REQUEST from {sender}: {threat}")
                # Learn from this for future threat detection
                self._learn_from_improvement_request(message)
                # Don't re-alert - just learn from it
    
    def _update_threat_intelligence_from_alert(self, alert: Dict[str, Any]):
        """Update threat intelligence based on alerts from other agents"""
        threat = alert.get("threat", "")
        severity = alert.get("severity", "medium")
        details = alert.get("details", {})
        
        # Add to known threats
        if threat:
            self.threat_intelligence["known_threats"].append({
                "threat": threat,
                "severity": severity,
                "details": details,
                "source": alert.get("source", "unknown"),
                "detected_at": datetime.datetime.now().isoformat()
            })
            
            # Update detection patterns if needed
            threat_lower = threat.lower()
            if "injection" in threat_lower and "sql.*injection" not in str(self.heuristic_patterns):
                self.heuristic_patterns.append(r"sql.*injection|command.*injection")
            if "xss" in threat_lower and "xss" not in str(self.heuristic_patterns):
                self.heuristic_patterns.append(r"xss|cross.*site.*scripting")
    
    def _learn_from_improvement_request(self, request: Dict[str, Any]):
        """Learn from security improvement requests to enhance detection"""
        vulnerability_types = request.get("vulnerability_type", [])
        threat = request.get("threat", "")
        
        # Add new patterns based on improvement requests
        for vuln_type in vulnerability_types:
            if vuln_type == "sql_injection" and "sql.*injection" not in str(self.heuristic_patterns):
                self.heuristic_patterns.append(r"sql.*injection|command.*injection")
                self.log_action(f"Added detection pattern for {vuln_type}")
            elif vuln_type == "xss" and "xss" not in str(self.heuristic_patterns):
                self.heuristic_patterns.append(r"xss|cross.*site.*scripting")
                self.log_action(f"Added detection pattern for {vuln_type}")
            elif vuln_type == "command_injection" and "command.*injection" not in str(self.heuristic_patterns):
                self.heuristic_patterns.append(r"command.*injection")
                self.log_action(f"Added detection pattern for {vuln_type}")
        
        # Log improvement learned
        self.log_action(f"📚 Learned from improvement request: {threat}")
    def scrape_threat_intelligence(self) -> Dict[str, Any]:
        """Scrape the internet for threat intelligence, hacking tactics, and security best practices"""
        self.log_action("🌐 Starting threat intelligence gathering from internet...")
        intelligence = {
            "threats": [],
            "scamming_tactics": [],
            "hacking_tactics": [],
            "security_best_practices": [],
            "cve_alerts": [],
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        # Source 1: CVE Database (MITRE CVE API)
        try:
            cve_data = self.fetch_cve_data()
            if cve_data:
                intelligence["cve_alerts"].extend(cve_data)
                self.log_action(f"Fetched {len(cve_data)} CVE alerts")
        except Exception as e:
            self.log_action(f"Error fetching CVE data: {e}")
        
        # Source 2: Security News & Blogs
        try:
            security_news = self.scrape_security_news()
            if security_news:
                intelligence["threats"].extend(security_news)
                self.log_action(f"Scraped {len(security_news)} security news items")
        except Exception as e:
            self.log_action(f"Error scraping security news: {e}")
        
        # Source 3: Hacking Forums & Security Communities
        try:
            hacking_tactics = self.scrape_hacking_tactics()
            if hacking_tactics:
                intelligence["hacking_tactics"].extend(hacking_tactics)
                self.log_action(f"Identified {len(hacking_tactics)} hacking tactics")
        except Exception as e:
            self.log_action(f"Error scraping hacking tactics: {e}")
        
        # Source 4: Scamming Tactics from Security Reports
        try:
            scamming_tactics = self.scrape_scamming_tactics()
            if scamming_tactics:
                intelligence["scamming_tactics"].extend(scamming_tactics)
                self.log_action(f"Identified {len(scamming_tactics)} scamming tactics")
        except Exception as e:
            self.log_action(f"Error scraping scamming tactics: {e}")
        
        # Source 5: Security Best Practices
        try:
            best_practices = self.scrape_security_best_practices()
            if best_practices:
                intelligence["security_best_practices"].extend(best_practices)
                self.log_action(f"Collected {len(best_practices)} security best practices")
        except Exception as e:
            self.log_action(f"Error scraping best practices: {e}")
        
        # Update threat intelligence database
        self.threat_intelligence["known_threats"].extend(intelligence["threats"])
        self.threat_intelligence["hacking_tactics"].extend(intelligence["hacking_tactics"])
        self.threat_intelligence["scamming_tactics"].extend(intelligence["scamming_tactics"])
        self.threat_intelligence["last_intelligence_update"] = datetime.datetime.now().isoformat()
        
        # Store exploit techniques from CVE data
        for cve in intelligence.get("cve_alerts", []):
            if cve.get("cvss_score", 0) >= 7.0:
                self.threat_intelligence["exploit_techniques"].append({
                    "cve_id": cve.get("cve_id"),
                    "severity": cve.get("severity"),
                    "description": cve.get("description", "")[:200]
                })
        
        # Update detection patterns based on learned threats
        self.update_detection_patterns(intelligence)
        
        # Update pentest knowledge base with new attack vectors
        all_tactics = intelligence.get("hacking_tactics", [])
        for tactic in all_tactics:
            tactic_name = tactic.get("tactic", "")
            if tactic_name and tactic_name not in self.pentest_knowledge["attack_vectors"]:
                self.pentest_knowledge["attack_vectors"].append(tactic_name)
        
        self.log_action(f"✅ Threat intelligence gathering complete: {len(intelligence['threats'])} threats, {len(intelligence['hacking_tactics'])} tactics")
        self.log_action(f"📚 Pentest knowledge base updated: {len(self.pentest_knowledge['attack_vectors'])} attack vectors learned")
        return intelligence
    
    def fetch_cve_data(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Fetch recent CVE (Common Vulnerabilities and Exposures) data"""
        cves = []
        try:
            # Use NIST NVD API for CVE data
            url = "https://services.nvd.nist.gov/rest/json/cves/2.0"
            params = {
                "resultsPerPage": limit,
                "startIndex": 0,
                "pubStartDate": (datetime.datetime.now() - datetime.timedelta(days=7)).strftime("%Y-%m-%dT%H:%M:%S.000"),
                "pubEndDate": datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S.000")
            }
            
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                vulnerabilities = data.get("vulnerabilities", [])
                
                for vuln in vulnerabilities:
                    cve_data = vuln.get("cve", {})
                    cve_id = cve_data.get("id", "unknown")
                    description = cve_data.get("descriptions", [{}])[0].get("value", "")
                    cvss_score = 0.0
                    
                    # Extract CVSS score
                    metrics = cve_data.get("metrics", {})
                    if "cvssMetricV31" in metrics:
                        cvss_score = metrics["cvssMetricV31"][0].get("cvssData", {}).get("baseScore", 0.0)
                    
                    cves.append({
                        "cve_id": cve_id,
                        "description": description,
                        "cvss_score": cvss_score,
                        "severity": "critical" if cvss_score >= 9.0 else "high" if cvss_score >= 7.0 else "medium",
                        "published": cve_data.get("published", "")
                    })
        except Exception as e:
            self.log_action(f"Error fetching CVE data: {e}")
        
        return cves
    
    def scrape_security_news(self) -> List[Dict[str, Any]]:
        """Scrape security news websites for threat information"""
        threats = []
        
        # Security news sources
        sources = [
            {
                "name": "KrebsOnSecurity",
                "url": "https://krebsonsecurity.com/",
                "selector": "article"
            },
            {
                "name": "BleepingComputer",
                "url": "https://www.bleepingcomputer.com/",
                "selector": "article"
            },
            {
                "name": "TheHackerNews",
                "url": "https://thehackernews.com/",
                "selector": "article"
            }
        ]
        
        for source in sources:
            try:
                # Rate limiting
                domain = source["url"].split("//")[1].split("/")[0]
                if domain in self.last_scrape_time:
                    time_since = time.time() - self.last_scrape_time[domain]
                    if time_since < self.scrape_delay:
                        time.sleep(self.scrape_delay - time_since)
                
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                }
                response = requests.get(source["url"], headers=headers, timeout=10)
                self.last_scrape_time[domain] = time.time()
                
                if response.status_code == 200 and BEAUTIFULSOUP_AVAILABLE:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    articles = soup.select(source["selector"])[:5]  # Limit to 5 articles
                    
                    for article in articles:
                        title_elem = article.select_one("h2, h3, .title, a")
                        title = title_elem.get_text(strip=True) if title_elem else "Unknown"
                        
                        # Extract threat keywords
                        text = article.get_text().lower()
                        threat_keywords = ["breach", "hack", "vulnerability", "exploit", "malware", "ransomware", 
                                          "phishing", "scam", "attack", "compromise"]
                        
                        if any(keyword in text for keyword in threat_keywords):
                            threats.append({
                                "source": source["name"],
                                "title": title,
                                "url": source["url"],
                                "threat_type": self._classify_threat_type(text),
                                "timestamp": datetime.datetime.now().isoformat()
                            })
            except Exception as e:
                self.log_action(f"Error scraping {source['name']}: {e}")
        
        return threats
    
    def scrape_hacking_tactics(self) -> List[Dict[str, Any]]:
        """Scrape information about current hacking tactics from ALL sources"""
        tactics = []
        
        try:
            # Comprehensive security research sources
            sources = [
                {
                    "name": "OWASP Attacks",
                    "url": "https://owasp.org/www-community/attacks/",
                    "type": "attack_patterns"
                },
                {
                    "name": "MITRE CWE",
                    "url": "https://cwe.mitre.org/data/definitions/699.html",
                    "type": "vulnerabilities"
                },
                {
                    "name": "MITRE ATT&CK",
                    "url": "https://attack.mitre.org/",
                    "type": "attack_framework"
                },
                {
                    "name": "Exploit-DB",
                    "url": "https://www.exploit-db.com/",
                    "type": "exploits"
                },
                {
                    "name": "PortSwigger Web Security",
                    "url": "https://portswigger.net/web-security",
                    "type": "web_attacks"
                },
                {
                    "name": "HackerOne Hacktivity",
                    "url": "https://hackerone.com/hacktivity",
                    "type": "bug_bounty"
                },
                {
                    "name": "Bugcrowd",
                    "url": "https://www.bugcrowd.com/resource/",
                    "type": "security_research"
                },
                {
                    "name": "PentesterLab",
                    "url": "https://pentesterlab.com/",
                    "type": "pentesting"
                }
            ]
            
            for source in sources:
                try:
                    # Rate limiting
                    domain = source["url"].split("//")[1].split("/")[0]
                    if domain in self.last_scrape_time:
                        time_since = time.time() - self.last_scrape_time[domain]
                        if time_since < self.scrape_delay:
                            time.sleep(self.scrape_delay - time_since)
                    
                    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
                    response = requests.get(source["url"], headers=headers, timeout=15)
                    self.last_scrape_time[domain] = time.time()
                    
                    if response.status_code == 200 and BEAUTIFULSOUP_AVAILABLE:
                        soup = BeautifulSoup(response.content, 'html.parser')
                        text = soup.get_text().lower()
                        
                        # Extended attack patterns based on source type
                        if source["type"] == "attack_patterns":
                            attack_patterns = [
                                "sql injection", "xss", "csrf", "command injection",
                                "directory traversal", "buffer overflow", "race condition",
                                "session hijacking", "man-in-the-middle", "ddos",
                                "xml injection", "ldap injection", "xpath injection",
                                "http header injection", "server-side request forgery",
                                "insecure deserialization", "xml external entity"
                            ]
                        elif source["type"] == "exploits":
                            attack_patterns = [
                                "remote code execution", "local file inclusion",
                                "remote file inclusion", "authentication bypass",
                                "privilege escalation", "sql injection", "xss",
                                "command injection", "path traversal", "arbitrary file upload"
                            ]
                        elif source["type"] == "web_attacks":
                            attack_patterns = [
                                "injection", "broken authentication", "sensitive data exposure",
                                "xml external entities", "broken access control",
                                "security misconfiguration", "xss", "insecure deserialization",
                                "using components with known vulnerabilities", "insufficient logging"
                            ]
                        elif source["type"] == "bug_bounty":
                            attack_patterns = [
                                "authentication bypass", "authorization bypass",
                                "information disclosure", "remote code execution",
                                "sql injection", "xss", "csrf", "ssrf",
                                "business logic flaw", "subdomain takeover"
                            ]
                        else:
                            attack_patterns = [
                                "sql injection", "xss", "csrf", "command injection",
                                "directory traversal", "buffer overflow", "race condition",
                                "session hijacking", "man-in-the-middle", "ddos"
                            ]
                        
                        for pattern in attack_patterns:
                            if pattern in text:
                                tactics.append({
                                    "tactic": pattern,
                                    "source": source["name"],
                                    "source_url": source["url"],
                                    "source_type": source["type"],
                                    "description": f"Identified {pattern} tactic from {source['name']}",
                                    "timestamp": datetime.datetime.now().isoformat()
                                })
                        
                        # Store in pentest knowledge base
                        if source["type"] in ["exploits", "attack_patterns", "web_attacks"]:
                            self.pentest_knowledge["attack_vectors"].extend([t["tactic"] for t in tactics[-len(attack_patterns):]])
                except Exception as e:
                    self.log_action(f"Error scraping hacking tactics from {source['name']}: {e}")
                    
            # Fetch MITRE ATT&CK techniques
            try:
                mitre_tactics = self._fetch_mitre_attack_tactics()
                tactics.extend(mitre_tactics)
                self.log_action(f"Fetched {len(mitre_tactics)} MITRE ATT&CK tactics")
            except Exception as e:
                self.log_action(f"Error fetching MITRE ATT&CK: {e}")
                
            # Fetch OWASP Top 10
            try:
                owasp_top10 = self._fetch_owasp_top_10()
                tactics.extend(owasp_top10)
                self.log_action(f"Fetched OWASP Top 10 vulnerabilities")
            except Exception as e:
                self.log_action(f"Error fetching OWASP Top 10: {e}")
                
        except Exception as e:
            self.log_action(f"Error in hacking tactics scraping: {e}")
        
        return tactics
    
    def scrape_scamming_tactics(self) -> List[Dict[str, Any]]:
        """Scrape information about scamming tactics"""
        tactics = []
        
        try:
            # Scrape from security awareness sources
            sources = [
                "https://www.fbi.gov/scams-and-safety/common-scams-and-crimes",
                "https://www.ftc.gov/tips-advice/business-center/small-businesses/cybersecurity"
            ]
            
            for url in sources:
                try:
                    headers = {"User-Agent": "Mozilla/5.0"}
                    response = requests.get(url, headers=headers, timeout=10)
                    
                    if response.status_code == 200 and BEAUTIFULSOUP_AVAILABLE:
                        soup = BeautifulSoup(response.content, 'html.parser')
                        text = soup.get_text().lower()
                        
                        # Extract scamming patterns
                        scam_patterns = [
                            "phishing", "social engineering", "business email compromise",
                            "ransomware", "fake invoice", "tech support scam",
                            "identity theft", "credit card fraud", "investment scam"
                        ]
                        
                        for pattern in scam_patterns:
                            if pattern in text:
                                tactics.append({
                                    "tactic": pattern,
                                    "source": url,
                                    "description": f"Identified {pattern} scam tactic",
                                    "timestamp": datetime.datetime.now().isoformat()
                                })
                except Exception as e:
                    self.log_action(f"Error scraping scamming tactics from {url}: {e}")
        except Exception as e:
            self.log_action(f"Error in scamming tactics scraping: {e}")
        
        return tactics
    
    def scrape_security_best_practices(self) -> List[Dict[str, Any]]:
        """Scrape security best practices from trusted sources"""
        practices = []
        
        try:
            # Scrape from security best practice sources
            sources = [
                "https://owasp.org/www-project-top-ten/",
                "https://www.cisa.gov/secure-our-world"
            ]
            
            for url in sources:
                try:
                    headers = {"User-Agent": "Mozilla/5.0"}
                    response = requests.get(url, headers=headers, timeout=10)
                    
                    if response.status_code == 200 and BEAUTIFULSOUP_AVAILABLE:
                        soup = BeautifulSoup(response.content, 'html.parser')
                        text = soup.get_text()
                        
                        # Extract security recommendations
                        recommendations = [
                            "input validation", "encryption", "authentication",
                            "access control", "secure coding", "vulnerability scanning",
                            "penetration testing", "security monitoring", "incident response"
                        ]
                        
                        for rec in recommendations:
                            if rec.lower() in text.lower():
                                practices.append({
                                    "practice": rec,
                                    "source": url,
                                    "description": f"Security best practice: {rec}",
                                    "timestamp": datetime.datetime.now().isoformat()
                                })
                except Exception as e:
                    self.log_action(f"Error scraping best practices from {url}: {e}")
        except Exception as e:
            self.log_action(f"Error in best practices scraping: {e}")
        
        return practices
    
    def _classify_threat_type(self, text: str) -> str:
        """Classify threat type from text"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ["breach", "compromise", "leak"]):
            return "data_breach"
        elif any(word in text_lower for word in ["ransomware", "malware", "virus"]):
            return "malware"
        elif any(word in text_lower for word in ["phishing", "scam", "fraud"]):
            return "social_engineering"
        elif any(word in text_lower for word in ["ddos", "dos", "flood"]):
            return "dos_attack"
        elif any(word in text_lower for word in ["vulnerability", "exploit", "cve"]):
            return "vulnerability"
        else:
            return "general_threat"
    
    def update_detection_patterns(self, intelligence: Dict[str, Any]):
        """Update detection patterns based on threat intelligence and learned hacking techniques"""
        # Add new patterns from learned threats
        new_patterns = []
        
        # Add patterns from CVE data
        for cve in intelligence.get("cve_alerts", []):
            if cve.get("cvss_score", 0) >= 7.0:
                description = cve.get("description", "").lower()
                # Extract key terms
                if "injection" in description:
                    new_patterns.append(r"injection.*vulnerability")
                if "buffer" in description or "overflow" in description:
                    new_patterns.append(r"buffer.*overflow")
        
        # Add patterns from hacking tactics
        for tactic in intelligence.get("hacking_tactics", []):
            tactic_name = tactic.get("tactic", "").lower()
            if "injection" in tactic_name:
                new_patterns.append(r"sql.*injection|command.*injection")
            if "xss" in tactic_name:
                new_patterns.append(r"xss|cross.*site.*scripting")
            # Add pattern based on tactic name
            pattern = tactic_name.replace(" ", ".*")
            if pattern not in str(self.heuristic_patterns):
                new_patterns.append(rf"{pattern}")
        
        # Add patterns from MITRE ATT&CK
        for tactic in intelligence.get("mitre_attack_tactics", []):
            tactic_name = tactic.get("tactic", "").lower().replace(" ", ".*")
            if tactic_name and tactic_name not in str(self.heuristic_patterns):
                new_patterns.append(rf"{tactic_name}")
        
        # Add patterns from OWASP Top 10
        for vuln in intelligence.get("owasp_top_10", []):
            vuln_name = vuln.get("tactic", "").lower().replace(" ", ".*")
            if vuln_name and vuln_name not in str(self.heuristic_patterns):
                new_patterns.append(rf"{vuln_name}")
        
        # Add new patterns to heuristic patterns
        for pattern in new_patterns:
            if pattern not in self.heuristic_patterns:
                self.heuristic_patterns.append(pattern)
                self.log_action(f"Added new detection pattern: {pattern}")
        
        # Store patterns in pentest knowledge
        self.pentest_knowledge["vulnerability_patterns"] = list(set([p for p in self.heuristic_patterns if isinstance(p, str)]))
        
        # Update threat levels with new keywords
        for threat in intelligence.get("threats", []):
            threat_type = threat.get("threat_type", "")
            if threat_type not in [kw for keywords in self.threat_levels.values() for kw in keywords]:
                if threat_type == "data_breach":
                    self.threat_levels["critical"].append("data_breach")
                elif threat_type == "malware":
                    self.threat_levels["critical"].append("ransomware")
    
    # ----------------------
    # Comprehensive System Testing
    # ----------------------
    def test_entire_nae_system(self) -> Dict[str, Any]:
        """Comprehensive security testing of the entire NAE system"""
        self.log_action("🔍 Starting comprehensive NAE system security test...")
        
        test_results = {
            "timestamp": datetime.datetime.now().isoformat(),
            "components_tested": [],
            "vulnerabilities_found": [],
            "security_score": 100.0,
            "recommendations": []
        }
        
        # Test 1: Agent Security
        agent_results = self.test_agent_security()
        test_results["components_tested"].append("agents")
        test_results["vulnerabilities_found"].extend(agent_results.get("vulnerabilities", []))
        test_results["security_score"] -= agent_results.get("score_deduction", 0)
        
        # Test 2: API Integration Security
        api_results = self.test_api_security()
        test_results["components_tested"].append("api_integrations")
        test_results["vulnerabilities_found"].extend(api_results.get("vulnerabilities", []))
        test_results["security_score"] -= api_results.get("score_deduction", 0)
        
        # Test 3: Secure Vault Security
        vault_results = self.test_vault_security()
        test_results["components_tested"].append("secure_vault")
        test_results["vulnerabilities_found"].extend(vault_results.get("vulnerabilities", []))
        test_results["security_score"] -= vault_results.get("score_deduction", 0)
        
        # Test 4: Logging Security
        logging_results = self.test_logging_security()
        test_results["components_tested"].append("logging")
        test_results["vulnerabilities_found"].extend(logging_results.get("vulnerabilities", []))
        test_results["security_score"] -= logging_results.get("score_deduction", 0)
        
        # Test 5: Communication Security
        comm_results = self.test_communication_security()
        test_results["components_tested"].append("communication")
        test_results["vulnerabilities_found"].extend(comm_results.get("vulnerabilities", []))
        test_results["security_score"] -= comm_results.get("score_deduction", 0)
        
        # Test 6: Data Storage Security
        storage_results = self.test_data_storage_security()
        test_results["components_tested"].append("data_storage")
        test_results["vulnerabilities_found"].extend(storage_results.get("vulnerabilities", []))
        test_results["security_score"] -= storage_results.get("score_deduction", 0)
        
        # Test 7: Network Security
        network_results = self.test_network_security()
        test_results["components_tested"].append("network")
        test_results["vulnerabilities_found"].extend(network_results.get("vulnerabilities", []))
        test_results["security_score"] -= network_results.get("score_deduction", 0)
        
        # Generate recommendations
        if test_results["vulnerabilities_found"]:
            test_results["recommendations"] = self.generate_security_recommendations(test_results["vulnerabilities_found"])
        
        # Ensure score doesn't go negative
        test_results["security_score"] = max(0.0, test_results["security_score"])
        
        self.log_action(f"✅ System security test complete: Score {test_results['security_score']:.1f}/100, {len(test_results['vulnerabilities_found'])} vulnerabilities found")
        
        return test_results
    
    def test_agent_security(self) -> Dict[str, Any]:
        """Test security of all agents"""
        vulnerabilities = []
        score_deduction = 0
        
        agent_files = [
            "agents/ralph.py",
            "agents/donnie.py",
            "agents/optimus.py",
            "agents/casey.py",
            "agents/bebop.py",
            "agents/rocksteady.py",
            "agents/phisher.py",
            "agents/splinter.py"
        ]
        
        for agent_file in agent_files:
            if os.path.exists(agent_file):
                result = self.audit_code_file(agent_file)
                
                # Check for vulnerabilities
                if result.get("engine") == "bandit" and result.get("report"):
                    issues = result["report"].get("results", [])
                    high_severity = sum(1 for issue in issues if issue.get("issue_severity") in ["HIGH", "MEDIUM"])
                    
                    if high_severity > 0:
                        vulnerabilities.append({
                            "component": agent_file,
                            "severity": "high" if high_severity > 3 else "medium",
                            "issue_count": len(issues),
                            "high_severity": high_severity
                        })
                        score_deduction += high_severity * 2
        
        return {"vulnerabilities": vulnerabilities, "score_deduction": score_deduction}
    
    def test_api_security(self) -> Dict[str, Any]:
        """Test security of API integrations"""
        vulnerabilities = []
        score_deduction = 0
        
        # Check API key storage
        try:
            from secure_vault import get_vault
            vault = get_vault()
            if vault:
                # Check if API keys are properly stored
                secrets = vault.list_secrets()
                if not secrets:
                    vulnerabilities.append({
                        "component": "api_keys",
                        "severity": "medium",
                        "issue": "API keys not properly secured in vault"
                    })
                    score_deduction += 5
        except Exception as e:
            vulnerabilities.append({
                "component": "api_keys",
                "severity": "high",
                "issue": f"Error checking vault: {e}"
            })
            score_deduction += 10
        
        # Check API integration files
        api_files = [
            "tools/data/api_integrations.py",
            "tools/data/web_scrapers.py"
        ]
        
        for api_file in api_files:
            if os.path.exists(api_file):
                result = self.audit_code_file(api_file)
                if result.get("engine") == "bandit" and result.get("report"):
                    issues = result["report"].get("results", [])
                    high_severity = sum(1 for issue in issues if issue.get("issue_severity") in ["HIGH"])
                    if high_severity > 0:
                        vulnerabilities.append({
                            "component": api_file,
                            "severity": "high",
                            "issue_count": high_severity
                        })
                        score_deduction += high_severity * 3
        
        return {"vulnerabilities": vulnerabilities, "score_deduction": score_deduction}
    
    def test_vault_security(self) -> Dict[str, Any]:
        """Test security of secure vault"""
        vulnerabilities = []
        score_deduction = 0
        
        # Check vault file permissions
        vault_files = [
            "config/.vault.encrypted",
            "config/.master.key"
        ]
        
        for vault_file in vault_files:
            if os.path.exists(vault_file):
                # Check file permissions (should be readable only by owner)
                stat_info = os.stat(vault_file)
                mode = stat_info.st_mode
                # Check if others can read
                if mode & 0o004:
                    vulnerabilities.append({
                        "component": vault_file,
                        "severity": "critical",
                        "issue": "Vault file readable by others"
                    })
                    score_deduction += 15
        
        # Check vault implementation
        vault_file = "secure_vault.py"
        if os.path.exists(vault_file):
            result = self.audit_code_file(vault_file)
            if result.get("engine") == "bandit" and result.get("report"):
                issues = result["report"].get("results", [])
                critical_issues = sum(1 for issue in issues if issue.get("issue_severity") == "HIGH")
                if critical_issues > 0:
                    vulnerabilities.append({
                        "component": vault_file,
                        "severity": "critical",
                        "issue_count": critical_issues
                    })
                    score_deduction += critical_issues * 5
        
        return {"vulnerabilities": vulnerabilities, "score_deduction": score_deduction}
    
    def test_logging_security(self) -> Dict[str, Any]:
        """Test security of logging system"""
        vulnerabilities = []
        score_deduction = 0
        
        # Check log files for sensitive data exposure
        logs_dir = "logs"
        if os.path.isdir(logs_dir):
            sensitive_patterns = [
                r"api[_-]?key\s*[=:]\s*['\"]?[\w-]+",
                r"password\s*[=:]\s*['\"]?[\w-]+",
                r"secret\s*[=:]\s*['\"]?[\w-]+",
                r"token\s*[=:]\s*['\"]?[\w-]+"
            ]
            
            for log_file in os.listdir(logs_dir):
                if log_file.endswith(".log"):
                    log_path = os.path.join(logs_dir, log_file)
                    try:
                        with open(log_path, "r", encoding="utf-8") as f:
                            content = f.read()
                            for pattern in sensitive_patterns:
                                matches = re.findall(pattern, content, re.IGNORECASE)
                                if matches:
                                    vulnerabilities.append({
                                        "component": log_path,
                                        "severity": "high",
                                        "issue": f"Sensitive data found in logs: {len(matches)} matches"
                                    })
                                    score_deduction += len(matches) * 2
                                    break
                    except Exception as e:
                        self.log_action(f"Error reading log file {log_file}: {e}")
        
        return {"vulnerabilities": vulnerabilities, "score_deduction": score_deduction}
    
    def test_communication_security(self) -> Dict[str, Any]:
        """Test security of agent communication"""
        vulnerabilities = []
        score_deduction = 0
        
        # Check communication security in agents
        comm_files = [
            "agents/optimus.py",
            "agents/donnie.py",
            "agents/casey.py"
        ]
        
        for comm_file in comm_files:
            if os.path.exists(comm_file):
                try:
                    with open(comm_file, "r", encoding="utf-8") as f:
                        content = f.read()
                        # Check for insecure communication patterns
                        # Only flag if HTTP is actually used in URLs (not just mentioned in comments)
                        # Look for actual HTTP usage patterns: http:// in URL assignments or requests
                        http_patterns = [
                            r'["\']http://',  # HTTP in string literals
                            r'=.*http://',    # HTTP in assignments
                            r'http://[^\s"\']+',  # HTTP URLs
                        ]
                        https_patterns = [
                            r'["\']https://',  # HTTPS in string literals
                            r'=.*https://',    # HTTPS in assignments
                            r'https://[^\s"\']+',  # HTTPS URLs
                        ]
                        
                        has_http = any(re.search(pattern, content) for pattern in http_patterns)
                        has_https = any(re.search(pattern, content) for pattern in https_patterns)
                        
                        # Only flag if HTTP is used AND HTTPS is not used
                        if has_http and not has_https:
                            vulnerabilities.append({
                                "component": comm_file,
                                "severity": "medium",
                                "issue": "HTTP communication detected (should use HTTPS)"
                            })
                            score_deduction += 3
                except Exception as e:
                    self.log_action(f"Error testing {comm_file}: {e}")
        
        return {"vulnerabilities": vulnerabilities, "score_deduction": score_deduction}
    
    def test_data_storage_security(self) -> Dict[str, Any]:
        """Test security of data storage"""
        vulnerabilities = []
        score_deduction = 0
        
        # Check data files for sensitive information
        data_files = [
            "config/api_keys.json",
            "config/settings.json"
        ]
        
        for data_file in data_files:
            if os.path.exists(data_file):
                # Check if file is in .gitignore
                gitignore_path = ".gitignore"
                if os.path.exists(gitignore_path):
                    with open(gitignore_path, "r") as f:
                        gitignore_content = f.read()
                        if data_file not in gitignore_content:
                            vulnerabilities.append({
                                "component": data_file,
                                "severity": "high",
                                "issue": "Sensitive file not in .gitignore"
                            })
                            score_deduction += 5
                
                # Check file permissions
                stat_info = os.stat(data_file)
                mode = stat_info.st_mode
                if mode & 0o004:  # Others can read
                    vulnerabilities.append({
                        "component": data_file,
                        "severity": "medium",
                        "issue": "Data file readable by others"
                    })
                    score_deduction += 3
        
        return {"vulnerabilities": vulnerabilities, "score_deduction": score_deduction}
    
    def test_network_security(self) -> Dict[str, Any]:
        """Test network security"""
        vulnerabilities = []
        score_deduction = 0
        
        # Check for exposed ports or insecure network configurations
        # This is a placeholder - in production would check actual network config
        
        # Check code for network security issues
        # Only flag if network keywords are used in actual network operations AND HTTPS/SSL is not used
        network_keywords = ["socket", "tcp", "udp"]
        agent_files = [
            "agents/optimus.py",
            "agents/ralph.py"
        ]
        
        for agent_file in agent_files:
            if os.path.exists(agent_file):
                try:
                    with open(agent_file, "r", encoding="utf-8") as f:
                        content = f.read()
                        content_lower = content.lower()
                        
                        # Check if network keywords are used in actual operations (not just imports/comments)
                        # Look for actual usage patterns: socket.socket(), socket.connect(), etc.
                        network_usage_patterns = [
                            r'socket\.socket\(',
                            r'socket\.connect\(',
                            r'socket\.bind\(',
                            r'socket\.listen\(',
                            r'\.connect\(.*[\'\"]tcp[\'"]',
                            r'\.connect\(.*[\'\"]udp[\'"]',
                        ]
                        
                        # Check if HTTPS/SSL/TLS is used (secure communication)
                        uses_secure = any(secure in content_lower for secure in ["https://", "ssl", "tls", "sslcontext", "ssl.wrap_socket", "requests.get", "requests.post"])
                        
                        # Only flag if insecure network operations are used AND secure communication is not used
                        uses_insecure_network = any(re.search(pattern, content, re.IGNORECASE) for pattern in network_usage_patterns)
                        
                        if uses_insecure_network and not uses_secure:
                            vulnerabilities.append({
                                "component": agent_file,
                                "severity": "medium",
                                "issue": "Network communication without SSL/TLS"
                            })
                            score_deduction += 3
                except Exception as e:
                    self.log_action(f"Error testing network security in {agent_file}: {e}")
        
        return {"vulnerabilities": vulnerabilities, "score_deduction": score_deduction}
    
    def generate_security_recommendations(self, vulnerabilities: List[Dict[str, Any]]) -> List[str]:
        """Generate security recommendations based on vulnerabilities"""
        recommendations = []
        
        # Group by component
        components = {}
        for vuln in vulnerabilities:
            component = vuln.get("component", "unknown")
            if component not in components:
                components[component] = []
            components[component].append(vuln)
        
        # Generate recommendations
        for component, vulns in components.items():
            severity_counts = {}
            for vuln in vulns:
                severity = vuln.get("severity", "medium")
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
            
            if severity_counts.get("critical", 0) > 0:
                recommendations.append(f"CRITICAL: Fix {severity_counts['critical']} critical vulnerabilities in {component}")
            elif severity_counts.get("high", 0) > 0:
                recommendations.append(f"HIGH: Address {severity_counts['high']} high-severity vulnerabilities in {component}")
            else:
                recommendations.append(f"Review {len(vulns)} vulnerabilities in {component}")
        
        return recommendations

# ----------------------
# Simple test harness when run directly
# ----------------------
if __name__ == "__main__":
    p = PhisherAgent()
    # test heuristic on this file
    print(p.audit_code_file(__file__))
    # test runtime log scan (safe if no logs exist)
    print(p.scan_runtime_logs("logs/optimus.log"))
    # simulated pen-test
    print(p.simulated_pen_test({"name": "local_test_target", "ip": "127.0.0.1"}))
