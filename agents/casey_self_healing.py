#!/usr/bin/env python3
"""
Casey Self-Healing System

Self-awareness and self-healing capabilities for Casey.
Continuously monitors and improves itself and NAE.
"""

import os
import sys
import json
import time
import logging
import threading
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import deque

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """System health status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


@dataclass
class SystemHealth:
    """System health metrics"""
    timestamp: datetime
    overall_health: HealthStatus
    health_score: float  # 0.0 to 1.0
    agent_health: Dict[str, float] = field(default_factory=dict)
    system_metrics: Dict[str, Any] = field(default_factory=dict)
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class HealingAction:
    """Represents a self-healing action"""
    action_id: str
    issue: str
    action_type: str
    description: str
    implementation: str
    priority: str
    created_at: datetime
    status: str = "pending"
    applied_at: Optional[datetime] = None
    result: Optional[str] = None


class CaseySelfHealing:
    """
    Self-healing system for Casey
    Monitors, diagnoses, and fixes issues automatically
    """
    
    def __init__(self, casey_agent):
        """Initialize self-healing system"""
        self.casey = casey_agent
        self.health_history: deque = deque(maxlen=1000)
        self.healing_actions: Dict[str, HealingAction] = {}
        self.issue_patterns: Dict[str, int] = {}  # Track recurring issues
        
        # Monitoring
        self.monitoring_active = False
        self.monitor_thread = None
        self.check_interval = 300  # 5 minutes
        
        # Self-awareness metrics
        self.self_awareness_score = 1.0
        self.improvement_history: deque = deque(maxlen=500)
        
        logger.info("🔧 Casey self-healing system initialized")
    
    def start_monitoring(self):
        """Start continuous health monitoring"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("🚀 Self-healing monitoring started")
    
    def stop_monitoring(self):
        """Stop health monitoring"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("🛑 Self-healing monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Check system health
                health = self.check_system_health()
                self.health_history.append(health)
                
                # Detect issues
                issues = self._detect_issues(health)
                
                # Auto-heal if possible
                for issue in issues:
                    if self._can_auto_heal(issue):
                        self._apply_healing(issue)
                
                # Update self-awareness
                self._update_self_awareness(health)
                
                time.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.check_interval)
    
    def check_system_health(self) -> SystemHealth:
        """Check overall system health"""
        health_score = 1.0
        issues = []
        recommendations = []
        agent_health = {}
        
        # Check agent health
        if hasattr(self.casey, 'monitored_agents'):
            for agent_name, agent_info in self._get_agent_info().items():
                agent_score = self._check_agent_health(agent_name, agent_info)
                agent_health[agent_name] = agent_score
                health_score = min(health_score, agent_score)
                
                if agent_score < 0.7:
                    issues.append(f"{agent_name} health degraded ({agent_score:.2f})")
                    recommendations.append(f"Investigate {agent_name} performance")
        
        # Check system resources
        try:
            import psutil
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            if cpu_percent > 80:
                health_score -= 0.2
                issues.append(f"High CPU usage: {cpu_percent}%")
                recommendations.append("Optimize resource usage")
            
            if memory.percent > 85:
                health_score -= 0.2
                issues.append(f"High memory usage: {memory.percent}%")
                recommendations.append("Free up memory")
            
        except Exception as e:
            logger.warning(f"Could not check system resources: {e}")
        
        # Check log errors
        error_count = self._count_recent_errors()
        if error_count > 10:
            health_score -= 0.1
            issues.append(f"High error rate: {error_count} errors in last hour")
            recommendations.append("Review error logs")
        
        # Determine overall health
        if health_score >= 0.8:
            overall_health = HealthStatus.HEALTHY
        elif health_score >= 0.5:
            overall_health = HealthStatus.DEGRADED
        elif health_score >= 0.3:
            overall_health = HealthStatus.UNHEALTHY
        else:
            overall_health = HealthStatus.CRITICAL
        
        health = SystemHealth(
            timestamp=datetime.now(),
            overall_health=overall_health,
            health_score=max(0.0, min(1.0, health_score)),
            agent_health=agent_health,
            system_metrics={
                "cpu_percent": cpu_percent if 'cpu_percent' in locals() else 0,
                "memory_percent": memory.percent if 'memory' in locals() else 0,
                "error_count": error_count
            },
            issues=issues,
            recommendations=recommendations
        )
        
        return health
    
    def _get_agent_info(self) -> Dict[str, Any]:
        """Get information about monitored agents"""
        agents = {}
        
        if hasattr(self.casey, 'monitored_agents'):
            if isinstance(self.casey.monitored_agents, dict):
                for name, process in self.casey.monitored_agents.items():
                    agents[name] = {"process": process}
            elif isinstance(self.casey.monitored_agents, list):
                for name, process in self.casey.monitored_agents:
                    agents[name] = {"process": process}
        
        return agents
    
    def _check_agent_health(self, agent_name: str, agent_info: Dict[str, Any]) -> float:
        """Check health of a specific agent"""
        score = 1.0
        
        try:
            process = agent_info.get("process")
            if process and hasattr(process, 'is_running'):
                if not process.is_running():
                    return 0.0  # Agent is dead
            
            # Check agent logs for errors
            log_file = f"logs/{agent_name.lower()}.log"
            if os.path.exists(log_file):
                error_count = self._count_errors_in_log(log_file)
                if error_count > 5:
                    score -= 0.3
                elif error_count > 0:
                    score -= 0.1
        
        except Exception as e:
            logger.warning(f"Error checking agent health for {agent_name}: {e}")
            score -= 0.2
        
        return max(0.0, min(1.0, score))
    
    def _count_recent_errors(self) -> int:
        """Count recent errors in logs"""
        error_count = 0
        log_dir = "logs"
        
        if not os.path.exists(log_dir):
            return 0
        
        cutoff_time = datetime.now() - timedelta(hours=1)
        
        for log_file in os.listdir(log_dir):
            if log_file.endswith('.log'):
                log_path = os.path.join(log_dir, log_file)
                error_count += self._count_errors_in_log(log_path, since=cutoff_time)
        
        return error_count
    
    def _count_errors_in_log(self, log_path: str, since: Optional[datetime] = None) -> int:
        """Count errors in a log file"""
        try:
            error_count = 0
            with open(log_path, 'r') as f:
                for line in f:
                    if any(keyword in line.lower() for keyword in ['error', 'exception', 'failed', 'critical']):
                        if since:
                            # Parse timestamp if available
                            try:
                                timestamp_str = line.split(']')[0].replace('[', '')
                                log_time = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                                if log_time >= since:
                                    error_count += 1
                            except:
                                error_count += 1
                        else:
                            error_count += 1
            return error_count
        except:
            return 0
    
    def _detect_issues(self, health: SystemHealth) -> List[str]:
        """Detect issues from health check"""
        issues = []
        
        # Add health issues
        issues.extend(health.issues)
        
        # Detect patterns
        if len(health.issues) > 0:
            for issue in health.issues:
                # Track recurring issues
                if issue in self.issue_patterns:
                    self.issue_patterns[issue] += 1
                else:
                    self.issue_patterns[issue] = 1
                
                # Flag recurring issues
                if self.issue_patterns[issue] > 3:
                    issues.append(f"RECURRING: {issue}")
        
        return issues
    
    def _can_auto_heal(self, issue: str) -> bool:
        """Check if issue can be auto-healed"""
        # Issues that can be auto-healed
        auto_healable = [
            "high cpu usage",
            "high memory usage",
            "agent health degraded",
            "log file too large",
            "cache needs clearing"
        ]
        
        issue_lower = issue.lower()
        return any(healable in issue_lower for healable in auto_healable)
    
    def _apply_healing(self, issue: str):
        """Apply healing action for an issue"""
        action_id = f"heal_{int(time.time())}"
        
        healing_action = HealingAction(
            action_id=action_id,
            issue=issue,
            action_type=self._determine_healing_type(issue),
            description=f"Auto-healing: {issue}",
            implementation=self._generate_healing_implementation(issue),
            priority="high" if "critical" in issue.lower() else "medium",
            created_at=datetime.now()
        )
        
        try:
            # Apply healing
            result = self._execute_healing(healing_action)
            healing_action.status = "completed" if result else "failed"
            healing_action.applied_at = datetime.now()
            healing_action.result = result
            
            self.healing_actions[action_id] = healing_action
            
            if result:
                self.casey.log_action(f"🔧 Auto-healed: {issue}")
            else:
                self.casey.log_action(f"❌ Auto-healing failed: {issue}")
        
        except Exception as e:
            healing_action.status = "failed"
            healing_action.result = str(e)
            logger.error(f"Error applying healing: {e}")
    
    def _determine_healing_type(self, issue: str) -> str:
        """Determine healing action type"""
        issue_lower = issue.lower()
        
        if "cpu" in issue_lower or "memory" in issue_lower:
            return "resource_optimization"
        elif "agent" in issue_lower:
            return "agent_restart"
        elif "log" in issue_lower:
            return "log_cleanup"
        elif "cache" in issue_lower:
            return "cache_clear"
        else:
            return "general"
    
    def _generate_healing_implementation(self, issue: str) -> str:
        """Generate healing implementation"""
        issue_lower = issue.lower()
        
        if "cpu" in issue_lower or "memory" in issue_lower:
            return "Optimize resource usage: Reduce monitoring frequency, clear caches"
        elif "agent" in issue_lower:
            return "Restart degraded agent or adjust its configuration"
        elif "log" in issue_lower:
            return "Rotate log files, clear old logs"
        elif "cache" in issue_lower:
            return "Clear file context cache and codebase index"
        else:
            return f"Investigate and fix: {issue}"
    
    def _execute_healing(self, action: HealingAction) -> bool:
        """Execute healing action"""
        try:
            if action.action_type == "resource_optimization":
                # Clear caches
                if hasattr(self.casey, 'file_context_cache'):
                    self.casey.file_context_cache.clear()
                if hasattr(self.casey, 'codebase_index'):
                    self.casey.codebase_index.clear()
                return True
            
            elif action.action_type == "log_cleanup":
                # Rotate logs
                log_dir = "logs"
                if os.path.exists(log_dir):
                    for log_file in os.listdir(log_dir):
                        if log_file.endswith('.log'):
                            log_path = os.path.join(log_dir, log_file)
                            # Keep last 1000 lines
                            try:
                                with open(log_path, 'r') as f:
                                    lines = f.readlines()
                                if len(lines) > 1000:
                                    with open(log_path, 'w') as f:
                                        f.writelines(lines[-1000:])
                            except:
                                pass
                return True
            
            elif action.action_type == "cache_clear":
                if hasattr(self.casey, 'file_context_cache'):
                    self.casey.file_context_cache.clear()
                if hasattr(self.casey, 'codebase_index'):
                    self.casey.codebase_index.clear()
                return True
            
            return False
        
        except Exception as e:
            logger.error(f"Error executing healing: {e}")
            return False
    
    def _update_self_awareness(self, health: SystemHealth):
        """Update self-awareness score"""
        # Self-awareness based on:
        # 1. Health monitoring accuracy
        # 2. Issue detection rate
        # 3. Healing success rate
        
        base_score = health.health_score
        
        # Adjust based on healing success
        if self.healing_actions:
            successful_heals = sum(1 for a in self.healing_actions.values() if a.status == "completed")
            total_heals = len(self.healing_actions)
            healing_rate = successful_heals / total_heals if total_heals > 0 else 1.0
            base_score = (base_score + healing_rate) / 2
        
        self.self_awareness_score = max(0.0, min(1.0, base_score))
    
    def get_self_awareness_report(self) -> Dict[str, Any]:
        """Get self-awareness report"""
        return {
            "self_awareness_score": self.self_awareness_score,
            "health_history_count": len(self.health_history),
            "healing_actions_count": len(self.healing_actions),
            "successful_heals": sum(1 for a in self.healing_actions.values() if a.status == "completed"),
            "recurring_issues": dict(sorted(self.issue_patterns.items(), key=lambda x: x[1], reverse=True)[:10]),
            "latest_health": asdict(self.health_history[-1]) if self.health_history else None
        }

