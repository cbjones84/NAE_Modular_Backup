#!/usr/bin/env python3
"""
Continuous Autonomous Operation System for NAE

Ensures NAE runs continuously, learns from errors, and improves holistically
in both dev/sandbox and prod/live modes.
"""

import os
import sys
import time
import json
import logging
import subprocess
import threading
import signal
import psutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
nae_ready_dir = os.path.abspath(os.path.join(current_dir, '../../..'))
if os.path.exists(nae_ready_dir):
    sys.path.insert(0, nae_ready_dir)
# Also add current directory structure
sys.path.insert(0, os.path.abspath(os.path.join(current_dir, '../..')))

logger = logging.getLogger(__name__)


class OperationMode(Enum):
    SANDBOX = "sandbox"
    LIVE = "live"
    DUAL = "dual"


class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


@dataclass
class ProcessHealth:
    """Health status of a process"""
    name: str
    pid: Optional[int]
    status: HealthStatus
    cpu_percent: float
    memory_mb: float
    uptime_seconds: float
    error_count: int
    last_error: Optional[str]
    last_check: str


@dataclass
class SystemHealth:
    """Overall system health"""
    timestamp: str
    mode: OperationMode
    overall_status: HealthStatus
    processes: List[ProcessHealth]
    system_resources: Dict[str, Any]
    error_rate: float
    learning_active: bool
    enhancements_applied: int


class ContinuousOperationManager:
    """
    Manages continuous autonomous operation of NAE
    """
    
    def __init__(self, mode: OperationMode = OperationMode.DUAL):
        self.mode = mode
        self.running = False
        self.processes: Dict[str, subprocess.Popen] = {}
        self.health_history: List[SystemHealth] = []
        self.error_log: List[Dict[str, Any]] = []
        self.learning_data: Dict[str, Any] = {}
        self.enhancement_count = 0
        
        # Configuration
        self.health_check_interval = 30  # seconds
        self.error_threshold = 10  # errors per hour
        self.restart_threshold = 3  # restarts before escalation
        self.learning_interval = 300  # 5 minutes
        
        # Process configurations
        self.process_configs = {
            "sandbox_optimus": {
                "command": ["python3", "-m", "agents.optimus"],
                "args": ["--sandbox"],
                "required": True,
                "restart_delay": 10,
                "max_restarts": 10000
            },
            "live_optimus": {
                "command": ["python3", "-m", "agents.optimus"],
                "args": ["--live"],
                "required": mode == OperationMode.LIVE or mode == OperationMode.DUAL,
                "restart_delay": 10,
                "max_restarts": 10000
            },
            "ralph": {
                "command": ["python3", "-m", "agents.ralph"],
                "args": [],
                "required": True,
                "restart_delay": 5,
                "max_restarts": 10000
            },
            "accelerator_controller": {
                "command": ["python3", "execution/integration/accelerator_controller.py"],
                # IMPORTANT: Must explicitly disable opposite mode since defaults are True
                # DUAL: both enabled, SANDBOX: sandbox only, LIVE: live only
                "args": ["--sandbox", "--live"] if mode == OperationMode.DUAL else (
                    ["--sandbox", "--no-live"] if mode == OperationMode.SANDBOX else 
                    ["--no-sandbox", "--live"]
                ),
                "required": True,
                "restart_delay": 10,
                "max_restarts": 10000
            },
            "master_controller": {
                "command": ["python3", "nae_autonomous_master.py"],
                "args": [],
                "required": True,
                "restart_delay": 5,
                "max_restarts": 10000
            }
        }
        
        # Threads
        self.health_monitor_thread: Optional[threading.Thread] = None
        self.error_monitor_thread: Optional[threading.Thread] = None
        self.learning_thread: Optional[threading.Thread] = None
        self.enhancement_thread: Optional[threading.Thread] = None
        self.healing_thread: Optional[threading.Thread] = None
        self.self_healing: Optional[Any] = None
        
        # Signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info(f"ContinuousOperationManager initialized in {mode.value} mode")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.stop()
        sys.exit(0)
    
    def start(self):
        """Start continuous operation"""
        logger.info("=" * 60)
        logger.info("STARTING CONTINUOUS AUTONOMOUS OPERATION")
        logger.info("=" * 60)
        logger.info(f"Mode: {self.mode.value}")
        logger.info(f"Health check interval: {self.health_check_interval}s")
        logger.info(f"Learning interval: {self.learning_interval}s")
        logger.info("=" * 60)
        
        self.running = True
        
        # Start all required processes
        self._start_all_processes()
        
        # Start monitoring threads
        self.health_monitor_thread = threading.Thread(
            target=self._health_monitor_loop,
            daemon=True
        )
        self.health_monitor_thread.start()
        
        self.error_monitor_thread = threading.Thread(
            target=self._error_monitor_loop,
            daemon=True
        )
        self.error_monitor_thread.start()
        
        self.learning_thread = threading.Thread(
            target=self._learning_loop,
            daemon=True
        )
        self.learning_thread.start()
        
        self.enhancement_thread = threading.Thread(
            target=self._enhancement_loop,
            daemon=True
        )
        self.enhancement_thread.start()
        
        # Start self-healing system
        try:
            # Try multiple import paths
            try:
                from execution.autonomous.self_healing import SelfHealingSystem
            except ImportError:
                # Try relative import
                sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
                from self_healing import SelfHealingSystem
            
            self.self_healing = SelfHealingSystem()
            self.healing_thread = threading.Thread(
                target=self._healing_loop,
                daemon=True
            )
            self.healing_thread.start()
            logger.info("✅ Self-healing system started")
        except Exception as e:
            logger.warning(f"Self-healing system not available: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            self.self_healing = None
        
        logger.info("✅ All systems started - NAE is now running autonomously")
        
        # Main loop
        try:
            while self.running:
                time.sleep(60)
                
                # Periodic status log
                if int(time.time()) % 3600 == 0:  # Every hour
                    health = self._check_system_health()
                    logger.info(f"System Status: {health.overall_status.value} | "
                              f"Processes: {len([p for p in health.processes if p.status == HealthStatus.HEALTHY])}/{len(health.processes)} | "
                              f"Enhancements: {self.enhancement_count}")
        except KeyboardInterrupt:
            logger.info("Shutdown requested")
            self.stop()
    
    def _start_all_processes(self):
        """Start all configured processes"""
        for name, config in self.process_configs.items():
            if config.get("required", False):
                if self.mode == OperationMode.SANDBOX and "live" in name:
                    continue
                if self.mode == OperationMode.LIVE and "sandbox" in name:
                    continue
                
                self._start_process(name, config)
                time.sleep(2)  # Stagger starts
    
    def _start_process(self, name: str, config: Dict[str, Any]) -> bool:
        """Start a single process"""
        try:
            cmd = config["command"] + config.get("args", [])
            log_file = Path("logs") / f"{name}.log"
            log_file.parent.mkdir(exist_ok=True)
            
            with open(log_file, "a") as log_f:
                process = subprocess.Popen(
                    cmd,
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                    cwd=Path.cwd(),
                    env=os.environ.copy()
                )
            
            self.processes[name] = process
            
            # Wait a moment to check if it started
            time.sleep(0.5)
            if process.poll() is not None:
                logger.error(f"❌ Process {name} exited immediately (exit code: {process.returncode})")
                return False
            
            logger.info(f"✅ Started {name} (PID: {process.pid})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start {name}: {e}")
            return False
    
    def _check_process_health(self, name: str) -> ProcessHealth:
        """Check health of a single process"""
        process = self.processes.get(name)
        
        if not process:
            return ProcessHealth(
                name=name,
                pid=None,
                status=HealthStatus.CRITICAL,
                cpu_percent=0.0,
                memory_mb=0.0,
                uptime_seconds=0.0,
                error_count=0,
                last_error=None,
                last_check=datetime.now().isoformat()
            )
        
        # Check if process is still running
        if process.poll() is not None:
            return ProcessHealth(
                name=name,
                pid=process.pid,
                status=HealthStatus.CRITICAL,
                cpu_percent=0.0,
                memory_mb=0.0,
                uptime_seconds=0.0,
                error_count=0,
                last_error=f"Process exited with code {process.returncode}",
                last_check=datetime.now().isoformat()
            )
        
        # Get process info
        try:
            proc = psutil.Process(process.pid)
            cpu_percent = proc.cpu_percent(interval=0.1)
            memory_mb = proc.memory_info().rss / 1024 / 1024
            uptime = (datetime.now() - datetime.fromtimestamp(proc.create_time())).total_seconds()
            
            # Check for errors in log
            error_count = self._count_recent_errors(name)
            last_error = self._get_last_error(name)
            
            # Determine status
            if error_count > self.error_threshold:
                status = HealthStatus.CRITICAL
            elif error_count > self.error_threshold / 2:
                status = HealthStatus.UNHEALTHY
            elif error_count > 0:
                status = HealthStatus.DEGRADED
            else:
                status = HealthStatus.HEALTHY
            
            return ProcessHealth(
                name=name,
                pid=process.pid,
                status=status,
                cpu_percent=cpu_percent,
                memory_mb=memory_mb,
                uptime_seconds=uptime,
                error_count=error_count,
                last_error=last_error,
                last_check=datetime.now().isoformat()
            )
            
        except psutil.NoSuchProcess:
            return ProcessHealth(
                name=name,
                pid=None,
                status=HealthStatus.CRITICAL,
                cpu_percent=0.0,
                memory_mb=0.0,
                uptime_seconds=0.0,
                error_count=0,
                last_error="Process not found",
                last_check=datetime.now().isoformat()
            )
    
    def _check_system_health(self) -> SystemHealth:
        """Check overall system health"""
        processes_health = [
            self._check_process_health(name)
            for name in self.process_configs.keys()
            if name in self.processes
        ]
        
        # System resources
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        system_resources = {
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "memory_available_gb": memory.available / (1024**3),
            "disk_percent": disk.percent,
            "disk_free_gb": disk.free / (1024**3)
        }
        
        # Calculate error rate (errors per hour)
        recent_errors = [
            e for e in self.error_log
            if (datetime.now() - datetime.fromisoformat(e["timestamp"])).total_seconds() < 3600
        ]
        error_rate = len(recent_errors)
        
        # Determine overall status
        critical_count = sum(1 for p in processes_health if p.status == HealthStatus.CRITICAL)
        unhealthy_count = sum(1 for p in processes_health if p.status == HealthStatus.UNHEALTHY)
        
        if critical_count > 0:
            overall_status = HealthStatus.CRITICAL
        elif unhealthy_count > len(processes_health) / 2:
            overall_status = HealthStatus.UNHEALTHY
        elif unhealthy_count > 0 or error_rate > self.error_threshold:
            overall_status = HealthStatus.DEGRADED
        else:
            overall_status = HealthStatus.HEALTHY
        
        health = SystemHealth(
            timestamp=datetime.now().isoformat(),
            mode=self.mode,
            overall_status=overall_status,
            processes=processes_health,
            system_resources=system_resources,
            error_rate=error_rate,
            learning_active=bool(self.learning_data),
            enhancements_applied=self.enhancement_count
        )
        
        self.health_history.append(health)
        
        # Keep only last 1000 health checks
        if len(self.health_history) > 1000:
            self.health_history = self.health_history[-1000:]
        
        return health
    
    def _health_monitor_loop(self):
        """Continuous health monitoring"""
        logger.info("Health monitor started")
        
        while self.running:
            try:
                health = self._check_system_health()
                
                # Restart unhealthy processes
                for proc_health in health.processes:
                    if proc_health.status == HealthStatus.CRITICAL:
                        logger.warning(f"⚠️ {proc_health.name} is CRITICAL - restarting...")
                        self._restart_process(proc_health.name)
                    elif proc_health.status == HealthStatus.UNHEALTHY:
                        logger.warning(f"⚠️ {proc_health.name} is UNHEALTHY - monitoring...")
                
                # Log status if degraded or worse
                if health.overall_status != HealthStatus.HEALTHY:
                    logger.warning(f"System health: {health.overall_status.value}")
                
                time.sleep(self.health_check_interval)
                
            except Exception as e:
                logger.error(f"Error in health monitor: {e}")
                time.sleep(self.health_check_interval)
    
    def _error_monitor_loop(self):
        """Monitor for errors and log them"""
        logger.info("Error monitor started")
        
        while self.running:
            try:
                # Check all process logs for errors
                for name in self.processes.keys():
                    self._scan_log_for_errors(name)
                
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in error monitor: {e}")
                time.sleep(60)
    
    def _scan_log_for_errors(self, process_name: str):
        """Scan process log for errors"""
        log_file = Path("logs") / f"{process_name}.log"
        
        if not log_file.exists():
            return
        
        try:
            # Read last 100 lines
            with open(log_file, "r") as f:
                lines = f.readlines()[-100:]
            
            # Look for error patterns
            error_patterns = [
                "ERROR", "Error", "error",
                "Exception", "Traceback",
                "CRITICAL", "FATAL",
                "failed", "Failed", "FAILED"
            ]
            
            for i, line in enumerate(lines):
                if any(pattern in line for pattern in error_patterns):
                    error_entry = {
                        "timestamp": datetime.now().isoformat(),
                        "process": process_name,
                        "error": line.strip(),
                        "line_number": len(lines) - 100 + i
                    }
                    
                    # Avoid duplicates
                    if error_entry not in self.error_log[-10:]:
                        self.error_log.append(error_entry)
                        
                        # Keep only last 1000 errors
                        if len(self.error_log) > 1000:
                            self.error_log = self.error_log[-1000:]
        
        except Exception as e:
            logger.debug(f"Error scanning log for {process_name}: {e}")
    
    def _count_recent_errors(self, process_name: str, minutes: int = 60) -> int:
        """Count recent errors for a process"""
        cutoff = datetime.now() - timedelta(minutes=minutes)
        return sum(
            1 for e in self.error_log
            if e["process"] == process_name
            and datetime.fromisoformat(e["timestamp"]) > cutoff
        )
    
    def _get_last_error(self, process_name: str) -> Optional[str]:
        """Get last error for a process"""
        process_errors = [e for e in self.error_log if e["process"] == process_name]
        if process_errors:
            return process_errors[-1]["error"]
        return None
    
    def _restart_process(self, name: str):
        """Restart a process"""
        config = self.process_configs.get(name)
        if not config:
            return
        
        logger.info(f"🔄 Restarting {name}...")
        
        # Stop existing process
        if name in self.processes:
            try:
                self.processes[name].terminate()
                self.processes[name].wait(timeout=5)
            except:
                try:
                    self.processes[name].kill()
                except:
                    pass
        
        # Wait before restart
        time.sleep(config.get("restart_delay", 10))
        
        # Start process
        self._start_process(name, config)
    
    def _learning_loop(self):
        """Continuous learning from errors and performance"""
        logger.info("Learning loop started")
        
        while self.running:
            try:
                # Analyze recent errors
                recent_errors = [
                    e for e in self.error_log
                    if (datetime.now() - datetime.fromisoformat(e["timestamp"])).total_seconds() < 3600
                ]
                
                if recent_errors:
                    # Learn from errors
                    self._learn_from_errors(recent_errors)
                
                # Analyze performance
                if len(self.health_history) > 10:
                    recent_health = self.health_history[-10:]
                    self._learn_from_performance(recent_health)
                
                time.sleep(self.learning_interval)
                
            except Exception as e:
                logger.error(f"Error in learning loop: {e}")
                time.sleep(self.learning_interval)
    
    def _learn_from_errors(self, errors: List[Dict[str, Any]]):
        """Learn patterns from errors"""
        # Group errors by process and type
        error_patterns = {}
        
        for error in errors:
            process = error["process"]
            error_msg = error["error"]
            
            # Extract error type
            error_type = "unknown"
            if "ImportError" in error_msg:
                error_type = "import"
            elif "AttributeError" in error_msg:
                error_type = "attribute"
            elif "KeyError" in error_msg:
                error_type = "key"
            elif "Connection" in error_msg:
                error_type = "connection"
            elif "Timeout" in error_msg:
                error_type = "timeout"
            
            key = f"{process}:{error_type}"
            if key not in error_patterns:
                error_patterns[key] = []
            error_patterns[key].append(error)
        
        # Store learning data
        self.learning_data["error_patterns"] = error_patterns
        self.learning_data["last_learning"] = datetime.now().isoformat()
        
        logger.debug(f"Learned from {len(errors)} errors: {len(error_patterns)} patterns")
    
    def _learn_from_performance(self, health_history: List[SystemHealth]):
        """Learn from performance patterns"""
        # Analyze trends
        avg_error_rate = sum(h.error_rate for h in health_history) / len(health_history)
        avg_cpu = sum(h.system_resources["cpu_percent"] for h in health_history) / len(health_history)
        avg_memory = sum(h.system_resources["memory_percent"] for h in health_history) / len(health_history)
        
        self.learning_data["performance"] = {
            "avg_error_rate": avg_error_rate,
            "avg_cpu": avg_cpu,
            "avg_memory": avg_memory,
            "timestamp": datetime.now().isoformat()
        }
    
    def _enhancement_loop(self):
        """Apply learned enhancements"""
        logger.info("Enhancement loop started")
        
        while self.running:
            try:
                # Check if we have learning data
                if not self.learning_data:
                    time.sleep(self.learning_interval)
                    continue
                
                # Apply enhancements based on learning
                enhancements_applied = self._apply_enhancements()
                
                if enhancements_applied > 0:
                    self.enhancement_count += enhancements_applied
                    logger.info(f"✅ Applied {enhancements_applied} enhancements")
                
                time.sleep(self.learning_interval)
                
            except Exception as e:
                logger.error(f"Error in enhancement loop: {e}")
                time.sleep(self.learning_interval)
    
    def _apply_enhancements(self) -> int:
        """Apply learned enhancements"""
        enhancements = 0
        
        # Example: Adjust health check interval based on error rate
        if "performance" in self.learning_data:
            perf = self.learning_data["performance"]
            if perf["avg_error_rate"] > self.error_threshold:
                # Increase monitoring frequency
                if self.health_check_interval > 10:
                    self.health_check_interval = max(10, self.health_check_interval - 5)
                    enhancements += 1
            elif perf["avg_error_rate"] < self.error_threshold / 2:
                # Decrease monitoring frequency to save resources
                if self.health_check_interval < 60:
                    self.health_check_interval = min(60, self.health_check_interval + 5)
                    enhancements += 1
        
        # Example: Pre-emptive restarts for problematic processes
        if "error_patterns" in self.learning_data:
            patterns = self.learning_data["error_patterns"]
            for pattern_key, errors in patterns.items():
                if len(errors) > 5:  # Frequent errors
                    process_name = pattern_key.split(":")[0]
                    logger.info(f"Pre-emptively restarting {process_name} due to error pattern")
                    self._restart_process(process_name)
                    enhancements += 1
        
        return enhancements
    
    def _healing_loop(self):
        """Self-healing loop"""
        logger.info("Self-healing loop started")
        
        while self.running:
            try:
                if self.self_healing:
                    # Heal from process logs
                    for name in self.processes.keys():
                        log_file = Path("logs") / f"{name}.log"
                        if log_file.exists():
                            fixed = self.self_healing.heal_from_log(str(log_file))
                            if fixed > 0:
                                logger.info(f"🔧 Fixed {fixed} issues in {name}")
                    
                    # Heal from error log
                    for error in self.error_log[-10:]:
                        issue = self.self_healing.detect_issue(
                            error["error"],
                            error.get("file")
                        )
                        if issue and issue.auto_fixable:
                            if self.self_healing.fix_issue(issue):
                                logger.info(f"🔧 Auto-fixed: {issue.issue_type.value}")
                
                time.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in healing loop: {e}")
                time.sleep(300)
    
    def stop(self):
        """Stop continuous operation"""
        logger.info("Stopping continuous operation...")
        self.running = False
        
        # Stop all processes
        for name, process in self.processes.items():
            try:
                process.terminate()
                process.wait(timeout=5)
                logger.info(f"Stopped {name}")
            except:
                try:
                    process.kill()
                except:
                    pass
        
        logger.info("✅ Shutdown complete")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status"""
        health = self._check_system_health()
        
        return {
            "running": self.running,
            "mode": self.mode.value,
            "overall_status": health.overall_status.value,
            "processes": {
                p.name: {
                    "status": p.status.value,
                    "pid": p.pid,
                    "uptime_seconds": p.uptime_seconds,
                    "error_count": p.error_count
                }
                for p in health.processes
            },
            "system_resources": health.system_resources,
            "error_rate": health.error_rate,
            "enhancements_applied": self.enhancement_count,
            "learning_active": health.learning_active
        }


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Continuous Autonomous Operation for NAE")
    parser.add_argument("--mode", choices=["sandbox", "live", "dual"], default="dual",
                       help="Operation mode")
    parser.add_argument("--health-interval", type=int, default=30,
                       help="Health check interval in seconds")
    parser.add_argument("--learning-interval", type=int, default=300,
                       help="Learning interval in seconds")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("logs/continuous_operation.log"),
            logging.StreamHandler()
        ]
    )
    
    # Create manager
    mode = OperationMode[args.mode.upper()]
    manager = ContinuousOperationManager(mode=mode)
    manager.health_check_interval = args.health_interval
    manager.learning_interval = args.learning_interval
    
    # Start continuous operation
    try:
        manager.start()
    except KeyboardInterrupt:
        logger.info("Shutdown requested")
    finally:
        manager.stop()


if __name__ == "__main__":
    main()

