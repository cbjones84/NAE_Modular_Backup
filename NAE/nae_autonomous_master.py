#!/usr/bin/env python3
"""
NAE Autonomous Master Controller
Ensures NAE runs continuously and autonomously by any means necessary
Production mode - runs all agents continuously
"""

import os
import sys
import time
import json
import logging
import threading
import subprocess
import signal
import psutil
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
import traceback

# Load environment variables from .env.prod or .env
def _load_env_file(filepath: str) -> bool:
    """Load environment variables from a file"""
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, _, value = line.partition('=')
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    os.environ[key] = value
        return True
    return False

# Load .env.prod first, fallback to .env
_script_dir = Path(__file__).parent
if _load_env_file(str(_script_dir / ".env.prod")):
    pass  # Loaded .env.prod
elif _load_env_file(str(_script_dir / ".env")):
    pass  # Loaded .env

# Setup logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / "nae_autonomous_master.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ProcessMonitor:
    """Monitors and manages NAE processes and agents"""
    
    def __init__(self):
        self.processes: Dict[str, subprocess.Popen] = {}
        self.process_configs = self._load_process_configs()
        self.running = True
        self.restart_counts: Dict[str, int] = {}
    
    def _load_process_configs(self) -> Dict:
        """Load process configurations"""
        configs = {}
        
        # Master API Server
        configs["master_api"] = {
            "script": "master_api/server.py",
            "restart_delay": 5,
            "max_restarts": 10000,
            "required": True
        }
        
        # Add agents if they exist
        agents_dir = Path("agents")
        if agents_dir.exists():
            for agent_file in agents_dir.glob("*.py"):
                if agent_file.name.startswith("__"):
                    continue
                agent_name = agent_file.stem
                # Prioritize Ralph GitHub research and Optimus (trading + accelerator)
                priority = agent_file.name.startswith("ralph_github")
                is_optimus = agent_file.name.startswith("optimus_launcher")
                configs[f"agent_{agent_name}"] = {
                    "script": str(agent_file),
                    "restart_delay": 10 if not (priority or is_optimus) else 5,
                    "max_restarts": 10000,
                    "required": priority or agent_file.name.startswith("ralph") or is_optimus
                }
        
        # Redis if needed
        configs["redis_check"] = {
            "script": "redis_kill_switch.py",
            "restart_delay": 30,
            "max_restarts": 100,
            "required": False,
            "check_only": True  # Just check, don't run continuously
        }
        
        return configs
    
    def start_process(self, name: str, config: Dict) -> bool:
        """Start a process"""
        try:
            script_path = Path(config["script"])
            if not script_path.exists():
                logger.warning(f"Script not found: {script_path} (skipping)")
                return False
            
            # Use Python from current environment
            python_cmd = sys.executable
            
            # Log file for this process
            log_file = log_dir / f"{name}.log"
            log_file.parent.mkdir(exist_ok=True)
            
            # Start process with output redirected to log file
            with open(log_file, "a") as log_f:
                log_f.write(f"\n{'='*60}\n")
                log_f.write(f"Started: {datetime.now().isoformat()}\n")
                log_f.write(f"{'='*60}\n")
                
                process = subprocess.Popen(
                    [python_cmd, str(script_path)],
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                    cwd=Path.cwd(),
                    env=os.environ.copy()
                )
            
            self.processes[name] = process
            self.restart_counts[name] = self.restart_counts.get(name, 0)
            
            # Check if process started successfully
            time.sleep(1)
            if process.poll() is not None:
                logger.error(f"❌ Process {name} exited immediately (exit code: {process.returncode})")
                if log_file.exists():
                    with open(log_file, "r") as f:
                        lines = f.readlines()
                        if lines:
                            logger.error(f"Last log lines: {''.join(lines[-10:])}")
                return False
            
            logger.info(f"✅ Started {name} (PID: {process.pid})")
            return True
            
        except Exception as e:
            logger.error(f"Error starting {name}: {e}")
            return False
    
    def stop_process(self, name: str):
        """Stop a process"""
        if name in self.processes:
            process = self.processes[name]
            try:
                process.terminate()
                process.wait(timeout=10)
                logger.info(f"✅ Stopped {name}")
            except subprocess.TimeoutExpired:
                process.kill()
                logger.warning(f"⚠️  Force killed {name}")
            except Exception as e:
                logger.error(f"Error stopping {name}: {e}")
            finally:
                del self.processes[name]
    
    def restart_process(self, name: str, config: Dict):
        """Restart a process"""
        if name in self.restart_counts:
            self.restart_counts[name] += 1
        else:
            self.restart_counts[name] = 1
        
        restart_count = self.restart_counts[name]
        max_restarts = config.get("max_restarts", 100)
        
        if restart_count > max_restarts:
            logger.error(f"❌ {name} exceeded max restarts ({max_restarts}), stopping")
            return
        
        logger.info(f"🔄 Restarting {name} (attempt {restart_count})")
        self.stop_process(name)
        time.sleep(config.get("restart_delay", 5))
        self.start_process(name, config)
    
    def check_process(self, name: str, config: Dict) -> bool:
        """Check if a process is running"""
        if name not in self.processes:
            return False
        
        process = self.processes[name]
        
        # Check if process is still alive
        if process.poll() is not None:
            # Process has exited
            exit_code = process.returncode
            logger.warning(f"⚠️  Process {name} exited (code: {exit_code})")
            
            if config.get("required", False):
                self.restart_process(name, config)
            else:
                del self.processes[name]
            
            return False
        
        return True
    
    def monitor_loop(self):
        """Continuous monitoring loop"""
        logger.info("🔍 Starting process monitor...")
        
        # Initial startup of all processes
        for name, config in self.process_configs.items():
            if not config.get("check_only", False):
                self.start_process(name, config)
                time.sleep(2)  # Stagger startup
        
        # Monitoring loop
        while self.running:
            try:
                for name, config in list(self.process_configs.items()):
                    if not config.get("check_only", False):
                        if not self.check_process(name, config):
                            # Process was restarted or removed
                            pass
                
                time.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Monitor loop error: {e}")
                time.sleep(10)
    
    def stop_all(self):
        """Stop all processes"""
        logger.info("Stopping all processes...")
        self.running = False
        for name in list(self.processes.keys()):
            self.stop_process(name)
        logger.info("✅ All processes stopped")


class HealthMonitor:
    """Monitors NAE system health"""
    
    def __init__(self):
        self.health_checks = []
        self.last_health_check = None
        self.health_status = "unknown"
    
    def check_system_health(self) -> Dict[str, Any]:
        """Perform comprehensive health check"""
        health = {
            "timestamp": datetime.now().isoformat(),
            "status": "healthy",
            "checks": {}
        }
        
        try:
            # Check disk space
            disk = psutil.disk_usage('/')
            health["checks"]["disk_space"] = {
                "free_gb": round(disk.free / (1024**3), 2),
                "percent_free": round((disk.free / disk.total) * 100, 2),
                "status": "ok" if disk.free > 10 * (1024**3) else "low"
            }
            
            # Check memory
            memory = psutil.virtual_memory()
            health["checks"]["memory"] = {
                "available_gb": round(memory.available / (1024**3), 2),
                "percent_used": memory.percent,
                "status": "ok" if memory.percent < 90 else "high"
            }
            
            # Check CPU
            cpu_percent = psutil.cpu_percent(interval=1)
            health["checks"]["cpu"] = {
                "percent": cpu_percent,
                "status": "ok" if cpu_percent < 90 else "high"
            }
            
            # Determine overall status
            if any(check["status"] != "ok" for check in health["checks"].values()):
                health["status"] = "degraded"
            
        except Exception as e:
            logger.error(f"Health check error: {e}")
            health["status"] = "error"
        
        self.last_health_check = health
        return health
    
    def monitor_loop(self):
        """Continuous health monitoring"""
        while True:
            try:
                health = self.check_system_health()
                if health["status"] != "healthy":
                    logger.warning(f"System health: {health['status']} - {health['checks']}")
                
                time.sleep(300)  # Check every 5 minutes
            
            except Exception as e:
                logger.error(f"Health check error: {e}")
                time.sleep(300)


class NAEAutonomousMaster:
    """Master controller for autonomous NAE operation"""
    
    def __init__(self):
        self.running = True
        self.process_monitor = ProcessMonitor()
        self.health_monitor = HealthMonitor()
        
        # Threads
        self.monitor_thread: Optional[threading.Thread] = None
        self.health_thread: Optional[threading.Thread] = None
        
        # Signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info("=" * 60)
        logger.info("NAE AUTONOMOUS MASTER CONTROLLER")
        logger.info("Production Mode - Continuous Operation")
        logger.info("=" * 60)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.stop()
        sys.exit(0)
    
    def start(self):
        """Start autonomous operation"""
        logger.info("🚀 Starting NAE Autonomous Master Controller")
        
        # Start process monitoring
        self.monitor_thread = threading.Thread(
            target=self.process_monitor.monitor_loop,
            daemon=False
        )
        self.monitor_thread.start()
        
        # Start health monitoring
        self.health_thread = threading.Thread(
            target=self.health_monitor.monitor_loop,
            daemon=True
        )
        self.health_thread.start()
        
        logger.info("✅ All monitoring systems started")
        logger.info("NAE is now running autonomously and continuously")
        
        # Keep main thread alive
        try:
            while self.running:
                time.sleep(60)
                
                # Periodic status log
                if int(time.time()) % 3600 == 0:  # Every hour
                    logger.info("NAE Autonomous Master: All systems operational")
                    health = self.health_monitor.check_system_health()
                    logger.info(f"System health: {health['status']}")
        
        except KeyboardInterrupt:
            logger.info("Shutdown requested")
            self.stop()
    
    def stop(self):
        """Stop autonomous operation"""
        logger.info("Stopping NAE Autonomous Master Controller...")
        self.running = False
        self.process_monitor.stop_all()
        logger.info("✅ Shutdown complete")


def main():
    """Main entry point"""
    master = NAEAutonomousMaster()
    
    try:
        master.start()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()

