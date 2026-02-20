#!/usr/bin/env python3
"""
NAE Autonomous Master Controller
Ensures NAE runs continuously and autonomously by any means necessary
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
    """Monitors and manages NAE processes"""
    
    def __init__(self):
        self.processes: Dict[str, subprocess.Popen] = {}
        self.process_configs = {
            "tradier_funds_activation": {
                "script": "execution/integration/tradier_funds_activation.py",
                "restart_delay": 10,
                "max_restarts": 10000,
                "restart_count": 0,
                "permanently_failed": False,
                "required": True
            },
            "continuous_trading_engine": {
                "script": "execution/integration/continuous_trading_engine.py",
                "restart_delay": 10,
                "max_restarts": 10000,
                "restart_count": 0,
                "permanently_failed": False,
                "required": True
            },
            "balance_monitor": {
                "script": "execution/monitoring/tradier_balance_monitor.py",
                "restart_delay": 5,
                "max_restarts": 10000,
                "restart_count": 0,
                "permanently_failed": False,
                "required": True
            },
            "optimus": {
                "module": "agents.optimus",
                "restart_delay": 10,
                "max_restarts": 10000,
                "restart_count": 0,
                "permanently_failed": False,
                "required": True,
                "env": {  # Mitigate numpy/BLAS SIGSEGV on macOS (Accelerate threading)
                    "OMP_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                    "NUMEXPR_NUM_THREADS": "1",
                }
            },
            "accelerator_controller": {
                "script": "execution/integration/accelerator_controller.py",
                "restart_delay": 10,
                "max_restarts": 5,
                "restart_count": 0,
                "required": False,
                "permanently_failed": False,
                "enabled": False,  # Disabled: SIGSEGV during Optimus/PyTorch init; set True when fixed
                "args": ["--live", "--no-sandbox", "--interval", "60"]
            }
        }
        self.running = True
    
    def start_process(self, name: str, config: Dict) -> bool:
        """Start a process"""
        try:
            python_cmd = sys.executable
            log_file = Path("logs") / f"{name}.log"
            log_file.parent.mkdir(exist_ok=True)

            # Support script path or Python module (-m)
            if "module" in config:
                cmd = [python_cmd, "-m", config["module"]]
            else:
                script_path = Path(config["script"])
                if not script_path.exists():
                    logger.error(f"Script not found: {script_path}")
                    return False
                cmd = [python_cmd, str(script_path)]

            if "args" in config:
                cmd.extend(config["args"])

            env = os.environ.copy()
            if "env" in config:
                env.update(config["env"])

            with open(log_file, "a") as log_f:
                process = subprocess.Popen(
                    cmd,
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                    cwd=Path.cwd(),
                    env=env
                )
            
            self.processes[name] = process
            
            # Check if process started successfully (wait a moment)
            time.sleep(0.5)
            if process.poll() is not None:
                # Process exited immediately - read error
                logger.error(f"❌ Process {name} exited immediately (exit code: {process.returncode})")
                # Try to read error from log
                if log_file.exists():
                    with open(log_file, "r") as f:
                        error_lines = f.readlines()[-10:]
                        logger.error(f"Last error lines from {name}:")
                        for line in error_lines:
                            logger.error(f"  {line.strip()}")
                return False
            
            logger.info(f"✅ Started {name} (PID: {process.pid})")
            return True
        
        except Exception as e:
            logger.error(f"Failed to start {name}: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def check_process(self, name: str) -> bool:
        """Check if process is running"""
        if name not in self.processes:
            return False
        
        process = self.processes[name]
        
        # Check if process is still alive
        if process.poll() is not None:
            # Process has terminated
            logger.warning(f"⚠️ Process {name} terminated (exit code: {process.returncode})")
            return False
        
        return True
    
    def restart_process(self, name: str, config: Dict) -> bool:
        """Restart a process"""
        config["restart_count"] += 1
        
        if config["restart_count"] > config["max_restarts"]:
            logger.error(f"❌ Max restarts reached for {name}")
            return False
        
        logger.info(f"🔄 Restarting {name} (attempt {config['restart_count']})")
        
        # Stop existing process
        if name in self.processes:
            try:
                self.processes[name].terminate()
                self.processes[name].wait(timeout=5)
            except (ProcessLookupError, OSError, TimeoutError):
                try:
                    self.processes[name].kill()
                except (ProcessLookupError, OSError):
                    pass
        
        # Wait before restart
        time.sleep(config["restart_delay"])
        
        # Start process
        return self.start_process(name, config)
    
    def monitor_loop(self):
        """Monitor all processes"""
        logger.info("Starting process monitoring loop")
        
        # Start all required processes (and enabled optional ones)
        for name, config in self.process_configs.items():
            if config.get("required", False) or config.get("enabled", False):
                self.start_process(name, config)
                time.sleep(2)  # Stagger starts
        
        # Monitor loop
        while self.running:
            try:
                for name, config in self.process_configs.items():
                    if config.get("permanently_failed", False):
                        continue
                    if not config.get("required", False) and not config.get("enabled", False):
                        continue  # Skip disabled optional processes
                    if name not in self.processes:
                        continue  # Never started (e.g. disabled)
                    if not self.check_process(name):
                        if config.get("required", False):
                            self.restart_process(name, config)
                        elif config.get("enabled", False) and config.get("restart_count", 0) < config.get("max_restarts", 10000):
                            self.restart_process(name, config)
                        elif config.get("enabled", False):
                            config["permanently_failed"] = True
                            logger.warning(f"⚠️ {name} exceeded max restarts ({config['max_restarts']}); stopping retries")
                
                time.sleep(30)  # Check every 30 seconds
            
            except KeyboardInterrupt:
                logger.info("Monitoring stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(30)
    
    def stop_all(self):
        """Stop all processes"""
        self.running = False
        for name, process in self.processes.items():
            try:
                process.terminate()
                process.wait(timeout=5)
                logger.info(f"Stopped {name}")
            except (ProcessLookupError, OSError, TimeoutError):
                try:
                    process.kill()
                except (ProcessLookupError, OSError):
                    pass


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
        
        # Check disk space
        disk = psutil.disk_usage('/')
        health["checks"]["disk_space"] = {
            "free_gb": disk.free / (1024**3),
            "percent_free": (disk.free / disk.total) * 100,
            "status": "ok" if disk.free > 10 * (1024**3) else "low"
        }
        
        # Check memory
        memory = psutil.virtual_memory()
        health["checks"]["memory"] = {
            "available_gb": memory.available / (1024**3),
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
        
        self.last_health_check = health
        return health
    
    def monitor_loop(self):
        """Continuous health monitoring"""
        while True:
            try:
                health = self.check_system_health()
                if health["status"] != "healthy":
                    logger.warning(f"System health degraded: {health}")
                
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
        logger.info("=" * 60)
        logger.info("Ensuring continuous autonomous operation")
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
            daemon=True
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

