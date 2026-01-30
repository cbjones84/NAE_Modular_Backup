#!/usr/bin/env python3
"""
NAE Master Automation Scheduler
Automates all agents and their roles within the NAE system
"""

import os
import sys
import time
import threading
import datetime
import json
from typing import Dict, List, Any, Optional
from pathlib import Path

# Try to import schedule module, use fallback if not available
try:
    import schedule
    SCHEDULE_AVAILABLE = True
except ImportError:
    SCHEDULE_AVAILABLE = False
    print("[WARNING] schedule module not available. Using simple time-based scheduling.")

# Add parent directory to path
sys.path.append(os.path.dirname(__file__))

# Import all agents
from agents.ralph import RalphAgent
from agents.donnie import DonnieAgent
from agents.optimus import OptimusAgent
from agents.casey import CaseyAgent
from agents.splinter import SplinterAgent
from agents.rocksteady import RocksteadyAgent
from agents.bebop import BebopAgent
from agents.phisher import PhisherAgent
from agents.genny import GennyAgent

# Try to import optional agents
try:
    from agents.april import AprilAgent
    APRIL_AVAILABLE = True
except ImportError:
    APRIL_AVAILABLE = False

try:
    from agents.mikey import MikeyAgent
    MIKEY_AVAILABLE = True
except ImportError:
    MIKEY_AVAILABLE = False

try:
    from agents.leo import LeoAgent
    LEO_AVAILABLE = True
except ImportError:
    LEO_AVAILABLE = False

try:
    from agents.shredder import ShredderAgent
    SHREDDER_AVAILABLE = True
except ImportError:
    SHREDDER_AVAILABLE = False

# ----------------------
# Configuration
# ----------------------
class AutomationConfig:
    """Configuration for agent automation"""
    
    # Core Trading Agents (run frequently)
    RALPH_INTERVAL_MINUTES = 30  # Generate strategies every 30 minutes
    DONNIE_INTERVAL_MINUTES = 30  # Process strategies every 30 minutes
    OPTIMUS_INTERVAL_SECONDS = 10  # Check for trades every 10 seconds
    
    # Monitoring Agents (run periodically)
    CASEY_INTERVAL_MINUTES = 120  # Check/build agents every 2 hours
    BEBOP_INTERVAL_MINUTES = 15  # Monitor system every 15 minutes
    SPLINTER_INTERVAL_MINUTES = 60  # Orchestrate every hour
    
    # Security Agents (run less frequently)
    ROCKSTEADY_INTERVAL_HOURS = 6  # Security sweep every 6 hours
    PHISHER_INTERVAL_MINUTES = 60  # Security scan every hour (increased frequency for compliance)
    
    # Other Agents
    GENNY_INTERVAL_MINUTES = 180  # Run every 3 hours
    APRIL_INTERVAL_MINUTES = 360  # Run every 6 hours (crypto operations)
    LEO_INTERVAL_MINUTES = 60  # Run every hour
    MIKEY_INTERVAL_MINUTES = 60  # Run every hour
    SHREDDER_INTERVAL_MINUTES = 60  # Run every hour
    
    # Enable/disable agents
    ENABLE_RALPH = True
    ENABLE_DONNIE = True
    ENABLE_OPTIMUS = True
    ENABLE_CASEY = True
    ENABLE_BEBOP = True
    ENABLE_SPLINTER = True
    ENABLE_ROCKSTEADY = True
    ENABLE_PHISHER = True
    ENABLE_GENNY = True
    ENABLE_APRIL = True
    ENABLE_LEO = True
    ENABLE_MIKEY = True
    ENABLE_SHREDDER = True

# ----------------------
# Agent Automation Wrappers
# ----------------------
class AgentAutomation:
    """Base class for agent automation"""
    
    def __init__(self, agent, name: str):
        self.agent = agent
        self.name = name
        self.last_run = None
        self.run_count = 0
        self.success_count = 0
        self.error_count = 0
        self.last_error = None
        self.enabled = True
        
    def run(self, *args, **kwargs):
        """Run the agent's cycle"""
        if not self.enabled:
            return {"status": "disabled", "agent": self.name}
        
        try:
            self.last_run = datetime.datetime.now()
            self.run_count += 1
            
            # Call the appropriate method based on agent type
            if hasattr(self.agent, 'run_cycle'):
                result = self.agent.run_cycle(*args, **kwargs)
            elif hasattr(self.agent, 'run'):
                result = self.agent.run(*args, **kwargs)
            else:
                result = {"status": "no_run_method", "agent": self.name}
            
            self.success_count += 1
            self.last_error = None
            return {"status": "success", "agent": self.name, "result": result}
            
        except Exception as e:
            self.error_count += 1
            self.last_error = str(e)
            print(f"[ERROR] {self.name} failed: {e}")
            return {"status": "error", "agent": self.name, "error": str(e)}
    
    def get_status(self):
        """Get automation status"""
        return {
            "name": self.name,
            "enabled": self.enabled,
            "last_run": self.last_run.isoformat() if self.last_run else None,
            "run_count": self.run_count,
            "success_count": self.success_count,
            "error_count": self.error_count,
            "success_rate": (self.success_count / self.run_count * 100) if self.run_count > 0 else 0,
            "last_error": self.last_error
        }

# ----------------------
# Master Automation Scheduler
# ----------------------
class NAEMasterScheduler:
    """Master scheduler for all NAE agents"""
    
    def __init__(self, config: AutomationConfig = None):
        self.config = config or AutomationConfig()
        self.agents: Dict[str, AgentAutomation] = {}
        self.running = False
        self.log_file = "logs/master_scheduler.log"
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        
        # Compliance checking
        self.compliance_check_interval_hours = 24  # Check compliance daily
        self.last_compliance_check = None
        
        # Initialize all agents
        self._initialize_agents()
        
        # Setup schedules
        self._setup_schedules()
    
    def _initialize_agents(self):
        """Initialize all agents"""
        self.log_action("[MASTER] Initializing agents with THRML support...")
        
        # Check THRML availability
        try:
            import thrml
            import jax
            self.log_action(f"[MASTER] ✅ THRML {thrml.__version__} and JAX {jax.__version__} available")
        except ImportError:
            self.log_action("[MASTER] ⚠️  THRML not available, using JAX fallback implementations")
        
        # Core Trading Agents
        if self.config.ENABLE_RALPH:
            try:
                ralph = RalphAgent()
                thrml_status = "✅" if (hasattr(ralph, 'thrml_enabled') and ralph.thrml_enabled) else "⚠️"
                self.agents['Ralph'] = AgentAutomation(ralph, 'Ralph')
                self.log_action(f"[MASTER] Ralph initialized {thrml_status} THRML")
            except Exception as e:
                self.log_action(f"[MASTER] ERROR: Ralph initialization failed: {e}")
        
        if self.config.ENABLE_DONNIE:
            try:
                donnie = DonnieAgent()
                thrml_status = "✅" if (hasattr(donnie, 'thrml_enabled') and donnie.thrml_enabled) else "⚠️"
                self.agents['Donnie'] = AgentAutomation(donnie, 'Donnie')
                self.log_action(f"[MASTER] Donnie initialized {thrml_status} THRML")
            except Exception as e:
                self.log_action(f"[MASTER] ERROR: Donnie initialization failed: {e}")
        
        if self.config.ENABLE_OPTIMUS:
            try:
                # Initialize OptimusAgent in LIVE mode (sandbox=False)
                # This connects to live Alpaca trading account
                optimus = OptimusAgent(sandbox=False)  # LIVE mode - connects to live Alpaca account
                thrml_status = "✅" if (hasattr(optimus, 'thrml_enabled') and optimus.thrml_enabled) else "⚠️"
                self.agents['Optimus'] = AgentAutomation(optimus, 'Optimus')
                self.log_action(f"[MASTER] Optimus initialized in LIVE mode {thrml_status} THRML")
            except Exception as e:
                self.log_action(f"[MASTER] ERROR: Optimus initialization failed: {e}")
        
        # Monitoring Agents
        if self.config.ENABLE_CASEY:
            try:
                casey = CaseyAgent()
                self.agents['Casey'] = AgentAutomation(casey, 'Casey')
                self.log_action("[MASTER] Casey initialized")
            except Exception as e:
                self.log_action(f"[MASTER] ERROR: Casey initialization failed: {e}")
        
        if self.config.ENABLE_BEBOP:
            try:
                bebop = BebopAgent()
                self.agents['Bebop'] = AgentAutomation(bebop, 'Bebop')
                self.log_action("[MASTER] Bebop initialized")
            except Exception as e:
                self.log_action(f"[MASTER] ERROR: Bebop initialization failed: {e}")
        
        if self.config.ENABLE_SPLINTER:
            try:
                splinter = SplinterAgent()
                self.agents['Splinter'] = AgentAutomation(splinter, 'Splinter')
                self.log_action("[MASTER] Splinter initialized")
            except Exception as e:
                self.log_action(f"[MASTER] ERROR: Splinter initialization failed: {e}")
        
        # Security Agents
        if self.config.ENABLE_ROCKSTEADY:
            try:
                rocksteady = RocksteadyAgent()
                self.agents['Rocksteady'] = AgentAutomation(rocksteady, 'Rocksteady')
                self.log_action("[MASTER] Rocksteady initialized")
            except Exception as e:
                self.log_action(f"[MASTER] ERROR: Rocksteady initialization failed: {e}")
        
        if self.config.ENABLE_PHISHER:
            try:
                phisher = PhisherAgent()
                # Connect Phisher to Bebop, Rocksteady, and Casey for alerting
                if 'Bebop' in self.agents:
                    phisher.bebop_agent = self.agents['Bebop'].agent
                if 'Rocksteady' in self.agents:
                    phisher.rocksteady_agent = self.agents['Rocksteady'].agent
                if 'Casey' in self.agents:
                    phisher.casey_agent = self.agents['Casey'].agent
                self.agents['Phisher'] = AgentAutomation(phisher, 'Phisher')
                self.log_action("[MASTER] Phisher initialized with security alerting to Bebop, Rocksteady, and Casey")
            except Exception as e:
                self.log_action(f"[MASTER] ERROR: Phisher initialization failed: {e}")
        
        # Connect all security agents bidirectionally
        if 'Bebop' in self.agents and 'Phisher' in self.agents and 'Rocksteady' in self.agents and 'Casey' in self.agents:
            try:
                bebop = self.agents['Bebop'].agent
                phisher = self.agents['Phisher'].agent
                rocksteady = self.agents['Rocksteady'].agent
                casey = self.agents['Casey'].agent
                
                # Connect Bebop to all security team members
                bebop.phisher_agent = phisher
                bebop.rocksteady_agent = rocksteady
                bebop.casey_agent = casey
                
                # Connect Rocksteady to all security team members
                rocksteady.phisher_agent = phisher
                rocksteady.bebop_agent = bebop
                rocksteady.casey_agent = casey
                
                # Connect Casey to all security team members
                casey.phisher_agent = phisher
                casey.bebop_agent = bebop
                casey.rocksteady_agent = rocksteady
                
                self.log_action("[MASTER] Bidirectional security alerting configured: Bebop ↔ Rocksteady ↔ Casey ↔ Phisher")
            except Exception as e:
                self.log_action(f"[MASTER] ERROR: Security agent connections failed: {e}")
        
        # Other Agents
        if self.config.ENABLE_GENNY:
            try:
                genny = GennyAgent()
                self.agents['Genny'] = AgentAutomation(genny, 'Genny')
                self.log_action("[MASTER] Genny initialized")
            except Exception as e:
                self.log_action(f"[MASTER] Genny not available: {e}")
        
        if self.config.ENABLE_APRIL and APRIL_AVAILABLE:
            try:
                april = AprilAgent()
                self.agents['April'] = AgentAutomation(april, 'April')
                self.log_action("[MASTER] April initialized")
            except Exception as e:
                self.log_action(f"[MASTER] ERROR: April initialization failed: {e}")
        
        if self.config.ENABLE_LEO and LEO_AVAILABLE:
            try:
                leo = LeoAgent()
                self.agents['Leo'] = AgentAutomation(leo, 'Leo')
                self.log_action("[MASTER] Leo initialized")
            except Exception as e:
                self.log_action(f"[MASTER] ERROR: Leo initialization failed: {e}")
        
        if self.config.ENABLE_MIKEY and MIKEY_AVAILABLE:
            try:
                mikey = MikeyAgent()
                self.agents['Mikey'] = AgentAutomation(mikey, 'Mikey')
                self.log_action("[MASTER] Mikey initialized")
            except Exception as e:
                self.log_action(f"[MASTER] ERROR: Mikey initialization failed: {e}")
        
        if self.config.ENABLE_SHREDDER and SHREDDER_AVAILABLE:
            try:
                shredder = ShredderAgent()
                self.agents['Shredder'] = AgentAutomation(shredder, 'Shredder')
                self.log_action("[MASTER] Shredder initialized")
            except Exception as e:
                self.log_action(f"[MASTER] ERROR: Shredder initialization failed: {e}")
        
        self.log_action(f"[MASTER] Initialized {len(self.agents)} agents")
    
    def _setup_schedules(self):
        """Setup scheduling for all agents"""
        if SCHEDULE_AVAILABLE:
            # Use schedule library if available
            # Ralph - Generate strategies every hour
            if self.config.ENABLE_RALPH:
                schedule.every(self.config.RALPH_INTERVAL_MINUTES).minutes.do(
                    self._run_ralph_cycle
                )
            
            # Donnie - Process strategies every 30 minutes
            if self.config.ENABLE_DONNIE:
                schedule.every(self.config.DONNIE_INTERVAL_MINUTES).minutes.do(
                    self._run_donnie_cycle
                )
            
            # Optimus - Check for trades every 10 seconds
            if self.config.ENABLE_OPTIMUS:
                schedule.every(self.config.OPTIMUS_INTERVAL_SECONDS).seconds.do(
                    self._run_optimus_cycle
                )
            
            # Casey - Check/build agents every 2 hours
            if self.config.ENABLE_CASEY:
                schedule.every(self.config.CASEY_INTERVAL_MINUTES).minutes.do(
                    self._run_casey_cycle
                )
            
            # Bebop - Monitor system every 15 minutes
            if self.config.ENABLE_BEBOP:
                schedule.every(self.config.BEBOP_INTERVAL_MINUTES).minutes.do(
                    self._run_bebop_cycle
                )
            
            # Splinter - Orchestrate every hour
            if self.config.ENABLE_SPLINTER:
                schedule.every(self.config.SPLINTER_INTERVAL_MINUTES).minutes.do(
                    self._run_splinter_cycle
                )
            
            # Rocksteady - Security sweep every 6 hours
            if self.config.ENABLE_ROCKSTEADY:
                schedule.every(self.config.ROCKSTEADY_INTERVAL_HOURS).hours.do(
                    self._run_rocksteady_cycle
                )
            
            # Phisher - Security scan every hour
            if self.config.ENABLE_PHISHER:
                schedule.every(self.config.PHISHER_INTERVAL_MINUTES).minutes.do(
                    self._run_phisher_cycle
                )
            
            # Genny - Run every 3 hours
            if self.config.ENABLE_GENNY and 'Genny' in self.agents:
                schedule.every(self.config.GENNY_INTERVAL_MINUTES).minutes.do(
                    self._run_genny_cycle
                )
            
            # April - Run every 6 hours (crypto operations)
            if self.config.ENABLE_APRIL and 'April' in self.agents:
                schedule.every(self.config.APRIL_INTERVAL_MINUTES).minutes.do(
                    self._run_april_cycle
                )
            
            # Leo - Run every hour
            if self.config.ENABLE_LEO and 'Leo' in self.agents:
                schedule.every(self.config.LEO_INTERVAL_MINUTES).minutes.do(
                    self._run_leo_cycle
                )
            
            # Mikey - Run every hour
            if self.config.ENABLE_MIKEY and 'Mikey' in self.agents:
                schedule.every(self.config.MIKEY_INTERVAL_MINUTES).minutes.do(
                    self._run_mikey_cycle
                )
            
            # Shredder - Run every hour
            if self.config.ENABLE_SHREDDER and 'Shredder' in self.agents:
                schedule.every(self.config.SHREDDER_INTERVAL_MINUTES).minutes.do(
                    self._run_shredder_cycle
                )
        else:
            # Fallback: Store timing info for manual scheduling
            self.last_run_times = {}
            self.run_intervals = {}
            
            if self.config.ENABLE_RALPH:
                self.run_intervals['Ralph'] = self.config.RALPH_INTERVAL_MINUTES * 60
                self.last_run_times['Ralph'] = 0
            
            if self.config.ENABLE_DONNIE:
                self.run_intervals['Donnie'] = self.config.DONNIE_INTERVAL_MINUTES * 60
                self.last_run_times['Donnie'] = 0
            
            if self.config.ENABLE_OPTIMUS:
                self.run_intervals['Optimus'] = self.config.OPTIMUS_INTERVAL_SECONDS
                self.last_run_times['Optimus'] = 0
            
            if self.config.ENABLE_CASEY:
                self.run_intervals['Casey'] = self.config.CASEY_INTERVAL_MINUTES * 60
                self.last_run_times['Casey'] = 0
            
            if self.config.ENABLE_BEBOP:
                self.run_intervals['Bebop'] = self.config.BEBOP_INTERVAL_MINUTES * 60
                self.last_run_times['Bebop'] = 0
            
            if self.config.ENABLE_SPLINTER:
                self.run_intervals['Splinter'] = self.config.SPLINTER_INTERVAL_MINUTES * 60
                self.last_run_times['Splinter'] = 0
            
            if self.config.ENABLE_ROCKSTEADY:
                self.run_intervals['Rocksteady'] = self.config.ROCKSTEADY_INTERVAL_HOURS * 3600
                self.last_run_times['Rocksteady'] = 0
            
            if self.config.ENABLE_PHISHER:
                self.run_intervals['Phisher'] = self.config.PHISHER_INTERVAL_MINUTES * 60
                self.last_run_times['Phisher'] = 0
            
            if self.config.ENABLE_GENNY and 'Genny' in self.agents:
                self.run_intervals['Genny'] = self.config.GENNY_INTERVAL_MINUTES * 60
                self.last_run_times['Genny'] = 0
            
            if self.config.ENABLE_APRIL and 'April' in self.agents:
                self.run_intervals['April'] = self.config.APRIL_INTERVAL_MINUTES * 60
                self.last_run_times['April'] = 0
            
            if self.config.ENABLE_LEO and 'Leo' in self.agents:
                self.run_intervals['Leo'] = self.config.LEO_INTERVAL_MINUTES * 60
                self.last_run_times['Leo'] = 0
            
            if self.config.ENABLE_MIKEY and 'Mikey' in self.agents:
                self.run_intervals['Mikey'] = self.config.MIKEY_INTERVAL_MINUTES * 60
                self.last_run_times['Mikey'] = 0
            
            if self.config.ENABLE_SHREDDER and 'Shredder' in self.agents:
                self.run_intervals['Shredder'] = self.config.SHREDDER_INTERVAL_MINUTES * 60
                self.last_run_times['Shredder'] = 0
    
    def _run_ralph_cycle(self):
        """Run Ralph's strategy generation cycle"""
        result = self.agents['Ralph'].run()
        self.log_action(f"Ralph cycle: {result}")
        
        # Pass strategies to Donnie
        if result.get('status') == 'success' and 'Donnie' in self.agents:
            ralph_agent = self.agents['Ralph'].agent
            if hasattr(ralph_agent, 'strategy_database') and ralph_agent.strategy_database:
                strategies = ralph_agent.top_strategies(5)
                self.agents['Donnie'].agent.receive_strategies(strategies)
                self.log_action(f"Passed {len(strategies)} strategies from Ralph to Donnie")
    
    def _run_donnie_cycle(self):
        """Run Donnie's strategy execution cycle"""
        donnie_agent = self.agents['Donnie'].agent
        optimus_agent = self.agents.get('Optimus')
        
        result = self.agents['Donnie'].run(
            sandbox=False,  # LIVE MODE ONLY - Tradier
            optimus_agent=optimus_agent.agent if optimus_agent else None
        )
        self.log_action(f"Donnie cycle: {result}")
    
    def _run_optimus_cycle(self):
        """Run Optimus's trade execution cycle"""
        # Check inbox for messages
        optimus_agent = self.agents['Optimus'].agent
        if hasattr(optimus_agent, 'inbox') and optimus_agent.inbox:
            execution_batch = []
            while optimus_agent.inbox:
                message = optimus_agent.inbox.pop(0)
                if isinstance(message, dict) and message.get('action') == 'execute_trade':
                    execution_batch.append(message)
            
            if execution_batch:
                result = self.agents['Optimus'].run(execution_batch)
                self.log_action(f"Optimus cycle: Executed {len(execution_batch)} trades")
    
    def _run_casey_cycle(self):
        """Run Casey's agent building cycle"""
        result = self.agents['Casey'].run()
        self.log_action(f"Casey cycle: {result}")
    
    def _run_bebop_cycle(self):
        """Run Bebop's monitoring cycle"""
        result = self.agents['Bebop'].run()
        self.log_action(f"Bebop cycle: {result}")
    
    def _run_splinter_cycle(self):
        """Run Splinter's orchestration cycle"""
        result = self.agents['Splinter'].run()
        self.log_action(f"Splinter cycle: {result}")
    
    def _run_rocksteady_cycle(self):
        """Run Rocksteady's security sweep"""
        result = self.agents['Rocksteady'].run()
        self.log_action(f"Rocksteady cycle: {result}")
    
    def _run_phisher_cycle(self):
        """Run Phisher's security scan"""
        result = self.agents['Phisher'].run()
        self.log_action(f"Phisher cycle: {result}")
    
    def _run_genny_cycle(self):
        """Run Genny's cycle"""
        if 'Genny' in self.agents:
            result = self.agents['Genny'].run()
            self.log_action(f"Genny cycle: {result}")
    
    def _run_april_cycle(self):
        """Run April's Bitcoin migration cycle"""
        if 'April' in self.agents:
            result = self.agents['April'].run()
            self.log_action(f"April cycle: {result}")
    
    def _run_leo_cycle(self):
        """Run Leo's cycle"""
        if 'Leo' in self.agents:
            result = self.agents['Leo'].run()
            self.log_action(f"Leo cycle: {result}")
    
    def _run_mikey_cycle(self):
        """Run Mikey's cycle"""
        if 'Mikey' in self.agents:
            result = self.agents['Mikey'].run()
            self.log_action(f"Mikey cycle: {result}")
    
    def _run_shredder_cycle(self):
        """Run Shredder's profit allocation cycle"""
        if 'Shredder' in self.agents:
            result = self.agents['Shredder'].run()
            self.log_action(f"Shredder cycle: {result}")
    
    def log_action(self, message: str):
        """Log action to file"""
        ts = datetime.datetime.now().isoformat()
        log_entry = f"[{ts}] {message}\n"
        with open(self.log_file, "a") as f:
            f.write(log_entry)
        print(f"[SCHEDULER] {message}")
    
    def start(self):
        """Start the scheduler"""
        self.running = True
        self.log_action("Master scheduler starting...")
        
        # Run initial compliance check
        self._run_compliance_check()
        
        print("[MASTER] NAE Master Scheduler Started")
        print("[MASTER] All agents automated and running")
        print("[MASTER] Press Ctrl+C to stop")
        
        try:
            while self.running:
                if SCHEDULE_AVAILABLE:
                    schedule.run_pending()
                else:
                    # Fallback: Manual time-based scheduling
                    current_time = time.time()
                    for agent_name, interval in self.run_intervals.items():
                        if agent_name in self.agents and (current_time - self.last_run_times[agent_name]) >= interval:
                            self._run_agent_by_name(agent_name)
                            self.last_run_times[agent_name] = current_time
                
                # Run compliance check daily
                if not self.last_compliance_check or (time.time() - self.last_compliance_check) >= (self.compliance_check_interval_hours * 3600):
                    self._run_compliance_check()
                
                time.sleep(1)
        except KeyboardInterrupt:
            self.log_action("Master scheduler stopped by user")
            print("\n[MASTER] Shutting down...")
            self.stop()
    
    def _run_compliance_check(self):
        """Run legal compliance check"""
        try:
            from legal_compliance_checker import LegalComplianceChecker
            checker = LegalComplianceChecker()
            results = checker.check_all_compliance()
            
            self.last_compliance_check = time.time()
            
            # Log results
            self.log_action(f"Compliance check: {results['summary']['passed']}/{results['summary']['total']} passed")
            
            # Alert on failures
            if results['summary']['failed'] > 0:
                self.log_action(f"⚠️  COMPLIANCE ALERT: {results['summary']['failed']} checks failed!")
                for check in results['checks']:
                    if check['status'] == 'fail':
                        self.log_action(f"  FAILED: {check['check_name']} - {check['message']}")
            
        except Exception as e:
            self.log_action(f"Error running compliance check: {e}")
    
    def _run_agent_by_name(self, agent_name: str):
        """Run agent by name (for fallback scheduling)"""
        if agent_name == 'Ralph':
            self._run_ralph_cycle()
        elif agent_name == 'Donnie':
            self._run_donnie_cycle()
        elif agent_name == 'Optimus':
            self._run_optimus_cycle()
        elif agent_name == 'Casey':
            self._run_casey_cycle()
        elif agent_name == 'Bebop':
            self._run_bebop_cycle()
        elif agent_name == 'Splinter':
            self._run_splinter_cycle()
        elif agent_name == 'Rocksteady':
            self._run_rocksteady_cycle()
        elif agent_name == 'Phisher':
            self._run_phisher_cycle()
        elif agent_name == 'Genny':
            self._run_genny_cycle()
        elif agent_name == 'April':
            self._run_april_cycle()
        elif agent_name == 'Leo':
            self._run_leo_cycle()
        elif agent_name == 'Mikey':
            self._run_mikey_cycle()
        elif agent_name == 'Shredder':
            self._run_shredder_cycle()
    
    def stop(self):
        """Stop the scheduler"""
        self.running = False
        self.log_action("Master scheduler stopped")
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of all agents"""
        return {
            "running": self.running,
            "agents": {
                name: agent.get_status()
                for name, agent in self.agents.items()
            },
            "schedules": (
                [
                    {
                        "job": str(job),
                        "next_run": job.next_run.isoformat() if job.next_run else None
                    }
                    for job in schedule.jobs
                ] if SCHEDULE_AVAILABLE else [
                    {
                        "agent": name,
                        "interval": interval,
                        "last_run": self.last_run_times.get(name, 0)
                    }
                    for name, interval in getattr(self, 'run_intervals', {}).items()
                ]
            )
        }
    
    def enable_agent(self, agent_name: str):
        """Enable an agent"""
        if agent_name in self.agents:
            self.agents[agent_name].enabled = True
            self.log_action(f"Enabled {agent_name}")
    
    def disable_agent(self, agent_name: str):
        """Disable an agent"""
        if agent_name in self.agents:
            self.agents[agent_name].enabled = False
            self.log_action(f"Disabled {agent_name}")

# ----------------------
# Main Entry Point
# ----------------------
if __name__ == "__main__":
    print("="*70)
    print("NAE Master Automation Scheduler")
    print("="*70)
    
    scheduler = NAEMasterScheduler()
    
    # Print initial status
    print("\nInitialized Agents:")
    for name, agent_auto in scheduler.agents.items():
        status = agent_auto.get_status()
        print(f"  - {name}: {'ENABLED' if status['enabled'] else 'DISABLED'}")
    
    print("\nScheduled Jobs:")
    for job in schedule.jobs:
        print(f"  - {job}")
    
    # Start scheduler
    scheduler.start()

