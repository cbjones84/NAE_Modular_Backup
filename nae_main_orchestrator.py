#!/usr/bin/env python3
"""
NAE Main Orchestrator - Comprehensive Autonomous System
Ensures all agents are synchronized, communicating, and self-improving

Features:
- Autonomous background operation
- Agent synchronization and communication
- Self-improvement mechanisms
- Error detection and recovery
- Continuous monitoring
- Performance optimization
"""

import os
import sys
import time
import signal
import threading
import datetime
import traceback
from typing import Dict, List, Any, Optional
from pathlib import Path

# Add NAE to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import core components
from nae_master_scheduler import NAEMasterScheduler, AutomationConfig
from agents.splinter import SplinterAgent
from agents.casey import CaseyAgent
from agents.ralph import RalphAgent
from agents.donnie import DonnieAgent
from agents.optimus import OptimusAgent
try:
    from tools.feedback_loops import FeedbackManager, PerformanceFeedbackLoop, RiskFeedbackLoop, ResearchFeedbackLoop
    FEEDBACK_LOOPS_AVAILABLE = True
except ImportError:
    try:
        from tools.feedback_loop import FeedbackManager, PerformanceFeedbackLoop, RiskFeedbackLoop, ResearchFeedbackLoop
        FEEDBACK_LOOPS_AVAILABLE = True
    except ImportError:
        FEEDBACK_LOOPS_AVAILABLE = False
        FeedbackManager = None
        PerformanceFeedbackLoop = None
        RiskFeedbackLoop = None
        ResearchFeedbackLoop = None

class NAEMainOrchestrator:
    """
    Main orchestrator for NAE autonomous operation
    
    Ensures:
    - All agents are synchronized
    - Proper communication between agents
    - Autonomous background operation
    - Self-improvement mechanisms
    - Error detection and recovery
    """
    
    def __init__(self):
        self.running = False
        self.scheduler = None
        self.splinter = None
        self.casey = None
        self.feedback_manager = None
        self.agent_instances = {}
        self.communication_bus = {}  # Centralized communication bus
        self.improvement_history = []
        self.error_history = []
        self.performance_metrics = {}
        self.log_file = "logs/nae_orchestrator.log"
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        
        # Initialize system
        self._initialize_system()
    
    def _initialize_system(self):
        """Initialize all system components"""
        self.log("="*80)
        self.log("NAE Main Orchestrator - Initializing...")
        self.log("="*80)
        
        try:
            # Initialize Master Scheduler
            config = AutomationConfig()
            self.scheduler = NAEMasterScheduler(config)
            self.log("✓ Master Scheduler initialized")
            
            # Initialize Splinter (orchestrator)
            self.splinter = SplinterAgent()
            self.log("✓ Splinter initialized")
            
            # Initialize Casey (monitor and builder)
            self.casey = CaseyAgent()
            self.log("✓ Casey initialized")
            
            # Get agent instances
            for name, agent_auto in self.scheduler.agents.items():
                self.agent_instances[name] = agent_auto.agent
                # Register with Splinter
                self.splinter.register_agents([name])
            
            self.log(f"✓ Registered {len(self.agent_instances)} agents")
            
            # Initialize communication bus
            self._initialize_communication_bus()
            
            # Initialize feedback loops
            self._initialize_feedback_loops()
            
            # Setup agent communication
            self._setup_agent_communication()
            
            # Initialize self-improvement mechanisms
            self._initialize_self_improvement()

            # Initialize Agent Loader (Flowise Integration)
            try:
                from core.agent_loader import AgentLoader
                definitions_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agents", "definitions")
                self.agent_loader = AgentLoader(definitions_dir=definitions_dir, orchestrator=self)
                self.agent_loader.start_watching()
                self.log("✓ Agent Loader initialized and watching agents/definitions/")
            except Exception as e:
                self.log(f"Warning: Failed to initialize AgentLoader: {e}")
                self.agent_loader = None
            
            self.log("="*80)
            self.log("✓ NAE Main Orchestrator initialized successfully")
            self.log("="*80)
            
        except Exception as e:
            self.log(f"ERROR: Failed to initialize system: {e}")
            self.log(traceback.format_exc())
            raise
    
    def _initialize_communication_bus(self):
        """Initialize centralized communication bus"""
        self.communication_bus = {
            "messages": [],
            "subscriptions": {},  # agent_name -> [message_types]
            "message_history": []
        }
        
        # Setup subscriptions
        for agent_name in self.agent_instances.keys():
            self.communication_bus["subscriptions"][agent_name] = ["strategy", "trade", "alert", "improvement"]
        
        self.log("✓ Communication bus initialized")
    
    def _initialize_feedback_loops(self):
        """Initialize and register feedback loops"""
        self.log("Initializing feedback loops...")
        
        if not FEEDBACK_LOOPS_AVAILABLE:
            self.log("⚠️  Feedback loops not available, skipping")
            self.feedback_manager = None
            return
        
        try:
            self.feedback_manager = FeedbackManager()
            
            # Performance feedback loop for Optimus
            if 'Optimus' in self.agent_instances:
                optimus = self.agent_instances['Optimus']
                if isinstance(optimus, OptimusAgent):
                    try:
                        perf_loop = PerformanceFeedbackLoop(agent=optimus)
                        risk_loop = RiskFeedbackLoop(agent=optimus)
                        self.feedback_manager.register(perf_loop)
                        self.feedback_manager.register(risk_loop)
                        self.log("✓ Performance and Risk feedback loops registered")
                    except Exception as e:
                        self.log(f"Warning: Could not initialize Optimus feedback loops: {e}")
            
            # Research feedback loop for Casey
            if self.casey:
                try:
                    research_loop = ResearchFeedbackLoop(self.casey)
                    self.feedback_manager.register(research_loop)
                    self.log("✓ Research feedback loop registered")
                except Exception as e:
                    self.log(f"Warning: Could not initialize Casey research loop: {e}")
            
            self.log("✓ Feedback loops initialized")
        except Exception as e:
            self.log(f"Warning: Could not initialize feedback loops: {e}")
            self.feedback_manager = None
    
    def _setup_agent_communication(self):
        """Setup bidirectional communication between agents"""
        self.log("Setting up agent communication...")
        
        # Ralph -> Donnie -> Optimus pipeline
        if 'Ralph' in self.agent_instances and 'Donnie' in self.agent_instances:
            ralph = self.agent_instances['Ralph']
            donnie = self.agent_instances['Donnie']
            
            # Ensure Ralph can send strategies to Donnie
            if hasattr(ralph, 'send_message') and hasattr(donnie, 'receive_message'):
                self.log("✓ Ralph -> Donnie communication configured")
        
        # Donnie -> Optimus pipeline
        if 'Donnie' in self.agent_instances and 'Optimus' in self.agent_instances:
            donnie = self.agent_instances['Donnie']
            optimus = self.agent_instances['Optimus']
            
            # Ensure Donnie can send execution requests to Optimus
            if hasattr(donnie, 'send_message') and hasattr(optimus, 'receive_message'):
                self.log("✓ Donnie -> Optimus communication configured")
        
        # Ralph -> Optimus direct communication channel (for high-confidence strategies)
        if 'Ralph' in self.agent_instances and 'Optimus' in self.agent_instances:
            ralph = self.agent_instances['Ralph']
            optimus = self.agent_instances['Optimus']
            
            # Register Optimus for direct high-confidence strategy communication
            if hasattr(ralph, 'register_optimus_channel'):
                ralph.register_optimus_channel(optimus)
                self.log("✓ Ralph -> Optimus direct communication channel registered")
        
        # Casey monitoring all agents
        if self.casey:
            for name, agent in self.agent_instances.items():
                if hasattr(agent, 'health_check'):
                    try:
                        # Register with Casey's monitoring
                        pid = os.getpid()
                        self.casey.monitor_process(name, pid)
                    except Exception as e:
                        self.log(f"Warning: Could not register {name} with Casey: {e}")
        
        # Splinter orchestration
        if self.splinter:
            for name in self.agent_instances.keys():
                self.splinter.register_agents([name])
        
        self.log("✓ Agent communication configured")
    
    def _initialize_self_improvement(self):
        """Initialize self-improvement mechanisms"""
        self.log("Initializing self-improvement mechanisms...")
        
        # Performance tracking
        self.performance_metrics = {
            "cycle_count": 0,
            "successful_cycles": 0,
            "failed_cycles": 0,
            "improvements_applied": 0,
            "last_improvement": None
        }
        
        self.log("✓ Self-improvement mechanisms initialized")
    
    def _agent_communication_loop(self):
        """Continuous agent communication monitoring"""
        while self.running:
            try:
                # Process messages in communication bus
                if self.communication_bus["messages"]:
                    messages = self.communication_bus["messages"].copy()
                    self.communication_bus["messages"] = []
                    
                    for message in messages:
                        self._route_message(message)
                
                # Process Ralph's outbox messages (strategies, GitHub discoveries)
                if 'Ralph' in self.agent_instances:
                    ralph = self.agent_instances['Ralph']
                    if hasattr(ralph, 'outbox') and ralph.outbox:
                        messages = ralph.outbox.copy()
                        ralph.outbox = []  # Clear outbox after processing
                        
                        for msg in messages:
                            recipient = msg.get("to", "")
                            message_content = msg.get("message", {})
                            
                            if recipient == "Donnie" and 'Donnie' in self.agent_instances:
                                donnie = self.agent_instances['Donnie']
                                if hasattr(donnie, 'receive_message'):
                                    donnie.receive_message(message_content)
                                    self.log(f"✓ Routed message from Ralph to Donnie: {message_content.get('type', 'unknown')}")
                            
                            elif recipient == "Optimus" and 'Optimus' in self.agent_instances:
                                optimus = self.agent_instances['Optimus']
                                # Direct Optimus communication (bypass Donnie for high-confidence strategies)
                                if msg.get("direct", False) and hasattr(optimus, 'receive_message'):
                                    optimus.receive_message(message_content)
                                    self.log(f"✓ Routed direct strategy signal from Ralph to Optimus: {message_content.get('strategy', {}).get('name', 'unknown')}")
                
                # Ensure Ralph sends strategies to Donnie (normal pipeline)
                if 'Ralph' in self.agent_instances and 'Donnie' in self.agent_instances:
                    ralph = self.agent_instances['Ralph']
                    donnie = self.agent_instances['Donnie']
                    
                    # Check if Ralph has new strategies
                    if hasattr(ralph, 'strategy_database') and ralph.strategy_database:
                        if hasattr(ralph, 'top_strategies'):
                            strategies = ralph.top_strategies(5)
                            if strategies and hasattr(donnie, 'receive_strategies'):
                                donnie.receive_strategies(strategies)
                                self.log(f"✓ Routed {len(strategies)} strategies from Ralph to Donnie")
                
                # Ensure Donnie sends execution requests to Optimus
                if 'Donnie' in self.agent_instances and 'Optimus' in self.agent_instances:
                    donnie = self.agent_instances['Donnie']
                    optimus = self.agent_instances['Optimus']
                    
                    # Check if Donnie has execution history
                    if hasattr(donnie, 'execution_history') and donnie.execution_history:
                        for execution in donnie.execution_history[-5:]:  # Last 5
                            if hasattr(optimus, 'receive_message'):
                                message = {
                                    "type": "execute_trade",
                                    "content": execution,
                                    "from": "Donnie",
                                    "to": "Optimus",
                                    "timestamp": datetime.datetime.now().isoformat()
                                }
                                optimus.receive_message(message)
                                self.log(f"✓ Routed execution request from Donnie to Optimus")
                
                time.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                self.log(f"Error in communication loop: {e}")
                time.sleep(10)
    
    def _route_message(self, message: Dict[str, Any]):
        """Route message to appropriate agent"""
        try:
            msg_type = message.get("type", "unknown")
            to_agent = message.get("to", "broadcast")
            
            if to_agent == "broadcast":
                # Broadcast to all subscribed agents
                for agent_name, subscriptions in self.communication_bus["subscriptions"].items():
                    if msg_type in subscriptions and agent_name in self.agent_instances:
                        agent = self.agent_instances[agent_name]
                        if hasattr(agent, 'receive_message'):
                            agent.receive_message(message)
            else:
                # Send to specific agent
                if to_agent in self.agent_instances:
                    agent = self.agent_instances[to_agent]
                    if hasattr(agent, 'receive_message'):
                        agent.receive_message(message)
            
            # Store in history
            self.communication_bus["message_history"].append({
                "message": message,
                "routed_at": datetime.datetime.now().isoformat()
            })
            
            # Keep only last 1000 messages
            if len(self.communication_bus["message_history"]) > 1000:
                self.communication_bus["message_history"] = self.communication_bus["message_history"][-1000:]
                
        except Exception as e:
            self.log(f"Error routing message: {e}")
    
    def _self_improvement_loop(self):
        """Continuous self-improvement monitoring"""
        while self.running:
            try:
                # Run feedback loops
                if self.feedback_manager and FEEDBACK_LOOPS_AVAILABLE:
                    context = {
                        "agents": self.agent_instances,
                        "performance_metrics": self.performance_metrics,
                        "timestamp": datetime.datetime.now().isoformat()
                    }
                    
                    # Run performance feedback
                    if 'performance' in [loop.name for loop in self.feedback_manager.loops]:
                        self.feedback_manager.run("performance", context)
                    
                    # Run risk feedback
                    if 'risk' in [loop.name for loop in self.feedback_manager.loops]:
                        self.feedback_manager.run("risk", context)
                    
                    # Run research feedback
                    if 'research' in [loop.name for loop in self.feedback_manager.loops]:
                        self.feedback_manager.run("research", context)
                
                # Analyze performance and suggest improvements
                self._analyze_and_improve()
                
                time.sleep(300)  # Run every 5 minutes
                
            except Exception as e:
                self.log(f"Error in self-improvement loop: {e}")
                time.sleep(60)
    
    def _analyze_and_improve(self):
        """Analyze system performance and apply improvements"""
        try:
            # Calculate success rate
            total_cycles = self.performance_metrics.get("cycle_count", 0)
            successful_cycles = self.performance_metrics.get("successful_cycles", 0)
            
            if total_cycles > 0:
                success_rate = (successful_cycles / total_cycles) * 100
                
                # If success rate is low, suggest improvements
                if success_rate < 80 and total_cycles > 10:
                    improvement = {
                        "type": "performance",
                        "description": f"Success rate is {success_rate:.1f}%, suggesting optimizations",
                        "timestamp": datetime.datetime.now().isoformat(),
                        "applied": False
                    }
                    self.improvement_history.append(improvement)
                    self.log(f"📊 Improvement suggestion: {improvement['description']}")
            
            # Check for recurring errors
            if len(self.error_history) > 10:
                error_counts = {}
                for error in self.error_history[-50:]:  # Last 50 errors
                    error_type = error.get("type", "unknown")
                    error_counts[error_type] = error_counts.get(error_type, 0) + 1
                
                # If an error type occurs frequently, suggest fix
                for error_type, count in error_counts.items():
                    if count > 5:
                        improvement = {
                            "type": "error_fix",
                            "description": f"Recurring error: {error_type} (occurred {count} times)",
                            "timestamp": datetime.datetime.now().isoformat(),
                            "applied": False
                        }
                        self.improvement_history.append(improvement)
                        self.log(f"🔧 Improvement suggestion: {improvement['description']}")
            
        except Exception as e:
            self.log(f"Error in analyze_and_improve: {e}")
    
    def _monitoring_loop(self):
        """Continuous system monitoring"""
        while self.running:
            try:
                # Monitor agent health
                for name, agent in self.agent_instances.items():
                    try:
                        if hasattr(agent, 'health_check'):
                            health = agent.health_check()
                            if health.get('status') != 'healthy':
                                self.log(f"⚠️  Agent {name} health check failed: {health}")
                    except Exception as e:
                        self.log(f"Error checking health for {name}: {e}")
                
                # Monitor Casey
                if self.casey:
                    try:
                        system_summary = self.casey.get_system_summary()
                        if system_summary.get('overall_health_score', 100) < 70:
                            self.log(f"⚠️  System health score low: {system_summary.get('overall_health_score')}")
                    except Exception:
                        pass
                
                # Monitor Splinter
                if self.splinter:
                    try:
                        # Splinter monitors agents
                        self.splinter.monitor_agents()
                    except Exception:
                        pass
                
                time.sleep(60)  # Monitor every minute
                
            except Exception as e:
                self.log(f"Error in monitoring loop: {e}")
                time.sleep(60)
    
    def start(self):
        """Start the orchestrator"""
        if self.running:
            self.log("System already running")
            return
        
        self.running = True
        self.log("="*80)
        self.log("NAE Main Orchestrator - STARTING")
        self.log("="*80)
        
        # Start scheduler thread
        scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
        scheduler_thread.start()
        self.log("✓ Scheduler thread started")
        
        # Start communication thread
        comm_thread = threading.Thread(target=self._agent_communication_loop, daemon=True)
        comm_thread.start()
        self.log("✓ Communication thread started")
        
        # Start self-improvement thread
        improvement_thread = threading.Thread(target=self._self_improvement_loop, daemon=True)
        improvement_thread.start()
        self.log("✓ Self-improvement thread started")
        
        # Start monitoring thread
        monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        monitor_thread.start()
        self.log("✓ Monitoring thread started")
        
        self.log("="*80)
        self.log("✓ NAE Main Orchestrator is RUNNING")
        self.log("="*80)
        self.log("All agents are synchronized and communicating")
        self.log("Self-improvement mechanisms are active")
        self.log("Press Ctrl+C to stop")
        self.log("="*80)
        
        # Keep main thread alive
        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            self.log("\nShutdown requested by user...")
            self.stop()
    
    def _run_scheduler(self):
        """Run the master scheduler"""
        try:
            if self.scheduler:
                self.scheduler.running = True
                while self.running and self.scheduler.running:
                    try:
                        # Run scheduled tasks
                        import schedule
                        schedule.run_pending()
                    except ImportError:
                        # Fallback scheduling
                        current_time = time.time()
                        for agent_name, interval in getattr(self.scheduler, 'run_intervals', {}).items():
                            if agent_name in self.scheduler.agents:
                                last_run = getattr(self.scheduler, 'last_run_times', {}).get(agent_name, 0)
                                if current_time - last_run >= interval:
                                    try:
                                        agent_auto = self.scheduler.agents[agent_name]
                                        if agent_auto.enabled:
                                            result = agent_auto.run()
                                            self.performance_metrics["cycle_count"] += 1
                                            if result.get("status") == "success":
                                                self.performance_metrics["successful_cycles"] += 1
                                            else:
                                                self.performance_metrics["failed_cycles"] += 1
                                                self.error_history.append({
                                                    "type": result.get("error", "unknown"),
                                                    "agent": agent_name,
                                                    "timestamp": datetime.datetime.now().isoformat()
                                                })
                                            if not hasattr(self.scheduler, 'last_run_times'):
                                                self.scheduler.last_run_times = {}
                                            self.scheduler.last_run_times[agent_name] = current_time
                                    except Exception as e:
                                        self.log(f"Error running {agent_name}: {e}")
                                        self.error_history.append({
                                            "type": str(e),
                                            "agent": agent_name,
                                            "timestamp": datetime.datetime.now().isoformat()
                                        })
                    
                    time.sleep(1)
        except Exception as e:
            self.log(f"Error in scheduler thread: {e}")
            self.log(traceback.format_exc())
    
    def stop(self):
        """Stop the orchestrator"""
        self.log("Stopping NAE Main Orchestrator...")
        self.running = False
        
        if self.scheduler:
            self.scheduler.stop()
        
        self.log("✓ NAE Main Orchestrator stopped")
    
    def log(self, message: str):
        """Log message to file and console"""
        timestamp = datetime.datetime.now().isoformat()
        log_entry = f"[{timestamp}] {message}\n"
        
        try:
            with open(self.log_file, "a") as f:
                f.write(log_entry)
        except Exception:
            pass
        
        print(f"[NAE-ORCH] {message}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get system status"""
        return {
            "running": self.running,
            "agents": {
                name: agent.health_check() if hasattr(agent, 'health_check') else {"status": "unknown"}
                for name, agent in self.agent_instances.items()
            },
            "performance_metrics": self.performance_metrics,
            "improvements": len(self.improvement_history),
            "errors": len(self.error_history),
            "communication": {
                "messages_in_queue": len(self.communication_bus.get("messages", [])),
                "total_messages": len(self.communication_bus.get("message_history", []))
            }
        }


def main():
    """Main entry point"""
    print("\n" + "="*80)
    print("NEURAL AGENCY ENGINE - MAIN ORCHESTRATOR")
    print("="*80)
    print("\nThis system ensures:")
    print("  • All agents are synchronized")
    print("  • Proper communication between agents")
    print("  • Autonomous background operation")
    print("  • Self-improvement mechanisms")
    print("  • Error detection and recovery")
    print("\n" + "="*80 + "\n")
    
    orchestrator = NAEMainOrchestrator()
    orchestrator.start()


if __name__ == "__main__":
    main()

