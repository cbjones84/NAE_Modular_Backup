# agents/splinter.py
# Splinter: Comprehensive Orchestrator / Mentor Agent for NAE
# Merged with main orchestrator functionality for autonomous operation

import datetime
import time
import threading
import traceback
from typing import Dict, Any, List, Optional

# Goals managed by GoalManager
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from goal_manager import get_nae_goals
GOALS = get_nae_goals()

# Import scheduler and feedback loops
try:
    from nae_master_scheduler import NAEMasterScheduler, AutomationConfig
    SCHEDULER_AVAILABLE = True
except ImportError:
    SCHEDULER_AVAILABLE = False
    NAEMasterScheduler = None
    AutomationConfig = None

try:
    from tools.feedback_loops import FeedbackManager, PerformanceFeedbackLoop, RiskFeedbackLoop, ResearchFeedbackLoop
    FEEDBACK_LOOPS_AVAILABLE = True
except ImportError:
    try:
        from tools.feedback_loop import FeedbackLoopSystem as FeedbackManager, PerformanceMetric
        FEEDBACK_LOOPS_AVAILABLE = True
        # Mock other classes if needed or handle gracefully
        PerformanceFeedbackLoop = None
        RiskFeedbackLoop = None
        ResearchFeedbackLoop = None
    except ImportError:
        FEEDBACK_LOOPS_AVAILABLE = False
        FeedbackManager = None

class SplinterAgent:
    def __init__(self, enable_autonomous_mode: bool = True):
        self.goals = GOALS
        self.long_term_plan = "docs/NAE_LONG_TERM_PLAN.md"
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
        
        self.managed_agents = []
        self.log_file = "logs/splinter.log"
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        
        # ----------------------
        # Messaging / AutoGen hooks
        # ----------------------
        self.inbox = []
        self.outbox = []
        
        # ----------------------
        # Autonomous Operation (merged from main orchestrator)
        # ----------------------
        self.running = False
        self.enable_autonomous_mode = enable_autonomous_mode
        self.scheduler = None
        self.casey = None
        self.feedback_manager = None
        self.agent_instances = {}
        self.communication_bus = {
            "messages": [],
            "subscriptions": {},  # agent_name -> [message_types]
            "message_history": []
        }
        self.improvement_history = []
        self.error_history = []
        self.performance_metrics = {
            "cycle_count": 0,
            "successful_cycles": 0,
            "failed_cycles": 0,
            "improvements_applied": 0,
            "last_improvement": None
        }
        
        # Initialize autonomous systems if enabled
        if self.enable_autonomous_mode:
            self._initialize_autonomous_systems()

    # ----------------------
    # Register agents under Splinter
    # ----------------------
    def register_agents(self, agents_list):
        if not agents_list:
            self.log_action("No agents provided for registration")
            return

        registered_names = []
        for entry in agents_list:
            if isinstance(entry, str):
                self.managed_agents.append({
                    "name": entry,
                    "instance": None,
                })
                registered_names.append(entry)
            else:
                agent_name = entry.__class__.__name__
                self.managed_agents.append({
                    "name": agent_name,
                    "instance": entry,
                })
                registered_names.append(agent_name)

        self.log_action(f"Registered agents: {registered_names}")

    # ----------------------
    # Messaging system
    # ----------------------
    def receive_message(self, msg: dict):
        """
        All messages to Splinter pass through here.
        msg structure:
        {
            "from": "AgentName",
            "to": "Splinter" / "broadcast",
            "timestamp": ISO timestamp,
            "type": "info|warning|task|data|improvement_recommendations|monitoring_update",
            "content": dict or string
        }
        """
        try:
            if not isinstance(msg, dict):
                self.log_action(f"Invalid message format: expected dict, got {type(msg)}")
                return
            
            msg_type = msg.get("type", "unknown")
            msg_from = msg.get("from", "unknown")
            content = msg.get("content", {})
            
            self.log_action(f"Received message from {msg_from} | Type: {msg_type}")
            
            # Handle improvement recommendations from feedback loop
            if msg_type == "improvement_recommendations":
                recommendations = content.get("recommendations", [])
                self.log_action(f"📊 Received {len(recommendations)} improvement recommendations")
                
                for rec in recommendations[:5]:  # Show top 5
                    priority = rec.get("priority", "unknown")
                    description = rec.get("description", "N/A")
                    agent = rec.get("agent_name", "Unknown")
                    self.log_action(f"   [{priority.upper()}] {agent}: {description}")
                
                # Broadcast high-priority recommendations to relevant agents
                for rec in recommendations:
                    if rec.get("priority") in ["high", "critical"]:
                        agent_name = rec.get("agent_name", "")
                        # Find agent and send message
                        for agent_entry in self.managed_agents:
                            if agent_entry.get("name") == agent_name:
                                self.send_message(agent_entry, "improvement", rec)
                                break
            
            # Handle monitoring updates
            elif msg_type == "monitoring_update":
                agent_statuses = content.get("agent_statuses", {})
                cycle_count = content.get("cycle_count", 0)
                strategies_executed = content.get("strategies_executed", 0)
                
                self.log_action(f"📈 Monitoring Update: Cycle {cycle_count}, {strategies_executed} strategies executed")
                for agent_name, status in agent_statuses.items():
                    self.log_action(f"   {agent_name}: {status}")
            
            # Handle task messages
            elif msg_type == "task":
                self.log_action(f"Task received from {msg_from}: {content}")
            
            # Handle other message types
            else:
                self.log_action(f"Message content: {content}")
                
        except Exception as e:
            self.log_action(f"Error processing message: {e}")

    def send_message(self, agent_entry: dict, msg_type: str, content):
        """
        Send a message to a single agent.
        """
        try:
            agent_name = agent_entry.get("name", "UnknownAgent")
            agent_instance = agent_entry.get("instance")

            if not agent_instance:
                self.log_action(f"Agent {agent_name} has no instance registered; cannot receive messages")
                return False

            if not hasattr(agent_instance, "receive_message"):
                self.log_action(f"Agent {agent_name} cannot receive messages")
                return False
            
            message = {
                "from": "Splinter",
                "to": agent_name,
                "timestamp": datetime.datetime.now().isoformat(),
                "type": msg_type,
                "content": content
            }
            
            agent_instance.receive_message(message)
            self.log_action(f"Sent message to {agent_name} | Type: {msg_type} | Content: {content}")
            return True
            
        except Exception as e:
            self.log_action(f"Error sending message: {e}")
            return False

    def broadcast(self, msg_type: str, content):
        """
        Send the same message to all managed agents.
        """
        try:
            if not self.managed_agents:
                self.log_action("No agents registered for broadcast")
                return
            
            success_count = 0
            for agent_entry in self.managed_agents:
                if self.send_message(agent_entry, msg_type, content):
                    success_count += 1
            
            self.log_action(f"Broadcast complete: {success_count}/{len(self.managed_agents)} agents notified")
            
        except Exception as e:
            self.log_action(f"Error in broadcast: {e}")

    # ----------------------
    # Monitor all agents and enforce 3 Goals
    # ----------------------
    def monitor_agents(self):
        try:
            if not self.managed_agents:
                self.log_action("No agents registered for monitoring")
                return
            
            monitoring_report = {
                "timestamp": datetime.datetime.now().isoformat(),
                "agents": {},
                "patterns": [],
                "improvements": []
            }
            
            for agent_entry in self.managed_agents:
                agent_name = agent_entry.get("name", "UnknownAgent")
                agent_instance = agent_entry.get("instance")
                agent_data = {}

                if not agent_instance:
                    agent_data["status"] = "unavailable"
                    monitoring_report["agents"][agent_name] = agent_data
                    self.log_action(f"Monitoring {agent_name}: {agent_data}")
                    continue
                
                # Check agent health if available
                if hasattr(agent_instance, 'health_check'):
                    health = agent_instance.health_check()
                    agent_data["health"] = health.get('status', 'unknown')
                
                # Monitor Optimus performance
                if agent_name == "OptimusAgent":
                    try:
                        if hasattr(agent_instance, 'get_trading_status'):
                            status = agent_instance.get_trading_status()
                            nav = status.get("nav", 25.0)
                            daily_pnl = status.get("daily_pnl", 0)
                            goal_progress = (nav / getattr(agent_instance, "target_goal", 5000000)) * 100
                            
                            agent_data.update({
                                "nav": nav,
                                "daily_pnl": daily_pnl,
                                "goal_progress": goal_progress,
                                "open_positions": status.get("open_positions", 0),
                                "consecutive_losses": status.get("consecutive_losses", 0),
                                "current_phase": getattr(agent_instance, "current_phase", "Phase 1")
                            })
                            
                            # Pattern detection: Goal progress
                            if goal_progress > 0.1:
                                monitoring_report["patterns"].append({
                                    "type": "progress",
                                    "description": f"Goal progress: {goal_progress:.4f}% toward $5M",
                                    "agent": agent_name
                                })
                            
                            # Pattern detection: Consecutive losses
                            if status.get("consecutive_losses", 0) >= 3:
                                monitoring_report["improvements"].append({
                                    "type": "risk_management",
                                    "priority": "high",
                                    "description": f"{status['consecutive_losses']} consecutive losses - recommend tightening risk",
                                    "agent": agent_name
                                })
                            
                            # Pattern detection: Daily loss
                            if daily_pnl < -1.0:
                                monitoring_report["improvements"].append({
                                    "type": "risk_management",
                                    "priority": "medium",
                                    "description": f"Daily loss: ${daily_pnl:.2f} - monitor closely",
                                    "agent": agent_name
                                })
                    
                    except Exception as e:
                        agent_data["error"] = str(e)
                
                # Monitor Ralph strategies
                elif agent_name == "RalphAgent":
                    try:
                        agent_data["approved_strategies"] = len(getattr(agent_instance, "strategy_database", []))
                        agent_data["candidate_pool"] = len(getattr(agent_instance, "candidate_pool", []))
                    except Exception as e:
                        agent_data["error"] = str(e)
                
                # Monitor Donnie validations
                elif agent_name == "DonnieAgent":
                    try:
                        agent_data["execution_history_size"] = len(getattr(agent_instance, "execution_history", []))
                    except Exception as e:
                        agent_data["error"] = str(e)
                
                monitoring_report["agents"][agent_name] = agent_data
                self.log_action(f"Monitoring {agent_name}: {agent_data}")
            
            # Log patterns and improvements
            if monitoring_report["patterns"]:
                for pattern in monitoring_report["patterns"]:
                    self.log_action(f"📊 PATTERN: {pattern['description']}")
            
            if monitoring_report["improvements"]:
                for improvement in monitoring_report["improvements"]:
                    self.log_action(f"💡 IMPROVEMENT: {improvement['description']} (Priority: {improvement['priority']})")
            
            return monitoring_report
                    
        except Exception as e:
            self.log_action(f"Error monitoring agents: {e}")
            return None

    # ----------------------
    # Logging
    # ----------------------
    def log_action(self, message):
        ts = datetime.datetime.now().isoformat()
        try:
            with open(self.log_file, "a", encoding='utf-8') as f:
                f.write(f"[{ts}] {message}\n")
        except Exception as e:
            print(f"Failed to write to log: {e}")
        # Safe print for Windows console
        try:
            print(f"[Splinter LOG] {message}")
        except UnicodeEncodeError:
            safe_message = message.encode('ascii', 'ignore').decode('ascii')
            print(f"[Splinter LOG] {safe_message}")

    # ----------------------
    # Autonomous System Initialization (merged from main orchestrator)
    # ----------------------
    def _initialize_autonomous_systems(self):
        """Initialize autonomous operation systems"""
        try:
            if SCHEDULER_AVAILABLE:
                config = AutomationConfig()
                self.scheduler = NAEMasterScheduler(config)
                self.log_action("✓ Master Scheduler initialized")
                
                # Get agent instances from scheduler
                for name, agent_auto in self.scheduler.agents.items():
                    self.agent_instances[name] = agent_auto.agent
                    # Register with Splinter
                    self.register_agents([name])
                
                self.log_action(f"✓ Registered {len(self.agent_instances)} agents from scheduler")
            
            # Initialize Casey
            try:
                from agents.casey import CaseyAgent
                self.casey = CaseyAgent()
                self.log_action("✓ Casey initialized")
            except Exception as e:
                self.log_action(f"Warning: Could not initialize Casey: {e}")
            
            # Initialize communication bus subscriptions
            for agent_name in self.agent_instances.keys():
                self.communication_bus["subscriptions"][agent_name] = ["strategy", "trade", "alert", "improvement"]
            
            # Initialize feedback loops
            self._initialize_feedback_loops()
            
            # Setup agent communication
            self._setup_agent_communication()
            
            self.log_action("✓ Autonomous systems initialized")
            
        except Exception as e:
            self.log_action(f"Error initializing autonomous systems: {e}")
            self.log_action(traceback.format_exc())
    
    def _initialize_feedback_loops(self):
        """Initialize and register feedback loops"""
        if not FEEDBACK_LOOPS_AVAILABLE:
            self.log_action("⚠️  Feedback loops not available, skipping")
            return
        
        try:
            self.feedback_manager = FeedbackManager()
            
            # Performance feedback loop for Optimus
            if 'Optimus' in self.agent_instances:
                try:
                    from agents.optimus import OptimusAgent
                    optimus = self.agent_instances['Optimus']
                    if isinstance(optimus, OptimusAgent):
                        perf_loop = PerformanceFeedbackLoop(agent=optimus)
                        risk_loop = RiskFeedbackLoop(agent=optimus)
                        self.feedback_manager.register(perf_loop)
                        self.feedback_manager.register(risk_loop)
                        self.log_action("✓ Performance and Risk feedback loops registered")
                except Exception as e:
                    self.log_action(f"Warning: Could not initialize Optimus feedback loops: {e}")
            
            # Research feedback loop for Casey
            if self.casey:
                try:
                    research_loop = ResearchFeedbackLoop(self.casey)
                    self.feedback_manager.register(research_loop)
                    self.log_action("✓ Research feedback loop registered")
                except Exception as e:
                    self.log_action(f"Warning: Could not initialize Casey research loop: {e}")
            
        except Exception as e:
            self.log_action(f"Warning: Could not initialize feedback loops: {e}")
    
    def _setup_agent_communication(self):
        """Setup bidirectional communication between agents"""
        # Ralph -> Donnie -> Optimus pipeline
        if 'Ralph' in self.agent_instances and 'Donnie' in self.agent_instances:
            ralph = self.agent_instances['Ralph']
            donnie = self.agent_instances['Donnie']
            if hasattr(ralph, 'send_message') and hasattr(donnie, 'receive_message'):
                self.log_action("✓ Ralph -> Donnie communication configured")
        
        # Donnie -> Optimus pipeline
        if 'Donnie' in self.agent_instances and 'Optimus' in self.agent_instances:
            donnie = self.agent_instances['Donnie']
            optimus = self.agent_instances['Optimus']
            if hasattr(donnie, 'send_message') and hasattr(optimus, 'receive_message'):
                self.log_action("✓ Donnie -> Optimus communication configured")
    
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
                
                # Ensure Ralph sends strategies to Donnie
                if 'Ralph' in self.agent_instances and 'Donnie' in self.agent_instances:
                    ralph = self.agent_instances['Ralph']
                    donnie = self.agent_instances['Donnie']
                    
                    if hasattr(ralph, 'strategy_database') and ralph.strategy_database:
                        if hasattr(ralph, 'top_strategies'):
                            strategies = ralph.top_strategies(5)
                            if strategies and hasattr(donnie, 'receive_strategies'):
                                donnie.receive_strategies(strategies)
                
                # Ensure Donnie sends execution requests to Optimus
                if 'Donnie' in self.agent_instances and 'Optimus' in self.agent_instances:
                    donnie = self.agent_instances['Donnie']
                    optimus = self.agent_instances['Optimus']
                    
                    if hasattr(donnie, 'execution_history') and donnie.execution_history:
                        for execution in donnie.execution_history[-5:]:
                            if hasattr(optimus, 'receive_message'):
                                message = {
                                    "type": "execute_trade",
                                    "content": execution,
                                    "from": "Donnie",
                                    "to": "Optimus",
                                    "timestamp": datetime.datetime.now().isoformat()
                                }
                                optimus.receive_message(message)
                
                time.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                self.log_action(f"Error in communication loop: {e}")
                time.sleep(10)
    
    def _route_message(self, message: Dict[str, Any]):
        """Route message to appropriate agent"""
        try:
            msg_type = message.get("type", "unknown")
            to_agent = message.get("to", "broadcast")
            
            if to_agent == "broadcast":
                for agent_name, subscriptions in self.communication_bus["subscriptions"].items():
                    if msg_type in subscriptions and agent_name in self.agent_instances:
                        agent = self.agent_instances[agent_name]
                        if hasattr(agent, 'receive_message'):
                            agent.receive_message(message)
            else:
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
            self.log_action(f"Error routing message: {e}")
    
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
                    
                    if self.feedback_manager.get('performance'):
                        self.feedback_manager.run("performance", context)
                    if self.feedback_manager.get('risk'):
                        self.feedback_manager.run("risk", context)
                    if self.feedback_manager.get('research'):
                        self.feedback_manager.run("research", context)
                
                # Analyze performance and suggest improvements
                self._analyze_and_improve()
                
                time.sleep(300)  # Run every 5 minutes
                
            except Exception as e:
                self.log_action(f"Error in self-improvement loop: {e}")
                time.sleep(60)
    
    def _analyze_and_improve(self):
        """Analyze system performance and apply improvements"""
        try:
            total_cycles = self.performance_metrics.get("cycle_count", 0)
            successful_cycles = self.performance_metrics.get("successful_cycles", 0)
            
            if total_cycles > 0:
                success_rate = (successful_cycles / total_cycles) * 100
                
                if success_rate < 80 and total_cycles > 10:
                    improvement = {
                        "type": "performance",
                        "description": f"Success rate is {success_rate:.1f}%, suggesting optimizations",
                        "timestamp": datetime.datetime.now().isoformat(),
                        "applied": False
                    }
                    self.improvement_history.append(improvement)
                    self.log_action(f"📊 Improvement suggestion: {improvement['description']}")
            
            # Check for recurring errors
            if len(self.error_history) > 10:
                error_counts = {}
                for error in self.error_history[-50:]:
                    error_type = error.get("type", "unknown")
                    error_counts[error_type] = error_counts.get(error_type, 0) + 1
                
                for error_type, count in error_counts.items():
                    if count > 5:
                        improvement = {
                            "type": "error_fix",
                            "description": f"Recurring error: {error_type} (occurred {count} times)",
                            "timestamp": datetime.datetime.now().isoformat(),
                            "applied": False
                        }
                        self.improvement_history.append(improvement)
                        self.log_action(f"🔧 Improvement suggestion: {improvement['description']}")
            
        except Exception as e:
            self.log_action(f"Error in analyze_and_improve: {e}")
    
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
                                self.log_action(f"⚠️  Agent {name} health check failed: {health}")
                    except Exception as e:
                        self.log_action(f"Error checking health for {name}: {e}")
                
                # Run standard monitoring
                self.monitor_agents()
                
                time.sleep(60)  # Monitor every minute
                
            except Exception as e:
                self.log_action(f"Error in monitoring loop: {e}")
                time.sleep(60)
    
    def start_autonomous(self):
        """Start autonomous operation mode"""
        if self.running:
            self.log_action("Autonomous mode already running")
            return
        
        if not self.enable_autonomous_mode:
            self.log_action("Autonomous mode not enabled")
            return
        
        self.running = True
        self.log_action("="*80)
        self.log_action("Splinter Autonomous Mode - STARTING")
        self.log_action("="*80)
        
        # Start communication thread
        comm_thread = threading.Thread(target=self._agent_communication_loop, daemon=True)
        comm_thread.start()
        self.log_action("✓ Communication thread started")
        
        # Start self-improvement thread
        improvement_thread = threading.Thread(target=self._self_improvement_loop, daemon=True)
        improvement_thread.start()
        self.log_action("✓ Self-improvement thread started")
        
        # Start monitoring thread
        monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        monitor_thread.start()
        self.log_action("✓ Monitoring thread started")
        
        # Start scheduler if available
        if self.scheduler:
            scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
            scheduler_thread.start()
            self.log_action("✓ Scheduler thread started")
        
        self.log_action("="*80)
        self.log_action("✓ Splinter Autonomous Mode is RUNNING")
        self.log_action("="*80)
    
    def _run_scheduler(self):
        """Run the master scheduler"""
        try:
            if self.scheduler:
                self.scheduler.running = True
                while self.running and self.scheduler.running:
                    try:
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
                                        self.log_action(f"Error running {agent_name}: {e}")
                    
                    time.sleep(1)
        except Exception as e:
            self.log_action(f"Error in scheduler thread: {e}")
    
    def stop_autonomous(self):
        """Stop autonomous operation"""
        self.log_action("Stopping Splinter Autonomous Mode...")
        self.running = False
        
        if self.scheduler:
            self.scheduler.stop()
        
        self.log_action("✓ Splinter Autonomous Mode stopped")
    
    # ----------------------
    # Run orchestrator loop
    # ----------------------
    def run(self) -> Dict[str, Any]:
        try:
            self.log_action("Splinter run cycle started")
            self.monitor_agents()
            
            result = {
                "status": "success",
                "agent": "Splinter",
                "agents_managed": len(self.managed_agents),
                "autonomous_mode": self.running,
                "timestamp": datetime.datetime.now().isoformat()
            }
            
            self.log_action(f"Splinter run cycle completed: {result}")
            return result
            
        except Exception as e:
            self.log_action(f"Error in Splinter run cycle: {e}")
            return {
                "status": "error",
                "agent": "Splinter",
                "error": str(e),
                "timestamp": datetime.datetime.now().isoformat()
            }
    
    def health_check(self) -> Dict[str, Any]:
        """Check agent health status"""
        return {
            "status": "healthy",
            "agent": "Splinter",
            "agents_managed": len(self.managed_agents),
            "autonomous_mode": self.running,
            "log_file": self.log_file,
            "performance_metrics": self.performance_metrics,
            "improvements": len(self.improvement_history),
            "errors": len(self.error_history)
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
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

# ----------------------
# Example usage
# ----------------------
def splinter_main_loop():
    """Splinter continuous operation loop - NEVER STOPS"""
    import traceback
    import logging
    
    logger = logging.getLogger(__name__)
    restart_count = 0
    
    while True:  # NEVER EXIT
        try:
            logger.info("=" * 70)
            logger.info(f"🚀 Starting Splinter Agent (Restart #{restart_count})")
            logger.info("=" * 70)
            
            splinter = SplinterAgent()
            logger.info("[Splinter] Orchestrator agent initialized.")
            
            # Main operation loop
            while True:
                try:
                    # Splinter's main operation - orchestrate agents continuously
                    time.sleep(30)  # Check every 30 seconds
                    
                except KeyboardInterrupt:
                    logger.warning("⚠️  KeyboardInterrupt - Continuing Splinter operation...")
                    time.sleep(5)
                except Exception as e:
                    logger.error(f"Error in Splinter main loop: {e}")
                    logger.error(traceback.format_exc())
                    time.sleep(30)
                    
        except KeyboardInterrupt:
            restart_count += 1
            logger.warning(f"⚠️  KeyboardInterrupt - RESTARTING Splinter (Restart #{restart_count})")
            time.sleep(5)
        except SystemExit:
            restart_count += 1
            logger.warning(f"⚠️  SystemExit - RESTARTING Splinter (Restart #{restart_count})")
            time.sleep(10)
        except Exception as e:
            restart_count += 1
            delay = min(60 * restart_count, 3600)
            logger.error(f"❌ Fatal error in Splinter (Restart #{restart_count}): {e}")
            logger.error(traceback.format_exc())
            logger.info(f"🔄 Restarting in {delay} seconds...")
            time.sleep(delay)


if __name__ == "__main__":
    splinter_main_loop()
    
    # NOTE: Code below is unreachable because splinter_main_loop() runs forever
    # If you need to run tests, create a separate test function or script
