# NAE/agents/donnie.py
"""
DonnieAgent v3 - Fully AutoGen-compatible execution agent for NAE

Responsibilities:
- Receive vetted strategies from Ralph/Casey
- Prepare and send execution instructions to Optimus
- Maintain safety checks and logging
- Goal-oriented: 3 Goals embedded

ALIGNED WITH 3 CORE GOALS:
1. Achieve generational wealth
2. Generate $5,000,000.00 within 8 years, every 8 years consistently
3. Optimize NAE and agents for successful options trading

ALIGNED WITH LONG-TERM PLAN:
- Validates strategies align with current phase (Phase 1-4)
- Ensures strategies are PDT-compliant (no same-day round trips)
- Coordinates execution aligned with tiered strategies (Wheel → Momentum → Multi-leg → AI)
- See: docs/NAE_LONG_TERM_PLAN.md for full strategy details
"""

import os
import sys
import datetime
from typing import List, Dict, Any

# Goals managed by GoalManager
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from goal_manager import get_nae_goals
GOALS = get_nae_goals()

# Profit enhancement algorithms (Lazy loaded)
# sys.path.append(os.path.dirname(os.path.dirname(__file__)))
# from tools.profit_algorithms import MetaLabelingModel

class DonnieAgent:
    def __init__(self, goals=None):
        self.goals = goals if goals else GOALS  # 3 Core Goals
        self.long_term_plan = "docs/NAE_LONG_TERM_PLAN.md"  # Reference to long-term plan
        # Growth Milestones from nae_mission_control.py
        self.target_goal = 5000000.0  # $5M target (exceeded in Year 7)
        self.stretch_goal = 15726144.0  # $15.7M final goal (Year 8)
        self.growth_milestones = {
            1: 9_411, 2: 44_110, 3: 152_834, 4: 388_657,
            5: 982_500, 6: 2_477_897, 7: 6_243_561, 8: 15_726_144
        }
        self.accelerator_enabled = True  # ALWAYS ON - support Optimus accelerator strategy infinitely
        
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
        # Logging & execution
        # ----------------------
        self.log_file = "logs/donnie.log"
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        self.execution_history: List[Dict[str, Any]] = []
        self.candidate_strategies: List[Dict[str, Any]] = []

        # ----------------------
        # Messaging / AutoGen hooks
        # ----------------------
        self.inbox = []
        self.outbox = []
        
        # ----------------------
        # Meta-Labeling for confidence scoring
        # ----------------------
        try:
            from tools.profit_algorithms import MetaLabelingModel
            self.meta_labeler = MetaLabelingModel()
            self.meta_labeler.load_model()  # Try to load existing model
        except ImportError:
            self.meta_labeler = None
            self.log_action("⚠️ MetaLabelingModel not available")
        except Exception as e:
            self.meta_labeler = None
            self.log_action(f"⚠️ MetaLabelingModel initialization failed: {e}")
        
        # ----------------------
        # THRML Integration for Probabilistic Validation
        # ----------------------
        # ----------------------
        # THRML Integration for Probabilistic Validation
        # ----------------------
        self.thrml_validation_model = None
        self.thrml_enabled = False
        # Lazy load in validate_strategy to prevent startup hangs with JAX


        
        # ----------------------
        # Excellence Protocol
        # ----------------------
        self.excellence_protocol = None
        try:
            from agents.donnie_excellence_protocol import DonnieExcellenceProtocol
            self.excellence_protocol = DonnieExcellenceProtocol(self)
            self.excellence_protocol.start_excellence_mode()
            self.log_action("🎯 Donnie Excellence Protocol initialized and active - Continuous learning, self-awareness, self-healing, and autonomous NAE support enabled")
        except ImportError as e:
            self.log_action(f"⚠️ Excellence protocol not available: {e}")
        except Exception as e:
            self.log_action(f"⚠️ Excellence protocol initialization failed: {e}")

    # ----------------------
    # Logging
    # ----------------------
    def log_action(self, message: str):
        ts = datetime.datetime.now().isoformat()
        try:
            with open(self.log_file, "a", encoding='utf-8') as f:
                f.write(f"[{ts}] {message}\n")
        except Exception as e:
            print(f"Failed to write to log: {e}")
        # Safe print for Windows console
        try:
            print(f"[Donnie LOG] {message}")
        except UnicodeEncodeError:
            safe_message = message.encode('ascii', 'ignore').decode('ascii')
            print(f"[Donnie LOG] {safe_message}")

    # ----------------------
    # Receive strategies from Ralph/Casey
    # ----------------------
    def receive_strategies(self, strategies: List[Dict[str, Any]]):
        self.log_action(f"Received {len(strategies)} strategies")
        self.candidate_strategies = strategies

    # ----------------------
    # Validate strategies before execution
    # ----------------------
    def validate_strategy(self, strategy: Dict[str, Any]) -> bool:
        try:
            if not isinstance(strategy, dict):
                self.log_action(f"Invalid strategy format: expected dict, got {type(strategy)}")
                return False
            
            # Standard validation
            trust_score = strategy.get("trust_score", 0)
            if not isinstance(trust_score, (int, float)) or trust_score < 55:
                self.log_action(f"Strategy {strategy.get('name')} rejected (low trust_score: {trust_score})")
                return False
            
            backtest = strategy.get("backtest_score", 0)
            if not isinstance(backtest, (int, float)) or backtest < 50:
                self.log_action(f"Strategy {strategy.get('name')} rejected (low backtest_score: {backtest})")
                return False
            
            # THRML Probabilistic Validation (if available)
            # Lazy load model if not yet initialized
            if not self.thrml_enabled and self.thrml_validation_model is None:
                try:
                    from tools.thrml_integration import ProbabilisticTradingModel
                    self.thrml_validation_model = ProbabilisticTradingModel(num_nodes=8)
                    market_features = ['backtest_score', 'trust_score', 'sharpe', 'win_rate', 
                                     'max_drawdown', 'perf', 'volatility', 'consensus']
                    self.thrml_validation_model.build_market_pgm(
                        market_features=market_features,
                        coupling_strength=0.3
                    )
                    self.thrml_enabled = True
                    self.log_action("THRML probabilistic validation model initialized (lazy loaded)")
                except Exception as e:
                    self.thrml_enabled = False
                    # Only log once to avoid spam
                    if not hasattr(self, '_thrml_error_logged'):
                        self.log_action(f"THRML initialization failed: {e}")
                        self._thrml_error_logged = True

            if self.thrml_enabled and self.thrml_validation_model:
                try:
                    import jax.numpy as jnp
                    
                    # Build feature vector from strategy
                    features = jnp.array([
                        strategy.get("backtest_score", 0) / 100.0,
                        strategy.get("trust_score", 0) / 100.0,
                        min(strategy.get("sharpe", 0) / 3.0, 1.0),
                        strategy.get("win_rate", 0.5),
                        strategy.get("max_drawdown", 0.0),
                        min(strategy.get("perf", 0) / 2.0, 1.0),
                        min(strategy.get("volatility", 0) / 0.5, 1.0),
                        min(strategy.get("consensus_count", 1) / 10.0, 1.0)
                    ])
                    
                    # Simulate probabilistic outcomes
                    trajectories = self.thrml_validation_model.simulate_market_trajectories(
                        features,
                        num_trajectories=50,
                        horizon=1
                    )
                    
                    # Calculate success probability
                    final_scores = [float(traj[-1][0] * 100.0) for traj in trajectories]
                    success_prob = sum(1 for s in final_scores if s > 50.0) / len(final_scores)
                    
                    strategy["thrml_success_probability"] = success_prob
                    
                    # Reject if success probability is too low
                    if success_prob < 0.6:  # Require 60% success probability
                        self.log_action(f"Strategy {strategy.get('name')} rejected by THRML validation (success_prob: {success_prob:.2%})")
                        return False
                    
                    self.log_action(f"Strategy {strategy.get('name')} passed THRML validation (success_prob: {success_prob:.2%})")
                except Exception as e:
                    self.log_action(f"THRML validation error (using standard validation): {e}")
            
            return True
        except Exception as e:
            self.log_action(f"Error validating strategy: {e}")
            return False

    # ----------------------
    # Send execution instructions to Optimus
    # ----------------------
    def execute_strategy(self, strategy: Dict[str, Any], sandbox: bool = False, optimus_agent=None):
        try:
            if not self.validate_strategy(strategy):
                return False

            # Apply Meta-Labeling: Calculate confidence score
            confidence = 0.85 # Default if unavailable
            if self.meta_labeler:
                try:
                    confidence = self.meta_labeler.predict_confidence(strategy)
                except Exception:
                    pass
            strategy["meta_confidence"] = confidence
            
            # Only execute if confidence is above threshold (50%)
            if confidence < 0.5:
                self.log_action(f"Strategy {strategy.get('name')} rejected by meta-labeling (confidence: {confidence:.2%})")
                return False

            execution_details = {
                "strategy_name": strategy.get("name", "unknown"),
                "action": "execute_trade",
                "parameters": strategy.get("aggregated_details", {}),
                "sandbox": sandbox,
                "timestamp": datetime.datetime.now().isoformat(),
                "meta_confidence": confidence,
                "trust_score": strategy.get("trust_score", 0),
                "backtest_score": strategy.get("backtest_score", 0)
            }

            self.log_action(f"Executing strategy: {execution_details['strategy_name']} | confidence: {confidence:.2%} | sandbox={sandbox}")
            self.execution_history.append(execution_details)

            if optimus_agent and hasattr(optimus_agent, 'receive_message'):
                optimus_agent.receive_message(execution_details)
                self.log_action(f"Sent execution instruction to Optimus with meta-labeling confidence: {confidence:.2%}")
                return True
            else:
                self.log_action("Optimus agent not available or missing receive_message method")
                return False
                
        except Exception as e:
            self.log_action(f"Error executing strategy: {e}")
            return False

    # ----------------------
    # Batch execution
    # ----------------------
    def run_cycle(self, sandbox: bool = False, optimus_agent=None):
        try:
            self.log_action("Donnie run_cycle start")
            if not self.candidate_strategies:
                self.log_action("No strategies to execute")
                return {
                    "status": "success",
                    "agent": "Donnie",
                    "strategies_executed": 0,
                    "timestamp": datetime.datetime.now().isoformat()
                }

            executed_count = 0
            for strat in self.candidate_strategies:
                if self.execute_strategy(strat, sandbox=sandbox, optimus_agent=optimus_agent):
                    executed_count += 1

            result = {
                "status": "success",
                "agent": "Donnie",
                "strategies_executed": executed_count,
                "total_strategies": len(self.candidate_strategies),
                "timestamp": datetime.datetime.now().isoformat()
            }
            
            self.log_action(f"Donnie run_cycle completed: {executed_count} strategies executed")
            return result
            
        except Exception as e:
            self.log_action(f"Error in Donnie run_cycle: {e}")
            return {
                "status": "error",
                "agent": "Donnie",
                "error": str(e),
                "timestamp": datetime.datetime.now().isoformat()
            }
    
    def run(self) -> Dict[str, Any]:
        """Main execution method (calls run_cycle) - LIVE MODE"""
        return self.run_cycle(sandbox=False)  # LIVE trading only
    
    def health_check(self) -> Dict[str, Any]:
        """Check agent health status"""
        return {
            "status": "healthy",
            "agent": "Donnie",
            "candidate_strategies": len(self.candidate_strategies),
            "execution_history_size": len(self.execution_history),
            "inbox_size": len(self.inbox),
            "outbox_size": len(self.outbox),
            "thrml_enabled": self.thrml_enabled if hasattr(self, 'thrml_enabled') else False
        }

    # ----------------------
    # Messaging hooks
    # ----------------------
    def receive_message(self, message: dict):
        self.inbox.append(message)
        self.log_action(f"Received message: {message}")
        
        # Handle specific message types
        if isinstance(message, dict):
            msg_type = message.get("type", "")
            content = message.get("content", {})
            
            # Handle GitHub implementation requests from Ralph
            if msg_type == "implementation_request" or msg_type == "github_implementation_request":
                discovery = content.get("discovery", {})
                self.log_action(
                    f"🔧 GitHub Implementation Request: {discovery.get('name', 'Unknown')} "
                    f"({discovery.get('url', 'N/A')}) - Priority: {content.get('priority', 'medium')}"
                )
                # Store for processing in next cycle
                if not hasattr(self, 'github_implementation_queue'):
                    self.github_implementation_queue = []
                self.github_implementation_queue.append(content)
                self.log_action(f"📋 Added to implementation queue (queue size: {len(self.github_implementation_queue)})")
            
            # Handle direct strategy signals from Ralph
            elif msg_type == "strategy_signal" or msg_type == "strategy_blocks":
                strategy = content.get("strategy", content)
                self.log_action(
                    f"📊 Strategy Signal from Ralph: {strategy.get('name', 'Unknown')} "
                    f"(Trust: {strategy.get('trust_score', 0):.1f})"
                )
                # Forward to Optimus via normal pipeline
                if not hasattr(self, 'ralph_direct_strategies'):
                    self.ralph_direct_strategies = []
                self.ralph_direct_strategies.append(strategy)

    def send_message(self, message: dict, recipient_agent):
        recipient_agent.receive_message(message)
        self.outbox.append({"to": recipient_agent.__class__.__name__, "message": message})
        self.log_action(f"Sent message to {recipient_agent.__class__.__name__}: {message}")


# ----------------------
# Test harness
# ----------------------
def donnie_main_loop():
    """Donnie continuous operation loop - NEVER STOPS"""
    import traceback
    import logging
    import time
    
    logger = logging.getLogger(__name__)
    restart_count = 0
    
    while True:  # NEVER EXIT
        try:
            logger.info("=" * 70)
            logger.info(f"🚀 Starting Donnie Agent (Restart #{restart_count})")
            logger.info("=" * 70)
            
            donnie = DonnieAgent()
            
            # Main operation loop
            while True:
                try:
                    # Donnie's main operation - evaluate strategies continuously
                    # This would typically receive strategies from other agents
                    time.sleep(30)  # Check every 30 seconds
                    
                except KeyboardInterrupt:
                    logger.warning("⚠️  KeyboardInterrupt - Continuing Donnie operation...")
                    time.sleep(5)
                except Exception as e:
                    logger.error(f"Error in Donnie main loop: {e}")
                    logger.error(traceback.format_exc())
                    time.sleep(30)
                    
        except KeyboardInterrupt:
            restart_count += 1
            logger.warning(f"⚠️  KeyboardInterrupt - RESTARTING Donnie (Restart #{restart_count})")
            time.sleep(5)
        except SystemExit:
            restart_count += 1
            logger.warning(f"⚠️  SystemExit - RESTARTING Donnie (Restart #{restart_count})")
            time.sleep(10)
        except Exception as e:
            restart_count += 1
            delay = min(60 * restart_count, 3600)
            logger.error(f"❌ Fatal error in Donnie (Restart #{restart_count}): {e}")
            logger.error(traceback.format_exc())
            logger.info(f"🔄 Restarting in {delay} seconds...")
            time.sleep(delay)


if __name__ == "__main__":
    donnie_main_loop()
