from typing import Dict, Any, List, Optional
import datetime
import logging

class DynamicAgent:
    """
    Base class for dynamically loaded agents (e.g. from Flowise definitions).
    """

    def __init__(self, name: str, config: Dict[str, Any], inputs: List[str], outputs: List[str]):
        self.name = name
        self.config = config
        self.inputs = inputs
        self.outputs = outputs
        self.version = config.get("version", "0.0.1")
        
        self.logger = logging.getLogger(f"NAE.Agents.{name}")
        self.status = "initialized"
        self.last_run_time = None
        self.message_queue = []
        
        # Standardize state for monitoring
        self.state = {
            "pnl": 0.0,
            "trades_count": 0,
            "lifecycle_stage": "sandbox", # sandbox, paper, production
            "health": "healthy"
        }
        
        self._setup_from_config()

    def _setup_from_config(self):
        """
        Override this to implement specific logic based on self.config.
        """
        pass

    def receive_message(self, message: Dict[str, Any]):
        """
        Handle incoming messages from the orchestrator or other agents.
        """
        self.message_queue.append(message)
        # In a real implementation, might process immediately or perform logic
        
    def health_check(self) -> Dict[str, Any]:
        """
        Returns health status for Casey/Splinter monitoring.
        """
        return {
            "name": self.name,
            "status": self.status,
            "health": "healthy" if self.status != "error" else "unhealthy",
            "last_run": self.last_run_time.isoformat() if self.last_run_time else None,
            "version": self.version
        }

    def run(self) -> Dict[str, Any]:
        """
        Main execution step, called by scheduler or triggered by events.
        """
        self.status = "running"
        self.last_run_time = datetime.datetime.now()
        
        try:
            # Placeholder for actual logic injection
            # In V2, this could run a generic strategy logic defined in config
            result = self.execute_logic()
            self.status = "idle"
            return {"status": "success", "result": result}
        except Exception as e:
            self.status = "error"
            self.logger.error(f"Error running agent {self.name}: {e}")
            return {"status": "error", "error": str(e)}

    def execute_logic(self):
        """
        The core logic of the agent. 
        For Flowise agents, this might interpret the 'logic' field from config.
        """
        # Example generic logic
        strategy_type = self.config.get("strategy_type", "unknown")
        # log activity
        return f"Executed {strategy_type}"

    def stop(self):
        """
        Cleanup when agent is unloaded/reloaded.
        """
        self.status = "stopped"
        self.logger.info(f"Agent {self.name} stopped.")
