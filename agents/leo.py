# NAE/agents/leo.py
"""
LeoAgent v2 - Open-Source & ML pipeline agent for NAE
Responsibilities:
- Interact with external APIs & open-source models
- Process and normalize data
- Provide insights to Ralph and Casey
- Fully goal-oriented and AutoGen-compatible
"""

import os
import sys
import datetime
from typing import Any, Dict, List

# 3 Goals embedded
# Goals managed by GoalManager
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from goal_manager import get_nae_goals
GOALS = get_nae_goals()

class LeoAgent:
    def __init__(self):
        self.goals = GOALS
        self.log_file = "logs/leo.log"
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        self.data_store: List[Dict[str, Any]] = []

        # ----------------------
        # Messaging / AutoGen hooks
        # ----------------------
        self.inbox = []
        self.outbox = []

    # ----------------------
    # Logging
    # ----------------------
    def log_action(self, message: str):
        ts = datetime.datetime.now().isoformat()
        with open(self.log_file, "a") as f:
            f.write(f"[{ts}] {message}\n")
        print(f"[Leo LOG] {message}")

    # ----------------------
    # Fetch external data / run ML pipelines
    # ----------------------
    def fetch_external_data(self):
        # Placeholder for open-source API calls / ML model inference
        data = [
            {"source": "open-source_model_1", "value": 123},
            {"source": "external_api_1", "value": 456}
        ]
        self.log_action(f"Fetched {len(data)} data points from external sources")
        self.data_store.extend(data)
        return data

    # ----------------------
    # Process & normalize data
    # ----------------------
    def normalize_data(self, raw_data: List[Dict[str, Any]]):
        # Placeholder: simple normalization example
        normalized = [{"source": d["source"], "value": d["value"]/1000} for d in raw_data]
        self.log_action(f"Normalized {len(normalized)} data points")
        self.data_store.extend(normalized)
        return normalized

    # ----------------------
    # Provide insights to other agents
    # ----------------------
    def provide_insights(self):
        insights = [{"insight": f"Processed data from {d['source']}", "value": d["value"]} 
                    for d in self.data_store]
        self.log_action(f"Providing {len(insights)} insights to other agents")
        return insights

    # ----------------------
    # Messaging hooks
    # ----------------------
    def receive_message(self, message: dict):
        self.inbox.append(message)
        self.log_action(f"Received message: {message}")

    def run(self) -> Dict[str, Any]:
        """Main execution cycle for Leo agent"""
        try:
            self.log_action("Leo run cycle started")
            
            # Fetch external data
            raw_data = self.fetch_external_data()
            
            # Normalize data
            normalized_data = self.normalize_data(raw_data)
            
            # Provide insights
            insights = self.provide_insights()
            
            # Process inbox messages
            processed_messages = []
            while self.inbox:
                message = self.inbox.pop(0)
                processed_messages.append(message)
                
                # Handle specific message types
                if isinstance(message, dict):
                    msg_type = message.get("type", "")
                    if msg_type == "fetch_data":
                        raw_data.extend(self.fetch_external_data())
                    elif msg_type == "request_insights":
                        insights.extend(self.provide_insights())
            
            result = {
                "status": "success",
                "agent": "Leo",
                "data_points": len(self.data_store),
                "insights_generated": len(insights),
                "messages_processed": len(processed_messages),
                "timestamp": datetime.datetime.now().isoformat()
            }
            
            self.log_action(f"Leo run cycle completed: {result}")
            return result
            
        except Exception as e:
            self.log_action(f"Error in Leo run cycle: {e}")
            return {
                "status": "error",
                "agent": "Leo",
                "error": str(e),
                "timestamp": datetime.datetime.now().isoformat()
            }
    
    def health_check(self) -> Dict[str, Any]:
        """Check agent health status"""
        return {
            "status": "healthy",
            "agent": "Leo",
            "data_store_size": len(self.data_store),
            "inbox_size": len(self.inbox),
            "outbox_size": len(self.outbox)
        }


# ----------------------
# Test harness
# ----------------------
if __name__ == "__main__":
    leo = LeoAgent()
    raw = leo.fetch_external_data()
    normalized = leo.normalize_data(raw)
    insights = leo.provide_insights()
    print(insights)

    # Example messaging test
    class DummyAgent:
        def __init__(self):
            self.inbox = []
        def receive_message(self, msg):
            self.inbox.append(msg)
    dummy = DummyAgent()
    leo.send_message({"data": "Test Insight"}, dummy)
    print("Dummy inbox:", dummy.inbox)
