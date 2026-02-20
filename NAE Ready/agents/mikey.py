# NAE/agents/mikey.py
"""
MikeyAgent v2 - Cloud Bridge Agent for NAE
Handles cloud synchronization, remote APIs, and external AI systems.
Reports to Splinter and Phisher for all cloud interactions.
"""

import os
import sys
import datetime
import requests
from typing import Dict, Any, List

# --------------------------
# Embedded 3 Main Goals
# --------------------------
# Goals managed by GoalManager
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from goal_manager import get_nae_goals
GOALS = get_nae_goals()


class MikeyAgent:
    def __init__(self):
        self.goals = GOALS
        self.log_file = "logs/mikey.log"
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        self.connected_services: Dict[str, Dict[str, Any]] = {}

        # ----------------------
        # Messaging / AutoGen hooks
        # ----------------------
        self.inbox = []
        self.outbox = []

    # --------------------------
    # Logging
    # --------------------------
    def log_action(self, message: str):
        ts = datetime.datetime.now().isoformat()
        with open(self.log_file, "a") as f:
            f.write(f"[{ts}] {message}\n")
        print(f"[Mikey LOG] {message}")

    # --------------------------
    # Register Cloud Service
    # --------------------------
    def connect_service(self, name: str, api_url: str, api_key: str):
        """Connect to a cloud or external API service"""
        self.connected_services[name] = {"url": api_url, "key": api_key}
        self.log_action(f"Connected to cloud service '{name}' at {api_url}")

    # --------------------------
    # Send Data to Cloud
    # --------------------------
    def upload_data(self, service_name: str, data: Dict[str, Any]):
        """Send data securely to a connected cloud service"""
        if service_name not in self.connected_services:
            self.log_action(f"Error: Unknown cloud service '{service_name}'")
            return {"status": "error", "detail": "Unknown service"}

        service = self.connected_services[service_name]
        headers = {"Authorization": f"Bearer {service['key']}", "Content-Type": "application/json"}

        try:
            response = requests.post(service["url"], headers=headers, json=data)
            self.log_action(f"Uploaded data to {service_name}, status: {response.status_code}")
            return response.json()
        except Exception as e:
            self.log_action(f"Failed to upload data to {service_name}: {e}")
            return {"status": "failed", "error": str(e)}

    # --------------------------
    # Download Data from Cloud
    # --------------------------
    def download_data(self, service_name: str, params: Dict[str, Any] = None):
        """Retrieve data securely from a connected cloud service"""
        if service_name not in self.connected_services:
            self.log_action(f"Error: Unknown cloud service '{service_name}'")
            return {"status": "error", "detail": "Unknown service"}

        service = self.connected_services[service_name]
        headers = {"Authorization": f"Bearer {service['key']}"}

        try:
            response = requests.get(service["url"], headers=headers, params=params)
            self.log_action(f"Downloaded data from {service_name}, status: {response.status_code}")
            return response.json()
        except Exception as e:
            self.log_action(f"Failed to download data from {service_name}: {e}")
            return {"status": "failed", "error": str(e)}

    # --------------------------
    # Cloud Sync Task
    # --------------------------
    def sync_with_cloud(self):
        """Run a periodic sync of local data and cloud systems"""
        self.log_action("Starting full cloud sync across all connected services...")
        for name in self.connected_services.keys():
            result = self.upload_data(name, {"sync_time": datetime.datetime.now().isoformat()})
            self.log_action(f"Sync result from {name}: {result}")
            # Send sync report to Splinter or Phisher if they exist in outbox
            self.send_message({"sync_result": result, "service": name}, recipient_agent=self.get_supervisor())
        self.log_action("Cloud sync complete.")

    # ----------------------
    # Messaging hooks
    # ----------------------
    def receive_message(self, message: dict):
        self.inbox.append(message)
        self.log_action(f"Received message: {message}")

    def run(self) -> Dict[str, Any]:
        """Main execution cycle for Mikey agent"""
        try:
            self.log_action("Mikey run cycle started")
            
            # Sync with cloud services
            sync_results = []
            if self.connected_services:
                self.sync_with_cloud()
                sync_results = [{"service": name, "status": "synced"} for name in self.connected_services.keys()]
            else:
                self.log_action("No cloud services connected")
            
            # Process inbox messages
            processed_messages = []
            while self.inbox:
                message = self.inbox.pop(0)
                processed_messages.append(message)
                
                # Handle specific message types
                if isinstance(message, dict):
                    msg_type = message.get("type", "")
                    content = message.get("content", {})
                    
                    if msg_type == "upload":
                        service = content.get("service")
                        data = content.get("data", {})
                        if service:
                            result = self.upload_data(service, data)
                            sync_results.append({"service": service, "action": "upload", "result": result})
                    
                    elif msg_type == "download":
                        service = content.get("service")
                        params = content.get("params", {})
                        if service:
                            result = self.download_data(service, params)
                            sync_results.append({"service": service, "action": "download", "result": result})
            
            result = {
                "status": "success",
                "agent": "Mikey",
                "services_connected": len(self.connected_services),
                "sync_results": sync_results,
                "messages_processed": len(processed_messages),
                "timestamp": datetime.datetime.now().isoformat()
            }
            
            self.log_action(f"Mikey run cycle completed: {result}")
            return result
            
        except Exception as e:
            self.log_action(f"Error in Mikey run cycle: {e}")
            return {
                "status": "error",
                "agent": "Mikey",
                "error": str(e),
                "timestamp": datetime.datetime.now().isoformat()
            }
    
    def health_check(self) -> Dict[str, Any]:
        """Check agent health status"""
        return {
            "status": "healthy",
            "agent": "Mikey",
            "services_connected": len(self.connected_services),
            "inbox_size": len(self.inbox),
            "outbox_size": len(self.outbox)
        }

    # ----------------------
    # Placeholder supervisor getter
    # ----------------------
    def get_supervisor(self):
        """
        In production, this would return the actual Splinter or Phisher agent object.
        For now, it returns a dummy agent for testing.
        """
        class DummySupervisor:
            def __init__(self):
                self.inbox = []
            def receive_message(self, msg):
                self.inbox.append(msg)
        return DummySupervisor()


# --------------------------
# Test Harness
# --------------------------
if __name__ == "__main__":
    mikey = MikeyAgent()
    mikey.connect_service("ExampleCloud", "https://api.example.com/upload", "12345-ABCDE")
    mikey.upload_data("ExampleCloud", {"test": "upload"})
    mikey.download_data("ExampleCloud", {"query": "latest"})
    mikey.sync_with_cloud()
