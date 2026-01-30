import os
import json
import time
import importlib
import threading
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

# Add NAE to path if needed, though usually handled by main orchestrator
import sys

class AgentLoader:
    """
    Monitors a directory for agent definition files (JSON) and dynamically loads/reloads them.
    Acts as the bridge between Flowise (which writes JSONs) and NAE (which runs python objects).
    """

    def __init__(self, definitions_dir: str, orchestrator: Any):
        self.definitions_dir = definitions_dir
        self.orchestrator = orchestrator
        self.loaded_agents: Dict[str, Any] = {}  # name -> agent_instance
        self.file_timestamps: Dict[str, float] = {}  # filename -> last_mtime
        self.logger = logging.getLogger("NAE.AgentLoader")
        self.running = False
        
        # Ensure directory exists
        os.makedirs(self.definitions_dir, exist_ok=True)

    def start_watching(self, interval: int = 5):
        """Start the background watcher thread"""
        self.running = True
        thread = threading.Thread(target=self._watch_loop, args=(interval,), daemon=True)
        thread.start()
        self.logger.info(f"AgentLoader started watching {self.definitions_dir}")

    def stop(self):
        self.running = False

    def _watch_loop(self, interval: int):
        while self.running:
            try:
                self._check_directory()
            except Exception as e:
                self.logger.error(f"Error in AgentLoader watch loop: {e}")
            time.sleep(interval)

    def _check_directory(self):
        """
        Scan directory for changes.
        """
        current_files = list(Path(self.definitions_dir).glob("*.json"))
        current_filenames = {f.name for f in current_files}
        
        # Check for new or modified files
        for file_path in current_files:
            fname = file_path.name
            mtime = file_path.stat().st_mtime
            
            if fname not in self.file_timestamps:
                # New file
                self.logger.info(f"New agent definition found: {fname}")
                self._load_agent(file_path)
                self.file_timestamps[fname] = mtime
            elif mtime > self.file_timestamps[fname]:
                # Modified file
                self.logger.info(f"Agent definition modified: {fname}")
                self._reload_agent(file_path)
                self.file_timestamps[fname] = mtime

        # Check for deleted files
        tracked_files = list(self.file_timestamps.keys())
        for fname in tracked_files:
            if fname not in current_filenames:
                self.logger.info(f"Agent definition deleted: {fname}")
                self._unload_agent_by_filename(fname)
                del self.file_timestamps[fname]

    def _load_agent(self, file_path: Path):
        """
        Read JSON and instantiate agent.
        """
        try:
            with open(file_path, 'r') as f:
                definition = json.load(f)
            
            agent_name = definition.get("name")
            if not agent_name:
                self.logger.error(f"Missing 'name' in {file_path}")
                return

            if not definition.get("enabled", True):
                self.logger.info(f"Agent {agent_name} is disabled in definition.")
                return

            class_path = definition.get("class", "agents.base_dynamic_agent.DynamicAgent")
            module_name, class_name = class_path.rsplit('.', 1)
            
            # Dynamic import
            module = importlib.import_module(module_name)
            agent_class = getattr(module, class_name)
            
            # Instantiate
            # Expecting __init__(name, config, inputs, outputs) signature for DynamicAgent
            # But standard NAE agents might have different sigs. 
            # We assume DynamicAgent or compatible for Flowise agents.
            
            config = definition.get("config", {})
            # Inject version
            config["version"] = definition.get("version", "0.0.1")
            
            inputs = definition.get("inputs", [])
            outputs = definition.get("outputs", [])
            
            new_agent = agent_class(
                name=agent_name,
                config=config,
                inputs=inputs,
                outputs=outputs
            )
            
            # Register with Orchestrator
            self.loaded_agents[agent_name] = new_agent
            self._register_with_orchestrator(agent_name, new_agent, file_path.name)
            
            self.logger.info(f"Successfully loaded agent: {agent_name}")

        except Exception as e:
            self.logger.error(f"Failed to load agent from {file_path}: {e}")

    def _reload_agent(self, file_path: Path):
        """
        Unload and verify load again.
        """
        # Find agent name associated with this file first? 
        # Simpler: just load it. If name matches existing, we overwrite.
        # But we need to clean up old one first if running.
        
        # Read definition to get name
        try:
            with open(file_path, 'r') as f:
                definition = json.load(f)
            agent_name = definition.get("name")
            
            if agent_name in self.loaded_agents:
                old_agent = self.loaded_agents[agent_name]
                if hasattr(old_agent, 'stop'):
                    old_agent.stop()
            
            self._load_agent(file_path)
            
        except Exception as e:
            self.logger.error(f"Failed to reload agent from {file_path}: {e}")

    def _unload_agent_by_filename(self, filename: str):
        # We need a mapping from filename to agent name or scan loaded agents?
        # For MVP, we won't strictly track filename->agent_name map in a separate DB,
        # but we can infer it if we store it. 
        # Let's rely on the user to not rename files for now, or just restart if messy.
        # To do it checks:
        pass 
        # (Implementing robust filename->agent mapping is skipped for brevity in this step, 
        # assuming 1:1 consistent mapping for now)

    def _register_with_orchestrator(self, name: str, agent: Any, filename: str):
        """
        Inject into main orchestrator's agent list and Splinter.
        """
        # 1. Add to orchestrator instances
        self.orchestrator.agent_instances[name] = agent
        
        # 2. Register with Splinter
        if self.orchestrator.splinter:
            try:
                self.orchestrator.splinter.register_agents([name])
                self.logger.info(f"Registered {name} with Splinter")
            except Exception as e:
                self.logger.error(f"Failed to register {name} with Splinter: {e}")
        
        # 3. Register with Casey (Monitoring)
        if self.orchestrator.casey:
            try:
                # Mock PID or just use system PID
                self.orchestrator.casey.monitor_process(name, os.getpid())
                self.logger.info(f"Registered {name} with Casey monitoring")
            except Exception as e:
                self.logger.error(f"Failed to register {name} with Casey: {e}")

    def get_loaded_agents(self) -> Dict[str, Any]:
        return self.loaded_agents
