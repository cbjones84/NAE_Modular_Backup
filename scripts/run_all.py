# NAE/run_all.py
"""
Launcher for Neural Agency Engine (NAE)
- Starts all agents in individual threads with Splinter orchestrating
- Registers all agent processes with CaseyAgent for CPU/memory monitoring
- Sends email alerts on crashes or high resource usage
"""

import threading
import time
import os
from agents import (
    casey, ralph, optimus, donnie, shredder, phisher,
    rocksteady, bebop, mikey, leo, splinter
)

# ----------------------
# Instantiate Casey first for monitoring
# ----------------------
casey_agent = casey.CaseyAgent()
casey_agent.log_action("Casey monitoring system initialized.")

# ----------------------
# Thread wrapper to run agents safely
# ----------------------
def start_agent(agent_instance, name):
    casey_agent.log_action(f"Starting {name}...")
    try:
        if hasattr(agent_instance, "run"):
            # Run agent's main loop
            agent_instance.run()
        else:
            # If no run() method, just loop to keep it alive
            while True:
                time.sleep(5)
    except Exception as e:
        casey_agent.log_action(f"{name} crashed: {e}")
        casey_agent.send_email_alert(
            f"NAE Crash Alert: {name}",
            f"Agent {name} crashed with error: {e}"
        )

# ----------------------
# Instantiate all other agents
# ----------------------
agents = {
    "Ralph": ralph.RalphAgent(),
    "Casey": casey_agent,
    "Optimus": optimus.OptimusAgent(),
    "Donnie": donnie.DonnieAgent(),
    "Shredder": shredder.ShredderAgent(),
    "Phisher": phisher.PhisherAgent(),
    "Rocksteady": rocksteady.RocksteadyAgent(),
    "Bebop": bebop.BebopAgent(),
    "Mikey": mikey.MikeyAgent(),
    "Leo": leo.LeoAgent(),
    "Splinter": splinter.SplinterAgent()
}

# ----------------------
# Register agents under Splinter for orchestration
# ----------------------
splinter_agent = agents["Splinter"]
splinter_agent.register_agents(list(agents.values()))
casey_agent.log_action("All agents registered under Splinter.")

# ----------------------
# Start all agents in separate threads
# ----------------------
threads = []
for name, agent_instance in agents.items():
    t = threading.Thread(target=start_agent, args=(agent_instance, name), daemon=True)
    t.start()
    threads.append(t)
    # Register agent process with Casey for monitoring
    casey_agent.monitor_process(name, os.getpid())  # Using current process; replace with agent PID if available

casey_agent.log_action("All agents launched and monitoring started.")

# ----------------------
# Keep main thread alive
# ----------------------
try:
    while True:
        time.sleep(5)
except KeyboardInterrupt:
    casey_agent.log_action("Shutdown requested. Exiting all agents...")
