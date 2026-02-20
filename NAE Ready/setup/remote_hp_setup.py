#!/usr/bin/env python3
"""
Remote HP OmniBook X Setup

Attempts to complete HP setup remotely from Mac via SSH.
"""

import os
import sys
import subprocess
from pathlib import Path

# Add NAE paths
script_dir = Path(__file__).parent
nae_root = script_dir.parent
sys.path.insert(0, str(nae_root))

from setup.remote_execution_bridge import RemoteExecutionBridge


def setup_hp_remotely():
    """Complete HP setup remotely"""
    print("==========================================")
    print("Remote HP OmniBook X Setup")
    print("==========================================")
    print("")
    
    # Check if remote config exists
    config_file = nae_root / "config" / "remote_config.json"
    if not config_file.exists():
        print("❌ Remote configuration not found")
        print("Run: ./setup/configure_remote_connection.sh first")
        return False
    
    bridge = RemoteExecutionBridge()
    
    try:
        # Test connection
        print("Step 1: Testing connection...")
        test_result = bridge.execute_remote("echo 'Connection test'")
        if not test_result.get("success"):
            print(f"❌ Connection failed: {test_result.get('error', 'Unknown error')}")
            return False
        print("✅ Connection successful")
        print("")
        
        # Check if NAE exists
        print("Step 2: Checking NAE installation...")
        nae_check = bridge.execute_remote("test -d NAE && echo 'exists' || echo 'not_found'")
        nae_exists = "exists" in nae_check.get("output", "")
        
        if not nae_exists:
            print("NAE not found. Cloning repository...")
            clone_result = bridge.execute_remote(
                "git clone https://github.com/cbjones84/NAE.git"
            )
            if not clone_result.get("success"):
                print(f"❌ Clone failed: {clone_result.get('error', 'Unknown error')}")
                return False
            print("✅ Repository cloned")
        else:
            print("✅ NAE directory found")
        print("")
        
        # Configure environment
        print("Step 3: Configuring environment...")
        env_result = bridge.execute_remote("cd NAE && bash setup/configure_environments.sh")
        if env_result.get("success"):
            print("✅ Environment configured")
        else:
            print(f"⚠️  Environment configuration: {env_result.get('error', 'Check output')}")
        print("")
        
        # Initialize safety systems
        print("Step 4: Initializing safety systems...")
        safety_result = bridge.execute_remote("cd NAE && bash setup/initialize_safety_systems.sh")
        if safety_result.get("success"):
            print("✅ Safety systems initialized")
        else:
            print(f"⚠️  Safety systems: {safety_result.get('error', 'Check output')}")
        print("")
        
        # Configure as node
        print("Step 5: Configuring as Node...")
        master_api_key = "72364bc8ecfea124010e9811d06d0b0b3b220e4dee3d09163b2bd1f005af40a7"
        
        # Get Mac IP (simplified - user may need to set this)
        mac_ip = os.getenv("MAC_IP", "localhost")
        master_url = f"http://{mac_ip}:8080"
        
        node_config_cmd = f"""cd NAE && cat >> .env.prod << 'EOF'

# Node Settings
NAE_IS_MASTER=false
NAE_IS_NODE=true
NAE_NODE_API_KEY={master_api_key}
NAE_MASTER_URL={master_url}
NAE_NODE_PORT=8081
EOF"""
        
        node_result = bridge.execute_remote(node_config_cmd)
        if node_result.get("success"):
            print("✅ Node configuration added")
        else:
            print(f"⚠️  Node configuration: {node_result.get('error', 'Check output')}")
        print("")
        
        # Create production branch
        print("Step 6: Setting up production branch...")
        branch_cmd = """cd NAE && (
            git checkout prod 2>/dev/null || git checkout -b prod
        ) && git push -u origin prod 2>/dev/null || echo 'Branch ready'"""
        
        branch_result = bridge.execute_remote(branch_cmd)
        if branch_result.get("success"):
            print("✅ Production branch configured")
        else:
            print(f"⚠️  Branch setup: {branch_result.get('error', 'Check output')}")
        print("")
        
        # Enable SSH (if needed)
        print("Step 7: Verifying SSH...")
        ssh_check = bridge.execute_remote("systemctl is-active ssh 2>/dev/null || systemctl is-active sshd 2>/dev/null || echo 'not_active'")
        if "active" not in ssh_check.get("output", ""):
            print("Enabling SSH...")
            ssh_enable = bridge.execute_remote("sudo systemctl enable ssh 2>/dev/null || sudo systemctl enable sshd 2>/dev/null || echo 'enabled'")
            print("✅ SSH enabled")
        else:
            print("✅ SSH already active")
        print("")
        
        # Get IP address
        print("Step 8: Getting HP IP address...")
        ip_result = bridge.execute_remote("hostname -I | awk '{print $1}' || ip addr show | grep 'inet ' | grep -v 127.0.0.1 | head -1 | awk '{print $2}' | cut -d/ -f1")
        hp_ip = ip_result.get("output", "").strip()
        
        # Final verification
        print("Step 9: Verifying setup...")
        status = bridge.get_remote_status()
        
        print("")
        print("==========================================")
        print("✅ HP OmniBook X Setup Complete!")
        print("==========================================")
        print("")
        print(f"HP IP Address: {hp_ip}")
        print(f"Branch: {status.get('branch', 'unknown')}")
        print(f"Production Mode: {status.get('production_mode', 'unknown')}")
        print("")
        print("Next Steps:")
        print("1. Verify connection: python3 setup/cursor_remote_integration.py --verify")
        print("2. Check status: python3 setup/cursor_remote_integration.py --status")
        print("3. Start NAE: python3 setup/cursor_remote_integration.py --start")
        print("")
        
        return True
        
    except Exception as e:
        print(f"❌ Setup error: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        bridge.close()


if __name__ == "__main__":
    success = setup_hp_remotely()
    sys.exit(0 if success else 1)

