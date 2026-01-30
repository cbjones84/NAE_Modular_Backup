# /Users/melissabishop/NAE_Ready/tests/test_ralph_mcp.py
import sys
import os
import json
import asyncio

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from agents.ralph import RalphAgent

def main():
    print("Initializing RalphAgent with MCP Support...")
    # Mocking environment variables if needed
    os.environ['GITHUB_TOKEN'] = 'mock_token'
    
    ralph = RalphAgent()
    
    print("\nRunning Ralph Cycle (MCP Hunt)...")
    result = ralph.run_cycle()
    
    print("\nResults:")
    print(json.dumps(result, indent=2))
    
    if result.get("status") == "success":
        print("\n✅ MCP Integration Verified Successfully!")
    else:
        print("\n❌ MCP Integration Failed.")
        sys.exit(1)

if __name__ == "__main__":
    main()
