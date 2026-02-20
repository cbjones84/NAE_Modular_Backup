#!/usr/bin/env python3
"""
Start Splinter in Autonomous Mode
Splinter now includes all orchestrator functionality
"""

import sys
import os
import time

# Add NAE to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.splinter import SplinterAgent

def main():
    print("\n" + "="*80)
    print("SPLINTER AUTONOMOUS MODE")
    print("="*80)
    print("\nSplinter is now the comprehensive orchestrator for NAE")
    print("Features:")
    print("  • Agent synchronization and communication")
    print("  • Autonomous background operation")
    print("  • Self-improvement mechanisms")
    print("  • Error detection and recovery")
    print("  • Continuous monitoring")
    print("\n" + "="*80 + "\n")
    
    # Initialize Splinter with autonomous mode enabled
    splinter = SplinterAgent(enable_autonomous_mode=True)
    
    # Start autonomous operation
    splinter.start_autonomous()
    
    # Keep running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\nShutting down...")
        splinter.stop_autonomous()
        print("✓ Splinter stopped")

if __name__ == "__main__":
    main()

