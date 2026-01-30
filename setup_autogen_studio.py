#!/usr/bin/env python3
"""
AutoGen Studio Setup Script for NAE
This script helps you set up AutoGen Studio for the Neural Agency Engine
"""

import os
import subprocess
import sys

def install_autogen_studio():
    """Install AutoGen Studio"""
    print("🚀 Installing AutoGen Studio...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "autogenstudio"], check=True)
        print("✅ AutoGen Studio installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing AutoGen Studio: {e}")
        return False

def start_autogen_studio():
    """Start AutoGen Studio"""
    print("🌐 Starting AutoGen Studio...")
    print("📱 AutoGen Studio will be available at: http://localhost:8081/")
    print("🔄 Press Ctrl+C to stop the server")
    
    try:
        subprocess.run(["autogenstudio", "ui", "--port", "8081"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error starting AutoGen Studio: {e}")
    except KeyboardInterrupt:
        print("\n🛑 AutoGen Studio stopped by user")

def main():
    """Main setup function"""
    print("=" * 60)
    print("🎯 NAE AutoGen Studio Setup")
    print("=" * 60)
    
    # Check if autogenstudio is already installed
    try:
        subprocess.run(["autogenstudio", "--version"], check=True, capture_output=True)
        print("✅ AutoGen Studio is already installed!")
    except (subprocess.CalledProcessError, FileNotFoundError):
        if not install_autogen_studio():
            return
    
    print("\n📋 Setup Instructions:")
    print("1. AutoGen Studio will start on port 8081")
    print("2. Open your browser to: http://localhost:8081/")
    print("3. Use the NAE agents in AutoGen Studio:")
    print("   - Casey Agent (Builder/Refiner)")
    print("   - Ralph Agent (Strategy Generator)")
    print("   - Donnie Agent (Strategy Executor)")
    print("   - Optimus Agent (Live Trading)")
    
    print("\n🔧 NAE Integration:")
    print("- Your NAE agents are already configured for AutoGen")
    print("- Use nae_autogen_integrated.py for full integration")
    print("- Use nae_casey_autogen_demo.py for testing")
    
    input("\nPress Enter to start AutoGen Studio...")
    start_autogen_studio()

if __name__ == "__main__":
    main()
