#!/usr/bin/env python3
"""
AutoGen Studio Launcher
Double-click or run this file to start AutoGen Studio and open it in your browser
"""

import os
import sys
import subprocess
import time
import webbrowser
import socket
from pathlib import Path

def check_port(port):
    """Check if a port is already in use"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('localhost', port))
    sock.close()
    return result == 0

def install_autogen_studio():
    """Install AutoGen Studio if not available"""
    print("📦 AutoGen Studio not found. Installing...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "autogenstudio"], 
                      check=True, capture_output=True)
        print("✅ AutoGen Studio installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install AutoGen Studio: {e}")
        return False

def check_autogen_studio():
    """Check if autogenstudio command is available"""
    try:
        subprocess.run(["autogenstudio", "--version"], 
                      check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def start_autogen_studio(port=8081):
    """Start AutoGen Studio server"""
    print("=" * 60)
    print("🚀 Starting AutoGen Studio for NAE")
    print("=" * 60)
    print()
    
    # Check if already installed
    if not check_autogen_studio():
        if not install_autogen_studio():
            print("\n❌ Please install AutoGen Studio manually:")
            print("   pip install autogenstudio")
            input("\nPress Enter to exit...")
            return False
    
    # Check if port is already in use
    if check_port(port):
        print(f"✅ AutoGen Studio appears to be running on port {port}")
        print("🌐 Opening browser...")
        time.sleep(1)
        webbrowser.open(f"http://localhost:{port}")
        print(f"\n✅ Browser opened! AutoGen Studio should be available at http://localhost:{port}")
        print("\nPress Ctrl+C to stop the server")
        input("\nPress Enter to exit...")
        return True
    
    print(f"🚀 Starting AutoGen Studio server on port {port}...")
    print(f"📱 AutoGen Studio will be available at: http://localhost:{port}/")
    print()
    print("⏳ Waiting for server to start...")
    print()
    
    # Start AutoGen Studio
    try:
        process = subprocess.Popen(
            ["autogenstudio", "ui", "--port", str(port)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=Path(__file__).parent
        )
        
        # Wait a bit for server to start
        time.sleep(5)
        
        # Check if process is still running
        if process.poll() is None:
            print("✅ AutoGen Studio server started!")
            print(f"🆔 Server PID: {process.pid}")
            print()
            print("🌐 Opening browser...")
            
            # Open browser
            time.sleep(2)
            webbrowser.open(f"http://localhost:{port}")
            
            print()
            print("=" * 60)
            print("✅ AutoGen Studio is running!")
            print("=" * 60)
            print()
            print(f"📱 Browser opened at: http://localhost:{port}")
            print()
            print("📋 Next steps:")
            print("   1. Create Casey agent using autogen_studio_config.json")
            print("   2. Create a workflow with Casey and User agents")
            print("   3. Start chatting with Casey!")
            print()
            print("🛑 To stop the server, press Ctrl+C")
            print()
            
            # Wait for process
            try:
                process.wait()
            except KeyboardInterrupt:
                print("\n🛑 Stopping AutoGen Studio...")
                process.terminate()
                process.wait()
                print("✅ Server stopped")
            
            return True
        else:
            stdout, stderr = process.communicate()
            print("❌ Failed to start AutoGen Studio server")
            print(f"Error: {stderr.decode() if stderr else 'Unknown error'}")
            return False
            
    except Exception as e:
        print(f"❌ Error starting AutoGen Studio: {e}")
        return False

if __name__ == "__main__":
    try:
        start_autogen_studio()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
        sys.exit(0)

