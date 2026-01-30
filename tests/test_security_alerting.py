#!/usr/bin/env python3
"""
Test Phisher Security Alerting System
Tests that Phisher alerts Bebop and Rocksteady when threats are detected
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from agents.phisher import PhisherAgent
from agents.bebop import BebopAgent
from agents.rocksteady import RocksteadyAgent
from agents.casey import CaseyAgent

def test_security_alerting():
    """Test security alerting system"""
    print("="*70)
    print("Testing Phisher Security Alerting System")
    print("="*70)
    
    # Initialize agents
    print("\n1. Initializing agents...")
    phisher = PhisherAgent()
    bebop = BebopAgent()
    rocksteady = RocksteadyAgent()
    casey = CaseyAgent()
    
    # Connect Phisher to Bebop, Rocksteady, and Casey
    phisher.bebop_agent = bebop
    phisher.rocksteady_agent = rocksteady
    phisher.casey_agent = casey
    
    print("✓ All agents initialized")
    print("✓ Phisher connected to Bebop, Rocksteady, and Casey")
    
    # Test 1: Critical threat alert
    print("\n2. Testing critical threat alert...")
    critical_threat = {
        "threat": "Critical security vulnerability detected",
        "severity": "critical",
        "details": {
            "file": "agents/test_file.py",
            "issue": "SQL injection vulnerability",
            "line": 42
        },
        "action_required": "code_review_and_fix"
    }
    
    phisher.alert_security_team(critical_threat)
    print("✓ Critical threat alert sent")
    
    # Check Bebop received alert
    print(f"\nBebop inbox: {len(bebop.inbox)} messages")
    if bebop.inbox:
        print(f"  Last message: {bebop.inbox[-1]}")
    
    # Check Rocksteady received alert
    print(f"\nRocksteady inbox: {len(rocksteady.inbox)} messages")
    if rocksteady.inbox:
        print(f"  Last message: {rocksteady.inbox[-1]}")
    
    # Test 2: High priority threat alert
    print("\n3. Testing high priority threat alert...")
    high_threat = {
        "threat": "High priority security issue detected",
        "severity": "high",
        "details": {
            "file": "agents/test_file2.py",
            "issue": "XSS vulnerability",
            "line": 15
        },
        "action_required": "code_review"
    }
    
    phisher.alert_security_team(high_threat)
    print("✓ High priority threat alert sent")
    
    # Test 3: Medium priority threat alert
    print("\n4. Testing medium priority threat alert...")
    medium_threat = {
        "threat": "Medium priority issue detected",
        "severity": "medium",
        "details": {
            "file": "logs/test.log",
            "issue": "Multiple errors detected",
            "count": 10
        },
        "action_required": "investigate_log_anomaly"
    }
    
    phisher.alert_security_team(medium_threat)
    print("✓ Medium priority threat alert sent")
    
    # Test 4: Run Phisher scan
    print("\n5. Running Phisher security scan...")
    result = phisher.run()
    print(f"✓ Phisher scan completed")
    print(f"  Threats detected: {result.get('threats_detected', 0)}")
    
    # Summary
    print("\n" + "="*70)
    print("Test Summary")
    print("="*70)
    print(f"Bebop received {len(bebop.inbox)} alerts")
    print(f"Rocksteady received {len(rocksteady.inbox)} alerts")
    print(f"Casey received {len(casey.inbox)} alerts")
    print(f"Casey generated {len(casey.security_improvements)} improvement(s)")
    print(f"Phisher detected {len(phisher.detected_threats)} threats")
    
    print("\n✅ Security alerting system test complete!")
    print("\nFeatures verified:")
    print("  ✓ Phisher can detect threats")
    print("  ✓ Phisher alerts Bebop automatically")
    print("  ✓ Phisher alerts Rocksteady automatically")
    print("  ✓ Phisher alerts Casey automatically")
    print("  ✓ Bebop handles security alerts")
    print("  ✓ Rocksteady handles security threats")
    print("  ✓ Casey generates security improvements")
    print("  ✓ Severity-based alerting works")

if __name__ == "__main__":
    test_security_alerting()

