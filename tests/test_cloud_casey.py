#!/usr/bin/env python3
"""
Cloud Casey Agent Test Script
Tests the cloud-based Casey agent for continuous NAE analysis
"""

import os
import sys
import time
import json
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

from cloud_casey_deployment import CloudCaseyAgent

def test_cloud_casey_capabilities():
    """Test all Cloud Casey capabilities"""
    print("🌩️ Testing Cloud Casey Agent Capabilities...")
    print("=" * 60)
    
    # Initialize Cloud Casey
    cloud_casey = CloudCaseyAgent()
    
    # Test 1: Continuous Analysis
    print("\n📊 Test 1: Continuous NAE Analysis")
    print("-" * 40)
    
    # Let it run for a short time to collect data
    time.sleep(5)
    
    status = cloud_casey.get_status()
    print(f"✅ Cloud Casey Status:")
    print(f"   - Status: {status['status']}")
    print(f"   - NAE System Status: {status['nae_system_status']}")
    print(f"   - Analysis Count: {status['analysis_count']}")
    print(f"   - Health Reports: {status['health_reports_count']}")
    
    # Test 2: Health Analysis
    print("\n❤️ Test 2: NAE Health Analysis")
    print("-" * 40)
    
    if cloud_casey.health_reports:
        latest_report = cloud_casey.health_reports[-1]
        print(f"✅ Latest Health Report:")
        print(f"   - Overall Health Score: {latest_report.overall_health_score:.1f}/100")
        print(f"   - Agent Health Scores:")
        for agent, score in latest_report.agent_health_scores.items():
            print(f"     • {agent}: {score:.1f}/100")
        print(f"   - Critical Issues: {len(latest_report.critical_issues)}")
        print(f"   - Recommendations: {len(latest_report.recommendations)}")
        
        if latest_report.recommendations:
            print(f"   - Top Recommendations:")
            for rec in latest_report.recommendations[:3]:
                print(f"     • {rec}")
    else:
        print("⏳ Health reports will be generated during continuous analysis")
    
    # Test 3: Code Analysis
    print("\n🔍 Test 3: Code Analysis")
    print("-" * 40)
    
    if cloud_casey.code_analyses:
        print(f"✅ Code Analysis Results:")
        print(f"   - Files Analyzed: {len(cloud_casey.code_analyses)}")
        
        for analysis in cloud_casey.code_analyses[:3]:
            print(f"   - {analysis.file_path}:")
            print(f"     • Complexity: {analysis.complexity_score:.1f}/100")
            print(f"     • Maintainability: {analysis.maintainability_score:.1f}/100")
            print(f"     • Performance: {analysis.performance_score:.1f}/100")
            print(f"     • Issues Found: {len(analysis.issues_found)}")
            print(f"     • Improvements: {len(analysis.improvements_suggested)}")
    else:
        print("⏳ Code analysis will be performed during continuous analysis")
    
    # Test 4: Improvement Suggestions
    print("\n💡 Test 4: Improvement Suggestions")
    print("-" * 40)
    
    if cloud_casey.improvement_suggestions:
        print(f"✅ Improvement Suggestions Generated:")
        print(f"   - Total Suggestions: {len(cloud_casey.improvement_suggestions)}")
        print(f"   - Top Suggestions:")
        for suggestion in cloud_casey.improvement_suggestions[:5]:
            print(f"     • {suggestion}")
    else:
        print("⏳ Improvement suggestions will be generated during analysis")
    
    # Test 5: System Status Monitoring
    print("\n📈 Test 5: System Status Monitoring")
    print("-" * 40)
    
    connectivity = cloud_casey._check_nae_connectivity()
    resources = cloud_casey._monitor_system_resources()
    
    print(f"✅ System Monitoring:")
    print(f"   - Connectivity Status: {connectivity['connectivity_status']}")
    print(f"   - Response Time: {connectivity['response_time']}s")
    print(f"   - CPU Usage: {resources['cpu_usage']}%")
    print(f"   - Memory Usage: {resources['memory_usage']}%")
    print(f"   - Disk Usage: {resources['disk_usage']}%")
    
    # Test 6: Report Generation
    print("\n📋 Test 6: Report Generation")
    print("-" * 40)
    
    report = cloud_casey._generate_nae_report()
    print(f"✅ NAE Report Generated:")
    print(f"   - Report Type: {report['report_type']}")
    print(f"   - System Status: {report['system_status']}")
    print(f"   - Overall Score: {report['health_summary']['overall_score']:.1f}")
    print(f"   - Agent Count: {report['health_summary']['agent_count']}")
    print(f"   - Critical Issues: {report['health_summary']['critical_issues']}")
    print(f"   - Analysis Count: {report['analysis_summary']['total_analyses']}")
    print(f"   - Recommendations: {len(report['recommendations'])}")
    
    print("\n🚀 Cloud Casey Agent Test Complete!")
    print("=" * 60)
    
    return cloud_casey

def demonstrate_continuous_analysis():
    """Demonstrate continuous analysis capabilities"""
    print("\n🔄 Demonstrating Continuous Analysis...")
    print("=" * 60)
    
    cloud_casey = CloudCaseyAgent()
    
    print("⏳ Running continuous analysis for 30 seconds...")
    
    # Run for 30 seconds to demonstrate continuous analysis
    start_time = time.time()
    while time.time() - start_time < 30:
        time.sleep(5)
        
        status = cloud_casey.get_status()
        print(f"📊 Analysis Progress:")
        print(f"   - Health Reports: {status['health_reports_count']}")
        print(f"   - Code Analyses: {status['code_analyses_count']}")
        print(f"   - Improvement Suggestions: {status['improvement_suggestions_count']}")
        
        if cloud_casey.health_reports:
            latest_health = cloud_casey.health_reports[-1].overall_health_score
            print(f"   - Latest Health Score: {latest_health:.1f}/100")
    
    print("\n✅ Continuous analysis demonstration complete!")
    
    # Show final results
    final_status = cloud_casey.get_status()
    print(f"\n📈 Final Results:")
    print(f"   - Total Analyses: {final_status['analysis_count']}")
    print(f"   - Health Reports: {final_status['health_reports_count']}")
    print(f"   - Code Analyses: {final_status['code_analyses_count']}")
    print(f"   - Improvement Suggestions: {final_status['improvement_suggestions_count']}")
    
    if cloud_casey.health_reports:
        latest_report = cloud_casey.health_reports[-1]
        print(f"\n🎯 Latest Health Analysis:")
        print(f"   - Overall Score: {latest_report.overall_health_score:.1f}/100")
        print(f"   - Agent Scores:")
        for agent, score in latest_report.agent_health_scores.items():
            status_emoji = "🟢" if score >= 80 else "🟡" if score >= 60 else "🔴"
            print(f"     {status_emoji} {agent}: {score:.1f}/100")
        
        if latest_report.recommendations:
            print(f"\n💡 Top Recommendations:")
            for i, rec in enumerate(latest_report.recommendations[:3], 1):
                print(f"   {i}. {rec}")

def show_deployment_options():
    """Show deployment options for Cloud Casey"""
    print("\n🚀 Cloud Casey Deployment Options")
    print("=" * 60)
    
    print("\n1. 🌩️ AWS Lambda Deployment")
    print("   - Serverless execution")
    print("   - Automatic scaling")
    print("   - Pay-per-execution")
    print("   - Continuous monitoring via CloudWatch Events")
    print("   - Cost-effective for intermittent analysis")
    
    print("\n2. 🐳 Docker Deployment")
    print("   - Container-based execution")
    print("   - Can run on any cloud provider")
    print("   - Persistent execution")
    print("   - Full control over environment")
    print("   - Suitable for continuous analysis")
    
    print("\n3. ☁️ Cloud Provider Options")
    print("   - AWS: Lambda, ECS, EC2")
    print("   - Google Cloud: Cloud Functions, Cloud Run, GKE")
    print("   - Azure: Functions, Container Instances, AKS")
    print("   - DigitalOcean: App Platform, Droplets")
    
    print("\n4. 🔧 Deployment Commands")
    print("   # AWS Lambda:")
    print("   python deploy_cloud_casey.py")
    print("   ")
    print("   # Docker:")
    print("   docker-compose -f docker-compose.cloud_casey.yml up -d")
    print("   ")
    print("   # Manual deployment:")
    print("   python cloud_casey_deployment.py")

if __name__ == "__main__":
    print("🌩️ Cloud Casey Agent - Continuous NAE Analysis")
    print("=" * 70)
    
    # Run capability tests
    cloud_casey = test_cloud_casey_capabilities()
    
    # Demonstrate continuous analysis
    demonstrate_continuous_analysis()
    
    # Show deployment options
    show_deployment_options()
    
    print("\n✨ Cloud Casey Agent is ready for continuous NAE analysis!")
    print("🚀 Deploy to cloud for persistent monitoring and optimization!")
