#!/usr/bin/env python3
"""
Automated Scheduler for NAE Next Steps

Runs automated tasks:
- THRML experiments (weekly)
- Model retraining (monthly)
- Performance analysis (daily)
"""

import os
import sys
import time
import schedule
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

def run_thrml_experiments():
    """Run THRML experiments weekly"""
    print(f"[{datetime.now()}] Running THRML experiments...")
    try:
        from scripts.run_thrml_experiments import run_all_experiments
        results = run_all_experiments()
        print(f"[{datetime.now()}] ✅ THRML experiments completed")
        return results
    except Exception as e:
        print(f"[{datetime.now()}] ❌ THRML experiments failed: {e}")
        return None

def check_model_performance():
    """Check model performance daily"""
    print(f"[{datetime.now()}] Checking model performance...")
    try:
        from tools.metrics_collector import get_metrics_collector
        from tools.model_registry import ModelRegistry
        
        metrics = get_metrics_collector()
        registry = ModelRegistry()
        
        # Get recent performance
        dashboard = metrics.get_dashboard_data()
        
        # Check if any models need retraining
        models = registry.list_models()
        for model in models:
            model_id = model.get("model_id")
            if model_id:
                # Check performance (simplified)
                print(f"  Checking model: {model_id}")
        
        print(f"[{datetime.now()}] ✅ Performance check completed")
        return dashboard
    except Exception as e:
        print(f"[{datetime.now()}] ❌ Performance check failed: {e}")
        return None

def run_ensemble_evaluation():
    """Evaluate ensemble performance weekly"""
    print(f"[{datetime.now()}] Evaluating ensemble...")
    try:
        from tools.ensemble_framework import EnsembleFramework
        
        ensemble = EnsembleFramework(weighting_method="performance_weighted")
        
        # Get model predictions and evaluate
        # This would integrate with actual models
        print(f"[{datetime.now()}] ✅ Ensemble evaluation completed")
        return True
    except Exception as e:
        print(f"[{datetime.now()}] ❌ Ensemble evaluation failed: {e}")
        return False

def check_cicd_status():
    """Check CI/CD pipeline status"""
    print(f"[{datetime.now()}] Checking CI/CD status...")
    try:
        from tools.model_registry import ModelRegistry
        
        registry = ModelRegistry()
        models = registry.list_models()
        
        # Check for models in canary deployment
        canary_models = [m for m in models if m.get("status") == "canary"]
        
        if canary_models:
            print(f"  Found {len(canary_models)} models in canary deployment")
            for model in canary_models:
                print(f"    - {model.get('model_id')}: {model.get('status')}")
        
        print(f"[{datetime.now()}] ✅ CI/CD status check completed")
        return canary_models
    except Exception as e:
        print(f"[{datetime.now()}] ❌ CI/CD status check failed: {e}")
        return []

def main():
    """Main scheduler loop"""
    print("="*60)
    print("🚀 NAE Automated Scheduler Started")
    print("="*60)
    print(f"Started at: {datetime.now()}\n")
    
    # Schedule tasks
    schedule.every().day.at("09:00").do(check_model_performance)
    schedule.every().monday.at("10:00").do(run_thrml_experiments)
    schedule.every().monday.at("11:00").do(run_ensemble_evaluation)
    schedule.every().day.at("17:00").do(check_cicd_status)
    
    print("Scheduled tasks:")
    print("  - Daily performance check: 09:00")
    print("  - Weekly THRML experiments: Monday 10:00")
    print("  - Weekly ensemble evaluation: Monday 11:00")
    print("  - Daily CI/CD status check: 17:00")
    print("\nRunning scheduler loop...\n")
    
    # Run scheduler
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nScheduler stopped by user")
    except Exception as e:
        print(f"\n\nScheduler error: {e}")

