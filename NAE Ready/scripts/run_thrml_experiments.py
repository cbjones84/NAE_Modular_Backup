#!/usr/bin/env python3
"""
Automated THRML Experiment Runner

Runs THRML experiments on schedule and reports results
"""

import os
import sys
import time
import json
from datetime import datetime, timedelta
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from experiments.thrml_sampling_experiment import THRMLExperiment


def run_all_experiments():
    """Run all THRML experiments and save results"""
    print("="*60)
    print("🚀 Starting Automated THRML Experiments")
    print("="*60)
    print(f"Timestamp: {datetime.now().isoformat()}\n")
    
    experiment = THRMLExperiment()
    results = {}
    
    # Experiment 1: Market State Sampler
    print("\n" + "="*60)
    print("Experiment 1: Market State Sampler")
    print("="*60)
    try:
        sampler_results = experiment.run_market_state_sampler_experiment(
            num_samples=1000,
            num_nodes=10
        )
        results["market_state_sampler"] = sampler_results
        experiment.save_results(sampler_results, "market_state_sampler")
        print("✅ Market state sampler experiment completed")
    except Exception as e:
        print(f"❌ Market state sampler experiment failed: {e}")
        results["market_state_sampler"] = {"error": str(e)}
    
    # Experiment 2: Options Valuation
    print("\n" + "="*60)
    print("Experiment 2: Options Valuation")
    print("="*60)
    try:
        options_results = experiment.run_options_valuation_experiment(
            spot_price=100.0,
            strike=100.0,
            time_to_expiry=0.25,
            num_paths=5000
        )
        results["options_valuation"] = options_results
        experiment.save_results(options_results, "options_valuation")
        print("✅ Options valuation experiment completed")
    except Exception as e:
        print(f"❌ Options valuation experiment failed: {e}")
        results["options_valuation"] = {"error": str(e)}
    
    # Save summary
    summary_file = "experiments/results/experiment_summary.json"
    os.makedirs(os.path.dirname(summary_file), exist_ok=True)
    
    summary = {
        "timestamp": datetime.now().isoformat(),
        "experiments": results,
        "status": "completed" if all("error" not in r for r in results.values()) else "partial"
    }
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print("\n" + "="*60)
    print("✅ All Experiments Complete")
    print("="*60)
    print(f"Results saved to: {summary_file}")
    
    return summary


if __name__ == "__main__":
    run_all_experiments()

