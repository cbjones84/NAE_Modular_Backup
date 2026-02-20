#!/usr/bin/env python3
"""
Run Ralph Cycle to Generate New Strategies
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from agents.ralph import RalphAgent
from env_loader import EnvLoader

print('='*70)
print('RUNNING RALPH CYCLE TO GENERATE NEW STRATEGIES')
print('='*70)
print()

# Setup environment
loader = EnvLoader()

# Initialize Ralph
print('Initializing Ralph Agent...')
ralph = RalphAgent()

# Run cycle
print('Running Ralph cycle (this may take a moment)...')
print()
result = ralph.run_cycle()

print()
print('='*70)
print('RALPH CYCLE COMPLETE')
print('='*70)
print(f"Approved strategies: {result.get('approved_count', 0)}")
print(f"Saved to: {result.get('path', 'N/A')}")
print()

# Show top strategies
if ralph.strategy_database:
    print('Top approved strategies:')
    top = ralph.top_strategies(5)
    for i, s in enumerate(top, 1):
        print(f"  {i}. {s.get('name', 'Unknown')} (Trust: {s.get('trust_score', 0):.1f})")
else:
    print('No strategies approved in this cycle')

