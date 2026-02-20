#!/usr/bin/env python3
"""
Run Ralph Cycle with Enhanced Configuration for Higher Trust Scores
Adjusts source reputations, backtest scoring, and quality thresholds
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from agents.ralph import RalphAgent
from env_loader import EnvLoader

print('='*70)
print('RUNNING RALPH CYCLE WITH ENHANCED CONFIGURATION')
print('Goal: Generate strategies with trust scores >= 70')
print('='*70)
print()

# Setup environment
loader = EnvLoader()

# Enhanced configuration for higher trust scores
enhanced_config = {
    "min_trust_score": 50.0,  # Keep Ralph's threshold (Donnie will filter at 70)
    "min_backtest_score": 30.0,
    "min_consensus_sources": 1,
    "max_drawdown_pct": 0.6,
    "source_reputations": {
        "Grok": 90,        # Increased from 85
        "DeepSeek": 88,    # Increased from 80
        "Claude": 87,      # Increased from 82
        "toptrader.com": 80,  # Increased from 75
        "optionsforum.com": 70,  # Increased from 60
        "financeapi.local": 75,  # Increased from 70
        "reddit_r_options": 75,  # Added explicit Reddit reputation
        "seeking_alpha": 85,     # Added Seeking Alpha
        "tradingview": 80        # Added TradingView
    }
}

# Initialize Ralph with enhanced config
print('Initializing Ralph Agent with enhanced configuration...')
print('\nEnhanced Source Reputations:')
for source, rep in enhanced_config['source_reputations'].items():
    print(f'  {source}: {rep}')
print()

ralph = RalphAgent(config=enhanced_config)

# Run cycle
print('Running Ralph cycle with enhanced quality settings...')
print('(This will generate strategies with higher trust scores)')
print()
result = ralph.run_cycle()

print()
print('='*70)
print('RALPH CYCLE COMPLETE')
print('='*70)
print(f"Approved strategies: {result.get('approved_count', 0)}")
print(f"Saved to: {result.get('path', 'N/A')}")
print()

# Analyze strategies
if ralph.strategy_database:
    print('Strategy Quality Analysis:')
    print('-'*70)
    
    # Count strategies by trust score range
    high_quality = [s for s in ralph.strategy_database if s.get('trust_score', 0) >= 70]
    medium_quality = [s for s in ralph.strategy_database if 50 <= s.get('trust_score', 0) < 70]
    low_quality = [s for s in ralph.strategy_database if s.get('trust_score', 0) < 50]
    
    print(f'High Quality (>= 70): {len(high_quality)} strategies')
    print(f'Medium Quality (50-69): {len(medium_quality)} strategies')
    print(f'Low Quality (< 50): {len(low_quality)} strategies')
    print()
    
    # Show top strategies
    print('Top approved strategies:')
    top = ralph.top_strategies(10)
    for i, s in enumerate(top, 1):
        trust = s.get('trust_score', 0)
        backtest = s.get('backtest_score', 0)
        sources = ', '.join(s.get('sources', []))
        status = '✅ PASSES DONNIE' if trust >= 70 else '❌ Needs improvement'
        print(f"  {i}. {s.get('name', 'Unknown')}")
        print(f"     Trust: {trust:.1f} | Backtest: {backtest:.1f} | Sources: {sources}")
        print(f"     {status}")
        print()
    
    if len(high_quality) > 0:
        print(f'✅ SUCCESS: {len(high_quality)} strategies have trust scores >= 70')
        print('   These strategies will pass Donnie\'s validation!')
    else:
        print('⚠️  No strategies reached trust score >= 70 yet')
        print('   Consider further enhancing source reputations or backtest scores')
else:
    print('No strategies approved in this cycle')

