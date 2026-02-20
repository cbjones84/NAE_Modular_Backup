#!/usr/bin/env python3
"""Analyze Ralph and Donnie success rates"""

import json
import os
import re
from glob import glob

print('='*80)
print('RALPH & DONNIE SUCCESS RATE ANALYSIS')
print('='*80)
print()

# Analyze Ralph's approved strategies
ralph_files = sorted(glob('logs/ralph_approved_strategies_*.json'))
print(f'📊 Ralph Strategy Files Found: {len(ralph_files)} cycles')
print()

if ralph_files:
    # Sample a few recent files
    sample_files = ralph_files[-5:]  # Last 5 cycles
    
    total_candidates = 0
    total_approved = 0
    
    for file in sample_files:
        try:
            with open(file, 'r') as f:
                data = json.load(f)
            
            # Handle both list and dict formats
            if isinstance(data, list):
                approved_count = len(data)
                print(f'   {os.path.basename(file)}: {approved_count} approved strategies')
                total_approved += approved_count
            elif isinstance(data, dict):
                approved = len(data.get('approved_strategies', []))
                candidates = len(data.get('all_candidates', []))
                validated = len(data.get('validated_candidates', []))
                print(f'   {os.path.basename(file)}: {approved} approved / {candidates} candidates / {validated} validated')
                total_approved += approved
                total_candidates += candidates
            else:
                print(f'   {os.path.basename(file)}: Unknown format')
        except Exception as e:
            print(f'   Error reading {file}: {e}')
    
    print()
    print('📊 RALPH SUCCESS METRICS (Last 5 cycles):')
    if total_candidates > 0:
        approval_rate = (total_approved / total_candidates) * 100
        print(f'   Total Candidates: {total_candidates}')
        print(f'   Total Approved: {total_approved}')
        print(f'   Approval Rate: {approval_rate:.1f}%')
    else:
        print(f'   Total Approved Strategies: {total_approved}')
        print(f'   Approval Rate: Cannot calculate (candidate data not in files)')

# Analyze Donnie's execution
print()
print('='*80)
print('DONNIE EXECUTION SUCCESS RATE')
print('='*80)

donnie_log = 'logs/donnie.log'
if os.path.exists(donnie_log):
    with open(donnie_log, 'r') as f:
        lines = f.readlines()
    
    cycles = [line for line in lines if 'run_cycle completed' in line]
    received = [line for line in lines if 'Received' in line and 'strategies' in line]
    executed = [line for line in lines if 'executed strategies' in line]
    rejected = [line for line in lines if 'rejected' in line.lower() and 'Strategy' in line]
    
    total_received = 0
    for line in received:
        # Extract number from "Received X strategies"
        match = re.search(r'Received (\d+)', line)
        if match:
            total_received += int(match.group(1))
    
    total_executed = 0
    for line in executed:
        # Extract number from "X executed strategies"
        match = re.search(r'(\d+) executed strategies', line)
        if match:
            total_executed += int(match.group(1))
    
    total_rejected = len(rejected)
    total_cycles = len(cycles)
    
    print(f'📊 Donnie Log Analysis:')
    print(f'   Total Cycles: {total_cycles}')
    print(f'   Total Strategies Received: {total_received}')
    print(f'   Total Strategies Executed: {total_executed}')
    print(f'   Total Strategies Rejected: {total_rejected}')
    
    if total_received > 0:
        execution_rate = (total_executed / total_received) * 100
        print(f'   Execution Success Rate: {execution_rate:.1f}%')
        print(f'   Rejection Rate: {(total_rejected / total_received) * 100:.1f}%')
    
    # Recent cycle analysis
    print()
    print('📊 RECENT CYCLE ANALYSIS (Last 10 cycles):')
    recent_cycles = cycles[-10:]
    recent_executed = 0
    for line in recent_cycles:
        match = re.search(r'(\d+) executed strategies', line)
        if match:
            recent_executed += int(match.group(1))
    
    print(f'   Recent Executed: {recent_executed} strategies')
    
    if total_cycles > 0:
        cycles_with_exec = len([c for c in cycles if '0 executed' not in c])
        cycle_success_rate = (cycles_with_exec / total_cycles) * 100
        print(f'   Cycles with Executions: {cycles_with_exec}/{total_cycles}')
        print(f'   Cycle Success Rate: {cycle_success_rate:.1f}%')
else:
    print('   Donnie log not found')

print()
print('='*80)
print('SUMMARY')
print('='*80)
print()
print('RALPH: Strategy Generation & Approval')
print('  - Generates candidates from multiple sources')
print('  - Filters based on: trust_score >= 50, backtest_score >= 30')
print('  - Approval rate depends on quality of source data')
print()
print('DONNIE: Strategy Execution Validation')
print('  - Receives approved strategies from Ralph')
print('  - Validates based on: trust_score >= 60, backtest_score >= 50')
print('  - This is STRICTER than Ralph, causing all rejections')
print()
print('⚠️  ISSUE IDENTIFIED:')
print('   Ralph approves strategies with trust_score >= 50')
print('   Donnie requires trust_score >= 60')
print('   This mismatch causes 0% execution rate')
print('='*80)


