#!/usr/bin/env python3
"""
Genny Tax Report Generator
Generate comprehensive tax reports and summaries
"""

import os
import sys
import json
import argparse
from datetime import datetime

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from agents.genny import GennyAgent


def main():
    parser = argparse.ArgumentParser(description="Generate Genny Tax Reports")
    parser.add_argument("--year", type=int, help="Tax year (default: current year)")
    parser.add_argument("--format", choices=["summary", "turbo_tax", "cpa", "json"], default="summary", help="Report format")
    parser.add_argument("--include-unrealized", action="store_true", help="Include unrealized gains")
    parser.add_argument("--output", type=str, help="Output file path")
    parser.add_argument("--nc-advice", type=str, help="Get NC tax advice for a question")
    
    args = parser.parse_args()
    
    # Initialize Genny
    genny = GennyAgent()
    
    if args.nc_advice:
        # Get NC tax advice
        advice = genny.get_nc_tax_advice(args.nc_advice)
        print("\n" + "=" * 60)
        print("NORTH CAROLINA TAX ADVICE")
        print("=" * 60)
        print(f"\nQuestion: {advice['question']}")
        print("\nNC-Specific Advice:")
        for note in advice.get("nc_specific", []):
            print(f"  • {note}")
        print("\nNC Tax Knowledge:")
        print(json.dumps(advice.get("nc_tax_knowledge", {}), indent=2, default=str))
        return
    
    # Generate tax summary
    tax_year = args.year or datetime.now().year
    
    print("\n" + "=" * 60)
    print(f"GENNY TAX REPORT - {tax_year}")
    print("=" * 60)
    
    if args.format == "summary":
        summary = genny.generate_tax_summary(tax_year, args.include_unrealized)
        
        print(f"\nTax Year: {tax_year}")
        print(f"Generated: {summary.get('generated_at', 'N/A')}")
        
        print("\n" + "-" * 60)
        print("TRADING ACTIVITY")
        print("-" * 60)
        trades = summary.get("trades", {})
        print(f"Total Trades: {trades.get('total_trades', 0)}")
        print(f"  Buy Trades: {trades.get('buy_trades', 0)}")
        print(f"  Sell Trades: {trades.get('sell_trades', 0)}")
        print(f"  Day Trades: {trades.get('day_trades', 0)}")
        print(f"\nTrades by Agent:")
        for agent, count in trades.get("by_agent", {}).items():
            print(f"  {agent}: {count}")
        
        print("\n" + "-" * 60)
        print("CAPITAL GAINS/LOSSES")
        print("-" * 60)
        gains = summary.get("capital_gains", {})
        print(f"Total Realized Gain: ${gains.get('total_realized_gain', 0):,.2f}")
        print(f"  Short-Term Gain: ${gains.get('short_term_gain', 0):,.2f}")
        print(f"  Long-Term Gain: ${gains.get('long_term_gain', 0):,.2f}")
        print(f"  Total Transactions: {gains.get('total_transactions', 0)}")
        print(f"\nGains by Asset Type:")
        for asset_type, amount in gains.get("by_asset_type", {}).items():
            print(f"  {asset_type}: ${amount:,.2f}")
        
        print("\n" + "-" * 60)
        print("BUSINESS EXPENSES")
        print("-" * 60)
        expenses = summary.get("expenses", {})
        print(f"Total Expenses: ${expenses.get('total_expenses', 0):,.2f}")
        print(f"Deductible Expenses: ${expenses.get('deductible_expenses', 0):,.2f}")
        print(f"\nExpenses by Category:")
        for category, amount in expenses.get("by_category", {}).items():
            print(f"  {category}: ${amount:,.2f}")
        
        print("\n" + "-" * 60)
        print("TAX ESTIMATES")
        print("-" * 60)
        tax_estimates = summary.get("tax_estimates", {})
        print(f"Federal Tax: ${tax_estimates.get('federal_tax', 0):,.2f}")
        print(f"NC State Tax: ${tax_estimates.get('nc_state_tax', 0):,.2f}")
        print(f"Total Estimated Tax: ${tax_estimates.get('total_tax', 0):,.2f}")
        print(f"Effective Rate: {tax_estimates.get('effective_rate', 0):.2f}%")
        
        print("\n" + "-" * 60)
        print("NORTH CAROLINA COMPLIANCE")
        print("-" * 60)
        nc_info = summary.get("north_carolina_tax", {})
        print(f"NC Tax Rate: {nc_info.get('rate', 0) * 100:.2f}%")
        print(f"Estimated NC Tax: ${nc_info.get('estimated_state_tax', 0):,.2f}")
        print(f"\nNotes:")
        for note in nc_info.get("notes", []):
            print(f"  • {note}")
        
        print("\n" + "-" * 60)
        print("NET AFTER TAX")
        print("-" * 60)
        print(f"Net After Tax: ${summary.get('net_after_tax', 0):,.2f}")
        
        print("\n" + "=" * 60)
        
        # Save to file if requested
        if args.output:
            with open(args.output, "w") as f:
                json.dump(summary, f, indent=2, default=str)
            print(f"\nReport saved to: {args.output}")
    
    elif args.format == "turbo_tax":
        data = genny.export_tax_data("turbo_tax", tax_year)
        output_file = args.output or f"tax_reports/turbo_tax_{tax_year}.json"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(data, f, indent=2, default=str)
        print(f"\nTurboTax format exported to: {output_file}")
    
    elif args.format == "cpa":
        data = genny.export_tax_data("cpa", tax_year)
        output_file = args.output or f"tax_reports/cpa_{tax_year}.json"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(data, f, indent=2, default=str)
        print(f"\nCPA format exported to: {output_file}")
    
    else:  # json
        summary = genny.generate_tax_summary(tax_year, args.include_unrealized)
        output_file = args.output or f"tax_reports/tax_summary_{tax_year}.json"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"\nJSON report saved to: {output_file}")


if __name__ == "__main__":
    main()

