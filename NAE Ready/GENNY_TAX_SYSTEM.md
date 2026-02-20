# Genny Tax Preparation and Assessment System

## Overview

Genny has been enhanced with comprehensive tax preparation and assessment capabilities. The system automatically tracks all trades, crypto transactions, fiat flows, and expenses to provide hands-free bookkeeping and automated tax preparation.

## Key Features

### ✅ Complete Trade Tracking
- **Optimus Trades**: Automatically tracks every trade Optimus makes
- **Crypto Transactions**: Tracks all crypto transactions from Shredder and April
- **Fiat Flows**: Monitors incoming/outgoing fiat from Donnie & Mikey
- **Real-time Recording**: All transactions recorded immediately

### ✅ Cost Basis Tracking
- **FIFO (First In, First Out)**: Default method
- **LIFO (Last In, First Out)**: Available option
- **HIFO (Highest In, First Out)**: Available option
- **Average Cost**: Available option
- **Configurable**: Can change method as needed

### ✅ Capital Gains Calculation
- **Short-Term vs Long-Term**: Automatic classification (>365 days = long-term)
- **Realized Gains**: Calculated on every sale
- **Unrealized Gains**: Can be included in reports
- **Day Trade Detection**: Automatically identifies day trades

### ✅ Expense Tracking
- **Deductible Expenses**: Tracks software, hardware, subscriptions
- **Business Use Percentage**: Configurable per expense
- **Receipt Management**: Links to receipt files
- **Category Organization**: Organized by expense type

### ✅ Tax Summaries
- **Quarterly Reports**: Auto-generated at end of each quarter
- **Annual Reports**: Comprehensive year-end summaries
- **TurboTax Export**: Ready for TurboTax import
- **CPA Format**: Professional format for CPAs

### ✅ North Carolina Tax Law Knowledge
- **NC Tax Rates**: 4.75% flat individual income tax
- **Capital Gains Treatment**: All gains treated as ordinary income
- **Business Expenses**: Deductible if ordinary and necessary
- **Day Trading**: May qualify as business income
- **Crypto Taxation**: Follows federal treatment (property)

## System Architecture

### Tax Module (`agents/genny_tax_module.py`)
- Core tax calculation engine
- Cost basis tracking
- Capital gains calculation
- Expense management
- Tax summary generation

### Integration Hooks (`agents/genny_tax_integration.py`)
- Automatic hooks for agent transactions
- Real-time tracking
- Event-driven updates

### Genny Agent Integration
- Tax module integrated into Genny
- Automatic tracking on all trades
- Quarterly tax summaries
- NC tax law knowledge base

## Usage

### Track a Trade (Automatic)

Trades are automatically tracked when Optimus executes:

```python
from agents.genny import GennyAgent

genny = GennyAgent()

# Optimus trade automatically tracked via hook
# No manual intervention needed
```

### Track Crypto Transaction

```python
# Shredder or April crypto transaction
crypto_data = {
    "transaction_id": "crypto_001",
    "timestamp": "2024-06-15T10:00:00Z",
    "crypto_symbol": "BTC",
    "action": "buy",
    "quantity": 0.5,
    "price": 50000.00,
    "fees": 10.00
}

genny.track_crypto_transaction(crypto_data, agent="shredder")
```

### Record Expense

```python
expense_data = {
    "expense_id": "exp_001",
    "timestamp": "2024-03-01T00:00:00Z",
    "category": "software",
    "description": "Trading platform subscription",
    "amount": 99.00,
    "deductible_pct": 100.0,
    "business_use_pct": 100.0
}

genny.record_deductible_expense(expense_data)
```

### Generate Tax Summary

```python
# Annual summary
summary = genny.generate_tax_summary(2024)

# Quarterly summary (with unrealized gains)
summary = genny.generate_tax_summary(2024, include_unrealized=True)
```

### Export for TurboTax

```python
# Export in TurboTax format
turbo_tax_data = genny.export_tax_data("turbo_tax", 2024)

# Save to file
import json
with open("turbo_tax_2024.json", "w") as f:
    json.dump(turbo_tax_data, f, indent=2)
```

### Export for CPA

```python
# Export in CPA format
cpa_data = genny.export_tax_data("cpa", 2024)

# Save to file
import json
with open("cpa_2024.json", "w") as f:
    json.dump(cpa_data, f, indent=2)
```

### Get NC Tax Advice

```python
# Ask Genny about NC tax law
advice = genny.get_nc_tax_advice("How are capital gains taxed in North Carolina?")

print(advice["nc_specific"])
```

### Set Cost Basis Method

```python
# Change cost basis method
genny.set_cost_basis_method("FIFO")  # or "LIFO", "HIFO", "AVERAGE"
```

## Command Line Interface

### Generate Tax Report

```bash
# Generate summary report
python3 scripts/genny_tax_report.py --year 2024

# Export for TurboTax
python3 scripts/genny_tax_report.py --year 2024 --format turbo_tax --output tax_reports/turbo_tax_2024.json

# Export for CPA
python3 scripts/genny_tax_report.py --year 2024 --format cpa --output tax_reports/cpa_2024.json

# Include unrealized gains
python3 scripts/genny_tax_report.py --year 2024 --include-unrealized

# Get NC tax advice
python3 scripts/genny_tax_report.py --nc-advice "How are crypto gains taxed in NC?"
```

## North Carolina Tax Law Knowledge

### Capital Gains
- **Rate**: 4.75% flat rate (same as income tax)
- **Treatment**: All capital gains treated as ordinary income
- **No Preferential Rate**: Long-term gains taxed same as short-term
- **Notes**: NC does NOT differentiate between short-term and long-term

### Business Expenses
- **Deductible**: Yes, if ordinary and necessary
- **Common Categories**:
  - Software subscriptions (trading platforms, data feeds)
  - Hardware (computers, servers)
  - Professional services (CPA, legal)
  - Office expenses
  - Internet and utilities (business portion)
  - Education and training

### Day Trading
- **Treatment**: May qualify as business income
- **Requirements**: Must meet IRS trader tax status
- **Benefits**: Business expense deductions available

### Cryptocurrency
- **Treatment**: Property (follows federal rules)
- **Capital Gains**: Apply to crypto sales
- **Mining**: Taxable as ordinary income
- **State**: NC follows federal treatment

### Filing Requirements
- **State Return**: Required
- **Due Date**: April 15
- **Extensions**: Available with federal extension
- **Estimated Payments**: Required if tax liability > $1000

## Data Storage

All tax data is stored in:
- `tools/data/genny/tax/trades.json` - All trade records
- `tools/data/genny/tax/positions.json` - Current positions
- `tools/data/genny/tax/expenses.json` - Expense records
- `tools/data/genny/tax/capital_gains.json` - Capital gains calculations

## Integration Points

### Optimus Integration
- Tracks every trade automatically
- Records symbol, quantity, price, fees
- Calculates cost basis and capital gains

### Shredder Integration
- Tracks crypto transactions
- Records Bitcoin purchases/sales
- Tracks profit allocations

### April Integration
- Tracks crypto conversions
- Records fiat-to-crypto transactions
- Tracks Ledger Live transactions

### Donnie & Mikey Integration
- Tracks fiat deposits/withdrawals
- Monitors account flows
- Records taxable income events

## Tax Report Formats

### Summary Format
- Human-readable summary
- Key metrics and totals
- NC compliance information
- Tax estimates

### TurboTax Format
- Schedule D (Capital Gains/Losses)
- Schedule C (Business Expenses)
- Ready for import

### CPA Format
- Professional format
- Detailed transactions
- Compliance notes
- NC-specific information

## Automated Features

### ✅ Automatic Trade Tracking
- All Optimus trades automatically recorded
- No manual intervention needed

### ✅ Automatic Expense Categorization
- Common expenses pre-categorized
- Business use percentage calculated

### ✅ Quarterly Summaries
- Auto-generated at end of each quarter
- Includes all transactions

### ✅ Annual Reports
- Comprehensive year-end reports
- Ready for tax filing

## Compliance Features

### ✅ Day Trade Tracking
- Automatically identifies day trades
- Tracks day trade count
- Compliance with PDT rules

### ✅ Holding Period Calculation
- Automatic short-term vs long-term classification
- Accurate holding period tracking

### ✅ Cost Basis Accuracy
- Multiple methods supported
- Accurate lot tracking
- FIFO/LIFO/HIFO/Average

### ✅ NC Tax Law Compliance
- Built-in NC tax knowledge
- Compliance notes included
- State-specific calculations

## Benefits

1. **Hands-Free Bookkeeping**: All transactions automatically tracked
2. **Accurate Accounting**: Precise cost basis and gain/loss calculations
3. **Tax Prep Ready**: Export formats for TurboTax and CPAs
4. **NC Compliance**: Built-in knowledge of NC tax laws
5. **Time Savings**: No manual data entry required
6. **Error Reduction**: Automated calculations reduce errors
7. **Audit Trail**: Complete transaction history
8. **Real-Time Tracking**: Up-to-date tax information

## Example Workflow

1. **Optimus executes trade** → Automatically tracked by Genny
2. **Shredder allocates crypto** → Crypto transaction tracked
3. **Expense incurred** → Recorded via expense tracking
4. **Quarter ends** → Genny generates quarterly summary
5. **Year ends** → Genny generates annual report
6. **Tax filing** → Export to TurboTax or provide to CPA

## Files Created

1. `agents/genny_tax_module.py` - Core tax module
2. `agents/genny_tax_integration.py` - Integration hooks
3. `scripts/genny_tax_report.py` - Report generator
4. `GENNY_TAX_SYSTEM.md` - This documentation

## Next Steps

1. **Start Tracking**: System automatically tracks all trades
2. **Record Expenses**: Use `record_deductible_expense()` for expenses
3. **Generate Reports**: Use tax report script quarterly/annually
4. **Export for Filing**: Export to TurboTax or provide to CPA

---

**Status**: ✅ Fully Operational
**Tax Tracking**: ✅ Automatic
**NC Compliance**: ✅ Built-in
**Export Formats**: ✅ TurboTax & CPA Ready

