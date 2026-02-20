# Optimus Option Trade Trigger

## Overview

The `trigger_optimus_option_trade.py` script enables automatic triggering of Optimus to evaluate and execute intelligent option trades at three key times during the trading day:
1. **Market Open (9:30 AM - 10:30 AM ET)**: Morning momentum strategies
2. **Midday (12:00 PM - 1:30 PM ET)**: Reversal/continuation strategies  
3. **Market Close (3:00 PM - 4:00 PM ET)**: End-of-day premium capture

This script leverages Optimus's advanced option analysis capabilities including IV forecasting, volatility ensemble, and liquidity analysis. Strategies are automatically selected based on the current trading phase.

## Features

- **Three Trading Phases**: Automatically detects and adapts to market open, midday, or close phases
- **Phase-Aware Strategy Selection**: Different strategies for each trading phase
- **Market Hours Monitoring**: Checks current time and trading phase
- **Multi-Symbol Analysis**: Analyzes multiple symbols to find the best option opportunity
- **Intelligent Opportunity Scoring**: Uses IV edge, liquidity, and volatility signals to score opportunities
- **Automatic Execution**: Executes trades through Optimus with full safety checks
- **Sandbox/Live Support**: Works in both sandbox and live modes

## Usage

### Basic Usage (Sandbox Mode)
```bash
python scripts/trigger_optimus_option_trade.py --sandbox
```

### Live Mode
```bash
python scripts/trigger_optimus_option_trade.py --live
```

### Custom Symbols
```bash
python scripts/trigger_optimus_option_trade.py --symbols SPY QQQ AAPL --sandbox
```

## How It Works

1. **Market Phase Detection**: Detects current trading phase:
   - **Morning**: 9:30 AM - 10:30 AM ET
   - **Midday**: 12:00 PM - 1:30 PM ET
   - **Close**: 3:00 PM - 4:00 PM ET
2. **Opportunity Analysis**: For each symbol:
   - Generates option signals using `_generate_option_signals()`
   - Calculates IV edge (current IV vs forecasted IV)
   - Checks liquidity (bid/ask spread)
   - Evaluates volatility forecasts
   - Scores the opportunity
3. **Best Opportunity Selection**: Selects the symbol with the highest opportunity score
4. **Phase-Specific Strategy Creation**: Creates intelligent execution details based on:
   - **Current trading phase** (morning/midday/close)
   - IV edge (positive = sell options, negative = buy options)
   - Option type (call/put) - selected based on phase
   - Strike price - adjusted based on phase (ITM/ATM/OTM)
   - Strategy type - phase-appropriate (momentum, reversal, premium capture)
5. **Trade Execution**: Executes through Optimus's `execute_trade()` with full:
   - Risk controls
   - Pre-trade validation
   - Position sizing (Kelly Criterion)
   - Audit logging

## Opportunity Scoring

The script scores opportunities based on:

- **IV Edge**: Positive IV edge (IV > forecast) favors selling options
- **Liquidity**: Narrow bid/ask spreads (< 10%) indicate good liquidity
- **Volatility**: Higher volatility provides higher option premiums
- **Dispersion Signals**: For index products (SPY, SPX), dispersion analysis

## Strategy Selection by Phase

### Morning Phase (9:30 AM - 10:30 AM)
**Focus**: Momentum plays, trend continuation
- **Elevated IV (> 3%)**: Sell morning volatility premium (2% OTM calls)
- **Low IV (< -3%)**: Buy momentum calls (slightly ITM for direction)
- **Neutral**: Covered calls (ATM)

### Midday Phase (12:00 PM - 1:30 PM)
**Focus**: Reversal plays, range trading, premium collection
- **Strong IV Edge (> 5%)**: Sell premium (1% OTM calls) - best time for credit spreads
- **Low IV (< -5%)**: Buy low IV options for afternoon move (puts if low vol, calls if high vol)
- **Neutral**: Cash-secured puts (2% OTM)

### Close Phase (3:00 PM - 4:00 PM)
**Focus**: Premium capture, end-of-day strategies (very conservative)
- **Any Positive IV Edge**: Sell premium (1% OTM covered calls)
- **Low IV (< -3%)**: Buy puts as hedge or speculative play (1% OTM)
- **Default**: ATM covered call

## Example Output

```
================================================================================
🎯 TRIGGERING OPTIMUS OPTION TRADE BEFORE MARKET CLOSE
================================================================================
Market Status:
  Current Time: 15:45:00
  Market Open: 09:30
  Market Close: 16:00
  Is Open: True
  Minutes Until Close: 15

🤖 Initializing Optimus...
✅ Trading enabled

📊 Analyzing 9 symbols for option opportunities...
  Analyzing SPY...
    SPY: IV Edge = 0.0234
    SPY: Spread = 2.45%
  ✅ SPY is currently best opportunity (score: 24.34)

================================================================================
✅ BEST OPPORTUNITY FOUND: SPY
   Opportunity Score: 24.34
   IV Edge: 0.0234
================================================================================

📝 Creating execution details...
✅ Execution details created:
   Strategy: IV Edge Covered Call
   Symbol: SPY
   Option Type: call
   Side: sell_to_open
   Strike: 450
   Trust Score: 70

🚀 EXECUTING OPTION TRADE...
================================================================================
📊 TRADE EXECUTION RESULT:
================================================================================
Status: executed
Symbol: SPY
✅ TRADE EXECUTED SUCCESSFULLY!
Order ID: 12345678
Execution Price: 2.45
Quantity: 1
================================================================================
```

## Trading Windows & Timing

The script automatically detects and adapts to three trading windows, but **also allows continuous trading throughout market hours**:

### Phase-Specific Windows (Optimized Strategies)
- **Morning Window**: 9:30 AM - 10:30 AM ET
  - Best for: Momentum plays, trend continuation
  - Strategy: Morning momentum calls, volatility premium sales
  
- **Midday Window**: 12:00 PM - 1:30 PM ET
  - Best for: Reversal plays, premium collection
  - Strategy: Midday reversal plays, IV edge premium
  
- **Close Window**: 3:00 PM - 4:00 PM ET
  - Best for: Premium capture, end-of-day strategies
  - Strategy: End-of-day premium capture, conservative covered calls

### Continuous Trading (Throughout Market Hours)
- **Between Windows**: Trading is enabled throughout all market hours (9:30 AM - 4:00 PM ET)
  - Uses general/balanced strategies based on current market conditions
  - Scans for opportunities every 5 minutes (configurable)
  - Adapts strategy based on IV edge, volatility, and liquidity
  - Still respects phase-aware logic when in specific windows

### Integration with Main System
The main NAE system (`run_nae_full_system.py`) now includes continuous opportunity scanning:
- **Active Scanning**: Optimus continuously scans for opportunities every 5 minutes during market hours
- **Proactive Execution**: Executes trades when opportunities meet minimum score threshold (30+)
- **Phase Awareness**: Still uses phase-specific strategies during optimal windows
- **Multi-Source**: Combines continuous scanning with messages from Ralph/Donnie pipeline

### Market Closed
- Will still analyze opportunities but won't execute (wait for next open)

**You can run the script at any time during market hours** - it will automatically detect the current phase and select appropriate strategies, or use general strategies between windows.

## Safety Features

- Full Optimus risk controls
- Pre-trade validation
- Kill switch respect
- Position sizing (Kelly Criterion)
- Audit logging
- Sandbox mode testing

## Integration with NAE System

This script can be:
- Run manually when needed
- Scheduled via cron/task scheduler before market close
- Integrated into the main NAE orchestrator for automatic daily execution
- Called from monitoring systems when opportunities are detected

## Notes

- Requires yfinance for option chain data
- Works best with Tradier broker adapter (options support)
- May take 30-60 seconds to analyze all symbols
- Sandbox mode recommended for initial testing

