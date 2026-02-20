# NAE Tradier Configuration

## ✅ Configuration Complete

Your Tradier credentials have been configured:

- **API Key**: `27Ymk28vtbgqY1LFYxhzaEmIuwJb`
- **Account ID**: `6YB66744`
- **Environment**: **PRODUCTION** ⚠️ (Live Trading Account)

## Quick Start

### Option 1: Use Setup Script (Recommended)

```bash
source NAE/agents/setup_tradier_env.sh
```

### Option 2: Manual Export

```bash
export TRADIER_API_KEY=27Ymk28vtbgqY1LFYxhzaEmIuwJb
export TRADIER_ACCOUNT_ID=6YB66744
export TRADIER_SANDBOX=false  # PRODUCTION account - live trading enabled
```

## Make Configuration Permanent

Add these lines to your shell profile (`~/.bashrc` or `~/.zshrc`):

```bash
# NAE Tradier Configuration (PRODUCTION)
export TRADIER_API_KEY=27Ymk28vtbgqY1LFYxhzaEmIuwJb
export TRADIER_ACCOUNT_ID=6YB66744
export TRADIER_SANDBOX=false
```

Then reload your shell:
```bash
source ~/.bashrc  # or source ~/.zshrc
```

## Current Account Status

✅ **Account Verified**: `6YB66744`
- **Type**: Cash Account
- **Status**: Active
- **Option Level**: 2
- **Current Equity**: $202.64
- **Cash Available**: $155.60
- **Current Position**: 1 share of SOXL

⚠️ **This is a PRODUCTION account** - All trades will be real!

## Verify Configuration

Run the test script:
```bash
python3 NAE/agents/fetch_tradier_account_id.py
```

Or test directly:
```python
from NAE.agents.ralph_github_continuous import TradierClient

client = TradierClient()
balances = client.get_balances()
print(f"Equity: ${balances['equity']:,.2f}")
```

## Next Steps

1. ✅ Environment variables configured
2. ✅ Ready to start trading
3. Run NAE trading system:
   ```bash
   python3 NAE/agents/ralph_github_continuous.py
   ```

## Security Notes

- Never commit API keys to version control
- Keep your API key secure
- Use sandbox mode for testing
- Review all trades before switching to live mode

