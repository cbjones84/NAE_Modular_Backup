# Kalshi External References

This directory contains local clones of official Kalshi repositories for reference purposes.
These are **not** included in the NAE repository to avoid nested git issues.

## How to Set Up (Development Only)

If you need to reference the official Kalshi code locally:

```bash
cd "NAE Ready/external/kalshi"
git clone https://github.com/Kalshi/tools-and-analysis.git
```

## Official Resources

1. **kalshi-python SDK** - Official Python SDK
   - PyPI: https://pypi.org/project/kalshi-python/
   - Install: `pip install kalshi-python`
   - Documentation: https://docs.kalshi.com/sdks/overview

2. **tools-and-analysis** - Community tools and examples
   - Repository: https://github.com/Kalshi/tools-and-analysis
   - Contains example analyses and client code

3. **API Documentation**
   - Main docs: https://docs.kalshi.com
   - Getting started: https://docs.kalshi.com/getting_started
   - API reference: https://docs.kalshi.com/api-reference

## NAE Integration

The NAE Kalshi integration (`adapters/kalshi.py` and `agents/kalshi_trader.py`) 
provides:

### Adapter Features
- RSA-PSS authentication (official Kalshi auth method)
- Full market discovery via API
- Order placement (limit and market orders)
- Position and balance tracking
- Bonding opportunity detection
- Cross-platform arbitrage with Polymarket

### Trader Features
- High-probability bonding strategy
- Cross-platform arbitrage (Kalshi ↔ Polymarket)
- LLM-based superforecasting
- Category-specific analysis (economics, politics, weather, etc.)
- Kelly criterion position sizing
- One best trade methodology

## Authentication Setup

1. **Create API Key on Kalshi**:
   - Log in to Kalshi
   - Go to Account Settings → API Keys
   - Generate new RSA key pair
   - Save the private key securely (it won't be shown again!)

2. **Configure NAE**:
   ```bash
   # Set environment variables
   export KALSHI_API_KEY_ID="your-key-id"
   export KALSHI_PRIVATE_KEY="-----BEGIN RSA PRIVATE KEY-----
   ... your private key content ...
   -----END RSA PRIVATE KEY-----"
   ```

   Or update `config/kalshi_config.json` with your credentials.

## Required Dependencies

```bash
pip install kalshi-python cryptography
```

Or use the requirements.txt:

```bash
pip install -r requirements.txt
```

## Demo Environment

For testing without real money:
- Use `demo=True` when initializing the adapter/trader
- Demo API: https://demo-api.kalshi.co/trade-api/v2
- Create demo account at: https://demo.kalshi.com

## Regulatory Information

Kalshi is the FIRST federally regulated exchange for event contracts in the US:

- **Regulator**: CFTC (Commodity Futures Trading Commission)
- **Type**: Designated Contract Market (DCM)
- **Tax Reporting**: 1099 forms issued for US users
- **Funds Custody**: Held at FDIC-insured banks
- **Legal Status**: Legal in most US states (some restrictions apply)

This provides significant advantages over unregulated prediction markets:
- Legal certainty for US traders
- Proper tax documentation
- FDIC-insured fund custody
- Regulatory oversight and consumer protection

## Categories

Kalshi offers markets in several categories:
- **Economics**: Fed rates, inflation, GDP, unemployment
- **Politics**: Elections, policy outcomes, government actions
- **Weather**: Temperature records, hurricanes, climate events
- **Finance**: Stock milestones, crypto prices
- **Science**: Discoveries, space events
- **Entertainment**: Awards, streaming records

