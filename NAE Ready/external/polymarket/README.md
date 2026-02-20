# Polymarket External References

This directory contains local clones of official Polymarket repositories for reference purposes.
These are **not** included in the NAE repository to avoid nested git issues.

## How to Set Up (Development Only)

If you need to reference the official Polymarket code locally:

```bash
cd "NAE Ready/external/polymarket"
git clone https://github.com/Polymarket/py-clob-client.git
git clone https://github.com/Polymarket/agents.git
```

## Official Repositories

1. **py-clob-client** - Official Python SDK for Polymarket CLOB
   - Repository: https://github.com/Polymarket/py-clob-client
   - Install: `pip install py-clob-client`

2. **agents** - AI agents framework for Polymarket
   - Repository: https://github.com/Polymarket/agents
   - Contains LLM-based trading strategies and utilities

## NAE Integration

The NAE Polymarket integration (`adapters/polymarket.py` and `agents/polymarket_trader.py`) 
has been **enhanced using patterns from these official repositories**:

- CLOB client initialization and authentication levels
- Gamma API for market discovery
- Web3 wallet integration
- LLM-based superforecasting methodology
- Optimal trade selection algorithms

## Required Dependencies

Install the Polymarket dependencies:

```bash
pip install py-clob-client web3 httpx langchain langchain-openai
```

Or use the requirements.txt:

```bash
pip install -r requirements.txt
```

