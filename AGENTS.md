# AGENTS.md

## Cursor Cloud specific instructions

### Project Overview

NAE (Neural Agency Engine) is an autonomous algorithmic trading system built in Python. The primary development codebase lives in `NAE Ready/`. All commands below should be run from that directory.

### Running Services

- **Redis** must be running before any agent or test. Start with: `redis-server --daemonize yes`
- Verify Redis: `redis-cli ping` (should return `PONG`)
- Kill switch health: `python3 redis_kill_switch.py --health`

### Key Commands

- **Tests**: `python3 -m pytest tests/ -v` (from `NAE Ready/`)
- **System test**: `python3 system_test.py` (from `NAE Ready/`)
- **Verify setup**: `python3 verify_setup.py` (from `NAE Ready/`)
- **Agent initialization**: Individual agents can be tested via `python3 agents/<agent>.py`

### Known Gotchas

- The `pyautogen` package (v0.9.0) imports as `autogen`, not `pyautogen`. The `verify_setup.py` script reports `pyautogen` as missing, but the actual code uses `import autogen` which works correctly.
- `ibapi>=10.0.0` is not installable on Python 3.12; the code handles this gracefully with try/except imports.
- `alpaca-trade-api` pins `websockets<11` which conflicts with `yfinance` and `polygon-api-client` requiring `websockets>=13`. Install `alpaca-trade-api` after other packages to let pip resolve the conflict (websockets will be downgraded to 10.x).
- The `config/api_keys.json` file does not ship with the repo. A placeholder file with `YOUR_*_HERE` values is needed for `verify_setup.py` and some tests to pass. The agents handle missing keys gracefully.
- `NAE_VAULT_PASSWORD` env var avoids interactive prompts from `secure_vault.py`. Set it to any string (e.g., `export NAE_VAULT_PASSWORD=dev`) for non-interactive use.
- Add `$HOME/.local/bin` to `PATH` to use pip-installed CLI tools (pytest, uvicorn, flask, etc.).
- Some tests (3 in `test_optimus_alpaca_paper.py`) error due to a missing `AlpacaClient` attribute in the optimus module — this is a pre-existing code issue, not a setup problem.
- Two tests fail because `GennyAgent.__init__` references an undefined `UniversalPortfolio` — also pre-existing.

### External API Keys

The system requires API keys for full functionality (trading, market data, LLM). Without them, agents initialize in degraded/demo mode. Key env vars: `TRADIER_API_KEY`, `TRADIER_ACCOUNT_ID`, `POLYGON_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`.
