# Alpaca API Credentials - Secure Setup Complete ✅

## Overview

Your Alpaca API credentials have been securely stored and configured for use with NAE.

**Endpoint**: `https://paper-api.alpaca.markets/v2` (Paper Trading)

## Security Measures

### ✅ Credentials Stored Securely

1. **Secure Vault** (Encrypted) - `config/.vault.encrypted`
   - Credentials are encrypted using Fernet encryption
   - Protected by master key (stored in `config/.master.key`)
   - Both files are in `.gitignore` (not committed to version control)

2. **Config File** - `config/api_keys.json`
   - Backup storage location
   - Also in `.gitignore` (not committed)

3. **Broker Adapter Config** - `config/broker_adapters.json`
   - References vault (not actual keys)
   - Safe to commit

### 🔒 Protected Files

All sensitive files are protected by `.gitignore`:
- `config/.vault.encrypted` - Encrypted credentials
- `config/.master.key` - Encryption key
- `config/api_keys.json` - API keys (plain text backup)
- `.env` files - Environment variables

## Credential Loading Priority

The `AlpacaAdapter` checks credentials in this order:

1. **Config dict** (passed to constructor)
2. **Environment variables** (`APCA_API_KEY_ID`, `APCA_API_SECRET_KEY`)
3. **Secure vault** (`alpaca.api_key`, `alpaca.api_secret`)
4. **Config file** (`config/api_keys.json`)

This ensures flexibility while maintaining security.

## Usage

### Automatic (Recommended)

The adapter automatically loads credentials from the secure vault:

```python
from adapters.alpaca import AlpacaAdapter

# Credentials automatically loaded from vault
adapter = AlpacaAdapter({"paper_trading": True})

if adapter.auth():
    account = adapter.get_account()
    print(f"Cash: ${account['cash']:,.2f}")
```

### Manual (If Needed)

You can also set environment variables:

```bash
export APCA_API_KEY_ID='PKQIXYQPWDKTGGQG7PQZ36JWGF'
export APCA_API_SECRET_KEY='EMPH6gEs5tSinsfb1BjB8ZD3p1HSugPq69rZMzUt942P'
```

### Using Adapter Manager

```python
from adapters.manager import AdapterManager

manager = AdapterManager()
alpaca = manager.get("ibkr")  # or "alpaca"

# Credentials automatically loaded
if alpaca.auth():
    result = alpaca.place_order({
        "symbol": "AAPL",
        "quantity": 1.0,
        "side": "buy",
        "type": "market"
    })
```

## Verification

### Test Connection

Run the connection test script:

```bash
python3 scripts/test_alpaca_connection.py
```

### Verify Credentials in Vault

```python
from secure_vault import get_vault

vault = get_vault()
api_key = vault.get_secret("alpaca", "api_key")
api_secret = vault.get_secret("alpaca", "api_secret")

print(f"API Key: {api_key[:10]}...{api_key[-5:]}")
print(f"Secret: {api_secret[:5]}...{api_secret[-5:]}")
```

## Next Steps

1. **Install Alpaca SDK** (if not already installed):
   ```bash
   pip install alpaca-py
   ```

2. **Test Connection**:
   ```bash
   python3 scripts/test_alpaca_connection.py
   ```

3. **Start Trading**:
   ```python
   from adapters.alpaca import AlpacaAdapter
   
   alpaca = AlpacaAdapter({"paper_trading": True})
   if alpaca.auth():
       # Your trading code here
   ```

## Security Notes

⚠️ **Important Security Reminders**:

- ✅ Credentials are encrypted in the vault
- ✅ All sensitive files are in `.gitignore`
- ✅ Never commit credentials to version control
- ✅ Use paper trading for testing
- ✅ Rotate keys periodically
- ✅ Use environment variables in production (if preferred)

## Troubleshooting

### Credentials Not Found

If you get "API_KEY and API_SECRET are required":

1. Check vault: `python3 -c "from secure_vault import get_vault; v=get_vault(); print(v.get_secret('alpaca', 'api_key'))"`
2. Check environment variables: `echo $APCA_API_KEY_ID`
3. Check config file: `cat config/api_keys.json | grep alpaca`

### Connection Issues

- Verify endpoint: `https://paper-api.alpaca.markets/v2`
- Check network connectivity
- Verify API key permissions (paper trading enabled)
- Check Alpaca dashboard for account status

## Support

For issues:
- Check `logs/` directory for error logs
- Review `docs/ALPACA_SDK_INTEGRATION.md` for full documentation
- Run `python3 scripts/test_alpaca_connection.py` for diagnostics

---

**Status**: ✅ **Credentials Securely Stored and Configured**

**Last Updated**: 2025-01-27

