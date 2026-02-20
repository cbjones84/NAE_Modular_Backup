# How to Get E*Trade Authorization URL

## Quick Answer

**Run this command to get the authorization URL:**

```bash
cd "/Users/melissabishop/Downloads/Neural Agency Engine/NAE"
python3 get_etrade_auth_url.py
```

This will:
- ✅ Get a fresh authorization URL
- ✅ Display it on screen
- ✅ Try to open it in your browser automatically
- ✅ Show you what to do next

## Alternative Methods

### Method 1: Quick Setup Script
```bash
python3 quick_setup_etrade_oauth.py
```
This interactive script walks you through the entire OAuth process.

### Method 2: Manual Python Script
```bash
python3 << 'EOF'
from agents.etrade_oauth import ETradeOAuth
from secure_vault import get_vault

vault = get_vault()
key = vault.get_secret('etrade', 'sandbox_api_key')
secret = vault.get_secret('etrade', 'sandbox_api_secret')

oauth = ETradeOAuth(key, secret, sandbox=True)
_, _, auth_url = oauth.get_request_token()
print(f"\nAuthorization URL:\n{auth_url}\n")
EOF
```

### Method 3: Check Previous Output
If you ran the setup script before, the authorization URL was displayed in the terminal output. Look for a line that starts with:
```
https://apisb.etrade.com/oauth/authorize?oauth_token=...
```

## What the Authorization URL Looks Like

```
https://apisb.etrade.com/oauth/authorize?oauth_token=ENCODED_TOKEN_HERE
```

Note: The token changes each time you request a new one, so you need a fresh URL each time.

## Using the Authorization URL

1. **Copy the URL** from the terminal output
2. **Paste it into your web browser**
3. **Log in** to your E*Trade account
4. **Authorize** the application
5. **Copy the verification code** (usually 8 digits)
6. **Complete OAuth** by running:
   ```bash
   python3 complete_etrade_oauth.py YOUR_VERIFICATION_CODE
   ```

## Important Notes

- ⚠️ **Each URL is unique** - Request tokens expire, so get a fresh URL if needed
- ⚠️ **URLs are single-use** - After you authorize, you'll need a new URL if you need to re-authorize
- ✅ **URLs are safe** - They're just authorization requests, not access tokens

## Current Status

**Issue:** The authorization URL is currently returning Error 999

**Possible Solutions:**
1. Check E*Trade Developer Portal: https://developer.etrade.com/
2. Verify API key status
3. Try the URL again
4. Contact E*Trade support

## Quick Reference

```bash
# Get authorization URL
python3 get_etrade_auth_url.py

# Complete OAuth (after authorization)
python3 complete_etrade_oauth.py YOUR_CODE

# Check status
python3 check_etrade_sandbox_readiness.py
```

