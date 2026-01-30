# Alpaca Authentication Troubleshooting

## Current Status

**Error:** 401 Unauthorized
**Status:** Authentication failing for both LIVE and PAPER modes

## Diagnostic Results

✅ Keys are correctly formatted:
- API Key: `AKPNTSZTJRSP7Y5DMPKIX7VK6R` (Live key format - AK prefix)
- API Secret: `72pWcuJpA8N8h5RQ5AtEFxe5UVbr2G`
- Endpoint: `https://api.alpaca.markets`

❌ Authentication failing with 401 error

## Common Causes & Solutions

### 1. API Access Not Enabled
**Check in Alpaca Dashboard:**
- Log into https://app.alpaca.markets/
- Go to **API Keys** section
- Find your API key
- Verify **"API Access"** toggle is **ON**
- Verify **"Trading"** permissions are enabled

**Solution:** Enable API access in dashboard

### 2. Keys Copied Incorrectly
**Check:**
- No extra spaces before/after keys
- No hidden characters
- Keys match exactly what's shown in dashboard

**Solution:** Re-copy keys directly from dashboard

### 3. Account Not Fully Activated
**Check:**
- Account is fully verified
- All required documents submitted
- Account status is "Active"
- No restrictions on account

**Solution:** Complete account activation in Alpaca dashboard

### 4. Keys Are for Different Environment
**Check:**
- Keys shown in dashboard match what you copied
- You're using the correct set of keys (not mixed paper/live)
- Keys haven't been regenerated since copying

**Solution:** Verify keys in dashboard match what's in config

### 5. IP Restrictions
**Check:**
- API keys don't have IP whitelist restrictions
- Your current IP is allowed

**Solution:** Remove IP restrictions or add your IP to whitelist

## Verification Steps

### Step 1: Verify Keys in Dashboard
1. Log into https://app.alpaca.markets/
2. Navigate to **API Keys**
3. Find the key starting with `AKPNTSZTJRSP7Y5DMPKI...`
4. Verify:
   - ✅ Key matches exactly
   - ✅ API Access is enabled
   - ✅ Trading permissions enabled
   - ✅ No IP restrictions

### Step 2: Test Keys Manually
Try using Alpaca's API directly:

```bash
curl -X GET "https://api.alpaca.markets/v2/account" \
  -H "APCA-API-KEY-ID: AKPNTSZTJRSP7Y5DMPKIX7VK6R" \
  -H "APCA-API-SECRET-KEY: 72pWcuJpA8N8h5RQ5AtEFxe5UVbr2G"
```

If this works, the keys are correct and the issue is in our adapter.

### Step 3: Check Account Status
In Alpaca dashboard:
- Go to **Account** section
- Verify account is "Active"
- Check for any restrictions or blocks
- Verify account is funded (if required)

### Step 4: Regenerate Keys (If Needed)
If keys still don't work:
1. In Alpaca dashboard, delete current keys
2. Generate new API keys
3. Ensure API Access is enabled immediately
4. Update `config/api_keys.json` with new keys
5. Test again

## Quick Fix Checklist

- [ ] API Access enabled in dashboard
- [ ] Trading permissions enabled
- [ ] Keys copied correctly (no spaces)
- [ ] Account is fully activated
- [ ] No IP restrictions on keys
- [ ] Keys match dashboard exactly
- [ ] Account status is "Active"

## Next Steps

1. **Verify in Dashboard:** Check all settings above
2. **Test Manually:** Use curl command to test keys directly
3. **Regenerate if Needed:** Create new keys if current ones don't work
4. **Update Config:** Once keys work, update `config/api_keys.json`
5. **Test Again:** Run `python3 verify_alpaca_keys.py`

## Alternative: Use Paper Trading First

If live keys continue to fail, you can:
1. Generate PAPER trading keys (PK prefix)
2. Test with paper trading first
3. Once paper works, switch to live keys

This helps isolate if the issue is with:
- The keys themselves
- The account setup
- The API access permissions

---

**Status:** Authentication failing - requires dashboard verification
**Action Required:** Check Alpaca dashboard settings and verify API access is enabled

