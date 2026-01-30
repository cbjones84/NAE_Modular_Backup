# NAE/ANTHROPIC_KEY_SETUP.md
"""
Anthropic API Key Auto-Retrieval Setup
"""

# 🔑 Anthropic API Key Auto-Retrieval

## **Quick Setup**

### **Step 1: Set Admin Key**
```bash
export ANTHROPIC_ADMIN_KEY="your-anthropic-admin-key"
```

### **Step 2: Run Retrieval Script**
```bash
python3 retrieve_anthropic_key.py
```

The script will:
1. ✅ Retrieve API key from Anthropic API
2. ✅ Save to `.env` file
3. ✅ Save to secure vault
4. ✅ Set in current environment

---

## **What the Script Does**

The `retrieve_anthropic_key.py` script:

1. **Retrieves API Key**: Uses your admin key to fetch the API key from Anthropic
2. **Saves to .env**: Automatically updates `.env` file
3. **Saves to Vault**: Stores in encrypted vault for security
4. **Sets Environment**: Makes it available immediately

---

## **Manual Method** (Alternative)

If you prefer to do it manually:

```bash
# Set admin key
export ANTHROPIC_ADMIN_KEY="your-admin-key"

# Retrieve API key
curl "https://api.anthropic.com/v1/organizations/api_keys/apikey_01Rj2N8SVvo6BePZj99NhmiT" \
  --header "anthropic-version: 2023-06-01" \
  --header "content-type: application/json" \
  --header "x-api-key: $ANTHROPIC_ADMIN_KEY"

# Then add to .env file:
# ANTHROPIC_API_KEY=retrieved-key-here
```

---

## **Verify**

After running the script:

```bash
python3 -c "from env_loader import get_env_loader; loader = get_env_loader(); print(loader.status())"
```

Should show:
```
ANTHROPIC_API_KEY: ✅ Set
```

---

**Status**: ✅ **Auto-retrieval script ready**  
**Next**: Set `ANTHROPIC_ADMIN_KEY` and run the script


