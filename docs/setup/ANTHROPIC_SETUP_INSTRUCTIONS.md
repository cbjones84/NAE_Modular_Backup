# NAE/ANTHROPIC_SETUP_INSTRUCTIONS.md
"""
Complete Anthropic API Key Setup Instructions
"""

# 🔑 Complete Anthropic API Key Setup

## **What I've Created**

I've created an automated script (`retrieve_anthropic_key.py`) that:
- ✅ Uses your curl command to retrieve the API key
- ✅ Automatically saves it to `.env` file
- ✅ Saves it to secure vault
- ✅ Sets it in environment
- ✅ Verifies it's working

---

## **How to Use**

### **Option 1: Automated (Recommended)**

1. **Set your admin key:**
   ```bash
   export ANTHROPIC_ADMIN_KEY="your-anthropic-admin-key"
   ```

2. **Run the script:**
   ```bash
   python3 retrieve_anthropic_key.py
   ```

That's it! The script handles everything automatically.

---

### **Option 2: Manual Curl (If You Prefer)**

If you want to run the curl command yourself:

```bash
# Set admin key
export ANTHROPIC_ADMIN_KEY="your-admin-key"

# Run curl command
curl "https://api.anthropic.com/v1/organizations/api_keys/apikey_01Rj2N8SVvo6BePZj99NhmiT" \
  --header "anthropic-version: 2023-06-01" \
  --header "content-type: application/json" \
  --header "x-api-key: $ANTHROPIC_ADMIN_KEY"

# Copy the API key from the response, then add to .env:
# ANTHROPIC_API_KEY=your-retrieved-key
```

---

## **What Happens**

When you run `retrieve_anthropic_key.py`:

1. ✅ Executes your curl command
2. ✅ Parses the JSON response
3. ✅ Extracts the API key
4. ✅ Saves to `.env` file (automatically updates)
5. ✅ Saves to secure vault (encrypted)
6. ✅ Sets in current environment
7. ✅ Verifies it's working

---

## **After Setup**

Once the key is retrieved:

- ✅ Auto-loader will find it automatically
- ✅ Model config will use it automatically
- ✅ All Anthropic agents (Ralph, Splinter, Genny, Leo) will work
- ✅ No manual configuration needed

---

## **Verify**

Check if it worked:

```bash
python3 -c "from env_loader import get_env_loader; loader = get_env_loader(); print(loader.status())"
```

Should show:
```
ANTHROPIC_API_KEY: ✅ Set
```

---

**Status**: ✅ **Ready to use**  
**Next Step**: Set `ANTHROPIC_ADMIN_KEY` and run `python3 retrieve_anthropic_key.py`


