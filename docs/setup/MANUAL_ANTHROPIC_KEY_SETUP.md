# NAE/MANUAL_ANTHROPIC_KEY_SETUP.md
"""
Manual Anthropic API Key Setup
Since the API only returns partial hints, you need to use the key you saved when created
"""

# 🔑 Manual Anthropic API Key Setup

## **Important Note**

The Anthropic API response shows:
- ✅ API Key ID: `apikey_01Rj2N8SVvo6BePZj99NhmiT`
- ✅ Partial Hint: `sk-ant-api03-R2D...igAA`
- ❌ Full Key: Not returned (security feature)

**Anthropic only shows the full API key value when it's first created.** After that, you can only see the partial hint.

---

## **Solution: Use Your Saved Key**

If you have the full API key saved somewhere (from when you created it), add it directly:

### **Option 1: Edit .env File**

```bash
# Edit .env file and add:
ANTHROPIC_API_KEY=sk-ant-api03-R2D...your-full-key-here...igAA
```

### **Option 2: Set Environment Variable**

```bash
export ANTHROPIC_API_KEY="sk-ant-api03-R2D...your-full-key-here...igAA"
```

### **Option 3: Add to Vault**

```python
python3 -c "
from secure_vault import get_vault
vault = get_vault()
vault.set_secret('anthropic', 'api_key', 'sk-ant-api03-R2D...your-full-key-here...igAA')
print('✅ Saved to vault')
"
```

---

## **If You Don't Have the Full Key**

If you lost the full key, you'll need to:

1. **Create a new API key** in Anthropic Console
2. **Save it immediately** when created (it's only shown once)
3. **Add it to NAE** using one of the methods above

---

## **Verify Setup**

After adding the key:

```bash
python3 -c "from env_loader import get_env_loader; loader = get_env_loader(); print(loader.status())"
```

Should show:
```
ANTHROPIC_API_KEY: ✅ Set
```

---

## **Key Information**

From your API response:
- **Key ID**: `apikey_01Rj2N8SVvo6BePZj99NhmiT`
- **Partial Hint**: `sk-ant-api03-R2D...igAA`
- **Status**: Active ✅
- **Name**: Developer Key
- **Created**: 2024-10-30

**The full key starts with `sk-ant-api03-R2D` and ends with `igAA`**

---

**Status**: ✅ **Ready to use**  
**Action**: Add your full API key (the one you saved when created) to .env file


