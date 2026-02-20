# NAE/QUICK_ENV_SETUP.md
"""
Quick Environment Variables Setup Guide
"""

# 🔧 Quick Environment Variables Setup

## **Automatic Setup (Easiest)**

The system now **automatically finds API keys** from multiple sources. Just add your keys to any of these:

### **Option 1: Edit .env File** (Recommended)

1. Open `.env` file in NAE directory
2. Replace placeholders:
   ```
   OPENAI_API_KEY=sk-your-actual-openai-key
   ANTHROPIC_API_KEY=sk-ant-your-actual-anthropic-key
   ```
3. Save the file
4. Load it: `source export_env.sh`

### **Option 2: Set Environment Variables**

```bash
export OPENAI_API_KEY="sk-your-openai-key"
export ANTHROPIC_API_KEY="sk-ant-your-anthropic-key"
```

### **Option 3: Add to Secure Vault**

```python
python3 -c "
from secure_vault import get_vault
vault = get_vault()
vault.set_secret('openai', 'api_key', 'your-openai-key')
vault.set_secret('anthropic', 'api_key', 'your-anthropic-key')
print('✅ Keys added to vault')
"
```

---

## **Get Your API Keys**

### **OpenAI API Key:**
1. Go to: https://platform.openai.com/api-keys
2. Sign in or create account
3. Click "Create new secret key"
4. Copy the key (starts with `sk-`)

### **Anthropic API Key:**
1. Go to: https://console.anthropic.com/
2. Sign in or create account
3. Navigate to API Keys section
4. Create new key
5. Copy the key (starts with `sk-ant-`)

---

## **Verify Setup**

After adding keys, verify:

```bash
python3 -c "from env_loader import get_env_loader; loader = get_env_loader(); print(loader.status())"
```

Should show:
```
OPENAI_API_KEY: ✅ Set
ANTHROPIC_API_KEY: ✅ Set
```

---

## **Auto-Loading**

The system automatically checks:
1. Environment variables (current shell)
2. `.env` file
3. Secure vault
4. Config files (backwards compatibility)

**Once you add keys anywhere, the system finds them automatically!**

---

**That's it!** The system handles the rest automatically.


