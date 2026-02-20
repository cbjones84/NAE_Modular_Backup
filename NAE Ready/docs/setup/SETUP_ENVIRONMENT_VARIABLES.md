# NAE/SETUP_ENVIRONMENT_VARIABLES.md
"""
Environment Variables Setup Guide
"""

# 🔧 Environment Variables Setup Guide

## Required Environment Variables

### 1. **OPENAI_API_KEY**
**Purpose**: GPT-4 Turbo models for most agents  
**Required For**: Casey, Donnie, Optimus, Bebop, Phisher, and others

**Setup:**
```bash
export OPENAI_API_KEY="sk-your-openai-api-key-here"
```

**Get Key**: https://platform.openai.com/api-keys

---

### 2. **ANTHROPIC_API_KEY**
**Purpose**: Claude Sonnet 4.5 models for complex reasoning  
**Required For**: Ralph, Splinter, Genny, Leo

**Setup:**
```bash
export ANTHROPIC_API_KEY="sk-ant-your-anthropic-api-key-here"
```

**Get Key**: https://console.anthropic.com/

---

### 3. **NAE_ENVIRONMENT**
**Purpose**: Sets current environment profile  
**Options**: `sandbox`, `paper`, `live`, `test`  
**Default**: `sandbox`

**Setup:**
```bash
export NAE_ENVIRONMENT="sandbox"  # or paper, live, test
```

---

### 4. **NAE_VAULT_PASSWORD** (Optional)
**Purpose**: Vault master password (if not using default)  
**Optional**: If not set, uses default password

**Setup:**
```bash
export NAE_VAULT_PASSWORD="your-secure-vault-password"
```

---

## Quick Setup Script

Create a `.env` file in the NAE directory:

```bash
cd "/Users/melissabishop/Downloads/Neural Agency Engine/NAE"
cat > .env << 'EOF'
# NAE Environment Configuration
NAE_ENVIRONMENT=sandbox

# LLM API Keys
OPENAI_API_KEY=your-openai-key-here
ANTHROPIC_API_KEY=your-anthropic-key-here

# Optional: Vault Password
NAE_VAULT_PASSWORD=your-vault-password
EOF
```

Then load it:
```bash
export $(cat .env | xargs)
```

---

## Permanent Setup (Mac/Linux)

Add to your `~/.bashrc` or `~/.zshrc`:

```bash
# NAE Environment Variables
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export NAE_ENVIRONMENT="sandbox"
```

Then reload:
```bash
source ~/.bashrc  # or source ~/.zshrc
```

---

## Verification

Test that variables are set:
```bash
echo $OPENAI_API_KEY
echo $ANTHROPIC_API_KEY
echo $NAE_ENVIRONMENT
```

Or test in Python:
```python
import os
print("OPENAI_API_KEY:", "✅ Set" if os.getenv("OPENAI_API_KEY") else "❌ Not set")
print("ANTHROPIC_API_KEY:", "✅ Set" if os.getenv("ANTHROPIC_API_KEY") else "❌ Not set")
print("NAE_ENVIRONMENT:", os.getenv("NAE_ENVIRONMENT", "sandbox"))
```


