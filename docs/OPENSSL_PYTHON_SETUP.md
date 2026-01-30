# OpenSSL Python Setup - Complete Guide

## ✅ **Updated: All Scripts Now Use OpenSSL**

All NAE scripts have been updated to use Python with OpenSSL (not LibreSSL) to avoid TLS handshake errors with APIs that have strict TLS settings.

---

## 🔧 **What Was Updated**

### **1. Python Wrapper Script**
Created `scripts/python_openssl.sh` - automatically finds and uses Python with OpenSSL

### **2. Monitor Scripts**
- `scripts/start_etrade_monitor.sh` - Starts monitor with OpenSSL Python
- `scripts/stop_etrade_monitor.sh` - Updated to handle OpenSSL Python processes
- `scripts/check_etrade_monitor.sh` - Checks monitor status

### **3. Helper Scripts**
- `scripts/get_python_openssl.py` - Finds Python with OpenSSL
- `scripts/find_python_openssl.sh` - Bash version of finder

---

## 🚀 **How It Works**

The wrapper script (`python_openssl.sh`) automatically:
1. Searches for Python with OpenSSL in common locations
2. Checks each Python to verify it uses OpenSSL
3. Falls back to regular python3 if none found

**Usage:**
```bash
# Instead of: python3 script.py
# Use: bash scripts/python_openssl.sh script.py

# Or make it executable and use directly:
./scripts/python_openssl.sh script.py
```

---

## 📋 **Updated Commands**

### **Start Monitor (with OpenSSL)**
```bash
bash scripts/start_etrade_monitor.sh
```

### **Stop Monitor**
```bash
bash scripts/stop_etrade_monitor.sh
```

### **Check Monitor Status**
```bash
bash scripts/check_etrade_monitor.sh
```

### **Run Any Script with OpenSSL Python**
```bash
bash scripts/python_openssl.sh scripts/your_script.py
```

---

## ✅ **Verifying OpenSSL**

Check if a Python uses OpenSSL:
```bash
# Method 1: Using wrapper
bash scripts/python_openssl.sh -c "import ssl; print(ssl.OPENSSL_VERSION)"

# Method 2: Direct check
python3 -c "import ssl; print(ssl.OPENSSL_VERSION)"
# Should show: OpenSSL X.X.X (not LibreSSL)
```

---

## 🔍 **Finding Python with OpenSSL**

### **Manual Check**
```bash
bash scripts/find_python_openssl.sh
```

### **Python Finder**
```bash
python3 scripts/get_python_openssl.py
```

---

## 📦 **Installing Python with OpenSSL**

If you don't have Python with OpenSSL:

```bash
# Install Python 3.11 (recommended)
brew install python@3.11

# Or Python 3.12
brew install python@3.12

# Verify it uses OpenSSL
/opt/homebrew/bin/python3.11 -c "import ssl; print(ssl.OPENSSL_VERSION)"
# Should show: OpenSSL X.X.X
```

---

## 🎯 **Why This Matters**

APIs with strict TLS settings (like many financial APIs) require:
- ✅ OpenSSL (fully compatible)
- ❌ LibreSSL (may cause handshake errors)

**Common errors with LibreSSL:**
- `SSL: CERTIFICATE_VERIFY_FAILED`
- `SSLError: [SSL: TLSV1_ALERT_PROTOCOL_VERSION]`
- Handshake failures with strict TLS APIs

**With OpenSSL:**
- ✅ Full TLS 1.3 support
- ✅ Better certificate validation
- ✅ Compatible with all financial APIs

---

## 📝 **Script Updates**

All scripts that run Python now:
1. Use the `python_openssl.sh` wrapper when started via scripts
2. Check for OpenSSL Python automatically
3. Warn if LibreSSL is detected

**Scripts Updated:**
- ✅ `scripts/start_etrade_monitor.sh`
- ✅ `scripts/monitor_etrade_status.py` (uses wrapper when started via script)
- ✅ All E*TRADE OAuth scripts (can use wrapper)

---

## 🔄 **Migration**

If you have scripts running with LibreSSL Python:

1. **Stop current processes**:
   ```bash
   bash scripts/stop_etrade_monitor.sh
   ```

2. **Start with OpenSSL Python**:
   ```bash
   bash scripts/start_etrade_monitor.sh
   ```

3. **Verify**:
   ```bash
   bash scripts/check_etrade_monitor.sh
   ```

---

## 💡 **Quick Reference**

```bash
# Check current Python SSL
python3 -c "import ssl; print(ssl.OPENSSL_VERSION)"

# Find OpenSSL Python
bash scripts/find_python_openssl.sh

# Start monitor (auto-uses OpenSSL)
bash scripts/start_etrade_monitor.sh

# Run any script with OpenSSL
bash scripts/python_openssl.sh your_script.py
```

---

**All scripts now use OpenSSL Python to avoid TLS handshake errors!** ✅


