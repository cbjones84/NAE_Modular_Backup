# E*TRADE Sandbox OAuth - Step-by-Step Guide

## 📋 Complete Walkthrough for First-Time Users

---

## **Step 1: Generate Authorization URL**

Run this command to start the OAuth process:

```bash
cd "/Users/melissabishop/Downloads/Neural Agency Engine/NAE"
python3 scripts/quick_complete_etrade_oauth.py
```

**What happens:**
- The script generates an authorization URL
- Request tokens are saved to `config/etrade_oauth_temp.json`
- You'll see a URL that looks like: `https://apisb.etrade.com/oauth/authorize?oauth_token=...`

---

## **Step 2: Open Authorization URL in Browser**

**Copy the URL** that was displayed and **open it in your web browser**.

**What you'll see:**

### **A) E*TRADE Login Page** (if not logged in)
- This looks like a normal E*TRADE login page
- You need to log in with your **E*TRADE sandbox account credentials**

**⚠️ Important:** You need an E*TRADE sandbox account. If you don't have one:
1. Go to: https://developer.etrade.com/home
2. Sign up for a developer account
3. Create a sandbox account (separate from production)

### **B) Authorization Page** (after logging in)
- You'll see a page asking you to authorize the application
- It will show your app name and what permissions it's requesting
- **Click "Accept" or "Authorize"** button

---

## **Step 3: Get Verification Code**

**After clicking "Accept/Authorize", you'll see:**

### **What to Look For:**
- A **large box or highlighted area** with a code
- The code might be labeled as:
  - "Verification Code"
  - "Verifier Code"
  - "Authorization Code"
  - Or just shown as numbers/letters

### **Examples of what the code looks like:**
- `123456`
- `ABC123`
- `9876543210`
- `XYZ789`

### **Where it appears:**
- Usually in the center of the page
- May be in a box or highlighted section
- Sometimes below text that says "Your verification code is:"

**⚠️ Important:** This code expires quickly (usually within a few minutes), so copy it right away!

---

## **Step 4: Complete OAuth with Verification Code**

Once you have the verification code, run:

```bash
python3 scripts/finish_etrade_oauth.py YOUR_VERIFICATION_CODE
```

**Example:**
```bash
python3 scripts/finish_etrade_oauth.py 123456
```

**What happens:**
- The script exchanges your verification code for access tokens
- Saves tokens to `config/etrade_tokens_sandbox.json`
- Tests the API connection
- Shows your E*TRADE sandbox accounts if successful

---

## **Step 5: Test the Adapter**

After OAuth is complete, test everything:

```bash
python3 scripts/test_etrade_adapter.py
```

This will test:
- ✅ Authentication
- ✅ Account information
- ✅ Positions
- ✅ Market quotes
- ✅ Full adapter functionality

---

## 🆘 **Troubleshooting**

### **Problem: "Verification code expired"**
**Solution:** Start over - run Step 1 again to get a new authorization URL

### **Problem: "Invalid verification code"**
**Solution:** 
- Double-check you copied the entire code
- Make sure there are no extra spaces
- Try getting a new code by starting over

### **Problem: "Can't log in to E*TRADE"**
**Solution:**
- Make sure you're using **sandbox** credentials (not production)
- You may need to create a sandbox account at: https://developer.etrade.com/home
- Sandbox and production accounts are separate

### **Problem: "Authorization URL doesn't work"**
**Solution:**
- The URL expires after a few minutes
- Generate a new one by running Step 1 again
- Make sure you're using the sandbox URL (apisb.etrade.com)

---

## 📸 **Visual Guide**

```
Step 1: Generate URL
┌─────────────────────────────────────┐
│ python3 scripts/quick_complete_... │
│ ✅ URL: https://apisb.etrade.com... │
└─────────────────────────────────────┘
                    ↓
Step 2: Open in Browser
┌─────────────────────────────────────┐
│ [E*TRADE Login Page]                │
│ Username: _____________             │
│ Password: _____________             │
│ [Login]                             │
└─────────────────────────────────────┘
                    ↓
Step 3: Authorize
┌─────────────────────────────────────┐
│ [Authorization Page]                │
│ Authorize NAE Trading Bot?         │
│ [Accept] [Cancel]                   │
└─────────────────────────────────────┘
                    ↓
Step 4: Get Code
┌─────────────────────────────────────┐
│ Verification Code:                  │
│ ┌─────────┐                         │
│ │ 123456  │  ← Copy this!           │
│ └─────────┘                         │
└─────────────────────────────────────┘
                    ↓
Step 5: Complete OAuth
┌─────────────────────────────────────┐
│ python3 finish_etrade_oauth.py 123456│
│ ✅ OAuth Complete!                   │
└─────────────────────────────────────┘
```

---

## ✅ **Success Indicators**

You'll know it worked when you see:
- ✅ "Access token obtained!"
- ✅ "Tokens saved to: config/etrade_tokens_sandbox.json"
- ✅ "API connection successful!"
- ✅ Your account ID(s) displayed

---

**That's it! Once complete, you can use the E*TRADE sandbox with NAE!** 🎉


