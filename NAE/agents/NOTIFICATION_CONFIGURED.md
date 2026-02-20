# ✅ NAE Notification Service - Configured

## Configuration Complete

The notification service has been configured with your Yahoo App Password.

---

## ✅ Current Configuration

- **Email Recipient**: cbjones84@yahoo.com
- **SMTP Server**: smtp.mail.yahoo.com
- **SMTP Port**: 587 (TLS)
- **App Name**: NAE
- **App Password**: Configured ✅

---

## 📧 Email Setup

The notification service will automatically send emails to **cbjones84@yahoo.com** when:

1. **Circuit Breaker Triggers** (Critical)
   - 10+ consecutive API errors
   - 50% drawdown exceeded
   - Trading automatically paused

2. **Daily Loss Limits** (Critical)
   - 35% daily loss threshold reached
   - Trading paused for the day

3. **Trading Pauses** (Critical)
   - Any automatic trading pause event

---

## 🚀 How to Use

### Option 1: Use the Setup Script (Recommended)

```bash
cd "/Users/melissabishop/Downloads/Neural Agency Engine/NAE/agents"
source setup_notifications.sh
```

### Option 2: Load from .env.notifications File

The system will automatically load credentials from `.env.notifications` file if it exists.

### Option 3: Set Environment Variables Manually

```bash
export NOTIFICATION_EMAIL_ENABLED=true
export NOTIFICATION_EMAIL_TO=cbjones84@yahoo.com
export NOTIFICATION_SMTP_USERNAME=cbjones84@yahoo.com
export NOTIFICATION_SMTP_PASSWORD=lbtfwssbsieswkft
```

---

## 🧪 Test the Notification Service

To test if emails are working:

```python
from ralph_github_continuous import NotificationService

service = NotificationService()
service.send(
    title="NAE Test Notification",
    message="This is a test message. If you receive this, the notification service is working correctly!",
    priority="critical"
)
```

---

## 📋 Email Format

You'll receive emails with:
- **Subject**: `[NAE CRITICAL] Circuit Breaker Triggered`
- **HTML Formatting**: Color-coded by priority
- **Details**: Full alert message with timestamp
- **Priority Level**: Critical, High, or Normal

---

## 🔒 Security Notes

1. **App Password**: The password `lbtfwssbsieswkft` is stored in `.env.notifications`
2. **Do NOT commit** `.env.notifications` to version control
3. **App Password** is safer than regular password - can be revoked independently
4. **TLS Encryption**: All emails sent over encrypted connection

---

## ✅ Status

- ✅ Email notifications configured
- ✅ SMTP credentials set
- ✅ Default recipient: cbjones84@yahoo.com
- ✅ Ready to send alerts

---

## 📝 Files Created

1. **`.env.notifications`** - Configuration file with credentials
2. **`setup_notifications.sh`** - Setup script
3. **`NOTIFICATION_CONFIGURED.md`** - This file

---

## 🎯 Next Steps

1. The notification service is ready to use
2. Emails will be sent automatically when critical events occur
3. Monitor your inbox at cbjones84@yahoo.com for alerts
4. Test the service using the test code above

---

*Configuration Date: 2025-12-09*  
*App Name: NAE*  
*Email: cbjones84@yahoo.com*

