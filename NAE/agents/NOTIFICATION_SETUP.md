# NAE Notification Service Setup

## Email Configuration for cbjones84@yahoo.com

The notification service has been configured to send emails to **cbjones84@yahoo.com** by default.

---

## Current Configuration

### Default Settings
- **Email Enabled**: ✅ Yes (default)
- **Email Recipient**: cbjones84@yahoo.com
- **Email From**: nae-trading@neuralagency.com
- **SMTP Server**: smtp.mail.yahoo.com
- **SMTP Port**: 587 (TLS)
- **SMTP TLS**: Enabled

---

## Setup Instructions

### Option 1: Using Yahoo Mail SMTP (Recommended)

1. **Enable App Password in Yahoo Account**:
   - Go to https://login.yahoo.com/account/security
   - Enable "Two-step verification" if not already enabled
   - Generate an "App Password" for "Mail"
   - Copy the generated app password

2. **Set Environment Variables**:
   ```bash
   export NOTIFICATION_EMAIL_ENABLED=true
   export NOTIFICATION_EMAIL_TO=cbjones84@yahoo.com
   export NOTIFICATION_EMAIL_FROM=cbjones84@yahoo.com  # Your Yahoo email
   export NOTIFICATION_SMTP_SERVER=smtp.mail.yahoo.com
   export NOTIFICATION_SMTP_PORT=587
   export NOTIFICATION_SMTP_USERNAME=cbjones84@yahoo.com
   export NOTIFICATION_SMTP_PASSWORD=your_app_password_here  # Use app password, not regular password
   export NOTIFICATION_SMTP_USE_TLS=true
   ```

### Option 2: Using Gmail SMTP (Alternative)

If you prefer to use Gmail:

```bash
export NOTIFICATION_EMAIL_ENABLED=true
export NOTIFICATION_EMAIL_TO=cbjones84@yahoo.com
export NOTIFICATION_EMAIL_FROM=your@gmail.com
export NOTIFICATION_SMTP_SERVER=smtp.gmail.com
export NOTIFICATION_SMTP_PORT=587
export NOTIFICATION_SMTP_USERNAME=your@gmail.com
export NOTIFICATION_SMTP_PASSWORD=your_gmail_app_password
export NOTIFICATION_SMTP_USE_TLS=true
```

### Option 3: Using SendGrid (Professional)

For production use, consider SendGrid:

```bash
export NOTIFICATION_EMAIL_ENABLED=true
export NOTIFICATION_EMAIL_TO=cbjones84@yahoo.com
export NOTIFICATION_SMTP_SERVER=smtp.sendgrid.net
export NOTIFICATION_SMTP_PORT=587
export NOTIFICATION_SMTP_USERNAME=apikey
export NOTIFICATION_SMTP_PASSWORD=your_sendgrid_api_key
export NOTIFICATION_SMTP_USE_TLS=true
```

---

## What Gets Notified

The notification service sends emails for:

1. **Circuit Breaker Triggers** (Critical Priority)
   - When 10+ consecutive errors occur
   - When drawdown exceeds 50%
   - When daily loss exceeds 35%

2. **Daily Loss Limits** (Critical Priority)
   - When daily loss reaches 35% threshold

3. **Trading Pauses** (Critical Priority)
   - Any automatic trading pause event

---

## Email Format

Emails include:
- **Subject**: `[NAE CRITICAL] Circuit Breaker Triggered`
- **Priority Level**: Critical, High, or Normal
- **Timestamp**: Exact time of the event
- **Detailed Message**: Full alert message
- **HTML Formatting**: Color-coded by priority

---

## Testing

To test the notification service:

```python
from ralph_github_continuous import NotificationService

service = NotificationService()
service.send(
    title="Test Notification",
    message="This is a test message from NAE",
    priority="critical"
)
```

---

## Troubleshooting

### Email Not Sending

1. **Check SMTP Credentials**:
   - Verify username and password are correct
   - For Yahoo: Use App Password, not regular password
   - For Gmail: Use App Password, not regular password

2. **Check Firewall/Network**:
   - Ensure port 587 (TLS) is not blocked
   - Check if SMTP server is accessible

3. **Check Logs**:
   - Look for error messages in logs
   - Check if email_enabled is True
   - Verify email_to is set correctly

### Common Errors

**"Authentication failed"**:
- Use App Password instead of regular password
- Enable 2FA on your email account

**"Connection refused"**:
- Check SMTP server address
- Verify port number (587 for TLS, 465 for SSL)
- Check firewall settings

**"Email not configured"**:
- Set NOTIFICATION_EMAIL_ENABLED=true
- Verify NOTIFICATION_EMAIL_TO is set

---

## Security Notes

1. **Never commit passwords to code**
   - Always use environment variables
   - Use App Passwords, not regular passwords
   - Consider using a secrets manager for production

2. **App Passwords**:
   - More secure than regular passwords
   - Can be revoked independently
   - Required for Yahoo and Gmail

3. **TLS Encryption**:
   - Always enabled by default
   - Ensures secure email transmission

---

## Current Status

✅ **Email notifications configured for**: cbjones84@yahoo.com  
✅ **Email enabled by default**  
✅ **SMTP ready** (requires credentials to be set)  
✅ **HTML email formatting**  
✅ **Priority-based alerts**  

---

## Next Steps

1. Set up SMTP credentials (see Option 1 above)
2. Test the notification service
3. Monitor emails for alerts
4. Adjust notification preferences as needed

---

*Last Updated: 2025-12-09*

