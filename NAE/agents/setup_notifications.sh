#!/bin/bash
# NAE Notification Service Setup Script
# Configures email notifications to cbjones84@yahoo.com

echo "🔔 Setting up NAE Notification Service..."
echo ""

# Set notification email configuration
export NOTIFICATION_EMAIL_ENABLED=true
export NOTIFICATION_EMAIL_TO=cbjones84@yahoo.com
export NOTIFICATION_EMAIL_FROM=cbjones84@yahoo.com
export NOTIFICATION_SMTP_SERVER=smtp.mail.yahoo.com
export NOTIFICATION_SMTP_PORT=587
export NOTIFICATION_SMTP_USERNAME=cbjones84@yahoo.com
export NOTIFICATION_SMTP_PASSWORD=lbtfwssbsieswkft
export NOTIFICATION_SMTP_USE_TLS=true

echo "✅ Notification service configured:"
echo "   Email: cbjones84@yahoo.com"
echo "   SMTP: smtp.mail.yahoo.com:587"
echo "   App Name: NAE"
echo ""
echo "📝 To make these settings permanent, add them to your shell profile:"
echo "   ~/.bashrc or ~/.zshrc"
echo ""
echo "Or create a .env file in the NAE directory with:"
echo "   NOTIFICATION_EMAIL_ENABLED=true"
echo "   NOTIFICATION_EMAIL_TO=cbjones84@yahoo.com"
echo "   NOTIFICATION_SMTP_USERNAME=cbjones84@yahoo.com"
echo "   NOTIFICATION_SMTP_PASSWORD=lbtfwssbsieswkft"
echo ""
echo "✅ Configuration complete!"

