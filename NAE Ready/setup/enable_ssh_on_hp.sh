#!/bin/bash
# Script to enable SSH on HP OmniBook X
# Copy this to HP and run, or run remotely if you have another way to access HP

echo "=========================================="
echo "Enable SSH on HP OmniBook X"
echo "=========================================="
echo ""

# Check if running as root or with sudo
if [ "$EUID" -ne 0 ]; then
    echo "⚠️  This script needs sudo privileges"
    echo "Run with: sudo bash enable_ssh_on_hp.sh"
    echo ""
    echo "Or run these commands manually:"
    echo "  sudo apt update"
    echo "  sudo apt install openssh-server -y"
    echo "  sudo systemctl enable ssh"
    echo "  sudo systemctl start ssh"
    echo "  sudo ufw allow 22"
    exit 1
fi

echo "Step 1: Updating package list..."
apt update -qq

echo "Step 2: Installing SSH server..."
apt install openssh-server -y -qq

echo "Step 3: Enabling SSH..."
systemctl enable ssh 2>/dev/null || systemctl enable sshd 2>/dev/null

echo "Step 4: Starting SSH..."
systemctl start ssh 2>/dev/null || systemctl start sshd 2>/dev/null

echo "Step 5: Configuring firewall..."
ufw allow 22 2>/dev/null || iptables -A INPUT -p tcp --dport 22 -j ACCEPT 2>/dev/null || echo "⚠️  Firewall configuration skipped"

echo "Step 6: Verifying SSH status..."
if systemctl is-active --quiet ssh || systemctl is-active --quiet sshd; then
    echo "✅ SSH is running"
else
    echo "⚠️  SSH status unclear - check manually: sudo systemctl status ssh"
fi

echo ""
echo "Step 7: Getting IP address..."
IP=$(hostname -I | awk '{print $1}' 2>/dev/null || ip addr show | grep "inet " | grep -v 127.0.0.1 | head -1 | awk '{print $2}' | cut -d/ -f1)
echo "IP Address: $IP"

echo ""
echo "=========================================="
echo "✅ SSH Setup Complete!"
echo "=========================================="
echo ""
echo "IP Address: $IP"
echo ""
echo "Test from Mac:"
echo "  ssh <username>@$IP 'echo Connected'"
echo ""

