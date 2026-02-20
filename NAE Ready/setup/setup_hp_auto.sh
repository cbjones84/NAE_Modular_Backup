#!/bin/bash
# Automated HP Setup - Non-Interactive Version
# Attempts to set up HP automatically or provides clear next steps

set -e

echo "=========================================="
echo "Automated HP OmniBook X Setup"
echo "=========================================="
echo ""

# Check for remote config
if [ -f "config/remote_config.json" ]; then
    echo "✅ Remote configuration found"
    echo "Attempting remote setup..."
    echo ""
    
    python3 setup/remote_hp_setup.py
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ HP setup completed remotely!"
        echo ""
        echo "Verifying connection..."
        python3 setup/cursor_remote_integration.py --verify
        echo ""
        echo "Getting production status..."
        python3 setup/cursor_remote_integration.py --status
        exit 0
    else
        echo ""
        echo "⚠️  Remote setup failed"
        echo ""
    fi
fi

# If remote setup not possible, provide instructions
echo "=========================================="
echo "HP Setup Instructions"
echo "=========================================="
echo ""
echo "To complete HP OmniBook X setup, choose one method:"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "METHOD 1: Standalone Script (RECOMMENDED)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "1. Copy this file to HP OmniBook X:"
echo "   $(pwd)/hp_local_setup.sh"
echo ""
echo "2. On HP OmniBook X, run:"
echo "   bash hp_local_setup.sh"
echo ""
echo "3. The script will:"
echo "   ✅ Clone NAE repository"
echo "   ✅ Configure environment (PRODUCTION=true)"
echo "   ✅ Initialize safety systems"
echo "   ✅ Configure as Node"
echo "   ✅ Create production branch"
echo "   ✅ Enable SSH"
echo "   ✅ Show HP IP address"
echo ""
echo "4. After HP setup, on Mac:"
echo "   ./setup/configure_remote_connection.sh"
echo "   (Enter HP IP address when prompted)"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "METHOD 2: Direct Setup on HP"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "On HP OmniBook X terminal:"
echo ""
echo "  git clone https://github.com/cbjones84/NAE.git"
echo "  cd NAE"
echo "  bash setup/automated_hp_setup.sh"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "METHOD 3: Remote Setup (If SSH Ready)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "If you have HP IP and SSH access:"
echo ""
echo "  1. ./setup/configure_remote_connection.sh"
echo "     (Enter HP IP, username, port)"
echo ""
echo "  2. Copy SSH key:"
echo "     ssh-copy-id <username>@<hp_ip>"
echo ""
echo "  3. Run remote setup:"
echo "     python3 setup/remote_hp_setup.py"
echo ""
echo "=========================================="
echo ""

# Show standalone script info
if [ -f "hp_local_setup.sh" ]; then
    echo "✅ Standalone script ready:"
    echo "   File: $(pwd)/hp_local_setup.sh"
    echo "   Size: $(du -h hp_local_setup.sh | cut -f1)"
    echo "   Lines: $(wc -l < hp_local_setup.sh)"
    echo ""
    echo "To copy to HP:"
    echo "  scp hp_local_setup.sh <username>@<hp_ip>:~/"
    echo "  (or use USB drive, network share, etc.)"
    echo ""
fi

echo "Master API Key (for HP setup):"
echo "  72364bc8ecfea124010e9811d06d0b0b3b220e4dee3d09163b2bd1f005af40a7"
echo ""

