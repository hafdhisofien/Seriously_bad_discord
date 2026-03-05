#!/bin/bash

# 🥒 Pickle Detection Bot - Installation Script
# This script installs all dependencies for the AI-powered pickle detection

echo "🥒 =========================================="
echo "   Pickle Detection Bot - Installation"
echo "=========================================="
echo ""

# Check Python version
echo "📋 Checking Python version..."
python3 --version || { echo "❌ Python 3 not found! Please install Python 3.8+"; exit 1; }
echo "✅ Python found"
echo ""

# Check pip
echo "📋 Checking pip..."
pip --version || pip3 --version || { echo "❌ pip not found! Please install pip"; exit 1; }
echo "✅ pip found"
echo ""

# Ask user which installation mode
echo "🎯 Choose installation mode:"
echo "   1) Full installation (with AI - ~2-4GB download)"
echo "   2) Minimal installation (text-only - ~50MB download)"
echo ""
read -p "Enter choice (1 or 2): " choice

case $choice in
    1)
        echo ""
        echo "🔄 Installing full version with AI..."
        echo "⏱️  This will take 5-10 minutes and download ~2-4GB"
        echo ""
        pip install -r requirements.txt
        ;;
    2)
        echo ""
        echo "🔄 Installing minimal version (text-only)..."
        echo "⚠️  Remember to set CLIP_ENABLED = False in RoleBot.py"
        echo ""
        pip install -r requirements-minimal.txt
        ;;
    *)
        echo "❌ Invalid choice. Exiting."
        exit 1
        ;;
esac

# Check if installation succeeded
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ =========================================="
    echo "   Installation Complete!"
    echo "=========================================="
    echo ""
    echo "📝 Next steps:"
    echo "   1. Edit RoleBot.py line 18:"
    echo "      PICKLE_RESTRICTED_USER_ID = YOUR_USER_ID"
    echo ""
    echo "   2. Make sure you have DISCORD_TOKEN in .env file"
    echo ""
    echo "   3. Run the bot:"
    echo "      python3 RoleBot.py"
    echo ""
    echo "📚 Documentation:"
    echo "   - QUICK_START.md - Fast setup guide"
    echo "   - README_PICKLE.md - Complete guide"
    echo "   - PICKLE_DETECTION_GUIDE.md - Detailed docs"
    echo ""
    echo "🎮 Test commands:"
    echo "   !clip_status - Check AI status"
    echo "   !test_pickle - Test detection on images"
    echo ""
    echo "🥒 Happy pickle hunting! 🤖"
else
    echo ""
    echo "❌ =========================================="
    echo "   Installation Failed!"
    echo "=========================================="
    echo ""
    echo "🔧 Try manual installation:"
    echo "   pip install -r requirements.txt"
    echo ""
    echo "Or for minimal version:"
    echo "   pip install -r requirements-minimal.txt"
    exit 1
fi
