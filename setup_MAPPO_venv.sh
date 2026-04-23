#!/bin/bash
# Setup script for MAPPO traffic control environment

echo "=========================================="
echo "MAPPO Traffic Control - Environment Setup"
echo "=========================================="
echo ""

# Check if SUMO is installed
if ! command -v sumo &> /dev/null; then
    echo "⚠️  Warning: SUMO not found. Please install SUMO first:"
    echo "   - Ubuntu/Debian: sudo apt-get install sumo sumo-tools"
    echo "   - macOS: brew install sumo"
    echo "   - Windows: Download from https://eclipse.dev/sumo/"
    echo ""
fi

# Check SUMO_HOME
if [ -z "$SUMO_HOME" ]; then
    echo "⚠️  Warning: SUMO_HOME not set. Please set it:"
    echo "   export SUMO_HOME=\"/usr/share/sumo\"  # Adjust path as needed"
    echo ""
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv MAPPO_venv

# Activate virtual environment
echo "Activating virtual environment..."
source MAPPO_venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install PyTorch (CPU version for compatibility)
echo "Installing PyTorch..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
echo "Installing dependencies..."
pip install numpy==1.24.3
pip install gymnasium==0.28.1
pip install pettingzoo==1.24.3
pip install sumo-rl==1.4.5
pip install pyyaml
pip install wandb
pip install tensorboard
pip install matplotlib
pip install seaborn

# Install Tianshou
echo "Installing Tianshou..."
pip install tianshou==0.5.1

echo ""
echo "=========================================="
echo "✅ Installation Complete!"
echo "=========================================="
echo ""
echo "To activate the environment in the future, run:"
echo "  source MAPPO_venv/bin/activate"
echo ""
echo "To test the installation, run:"
echo "  python scripts/train_mappo.py --debug"
echo ""
