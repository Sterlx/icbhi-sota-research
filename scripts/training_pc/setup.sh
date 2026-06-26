#!/bin/bash
# ============================================================
# Training PC Setup Script (REFERENCE ONLY)
# Your actual environment is already configured:
#   Python 3.12.7 | PyTorch 2.12.0+cu126 | torchaudio 2.11.0 | torchvision 0.27.0+cu126
# ============================================================

set -e

echo "🖥️  ICBHI Training Environment Setup (Reference)"
echo "================================================="
echo ""
echo "⚠️  Your training environment is already set up with:"
echo "    Python 3.12.7"
echo "    PyTorch 2.12.0+cu126"
echo "    torchaudio 2.11.0"
echo "    torchvision 0.27.0+cu126"
echo ""
echo "This script documents the setup. Skip the install steps"
echo "if your environment is already working."
echo ""

# Install system dependencies (skip if already done)
echo "📦 System dependencies (skip if installed)..."
# sudo apt-get update
# sudo apt-get install -y python3.12 python3.12-venv python3-pip git

# Virtual environment (skip - already exists)
# echo "🐍 Creating Python virtual environment..."
# python3.12 -m venv venv
# source venv/bin/activate

# PyTorch with CUDA 12.6 (skip - already installed)
# echo "🔥 PyTorch 2.12.0+cu126 (already installed)..."
# pip install torch==2.12.0 torchvision==0.27.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu126

# Install additional ML packages
echo "📚 Installing additional ML packages..."
pip install transformers accelerate
pip install wandb tensorboard
pip install librosa soundfile audiomentations
pip install scikit-learn pandas numpy matplotlib seaborn
pip install pyyaml tqdm

echo ""
echo "✅ Environment setup complete!"
echo ""

# Set up systemd service
echo "🔧 Setting up systemd service for auto-training..."

SERVICE_FILE="/etc/systemd/system/icbhi-agent.service"
sudo tee $SERVICE_FILE > /dev/null << EOF
[Unit]
Description=ICBHI Lung Sound Research Agent
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$(pwd)
ExecStart=$(pwd)/scripts/training_pc/watchdog.sh
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
echo ""
echo "To enable auto-start on boot:"
echo "  sudo systemctl enable icbhi-agent"
echo ""
echo "To start now:"
echo "  sudo systemctl start icbhi-agent"
echo ""
echo "To check status:"
echo "  sudo systemctl status icbhi-agent"
