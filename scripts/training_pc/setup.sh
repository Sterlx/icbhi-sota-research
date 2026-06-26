#!/bin/bash
# ============================================================
# Training PC Setup Script
# Run once on the training computer to set up the environment
# ============================================================

set -e

echo "🖥️  Setting up ICBHI Training Environment"
echo "========================================="

# Install system dependencies
echo "📦 Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y python3.10 python3.10-venv python3-pip git

# Create virtual environment
echo "🐍 Creating Python virtual environment..."
python3.10 -m venv venv
source venv/bin/activate

# Install PyTorch with CUDA
echo "🔥 Installing PyTorch with CUDA..."
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
echo "📚 Installing ML packages..."
pip install transformers accelerate deepspeed
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
