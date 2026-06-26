#!/bin/bash
# ============================================================
# Training PC Setup — Install additional ML packages only
# Your environment is already configured:
#   Python 3.12.7 | PyTorch 2.12.0+cu126 | torchaudio 2.11.0 | torchvision 0.27.0+cu126
# Run with Git Bash on Windows:  ./scripts/training_pc/setup.sh
# ============================================================

echo "🖥️  ICBHI Training — Installing ML packages"
echo "============================================"
echo ""
echo "Your base environment (skip — already installed):"
echo "  Python 3.12.7"
echo "  PyTorch 2.12.0+cu126"
echo "  torchaudio 2.11.0"
echo "  torchvision 0.27.0+cu126"
echo ""

# Install additional ML packages
echo "Installing additional ML packages..."
pip install transformers accelerate
pip install wandb tensorboard
pip install librosa soundfile audiomentations
pip install scikit-learn pandas numpy matplotlib seaborn
pip install pyyaml tqdm

echo ""
echo "Done. To start the watchdog, run:"
echo "  ./scripts/training_pc/watchdog.sh"
echo "To enable auto-start on boot:"
echo "  sudo systemctl enable icbhi-agent"
echo ""
echo "To start now:"
echo "  sudo systemctl start icbhi-agent"
echo ""
echo "To check status:"
echo "  sudo systemctl status icbhi-agent"
