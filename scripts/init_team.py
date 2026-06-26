"""
Initialize the AI Agent Research Team.
Sets up environment, downloads dependencies, configures agents.
"""
import os
import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).parent.parent


def check_prerequisites():
    """Check all prerequisites are installed."""
    print("🔍 Checking prerequisites...\n")
    
    checks = {
        "Python >= 3.9": _check_python(),
        "Git": _check_git(),
        "PyTorch": _check_pytorch(),
        "Transformers": _check_package("transformers"),
        "Librosa": _check_package("librosa"),
        "PyYAML": _check_package("yaml"),
        "Scikit-learn": _check_package("sklearn"),
    }
    
    for name, ok in checks.items():
        status = "✅" if ok else "❌"
        print(f"  {status} {name}")
    
    all_ok = all(checks.values())
    if not all_ok:
        print("\n❌ Some prerequisites are missing. Please install them:")
        print("  pip install torch torchaudio transformers librosa scikit-learn pyyaml")
        return False
    return True


def _check_python():
    return sys.version_info >= (3, 9)


def _check_git():
    try:
        subprocess.run(["git", "--version"], capture_output=True, check=True)
        return True
    except:
        return False


def _check_pytorch():
    try:
        import torch
        return True
    except ImportError:
        return False


def _check_package(name):
    try:
        __import__(name)
        return True
    except ImportError:
        return False


def setup_env_file():
    """Create .env template if not exists."""
    env_path = ROOT / ".env"
    if not env_path.exists():
        env_template = """# DeepSeek API Key (for AI agents)
DEEPSEEK_API_KEY=sk-your-key-here

# Weights & Biases (for experiment tracking)
WANDB_API_KEY=your-wandb-key
WANDB_PROJECT=icbhi2017-sota

# GitHub token (for git operations)
GITHUB_TOKEN=ghp_your-token-here

# Training PC connection
TRAINING_PC_SSH=user@training-pc-ip
TRAINING_PC_PORT=22

# ICBHI 2017 Dataset path
ICBHI_DATA_PATH=/path/to/ICBHI_2017
"""
        env_path.write_text(env_template)
        print("  📝 Created .env template. Please fill in your API keys.")


def setup_git():
    """Initialize git if not already done."""
    git_dir = ROOT / ".git"
    if not git_dir.exists():
        print("  📦 Initializing git repository...")
        subprocess.run(["git", "init"], cwd=ROOT, check=True)
        
        # Create .gitignore
        gitignore = """# Python
__pycache__/
*.py[cod]
*.egg-info/
dist/
build/

# Environments
.env
.venv/
venv/

# Experiments (large files)
experiments/*/checkpoints/
*.pth
*.pt

# IDE
.vscode/
.idea/

# Data (don't commit dataset)
ICBHI_2017/
data/raw/

# Outputs
*.log
wandb/
"""
        (ROOT / ".gitignore").write_text(gitignore)
        print("  📝 Created .gitignore")


def setup_directories():
    """Create necessary directories."""
    dirs = [
        "experiments",
        "results",
        "paper/figures",
        ".agent_tasks",
    ]
    for d in dirs:
        (ROOT / d).mkdir(parents=True, exist_ok=True)
    print("  📁 Created directory structure")


def print_next_steps():
    """Print next steps for the user."""
    print(f"""
{'='*60}
  ✅ Team initialization complete!
{'='*60}

Next Steps:
  1. Fill in your .env file with API keys:
     → Edit {ROOT / '.env'}

  2. Place ICBHI 2017 dataset in:
     → {ROOT / 'ICBHI_2017/'}
     (Or update ICBHI_DATA_PATH in .env)

  3. Set up GitHub remote:
     → git remote add origin <your-repo-url>
     → git push -u origin main

  4. On Training Computer:
     → git clone <your-repo-url>
     → ./scripts/training_pc/setup.sh
     → sudo systemctl enable icbhi-agent
     → sudo systemctl start icbhi-agent

  5. Start autonomous research:
     → python workflows/autonomous_research.py --mode semi-auto

  6. Monitor progress:
     → Check results/STATUS.md
     → wandb dashboard (if configured)

{'='*60}
""")


def main():
    print("\n" + "="*60)
    print("  🫁 Lung Sound SOTA Research Team — Initialization")
    print("="*60 + "\n")
    
    if not check_prerequisites():
        sys.exit(1)
    
    print("\n⚙️  Setting up...\n")
    
    setup_env_file()
    setup_git()
    setup_directories()
    
    print_next_steps()


if __name__ == "__main__":
    main()
