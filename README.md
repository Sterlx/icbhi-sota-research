# 🫁 Lung Sound SOTA Research Team — AI Agent System

> **ICBHI 2017 Dataset** | Abnormal Lung Sound Detection (Wheeze, Crackles, Normal, Both)  
> Official 60/40 Split + Per Respiratory Cycle Detection | Powered by DeepSeek V4 Pro

---

## 📋 Overview

This is an **AI-powered autonomous research team** designed to discover and build **State-of-the-Art (SOTA)** models for lung sound classification. The team consists of specialized AI agents that collaborate to:

1. **Research** current SOTA methods in lung sound classification
2. **Design** novel architectures using pretrained models + Transformers
3. **Train** models on a dedicated GPU training computer
4. **Evaluate** results against ICBHI 2017 benchmarks
5. **Write** academic papers automatically
6. **Iterate** until SOTA performance is achieved

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    👤 YOU (Researcher)                           │
│               Check results • Approve • Intervene                │
└──────────────────────┬──────────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────────┐
│              🎯 ORCHESTRATOR AGENT (Laptop)                      │
│  ┌─────────┐ ┌──────────┐ ┌───────────┐ ┌──────────┐           │
│  │Research │ │  Data    │ │  Model    │ │  Paper   │           │
│  │ Agent   │ │  Agent   │ │ Architect │ │  Writer  │           │
│  └────┬────┘ └────┬─────┘ └─────┬─────┘ └────┬─────┘           │
│       │           │             │             │                  │
│       └───────────┴──────┬──────┴─────────────┘                 │
│                          │                                       │
│              ┌───────────▼──────────┐                            │
│              │  GitHub Repository   │                            │
│              └───────────┬──────────┘                            │
└──────────────────────────┼──────────────────────────────────────┘
                           │  git push / pull
┌──────────────────────────▼──────────────────────────────────────┐
│           🖥️ TRAINING COMPUTER (GPU)                             │
│  ┌────────────────┐  ┌──────────────────┐  ┌────────────────┐  │
│  │ Training       │  │ Evaluation       │  │ Result         │  │
│  │ Controller     │  │ Agent            │  │ Reporter       │  │
│  └────────────────┘  └──────────────────┘  └────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## 🤖 Agent Team

| Agent | Role | Location |
|-------|------|----------|
| **Orchestrator** | Coordinates all agents, manages workflow | Laptop |
| **Research Agent** | Searches papers, GitHub repos, finds SOTA | Laptop |
| **Data Agent** | ICBHI 2017 preprocessing, augmentation | Laptop |
| **Model Architect** | Designs novel architectures | Laptop |
| **Training Controller** | Manages GPU training jobs | Training PC |
| **Evaluation Agent** | Analyzes results, benchmarks | Training PC |
| **Paper Writer** | Drafts LaTeX papers, figures | Laptop |

## 🚀 Quick Start

### Prerequisites

```bash
# Laptop (Development)
pip install torch torchaudio transformers librosa audiomentations
pip install scikit-learn pandas numpy matplotlib seaborn
pip install wandb tensorboard pyyaml tqdm

# Training Computer (GPU)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers accelerate deepspeed
pip install wandb tensorboard
```

### 1. Set Environment Variables

```bash
# .env file
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxx
WANDB_API_KEY=your_wandb_key
GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxx
TRAINING_PC_SSH=user@training-pc-ip
TRAINING_PC_PORT=22
ICBHI_DATA_PATH=/path/to/ICBHI_2017
```

### 2. Initialize the Agent Team

```bash
python scripts/init_team.py
```

### 3. Start Autonomous Research

```bash
python workflows/autonomous_research.py --mode auto
```

## 📁 Project Structure

```
MyResearchTeam/
├── agents/              # AI Agent definitions (.agent.md)
├── skills/              # Domain-specific skills (SKILL.md)
├── workflows/           # Automation workflows
├── src/                 # ML source code
│   ├── data/            # Data preprocessing & loading
│   ├── models/          # Model architectures
│   ├── training/        # Training loops & configs
│   ├── evaluation/      # Metrics & benchmarking
│   └── utils/           # Utilities
├── configs/             # Experiment configurations
├── scripts/             # Automation scripts
├── experiments/         # Experiment logs & checkpoints
├── results/             # Final results & comparisons
├── paper/               # Paper templates & drafts
└── .github/workflows/   # CI/CD for auto-training
```

## 📊 ICBHI 2017 Dataset

- **Classes**: Normal, Wheeze, Crackles, Both (Wheeze+Crackles)
- **Split**: Official 60/40 train/test split
- **Task**: Per respiratory cycle classification
- **Metric**: Sensitivity, Specificity, Score (average of Se/Sp)

## 🔬 Research Strategy

1. **Audio Pretrained Models**: AST, HuBERT, wav2vec2, CLAP, Whisper
2. **Architecture Innovations**: Multi-scale attention, cross-modal fusion
3. **Data Strategies**: MixMatch, SpecAugment, respiratory cycle alignment
4. **Ensemble Methods**: Multi-model voting, test-time augmentation

---

*Built with DeepSeek V4 Pro • Autonomous ML Research*
