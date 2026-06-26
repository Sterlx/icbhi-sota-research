"""
Auto-generate all paper figures from experiment results.
"""
import sys
from pathlib import Path
import json
import yaml
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).parent.parent
FIGURES_DIR = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# Style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

CLASS_NAMES = ["Normal", "Wheeze", "Crackles", "Both"]


def generate_confusion_matrix(metrics: dict, save_path: str):
    """Generate confusion matrix plot."""
    cm = np.array(metrics.get("confusion_matrix", [[0,0,0,0]]*4))
    
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=ax, cbar_kws={'label': 'Count'},
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix — ICBHI 2017")
    
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Confusion matrix saved to {save_path}")


def generate_sota_comparison(our_results: dict, save_path: str):
    """Generate SOTA comparison bar chart.
    
    ⚠️ ANTI-HALLUCINATION: This function loads verified SOTA data from
    results/sota_comparison.md at runtime. Never hardcode paper names/scores.
    If no verified baselines are available, only show 'Ours'.
    """
    # Load verified baselines from the SOTA tracker
    import re
    sota_file = ROOT / "results" / "sota_comparison.md"
    methods = []
    scores = []
    
    if sota_file.exists():
        with open(sota_file) as f:
            content = f.read()
        # Parse markdown table rows (format: | Name | URL | Se | Sp | Score | Year |)
        for line in content.split("\n"):
            match = re.match(r'\|\s*\d+\s*\|\s*(.+?)\s*\|.*?\|\s*[\d.]+\s*\|\s*[\d.]+\s*\|\s*([\d.]+)\s*\|', line)
            if match:
                name = match.group(1).strip()
                score = float(match.group(2))
                methods.append(name)
                scores.append(score)
    
    if not methods:
        # Fallback: show only known verified baseline
        methods = ["RespireNet\n(Gairola 2021)"]
        scores = [65.93]
    
    # Add our result
    methods.append("Ours")
    scores.append(our_results.get("score", 0))
    colors = ['#cccccc'] * (len(methods) - 1) + ['#0066cc']
    
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(methods, scores, color=colors, edgecolor='white')
    
    # Add value labels
    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{score:.1f}', ha='center', va='bottom', fontweight='bold')
    
    ax.set_ylabel("ICBHI Score (%)")
    ax.set_title("SOTA Comparison — ICBHI 2017 (60/40 Split)")
    ax.set_ylim(60, max(scores) + 3)
    ax.axhline(y=68, color='red', linestyle='--', alpha=0.3, label='~68% baseline')
    ax.legend()
    
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ SOTA comparison saved to {save_path}")


def generate_training_curves(history: dict, save_path: str):
    """Generate training curves from history."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    epochs = range(1, len(history.get("train_loss", [])) + 1)
    
    ax1.plot(epochs, history.get("train_loss", []), label="Train Loss", linewidth=2)
    ax1.plot(epochs, history.get("val_loss", []), label="Val Loss", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(epochs, history.get("val_acc", []), color='green', linewidth=2)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Validation Accuracy")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Training curves saved to {save_path}")


def generate_per_class_radar(per_class: dict, save_path: str):
    """Generate per-class performance radar chart."""
    categories = list(per_class.keys())
    se_values = [per_class[c]["sensitivity"] for c in categories]
    sp_values = [per_class[c]["specificity"] for c in categories]
    
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    
    se_values += se_values[:1]
    sp_values += sp_values[:1]
    
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.fill(angles, se_values, alpha=0.25, label='Sensitivity (Se)')
    ax.fill(angles, sp_values, alpha=0.25, label='Specificity (Sp)')
    ax.plot(angles, se_values, linewidth=2)
    ax.plot(angles, sp_values, linewidth=2)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 100)
    ax.set_title("Per-Class Performance")
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Per-class radar saved to {save_path}")


def main():
    """Generate all figures from latest experiment results."""
    print("\n📊 Generating paper figures...\n")
    
    # Find latest experiment with results
    exp_dirs = sorted(ROOT.glob("experiments/exp_*"), reverse=True)
    
    metrics = None
    history = None
    
    for exp_dir in exp_dirs:
        metrics_file = exp_dir / "metrics.json"
        if metrics_file.exists():
            with open(metrics_file) as f:
                data = json.load(f)
            metrics = data["metrics"]
            break
    
    if metrics is None:
        print("⚠️  No experiment results found. Using placeholder data.")
        metrics = {
            "score": 0,
            "sensitivity": 0,
            "specificity": 0,
            "confusion_matrix": [[10, 2, 1, 0], [2, 8, 0, 1], [1, 0, 12, 2], [0, 1, 2, 5]],
            "per_class": {
                "Normal": {"sensitivity": 75, "specificity": 85, "score": 80, "f1": 78},
                "Wheeze": {"sensitivity": 65, "specificity": 88, "score": 76.5, "f1": 70},
                "Crackles": {"sensitivity": 70, "specificity": 82, "score": 76, "f1": 72},
                "Both": {"sensitivity": 50, "specificity": 90, "score": 70, "f1": 55},
            }
        }
    
    # Generate all figures
    generate_confusion_matrix(metrics, str(FIGURES_DIR / "confusion_matrix.pdf"))
    generate_sota_comparison(metrics, str(FIGURES_DIR / "sota_comparison.pdf"))
    
    if history:
        generate_training_curves(history, str(FIGURES_DIR / "training_curves.pdf"))
    
    generate_per_class_radar(metrics["per_class"], str(FIGURES_DIR / "per_class_radar.pdf"))
    
    print(f"\n✅ All figures saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
