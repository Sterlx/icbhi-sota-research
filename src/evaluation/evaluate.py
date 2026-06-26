"""
Evaluation script for ICBHI 2017 lung sound classification.
Computes official metrics and generates reports.
"""
import os
import sys
import argparse
import yaml
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_auc_score, f1_score, accuracy_score
)

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.dataset import create_dataloaders
from src.models import get_model


def icbhi_score(y_true: np.ndarray, y_pred: np.ndarray, 
                num_classes: int = 4) -> Dict:
    """Calculate official ICBHI 2017 metrics.
    
    Score = (macro_avg_sensitivity + macro_avg_specificity) / 2
    """
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    
    sensitivities = []
    specificities = []
    per_class = {}
    class_names = ["Normal", "Wheeze", "Crackles", "Both"]
    
    for c in range(num_classes):
        tp = cm[c, c]
        fn = cm[c, :].sum() - tp
        fp = cm[:, c].sum() - tp
        tn = cm.sum() - tp - fn - fp
        
        se = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        sp = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        sensitivities.append(se)
        specificities.append(sp)
        per_class[class_names[c]] = {
            "sensitivity": round(se * 100, 2),
            "specificity": round(sp * 100, 2),
            "score": round((se + sp) * 50, 2),
            "f1": round(f1_score(y_true == c, y_pred == c) * 100, 2),
        }
    
    macro_se = np.mean(sensitivities) * 100
    macro_sp = np.mean(specificities) * 100
    score = (macro_se + macro_sp) / 2
    
    return {
        "sensitivity": round(macro_se, 2),
        "specificity": round(macro_sp, 2),
        "score": round(score, 2),
        "accuracy": round(accuracy_score(y_true, y_pred) * 100, 2),
        "macro_f1": round(f1_score(y_true, y_pred, average="macro") * 100, 2),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }


def mcnemar_test(y_true: np.ndarray, y_pred_a: np.ndarray, 
                 y_pred_b: np.ndarray) -> Dict:
    """McNemar's test for paired classifier comparison."""
    both_correct = (y_pred_a == y_true) & (y_pred_b == y_true)
    a_correct_b_wrong = (y_pred_a == y_true) & (y_pred_b != y_true)
    a_wrong_b_correct = (y_pred_a != y_true) & (y_pred_b == y_true)
    
    n1 = a_correct_b_wrong.sum()
    n2 = a_wrong_b_correct.sum()
    
    chi2 = (abs(n1 - n2) - 1)**2 / (n1 + n2) if (n1 + n2) > 0 else 0
    
    from scipy.stats import chi2 as chi2_dist
    p_value = 1 - chi2_dist.cdf(chi2, df=1)
    
    return {
        "chi2_statistic": round(chi2, 4),
        "p_value": round(p_value, 6),
        "significant_at_0.05": p_value < 0.05,
    }


def bootstrap_confidence_interval(
    y_true: np.ndarray, 
    y_pred: np.ndarray,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
) -> Dict:
    """Bootstrap 95% confidence intervals for Score."""
    scores = []
    n = len(y_true)
    
    for _ in range(n_bootstrap):
        indices = np.random.choice(n, n, replace=True)
        metrics = icbhi_score(y_true[indices], y_pred[indices])
        scores.append(metrics["score"])
    
    lower = np.percentile(scores, alpha / 2 * 100)
    upper = np.percentile(scores, (1 - alpha / 2) * 100)
    
    return {
        "mean": round(np.mean(scores), 2),
        "std": round(np.std(scores), 2),
        "ci_lower": round(lower, 2),
        "ci_upper": round(upper, 2),
        "ci_95": f"[{lower:.2f}, {upper:.2f}]",
    }


@torch.no_grad()
def evaluate_model(model, dataloader, device: torch.device) -> tuple:
    """Run inference on test set."""
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []
    
    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        probs = torch.softmax(outputs, dim=1)
        _, predicted = outputs.max(1)
        
        all_preds.append(predicted.cpu())
        all_targets.append(targets.cpu())
        all_probs.append(probs.cpu())
    
    return (
        torch.cat(all_preds).numpy(),
        torch.cat(all_targets).numpy(),
        torch.cat(all_probs).numpy(),
    )


def generate_report(
    exp_dir: Path,
    metrics: Dict,
    bootstrap: Dict,
    config: Dict,
):
    """Generate evaluation report."""
    report = f"""# Evaluation Report — {exp_dir.name}

## Model: {config.get('model', {}).get('name', 'Unknown')}

## Official ICBHI 2017 Metrics (60/40 Split)

| Metric | Value |
|--------|-------|
| **Score** (Se+Sp)/2 | **{metrics['score']}%** |
| Macro Sensitivity (Se) | {metrics['sensitivity']}% |
| Macro Specificity (Sp) | {metrics['specificity']}% |
| Accuracy | {metrics['accuracy']}% |
| Macro F1 | {metrics['macro_f1']}% |

### 95% Confidence Interval (Bootstrap)
- Score: {bootstrap['ci_95']}
- Mean: {bootstrap['mean']}% ± {bootstrap['std']}%

## Per-Class Performance

| Class | Se (%) | Sp (%) | Score (%) | F1 (%) |
|-------|--------|--------|-----------|--------|
"""
    for cls_name, cls_metrics in metrics["per_class"].items():
        report += f"| {cls_name} | {cls_metrics['sensitivity']} | "
        report += f"{cls_metrics['specificity']} | {cls_metrics['score']} | "
        report += f"{cls_metrics['f1']} |\n"
    
    report += f"""
## Confusion Matrix
```
{np.array(metrics['confusion_matrix'])}
```
"""
    
    with open(exp_dir / "evaluation_report.md", "w") as f:
        f.write(report)
    
    # Also save as JSON
    full_results = {
        "metrics": metrics,
        "bootstrap": bootstrap,
        "config_summary": {
            "model": config.get("model", {}).get("name"),
            "backbone": config.get("model", {}).get("backbone"),
        },
    }
    with open(exp_dir / "metrics.json", "w") as f:
        json.dump(full_results, f, indent=2)
    
    return report


def main():
    parser = argparse.ArgumentParser(description="Evaluate lung sound model")
    parser.add_argument("--exp-id", type=str, required=True, help="Experiment ID")
    parser.add_argument("--checkpoint", type=str, default=None, 
                       help="Path to checkpoint (default: experiments/{exp_id}/best_model.pt)")
    args = parser.parse_args()
    
    exp_dir = Path("experiments") / args.exp_id
    
    # Load config
    with open(exp_dir / "config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Load checkpoint
    checkpoint_path = args.checkpoint or (exp_dir / "best_model.pt")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    model_config = config.get("model", {})
    model = get_model(model_config.get("name", "AST-RespAdapter"), model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    
    # Load test data
    data_config = config.get("data", {})
    _, test_loader, _ = create_dataloaders(
        data_dir=data_config.get("data_dir", "./ICBHI_2017"),
        batch_size=config.get("batch_size", 32),
        sample_rate=data_config.get("sample_rate", 16000),
        cycle_duration=data_config.get("cycle_duration", 4.0),
        num_workers=data_config.get("num_workers", 4),
        return_waveform=model_config.get("input_type") == "waveform",
    )
    
    # Evaluate
    print("\nRunning evaluation...")
    y_pred, y_true, y_probs = evaluate_model(model, test_loader, device)
    
    # Compute metrics
    metrics = icbhi_score(y_true, y_pred)
    bootstrap = bootstrap_confidence_interval(y_true, y_pred)
    
    # Generate report
    report = generate_report(exp_dir, metrics, bootstrap, config)
    print(report)
    
    # Print summary
    print(f"\n{'='*40}")
    print(f"  ICBHI SCORE: {metrics['score']}%")
    print(f"  Se: {metrics['sensitivity']}% | Sp: {metrics['specificity']}%")
    print(f"  95% CI: {bootstrap['ci_95']}")
    print(f"{'='*40}")


if __name__ == "__main__":
    main()
