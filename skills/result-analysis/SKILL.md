# Skill: Result Analysis & SOTA Comparison

## Official ICBHI Metric Calculation

```python
import numpy as np
from sklearn.metrics import confusion_matrix

def icbhi_score(y_true, y_pred, num_classes=4):
    """
    Calculate official ICBHI 2017 Score.
    Score = (macro_avg_sensitivity + macro_avg_specificity) / 2
    """
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    
    sensitivities = []
    specificities = []
    
    for c in range(num_classes):
        tp = cm[c, c]
        fn = cm[c, :].sum() - tp
        fp = cm[:, c].sum() - tp
        tn = cm.sum() - tp - fn - fp
        
        se = tp / (tp + fn) if (tp + fn) > 0 else 0
        sp = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        sensitivities.append(se)
        specificities.append(sp)
    
    macro_se = np.mean(sensitivities) * 100
    macro_sp = np.mean(specificities) * 100
    score = (macro_se + macro_sp) / 2
    
    return {
        "sensitivity": macro_se,
        "specificity": macro_sp,
        "score": score,
        "per_class_se": [s * 100 for s in sensitivities],
        "per_class_sp": [s * 100 for s in specificities],
        "confusion_matrix": cm.tolist()
    }
```

## SOTA Claim Requirements

Before claiming SOTA, verify ALL:
1. ✅ Official 60/40 patient-level split
2. ✅ 4-class classification (Normal, Wheeze, Crackles, Both)
3. ✅ Cycle-level evaluation
4. ✅ Score = (Se + Sp) / 2, macro-averaged
5. ✅ Reproducible with fixed seed
6. ✅ No test data used in training/validation

## Statistical Significance

```python
def mcnemar_test(y_true, y_pred_a, y_pred_b):
    """McNemar's test for paired classifier comparison."""
    both_correct = (y_pred_a == y_true) & (y_pred_b == y_true)
    a_correct_b_wrong = (y_pred_a == y_true) & (y_pred_b != y_true)
    a_wrong_b_correct = (y_pred_a != y_true) & (y_pred_b == y_true)
    both_wrong = (y_pred_a != y_true) & (y_pred_b != y_true)
    
    n1 = a_correct_b_wrong.sum()
    n2 = a_wrong_b_correct.sum()
    
    # Chi-squared test
    chi2 = (abs(n1 - n2) - 1)**2 / (n1 + n2) if (n1 + n2) > 0 else 0
    from scipy.stats import chi2 as chi2_dist
    p_value = 1 - chi2_dist.cdf(chi2, df=1)
    
    return {"chi2": chi2, "p_value": p_value, "significant": p_value < 0.05}
```

## Results Dashboard

Generate `results/dashboard.md` with:
- Current best model metrics
- SOTA comparison table (auto-updated)
- Per-class radar chart
- Training progress charts
- Experiment history timeline
