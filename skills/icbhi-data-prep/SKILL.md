# Skill: ICBHI 2017 Data Preparation

## Dataset Overview
- **Source**: ICBHI 2017 Challenge (International Conference on Biomedical and Health Informatics)
- **Audio**: 5.5 hours, 920 recordings, 126 patients
- **Annotations**: Respiratory cycle-level (6898 cycles)
- **Classes**: Normal, Wheeze, Crackles, Both

## Data Loading Recipe

### Step 1: Parse Annotations
```python
import pandas as pd
import numpy as np
from pathlib import Path

def parse_icbhi_annotations(annot_dir: Path) -> pd.DataFrame:
    """Parse all .txt annotation files into a DataFrame."""
    records = []
    for txt_file in annot_dir.glob("*.txt"):
        # Skip log files
        if "_Log" in txt_file.name:
            continue
        patient_id = txt_file.stem
        with open(txt_file) as f:
            for line in f:
                start, end, crackles, wheezes = line.strip().split()
                records.append({
                    "patient_id": patient_id,
                    "start": float(start),
                    "end": float(end),
                    "crackles": int(crackles),
                    "wheezes": int(wheezes),
                    "label": get_label(int(crackles), int(wheezes))
                })
    return pd.DataFrame(records)

def get_label(crackles: int, wheezes: int) -> int:
    if crackles == 0 and wheezes == 0: return 0  # Normal
    if crackles == 0 and wheezes == 1: return 1  # Wheeze
    if crackles == 1 and wheezes == 0: return 2  # Crackles
    return 3  # Both
```

### Step 2: Patient-Level Split
```python
def official_split(patients: list, train_ratio=0.6, seed=42):
    """Official 60/40 patient-level split."""
    np.random.seed(seed)
    patients = sorted(set(patients))
    np.random.shuffle(patients)
    split_idx = int(len(patients) * train_ratio)
    return patients[:split_idx], patients[split_idx:]
```

### Step 3: Audio Preprocessing
- Resample all to 16kHz
- Bandpass filter: 100-2000 Hz (lung sound range)
- Normalize to [-1, 1]
- Extract cycles based on annotations

## Critical Rules
1. **NEVER split cycles randomly** — always split by patient
2. Handle sampling rate variations (4kHz, 10kHz, 44.1kHz in dataset)
3. Some cycles overlap — handle carefully
4. Class imbalance is severe — use weighted sampling/loss
