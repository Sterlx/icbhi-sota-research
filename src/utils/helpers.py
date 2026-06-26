"""
Utility functions for the research pipeline.
"""
import os
import yaml
import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent


def load_yaml(path: str) -> Dict:
    """Load a YAML file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_yaml(data: Dict, path: str):
    """Save data to a YAML file."""
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)


def load_json(path: str) -> Dict:
    """Load a JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def save_json(data: Dict, path: str):
    """Save data to a JSON file."""
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def get_git_hash() -> str:
    """Get current git commit hash."""
    import subprocess
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], 
            cwd=get_project_root()
        ).decode().strip()
    except:
        return "unknown"


def get_timestamp() -> str:
    """Get ISO timestamp."""
    return datetime.now().isoformat()


def compute_file_hash(filepath: str) -> str:
    """Compute SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def check_stop_signal() -> bool:
    """Check if STOP_SIGNAL file exists (human intervention)."""
    return (get_project_root() / "STOP_SIGNAL").exists()


def update_status_file(exp_id: str, status: str, metrics: Optional[Dict] = None):
    """Update the status file for the Orchestrator."""
    status_file = get_project_root() / "results" / "STATUS.md"
    
    status_content = f"""# Research Status — {get_timestamp()}

**Current Experiment**: {exp_id}
**Status**: {status}

"""
    if metrics:
        status_content += f"""## Latest Results
| Metric | Value |
|--------|-------|
| Score | {metrics.get('score', 'N/A')}% |
| Sensitivity | {metrics.get('sensitivity', 'N/A')}% |
| Specificity | {metrics.get('specificity', 'N/A')}% |
"""
    
    with open(status_file, "w") as f:
        f.write(status_content)


def get_latest_experiment() -> Optional[str]:
    """Get the most recent experiment ID."""
    exp_dir = get_project_root() / "experiments"
    if not exp_dir.exists():
        return None
    
    experiments = sorted(
        [d for d in exp_dir.iterdir() if d.is_dir() and d.name.startswith("exp_")],
        key=lambda x: x.stat().st_mtime,
        reverse=True,
    )
    return experiments[0].name if experiments else None


def format_metrics_table(metrics_list: list) -> str:
    """Format a list of metric dictionaries as a markdown table."""
    if not metrics_list:
        return "No metrics available."
    
    headers = list(metrics_list[0].keys())
    header_line = "| " + " | ".join(headers) + " |"
    sep_line = "|" + "|".join(["------" for _ in headers]) + "|"
    
    rows = []
    for m in metrics_list:
        row = "| " + " | ".join(str(m.get(h, "")) for h in headers) + " |"
        rows.append(row)
    
    return "\n".join([header_line, sep_line] + rows)
