#!/bin/bash
# ============================================================
# Sync: Pull results from GitHub to laptop
# Used by Orchestrator to check for completed experiments
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo "📥 Pulling latest results..."

git pull origin main

# Check for newly completed experiments
python3 -c "
import yaml
from pathlib import Path

try:
    with open('experiments/TRIGGER.yaml') as f:
        trigger = yaml.safe_load(f)
    
    status = trigger.get('status', 'unknown')
    exp_id = trigger.get('experiment_id', 'unknown')
    
    if status == 'completed':
        print(f'  ✅ Experiment {exp_id} completed!')
        
        # Check for results
        results_path = Path('experiments') / exp_id / 'metrics.json'
        if results_path.exists():
            import json
            with open(results_path) as f:
                metrics = json.load(f)
            score = metrics['metrics']['score']
            print(f'  Score: {score}%')
        else:
            print('  ⚠️  No metrics.json found yet')
    elif status == 'running':
        print(f'  ⏳ Experiment {exp_id} still running...')
    elif status == 'failed':
        print(f'  ❌ Experiment {exp_id} failed!')
    else:
        print(f'  Status: {status}')
except FileNotFoundError:
    print('  No active trigger found')
except Exception as e:
    print(f'  Error: {e}')
"

echo "  ✅ Pull complete"
