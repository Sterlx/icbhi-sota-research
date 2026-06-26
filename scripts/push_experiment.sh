#!/bin/bash
# ============================================================
# Sync: Push experiment from laptop to GitHub 
# Used by Orchestrator to start a new experiment
# ============================================================

set -e

EXP_ID=$1
DESCRIPTION=${2:-"Experiment $EXP_ID"}

if [ -z "$EXP_ID" ]; then
    echo "Usage: $0 <experiment_id> [description]"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo "📤 Pushing experiment $EXP_ID to GitHub..."

# Stage changes
git add src/ configs/ "experiments/${EXP_ID}/"

# Check if there's anything to commit
if git diff --staged --quiet; then
    echo "   No changes to commit"
else
    git commit -m "[AUTO] ${EXP_ID}: ${DESCRIPTION}"
fi

# Create/update trigger
python3 -c "
import yaml
from datetime import datetime

trigger = {
    'experiment_id': '${EXP_ID}',
    'timestamp': datetime.now().isoformat(),
    'status': 'queued',
}
with open('experiments/TRIGGER.yaml', 'w') as f:
    yaml.dump(trigger, f)
"

git add experiments/TRIGGER.yaml
git commit -m "[TRIGGER] Start ${EXP_ID}" 2>/dev/null || true

# Push
git push origin main

echo "   ✅ Experiment $EXP_ID pushed!"
echo "   Training PC will pick it up within 60 seconds."
