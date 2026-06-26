#!/bin/bash
# ============================================================
# Training PC Watchdog
# Polls GitHub for new experiments, runs training, pushes results
# Run with Git Bash:  ./scripts/training_pc/watchdog.sh
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Use 'python' (works on Windows venv + Linux)
PYTHON=$(which python 2>/dev/null || which python3 2>/dev/null)
if [ -z "$PYTHON" ]; then
    echo "ERROR: Python not found! Activate your venv first."
    exit 1
fi

echo " ICBHI Training Watchdog started"
echo "   Project: $PROJECT_DIR"
echo "   Python:  $PYTHON"
echo "   Time:    $(date)"
echo ""

while true; do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Checking for new experiments..."
    
    git pull origin main 2>/dev/null || true
    
    if [ -f "experiments/TRIGGER.yaml" ]; then
        STATUS=$($PYTHON -c "
import yaml
try:
    with open('experiments/TRIGGER.yaml') as f:
        t = yaml.safe_load(f)
    print(t.get('status', 'unknown'))
except Exception as e:
    print(f'error: {e}')
")
        echo "   Trigger status: $STATUS"
        
        if [ "$STATUS" == "queued" ]; then
            EXP_ID=$($PYTHON -c "
import yaml
with open('experiments/TRIGGER.yaml') as f:
    t = yaml.safe_load(f)
print(t.get('experiment_id', 'unknown'))
")
            
            echo ""
            echo "$(printf '=%.0s' {1..60})"
            echo "  STARTING experiment: $EXP_ID"
            echo "$(printf '=%.0s' {1..60})"
            
            # Update status to running
            $PYTHON -c "
import yaml
from datetime import datetime
with open('experiments/TRIGGER.yaml') as f:
    t = yaml.safe_load(f)
t['status'] = 'running'
t['started_at'] = datetime.now().isoformat()
with open('experiments/TRIGGER.yaml', 'w') as f:
    yaml.dump(t, f)
"
            git add experiments/TRIGGER.yaml
            git commit -m "[STATUS] Running $EXP_ID" 2>/dev/null || true
            git push origin main 2>/dev/null || true
            
            CONFIG_FILE="experiments/${EXP_ID}/config.yaml"
            if [ ! -f "$CONFIG_FILE" ]; then
                CONFIG_FILE=$(ls configs/models/*.yaml 2>/dev/null | head -1)
            fi
            
            if [ -f "$CONFIG_FILE" ]; then
                echo "   Config: $CONFIG_FILE"
                echo "   Training..."
                
                $PYTHON src/training/train.py --config "$CONFIG_FILE" --exp-id "$EXP_ID"
                
                if [ $? -eq 0 ]; then
                    echo "   Evaluating..."
                    $PYTHON src/evaluation/evaluate.py --exp-id "$EXP_ID"
                    
                    $PYTHON -c "
import yaml
from datetime import datetime
with open('experiments/TRIGGER.yaml') as f:
    t = yaml.safe_load(f)
t['status'] = 'completed'
t['completed_at'] = datetime.now().isoformat()
with open('experiments/TRIGGER.yaml', 'w') as f:
    yaml.dump(t, f)
"
                    echo "   Pushing results..."
                    git add "experiments/${EXP_ID}/" experiments/TRIGGER.yaml results/
                    git commit -m "[RESULTS] $EXP_ID completed" 2>/dev/null || true
                    git push origin main 2>/dev/null || true
                    
                    echo "   DONE: Experiment $EXP_ID complete!"
                else
                    echo "   TRAINING FAILED!"
                    $PYTHON -c "
import yaml
with open('experiments/TRIGGER.yaml') as f:
    t = yaml.safe_load(f)
t['status'] = 'failed'
with open('experiments/TRIGGER.yaml', 'w') as f:
    yaml.dump(t, f)
"
                    git add experiments/TRIGGER.yaml
                    git commit -m "[FAILED] $EXP_ID" 2>/dev/null || true
                    git push origin main 2>/dev/null || true
                fi
            else
                echo "   ERROR: No config file found!" 
            fi
            echo ""
        fi
    fi
    
    sleep 60
done
