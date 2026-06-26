#!/bin/bash
# ============================================================
# Training PC Watchdog
# Polls GitHub for new experiments, runs training, pushes results
# ============================================================
# Usage: ./watchdog.sh
# Or install as systemd service for persistent running
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "🖥️  ICBHI Training Watchdog started"
echo "   Project: $PROJECT_DIR"
echo "   Time: $(date)"
echo ""

# Load environment
if [ -f ".env" ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Main loop
while true; do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Checking for new experiments..."
    
    # Pull latest from GitHub
    git pull origin main 2>/dev/null || true
    
    # Check for trigger file
    if [ -f "experiments/TRIGGER.yaml" ]; then
        # Read trigger status using Python
        STATUS=$(python3 -c "
import yaml
try:
    with open('experiments/TRIGGER.yaml') as f:
        t = yaml.safe_load(f)
    print(t.get('status', 'unknown'))
except:
    print('error')
" 2>/dev/null)
        
        if [ "$STATUS" == "queued" ]; then
            EXP_ID=$(python3 -c "
import yaml
with open('experiments/TRIGGER.yaml') as f:
    t = yaml.safe_load(f)
print(t.get('experiment_id', 'unknown'))
")
            
            echo ""
            echo "="'='*60
            echo "  🚀 Starting experiment: $EXP_ID"
            echo "="'='*60
            
            # Update status to running
            python3 -c "
import yaml
with open('experiments/TRIGGER.yaml') as f:
    t = yaml.safe_load(f)
t['status'] = 'running'
t['started_at'] = '$(date -Iseconds)'
with open('experiments/TRIGGER.yaml', 'w') as f:
    yaml.dump(t, f)
"
            git add experiments/TRIGGER.yaml
            git commit -m "[STATUS] Running $EXP_ID" 2>/dev/null || true
            git push origin main 2>/dev/null || true
            
            # Find config file
            CONFIG_FILE="experiments/${EXP_ID}/config.yaml"
            if [ ! -f "$CONFIG_FILE" ]; then
                # Use latest model config
                CONFIG_FILE=$(ls configs/models/*.yaml 2>/dev/null | head -1)
            fi
            
            if [ -f "$CONFIG_FILE" ]; then
                echo "   Config: $CONFIG_FILE"
                
                # Run training
                echo "   Training..."
                python3 src/training/train.py --config "$CONFIG_FILE" --exp-id "$EXP_ID"
                
                # Run evaluation
                echo "   Evaluating..."
                python3 src/evaluation/evaluate.py --exp-id "$EXP_ID"
                
                # Mark as completed
                python3 -c "
import yaml
with open('experiments/TRIGGER.yaml') as f:
    t = yaml.safe_load(f)
t['status'] = 'completed'
t['completed_at'] = '$(date -Iseconds)'
with open('experiments/TRIGGER.yaml', 'w') as f:
    yaml.dump(t, f)
"
                
                # Push results
                echo "   Pushing results..."
                git add "experiments/${EXP_ID}/" experiments/TRIGGER.yaml results/
                git commit -m "[RESULTS] $EXP_ID completed" 2>/dev/null || true
                git push origin main 2>/dev/null || true
                
                echo "   ✅ Experiment $EXP_ID complete!"
            else
                echo "   ❌ No config file found!"
                
                # Mark as failed
                python3 -c "
import yaml
with open('experiments/TRIGGER.yaml') as f:
    t = yaml.safe_load(f)
t['status'] = 'failed'
t['error'] = 'No config file found'
with open('experiments/TRIGGER.yaml', 'w') as f:
    yaml.dump(t, f)
"
                git add experiments/TRIGGER.yaml
                git commit -m "[FAILED] $EXP_ID - no config" 2>/dev/null || true
                git push origin main 2>/dev/null || true
            fi
            
            echo ""
        fi
    fi
    
    # Wait before next poll
    sleep 60
done
