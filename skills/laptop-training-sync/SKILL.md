# Skill: Laptop ↔ Training Computer Synchronization

## Architecture

```
LAPTOP (Development)                    TRAINING PC (GPU)
├── src/                               ├── src/
├── configs/                           ├── configs/
├── agents/                            ├── experiments/
├── paper/                             └── results/
└── experiments/
      │                                      │
      │  git push                            │  git pull
      │  (code + triggers)                   │  (code + triggers)
      ▼                                      ▼
┌──────────────────────────────────────────────────┐
│                  GITHUB REPO                       │
│  ┌─────────┐  ┌──────────┐  ┌────────────────┐  │
│  │  src/   │  │ configs/ │  │  experiments/  │  │
│  └─────────┘  └──────────┘  └────────────────┘  │
└──────────────────────────────────────────────────┘
      ▲                                      │
      │  git pull                            │  git push
      │  (results + checkpoints)             │  (results + checkpoints)
      │                                      ▼
LAPTOP (Analysis)                       TRAINING PC (Done)
```

## Sync Protocol

### 1. Laptop → GitHub → Training PC

```bash
# scripts/sync_push_experiment.sh
#!/bin/bash
# Called by Orchestrator when starting new experiment

EXP_ID=$1
git add src/ configs/ experiments/${EXP_ID}/
git commit -m "[AUTO] Experiment ${EXP_ID}: ${DESCRIPTION}"
git push origin main

# Create trigger file
echo "experiment_id: ${EXP_ID}" > experiments/TRIGGER.yaml
echo "timestamp: $(date -Iseconds)" >> experiments/TRIGGER.yaml
echo "status: queued" >> experiments/TRIGGER.yaml
git add experiments/TRIGGER.yaml
git commit -m "[TRIGGER] Start experiment ${EXP_ID}"
git push origin main
```

### 2. Training PC: Watch & Execute

```bash
# scripts/training_pc/watchdog.sh
#!/bin/bash
# Runs as systemd service on training PC
# Polls GitHub for new triggers every 60 seconds

while true; do
    git pull origin main
    
    if [ -f "experiments/TRIGGER.yaml" ]; then
        STATUS=$(python -c "import yaml; print(yaml.safe_load(open('experiments/TRIGGER.yaml'))['status'])")
        if [ "$STATUS" == "queued" ]; then
            # Update status to running
            python -c "
import yaml
t = yaml.safe_load(open('experiments/TRIGGER.yaml'))
t['status'] = 'running'
yaml.dump(t, open('experiments/TRIGGER.yaml','w'))
"
            git add experiments/TRIGGER.yaml
            git commit -m "[STATUS] Running experiment ${EXP_ID}"
            git push origin main
            
            # Read experiment ID
            EXP_ID=$(python -c "import yaml; print(yaml.safe_load(open('experiments/TRIGGER.yaml'))['experiment_id'])")
            
            # Run training
            python src/training/train.py --exp-id ${EXP_ID}
            
            # Run evaluation
            python src/evaluation/evaluate.py --exp-id ${EXP_ID}
            
            # Mark as completed & push results
            python -c "
import yaml
t = yaml.safe_load(open('experiments/TRIGGER.yaml'))
t['status'] = 'completed'
yaml.dump(t, open('experiments/TRIGGER.yaml','w'))
"
            git add experiments/${EXP_ID}/ results/
            git commit -m "[RESULTS] Experiment ${EXP_ID} completed"
            git push origin main
        fi
    fi
    
    sleep 60
done
```

### 3. Laptop: Pull Results

```bash
# scripts/sync_pull_results.sh
#!/bin/bash
# Called by Orchestrator to check for completed experiments

git pull origin main

# Check for newly completed experiments
python scripts/check_completed_experiments.py
```

## SSH Alternative (Direct Laptop ↔ Training PC)

```bash
# scripts/training_pc/remote_trigger.sh
#!/bin/bash
# If direct SSH is available, trigger training directly

ssh ${TRAINING_PC_SSH} -p ${TRAINING_PC_PORT} \
    "cd ~/MyResearchTeam && \
     git pull origin main && \
     nohup python src/training/train.py --exp-id ${EXP_ID} > train_${EXP_ID}.log 2>&1 &"
```

## Systemd Service (Training PC)

```ini
# /etc/systemd/system/icbhi-agent.service
[Unit]
Description=ICBHI Research Agent Watchdog
After=network.target

[Service]
Type=simple
User=researcher
WorkingDirectory=/home/researcher/MyResearchTeam
ExecStart=/home/researcher/MyResearchTeam/scripts/training_pc/watchdog.sh
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

## File Locking

To prevent race conditions:
- Laptop writes TRIGGER only if no trigger with status "queued" or "running" exists
- Training PC sets status to "running" before starting
- Use `experiments/.lock` file for mutual exclusion
