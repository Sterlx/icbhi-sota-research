# Skill: Model Experiment Design

## Experiment Workflow

### 1. Define Experiment Config
```yaml
# experiments/{exp_id}/config.yaml
experiment:
  id: "exp_XXX"
  name: "Short description"
  hypothesis: "What we expect to improve"
  
model:
  architecture: "AST-RespAdapter"
  backbone: "MIT/ast-finetuned-audioset-10-10-0.4593"
  pretrained: true
  
training:
  batch_size: 32
  epochs: 100
  optimizer: "AdamW"
  lr: 1e-4
  scheduler: "cosine_with_warmup"
  warmup_epochs: 5
  
data:
  augmentations: ["specaugment", "mixup"]
  mixup_alpha: 0.4
  
evaluation:
  metrics: ["se", "sp", "score", "f1", "auc"]
  per_class: true
```

### 2. Run Training
```bash
python src/training/train.py --config experiments/{exp_id}/config.yaml
```

### 3. Log Everything
- WandB project: "icbhi2017-sota"
- TensorBoard logs
- Git commit hash tracked

## Experiment Naming Convention
```
exp_{NNN}_{model_short}_{key_innovation}
Example: exp_042_ast_respadapter_v2
```

## Hyperparameter Tuning Strategy
1. **Phase 1**: Coarse grid search (LR, batch size, dropout)
2. **Phase 2**: Architecture-specific (adapter dim, num heads)
3. **Phase 3**: Augmentation mix

## Reproducibility Checklist
- [ ] Seed fixed (42)
- [ ] Data split saved
- [ ] Config YAML saved with results
- [ ] Model checkpoint saved
- [ ] WandB run ID recorded
- [ ] GPU type noted
