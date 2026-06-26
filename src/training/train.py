"""
Main training script for ICBHI 2017 lung sound classification.
Supports all registered models with configurable training pipeline.
"""
import os
import sys
import argparse
import yaml
import json
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LinearLR, SequentialLR

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.dataset import create_dataloaders
from src.data.preprocessing import SpecAugment, MixUp
from src.models import get_model


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(
    model: nn.Module,
    dataloader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    scaler: GradScaler,
    device: torch.device,
    specaugment: SpecAugment = None,
    mixup: MixUp = None,
    class_weights: torch.Tensor = None,
    gradient_accumulation_steps: int = 1,
) -> dict:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        # Apply SpecAugment
        if specaugment is not None and inputs.dim() >= 3:
            inputs = specaugment(inputs)
        
        with autocast():
            outputs = model(inputs)
            
            if class_weights is not None:
                class_weights = class_weights.to(device)
                loss = criterion(outputs, targets) * class_weights[targets]
                loss = loss.mean()
            else:
                loss = criterion(outputs, targets)
            
            loss = loss / gradient_accumulation_steps
        
        scaler.scale(loss).backward()
        
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        total_loss += loss.item() * gradient_accumulation_steps
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    return {
        "loss": total_loss / len(dataloader),
        "accuracy": 100.0 * correct / total,
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    all_preds = []
    all_targets = []
    
    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        all_preds.append(predicted.cpu())
        all_targets.append(targets.cpu())
    
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    
    return {
        "loss": total_loss / len(dataloader),
        "accuracy": 100.0 * correct / total,
        "predictions": all_preds,
        "targets": all_targets,
    }


def train(config: dict, exp_dir: Path):
    """Full training pipeline."""
    
    # Setup
    set_seed(config.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Data
    data_config = config.get("data", {})
    train_loader, test_loader, class_weights = create_dataloaders(
        data_dir=data_config.get("data_dir", "./ICBHI_2017"),
        split_file=data_config.get("split_file", "./src/data/official_split.txt"),
        batch_size=config.get("batch_size", 32),
        sample_rate=data_config.get("sample_rate", 16000),
        cycle_duration=data_config.get("cycle_duration", 4.0),
        num_workers=data_config.get("num_workers", 4),
        return_waveform=config.get("model", {}).get("input_type") == "waveform",
    )
    
    # Model
    model_config = config.get("model", {})
    model_name = model_config.get("name", "AST-RespAdapter")
    model = get_model(model_name, model_config)
    model = model.to(device)
    
    print(f"Model: {model_name}")
    print(f"Trainable parameters: {model.num_parameters:,}")
    print(f"Total parameters: {model.total_parameters:,}")
    
    # Training config
    training_config = config.get("training", {})
    epochs = training_config.get("epochs", 100)
    lr = training_config.get("learning_rate", 1e-4)
    weight_decay = training_config.get("weight_decay", 1e-4)
    warmup_epochs = training_config.get("warmup_epochs", 5)
    grad_accum = training_config.get("gradient_accumulation_steps", 1)
    patience = training_config.get("early_stopping_patience", 15)
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    
    # Scheduler: linear warmup + cosine
    warmup_scheduler = LinearLR(
        optimizer, start_factor=0.1, total_iters=warmup_epochs
    )
    cosine_scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=epochs - warmup_epochs, T_mult=1
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs],
    )
    
    # Loss
    criterion = nn.CrossEntropyLoss()
    
    # Mixed precision
    scaler = GradScaler(device.type)
    
    # Augmentation
    aug_config = config.get("augmentation", {})
    specaugment = SpecAugment(
        time_mask_param=aug_config.get("time_mask_param", 20),
        freq_mask_param=aug_config.get("freq_mask_param", 12),
    ) if aug_config.get("use_specaugment", True) else None
    
    mixup = MixUp(alpha=aug_config.get("mixup_alpha", 0.4)) \
        if aug_config.get("use_mixup", False) else None
    
    # Training loop
    best_val_acc = 0.0
    best_epoch = 0
    no_improve = 0
    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        # Train
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler,
            device, specaugment, mixup, class_weights, grad_accum,
        )
        
        # Validate
        val_metrics = validate(model, test_loader, criterion, device)
        
        # Step scheduler
        scheduler.step()
        
        # Log
        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])
        
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"LR: {current_lr:.2e} | "
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.2f}%"
        )
        
        # Save best
        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            best_epoch = epoch
            no_improve = 0
            
            # Save checkpoint
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_accuracy": best_val_acc,
                "config": config,
            }
            torch.save(checkpoint, exp_dir / "best_model.pt")
            
            print(f"  ✓ New best model! Acc: {best_val_acc:.2f}%")
        else:
            no_improve += 1
        
        # Early stopping
        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break
    
    training_time = time.time() - start_time
    
    # Save training history
    with open(exp_dir / "train_history.json", "w") as f:
        json.dump(history, f, indent=2)
    
    # Return results
    return {
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_acc,
        "total_epochs": epoch,
        "training_time_hours": training_time / 3600,
        "train_history": history,
    }


def main():
    parser = argparse.ArgumentParser(description="Train lung sound model")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--exp-id", type=str, default=None, help="Experiment ID")
    args = parser.parse_args()
    
    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    # Setup experiment directory
    exp_id = args.exp_id or datetime.now().strftime("exp_%Y%m%d_%H%M%S")
    exp_dir = Path("experiments") / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(exp_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)
    
    print(f"\n{'='*60}")
    print(f"Experiment: {exp_id}")
    print(f"Config: {args.config}")
    print(f"{'='*60}\n")
    
    # Train
    results = train(config, exp_dir)
    
    # Save results
    results["experiment_id"] = exp_id
    results["config_file"] = args.config
    
    with open(exp_dir / "results.yaml", "w") as f:
        yaml.dump(results, f)
    
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Best validation accuracy: {results['best_val_accuracy']:.2f}%")
    print(f"Best epoch: {results['best_epoch']}")
    print(f"Training time: {results['training_time_hours']:.2f} hours")
    print(f"Results saved to: {exp_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
