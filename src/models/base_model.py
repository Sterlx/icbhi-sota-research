"""
Base model interface and model registry for lung sound classification.
"""
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple


class BaseLungSoundModel(nn.Module, ABC):
    """Abstract base class for all lung sound classification models."""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.num_classes = config.get("num_classes", 4)
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor. Shape depends on model type:
               - Waveform models: (B, T)
               - Spectrogram models: (B, 1, T, F) or (B, T, F)
        Returns:
            Logits: (B, num_classes)
        """
        pass
    
    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Extract feature embeddings (before classifier)."""
        raise NotImplementedError
    
    @property
    def num_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    @property 
    def total_parameters(self) -> int:
        """Count all parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def freeze_backbone(self):
        """Freeze backbone parameters (for fine-tuning)."""
        raise NotImplementedError
    
    def unfreeze_backbone(self):
        """Unfreeze backbone parameters."""
        raise NotImplementedError


MODEL_REGISTRY = {}


def register_model(name: str):
    """Decorator to register a model in the registry."""
    def wrapper(cls):
        MODEL_REGISTRY[name] = cls
        return cls
    return wrapper


def get_model(name: str, config: Dict) -> BaseLungSoundModel:
    """Get a model by name from the registry."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{name}' not found. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name](config)
