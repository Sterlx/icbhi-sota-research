"""
AST (Audio Spectrogram Transformer) with Respiratory Cycle Adapter.
A novel architecture combining pretrained AST with cycle-aware attention.
"""
import torch
import torch.nn as nn
import math
from typing import Dict, Optional
from .base_model import BaseLungSoundModel, register_model


class RespiratoryCycleEmbedding(nn.Module):
    """Learnable embedding that encodes position within a respiratory cycle."""
    
    def __init__(self, max_cycles: int = 4, embed_dim: int = 64):
        super().__init__()
        self.cycle_embed = nn.Embedding(max_cycles, embed_dim)
        self.position_in_cycle = nn.Parameter(
            torch.randn(1, 1, embed_dim) * 0.02
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add cycle-aware embeddings.
        
        Args:
            x: (B, N_patches, D) patch features
        Returns:
            (B, N_patches, D) with cycle embeddings added
        """
        B, N, D = x.shape
        # Create a simple sinusoidal pattern mimicking respiratory cycles
        t = torch.linspace(0, 4 * math.pi, N, device=x.device)
        cycle_pattern = torch.sin(t).unsqueeze(0).unsqueeze(-1)  # (1, N, 1)
        cycle_embed = cycle_pattern * self.position_in_cycle[:, :N, :]
        return x + cycle_embed


class RespiratoryAdapter(nn.Module):
    """Adapter module that adds respiratory cycle awareness to AST features."""
    
    def __init__(
        self,
        input_dim: int = 768,
        adapter_dim: int = 256,
        num_cycles: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.adapter_dim = adapter_dim
        
        # Down-projection
        self.down = nn.Linear(input_dim, adapter_dim)
        
        # Cycle-aware attention
        self.cycle_embed = RespiratoryCycleEmbedding(num_cycles, adapter_dim)
        self.cycle_attention = nn.MultiheadAttention(
            adapter_dim, num_heads=4, dropout=dropout, batch_first=True
        )
        
        # Up-projection with residual
        self.up = nn.Linear(adapter_dim, input_dim)
        self.norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Activation
        self.act = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply respiratory adapter.
        
        Args:
            x: (B, N, D) AST hidden states
        Returns:
            (B, N, D) Adapted features
        """
        residual = x
        
        # Down-project
        h = self.down(x)
        h = self.act(h)
        
        # Add cycle embeddings
        h = self.cycle_embed(h)
        
        # Self-attention with cycle awareness
        h, _ = self.cycle_attention(h, h, h)
        
        # Up-project with residual
        h = self.up(h)
        h = self.dropout(h)
        h = self.norm(residual + h)
        
        return h


@register_model("AST-RespAdapter")
class ASTRespAdapter(BaseLungSoundModel):
    """AST with Respiratory Cycle Adapter for lung sound classification.
    
    Architecture:
        Audio → Mel Spectrogram → AST (pretrained) → RespiratoryAdapter → Classifier
    """
    
    def __init__(self, config: Dict):
        super().__init__(config)
        
        backbone_name = config.get("backbone", "MIT/ast-finetuned-audioset-10-10-0.4593")
        adapter_config = config.get("adapter", {})
        classifier_config = config.get("classifier", {})
        
        self.freeze_backbone_flag = config.get("freeze_backbone", False)
        self.input_type = "mel"  # AST takes mel spectrograms
        
        # AST backbone (from HuggingFace)
        try:
            from transformers import ASTModel, ASTConfig
            self.ast = ASTModel.from_pretrained(backbone_name)
            self.hidden_size = self.ast.config.hidden_size
        except Exception as e:
            print(f"Warning: Could not load AST from HuggingFace ({e})")
            print("Using placeholder — will need pretrained weights")
            self.hidden_size = 768
            self.ast = self._build_placeholder_ast()
        
        # Respiratory adapter
        adapter_dim = adapter_config.get("adapter_dim", 256)
        num_cycles = adapter_config.get("num_cycles", 4)
        adapter_dropout = adapter_config.get("dropout", 0.2)
        
        self.adapter = RespiratoryAdapter(
            input_dim=self.hidden_size,
            adapter_dim=adapter_dim,
            num_cycles=num_cycles,
            dropout=adapter_dropout,
        )
        
        # Classifier head
        hidden_dims = classifier_config.get("hidden_dims", [768, 256, 128])
        dropout = classifier_config.get("dropout", 0.3)
        
        layers = []
        in_dim = self.hidden_size
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, self.num_classes))
        self.classifier = nn.Sequential(*layers)
        
        # Initialize
        self._init_weights()
        
        if self.freeze_backbone_flag:
            self.freeze_backbone()
    
    def _build_placeholder_ast(self):
        """Build a simplified AST-like model if HuggingFace version unavailable."""
        from transformers import ViTConfig, ViTModel
        config = ViTConfig(
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            image_size=128,
            patch_size=16,
            num_channels=1,
        )
        return ViTModel(config)
    
    def _init_weights(self):
        """Initialize adapter and classifier weights."""
        for module in [self.adapter, self.classifier]:
            for p in module.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: (B, 1, T, F) mel spectrogram or (B, T, F) mel spectrogram
        Returns:
            Logits: (B, num_classes)
        """
        # Ensure correct shape for AST
        if x.dim() == 3:
            x = x.unsqueeze(1)  # (B, 1, T, F)
        
        # AST expects (B, C, H, W) where H=time, W=freq
        # Our input: (B, 1, T, F) — T=time frames, F=mel bins
        # AST uses ViT: patches over (T, F) grid
        
        # Get AST hidden states
        outputs = self.ast(x)
        hidden_states = outputs.last_hidden_state  # (B, N_patches+1, D)
        
        # Remove CLS token for adapter
        patch_features = hidden_states[:, 1:, :]  # (B, N_patches, D)
        
        # Apply respiratory adapter
        adapted_features = self.adapter(patch_features)
        
        # Global pooling
        pooled = adapted_features.mean(dim=1)  # (B, D)
        
        # Classify
        logits = self.classifier(pooled)
        
        return logits
    
    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Get feature embeddings before classifier."""
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        outputs = self.ast(x)
        patch_features = outputs.last_hidden_state[:, 1:, :]
        adapted = self.adapter(patch_features)
        return adapted.mean(dim=1)
    
    def freeze_backbone(self):
        """Freeze AST backbone."""
        for param in self.ast.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze AST backbone."""
        for param in self.ast.parameters():
            param.requires_grad = True
