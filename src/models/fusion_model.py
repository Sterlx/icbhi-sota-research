"""
Multi-Model Cross-Attention Fusion Architecture.
Combines multiple pretrained audio models with cross-attention fusion.
"""
import torch
import torch.nn as nn
from typing import Dict, List, Optional
from .base_model import BaseLungSoundModel, register_model


class CrossAttentionFusion(nn.Module):
    """Cross-attention fusion of multiple model representations."""
    
    def __init__(
        self,
        dims: List[int],
        fusion_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_models = len(dims)
        
        # Project each model to common dimension
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d, fusion_dim),
                nn.LayerNorm(fusion_dim),
                nn.GELU(),
            ) for d in dims
        ])
        
        # Cross-attention between model representations
        self.cross_attention = nn.ModuleList([
            nn.MultiheadAttention(
                fusion_dim, num_heads, dropout=dropout, batch_first=True
            ) for _ in range(self.num_models)
        ])
        
        self.norm = nn.LayerNorm(fusion_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """Fuse multiple model features with cross-attention.
        
        Args:
            features: List of (B, D_i) features from different models
        Returns:
            (B, fusion_dim) fused representation
        """
        # Project to common dimension
        projected = [
            proj(f) for f, proj in zip(features, self.projections)
        ]  # Each: (B, fusion_dim)
        
        # Stack as sequence: (B, num_models, fusion_dim)
        stacked = torch.stack(projected, dim=1)
        
        # Cross-attention: each model attends to all others
        attended = []
        for i, attn in enumerate(self.cross_attention):
            out, _ = attn(stacked[:, i:i+1, :], stacked, stacked)
            attended.append(out.squeeze(1))
        
        # Average fusion
        fused = torch.stack(attended, dim=1).mean(dim=1)  # (B, fusion_dim)
        fused = self.norm(fused)
        
        return fused


@register_model("MultiModelFusion")
class MultiModelFusion(BaseLungSoundModel):
    """Multi-model fusion architecture.
    
    Combines multiple pretrained audio models with cross-attention:
        - AST (mel spectrogram understanding)
        - wav2vec2 (waveform features)
        - EfficientNet (spectral patterns)
    """
    
    def __init__(self, config: Dict):
        super().__init__(config)
        
        fusion_config = config.get("fusion", {})
        classifier_config = config.get("classifier", {})
        fusion_dim = fusion_config.get("fusion_dim", 512)
        
        # Model dimensions
        self.model_dims = []
        self.models = nn.ModuleDict()
        
        # AST branch
        if config.get("use_ast", True):
            try:
                from transformers import ASTModel
                self.models["ast"] = ASTModel.from_pretrained(
                    "MIT/ast-finetuned-audioset-10-10-0.4593"
                )
                self.model_dims.append(self.models["ast"].config.hidden_size)
            except:
                self.model_dims.append(768)
        
        # wav2vec2 branch
        if config.get("use_wav2vec2", True):
            try:
                from transformers import Wav2Vec2Model
                self.models["wav2vec2"] = Wav2Vec2Model.from_pretrained(
                    "facebook/wav2vec2-base"
                )
                self.model_dims.append(
                    self.models["wav2vec2"].config.hidden_size
                )
            except:
                self.model_dims.append(768)
        
        # EfficientNet branch (as spectral encoder)
        if config.get("use_efficientnet", True):
            try:
                import timm
                self.models["efficientnet"] = timm.create_model(
                    "efficientnet_b0", pretrained=True, num_classes=0
                )
                self.model_dims.append(1280)  # efficientnet_b0 feature dim
            except:
                self.model_dims.append(1280)
        
        # Cross-attention fusion
        self.fusion = CrossAttentionFusion(
            dims=self.model_dims,
            fusion_dim=fusion_dim,
            num_heads=fusion_config.get("num_heads", 8),
            dropout=fusion_config.get("dropout", 0.2),
        )
        
        # Classifier
        hidden_dims = classifier_config.get("hidden_dims", [512, 256, 128])
        dropout = classifier_config.get("dropout", 0.3)
        
        layers = []
        in_dim = fusion_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, self.num_classes))
        self.classifier = nn.Sequential(*layers)
    
    def _extract_ast_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract AST features from mel spectrogram."""
        outputs = self.models["ast"](x)
        return outputs.pooler_output  # (B, D)
    
    def _extract_wav2vec2_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract wav2vec2 features from waveform."""
        outputs = self.models["wav2vec2"](x)
        return outputs.last_hidden_state.mean(dim=1)  # (B, D)
    
    def _extract_efficientnet_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract EfficientNet features from spectrogram."""
        # EfficientNet expects 3-channel input
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        return self.models["efficientnet"](x)  # (B, D)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. x should be a tuple of (mel_spec, waveform)."""
        if isinstance(x, (list, tuple)):
            mel_spec, waveform = x
        else:
            # If only one input, assume mel spectrogram
            mel_spec = x
            waveform = x  # Fallback
        
        features = []
        
        if "ast" in self.models:
            features.append(self._extract_ast_features(mel_spec))
        
        if "wav2vec2" in self.models:
            features.append(self._extract_wav2vec2_features(waveform))
        
        if "efficientnet" in self.models:
            features.append(self._extract_efficientnet_features(
                mel_spec.unsqueeze(1) if mel_spec.dim() == 3 else mel_spec
            ))
        
        # Fuse features
        fused = self.fusion(features)
        
        # Classify
        logits = self.classifier(fused)
        
        return logits
    
    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(x, (list, tuple)):
            mel_spec, waveform = x
        else:
            mel_spec = waveform = x
        
        features = []
        if "ast" in self.models:
            features.append(self._extract_ast_features(mel_spec))
        if "wav2vec2" in self.models:
            features.append(self._extract_wav2vec2_features(waveform))
        if "efficientnet" in self.models:
            features.append(self._extract_efficientnet_features(
                mel_spec.unsqueeze(1) if mel_spec.dim() == 3 else mel_spec
            ))
        
        return self.fusion(features)
    
    def freeze_backbone(self):
        for model in self.models.values():
            for param in model.parameters():
                param.requires_grad = False
    
    def unfreeze_backbone(self):
        for model in self.models.values():
            for param in model.parameters():
                param.requires_grad = True
