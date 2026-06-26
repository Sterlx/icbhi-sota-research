"""
Whisper Encoder with Medical Adapter for lung sound classification.
"""
import torch
import torch.nn as nn
from typing import Dict
from .base_model import BaseLungSoundModel, register_model


@register_model("WhisperMedical")
class WhisperMedical(BaseLungSoundModel):
    """Whisper encoder adapted for medical audio classification.
    
    Uses frozen Whisper encoder + trainable medical adapter.
    Efficient: only adapter & classifier are trained.
    """
    
    def __init__(self, config: Dict):
        super().__init__(config)
        
        whisper_size = config.get("whisper_size", "small")  # tiny, small, medium
        adapter_config = config.get("adapter", {})
        classifier_config = config.get("classifier", {})
        
        self.freeze_backbone_flag = config.get("freeze_backbone", True)
        
        # Load Whisper encoder
        try:
            from transformers import WhisperModel
            model_name = f"openai/whisper-{whisper_size}"
            self.whisper = WhisperModel.from_pretrained(model_name)
            encoder_dim = self.whisper.config.d_model
            
            # We only need the encoder
            self.whisper.decoder = None
            
        except Exception as e:
            print(f"Warning: Could not load Whisper ({e})")
            encoder_dim = 768 if whisper_size == "small" else 384
            self.whisper = None
        
        self.encoder_dim = encoder_dim
        
        # Medical adapter (LoRA-style)
        adapter_dim = adapter_config.get("adapter_dim", 128)
        self.adapter_down = nn.Linear(encoder_dim, adapter_dim, bias=False)
        self.adapter_act = nn.GELU()
        self.adapter_up = nn.Linear(adapter_dim, encoder_dim, bias=False)
        self.adapter_norm = nn.LayerNorm(encoder_dim)
        
        # Initialize adapter with small values
        nn.init.normal_(self.adapter_down.weight, std=0.02)
        nn.init.zeros_(self.adapter_up.weight)
        
        # Classifier
        hidden_dims = classifier_config.get("hidden_dims", [512, 256])
        dropout = classifier_config.get("dropout", 0.3)
        
        layers = []
        in_dim = encoder_dim
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
        
        if self.freeze_backbone_flag:
            self.freeze_backbone()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: (B, T) waveform or (B, 1, T, F) mel spectrogram
        Returns:
            Logits: (B, num_classes)
        """
        if self.whisper is not None:
            # Whisper expects mel features: our preprocessing already gives mel
            # Use Whisper's feature extractor via the model
            if x.dim() == 4:
                x = x.squeeze(1)  # (B, T, F)
            
            # Get encoder outputs
            encoder_outputs = self.whisper.encoder(x)
            hidden = encoder_outputs.last_hidden_state  # (B, N, D)
        else:
            # Placeholder
            hidden = x.mean(dim=1, keepdim=True).expand(-1, 50, -1)
            if hidden.shape[-1] != self.encoder_dim:
                hidden = nn.Linear(hidden.shape[-1], self.encoder_dim).to(x.device)(hidden)
        
        # Apply medical adapter (parallel to frozen backbone)
        adapted = self.adapter_down(hidden)
        adapted = self.adapter_act(adapted)
        adapted = self.adapter_up(adapted)
        adapted = self.adapter_norm(hidden + adapted)
        
        # Global pooling
        pooled = adapted.mean(dim=1)  # (B, D)
        
        # Classify
        logits = self.classifier(pooled)
        
        return logits
    
    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Get adapted embeddings."""
        if self.whisper is not None:
            if x.dim() == 4:
                x = x.squeeze(1)
            encoder_outputs = self.whisper.encoder(x)
            hidden = encoder_outputs.last_hidden_state
        else:
            hidden = x.mean(dim=1, keepdim=True).expand(-1, 50, -1)
        
        adapted = self.adapter_down(hidden)
        adapted = self.adapter_act(adapted)
        adapted = self.adapter_up(adapted)
        adapted = self.adapter_norm(hidden + adapted)
        
        return adapted.mean(dim=1)
    
    def freeze_backbone(self):
        if self.whisper is not None:
            for param in self.whisper.parameters():
                param.requires_grad = False
    
    def unfreeze_backbone(self):
        if self.whisper is not None:
            for param in self.whisper.parameters():
                param.requires_grad = True
