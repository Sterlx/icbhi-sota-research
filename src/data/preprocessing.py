"""
Audio preprocessing and augmentation pipeline for ICBHI 2017
"""
import numpy as np
import librosa
from typing import Dict, Optional
import torch
import torch.nn as nn


class AudioPreprocessor:
    """Audio preprocessing with configurable pipeline."""
    
    def __init__(self, config: Dict):
        self.sample_rate = config.get("sample_rate", 16000)
        self.mel_bins = config.get("mel_bins", 128)
        self.n_fft = config.get("n_fft", 1024)
        self.hop_length = config.get("hop_length", 160)
        self.win_length = config.get("win_length", 400)
        self.fmin = config.get("fmin", 50)
        self.fmax = config.get("fmax", 2000)
    
    def waveform_to_mel(self, waveform: np.ndarray) -> np.ndarray:
        """Convert waveform to log-mel spectrogram."""
        mel = librosa.feature.melspectrogram(
            y=waveform,
            sr=self.sample_rate,
            n_mels=self.mel_bins,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            fmin=self.fmin,
            fmax=self.fmax,
        )
        return librosa.power_to_db(mel, ref=np.max).astype(np.float32)
    
    def waveform_to_mfcc(self, waveform: np.ndarray, n_mfcc: int = 40) -> np.ndarray:
        """Convert waveform to MFCC features."""
        mfcc = librosa.feature.mfcc(
            y=waveform, sr=self.sample_rate,
            n_mfcc=n_mfcc, n_fft=self.n_fft,
            hop_length=self.hop_length,
        )
        return mfcc.astype(np.float32)


class SpecAugment(nn.Module):
    """SpecAugment for mel spectrograms (time & frequency masking)."""
    
    def __init__(
        self,
        time_mask_param: int = 20,
        freq_mask_param: int = 12,
        num_time_masks: int = 2,
        num_freq_masks: int = 2,
        prob: float = 0.5,
    ):
        super().__init__()
        self.time_mask_param = time_mask_param
        self.freq_mask_param = freq_mask_param
        self.num_time_masks = num_time_masks
        self.num_freq_masks = num_freq_masks
        self.prob = prob
    
    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """Apply SpecAugment.
        
        Args:
            spec: (B, C, T, F) or (C, T, F) spectrogram
        Returns:
            Augmented spectrogram
        """
        if not self.training or torch.rand(1).item() > self.prob:
            return spec
        
        # Handle batch dimension
        single = spec.dim() == 3
        if single:
            spec = spec.unsqueeze(0)
        
        B, C, T, F = spec.shape
        
        for _ in range(self.num_freq_masks):
            f = torch.randint(0, self.freq_mask_param, (1,)).item()
            f0 = torch.randint(0, F - f, (1,)).item()
            spec[:, :, :, f0:f0+f] = spec.mean()
        
        for _ in range(self.num_time_masks):
            t = torch.randint(0, self.time_mask_param, (1,)).item()
            t0 = torch.randint(0, T - t, (1,)).item()
            spec[:, :, t0:t0+t, :] = spec.mean()
        
        if single:
            spec = spec.squeeze(0)
        
        return spec


class MixUp:
    """MixUp augmentation for audio."""
    
    def __init__(self, alpha: float = 0.4):
        self.alpha = alpha
    
    def __call__(self, batch: torch.Tensor, labels: torch.Tensor
                 ) -> tuple:
        """Apply MixUp to a batch.
        
        Args:
            batch: (B, ...) input batch
            labels: (B,) labels
        Returns:
            Mixed batch, mixed labels, lambda
        """
        lam = np.random.beta(self.alpha, self.alpha)
        batch_size = batch.size(0)
        index = torch.randperm(batch_size)
        
        mixed_batch = lam * batch + (1 - lam) * batch[index]
        mixed_labels = (labels, labels[index], lam)
        
        return mixed_batch, mixed_labels
