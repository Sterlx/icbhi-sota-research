"""
ICBHI 2017 Dataset Loader
Uses OFFICIAL recording-level 60/40 split, per respiratory cycle detection
"""
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import librosa
import soundfile as sf
from typing import Tuple, Optional


class ICBHIDataset(Dataset):
    """ICBHI 2017 respiratory sound dataset with cycle-level annotations.
    
    Uses the OFFICIAL recording-level train/test split.
    IMPORTANT: The same patient may appear in both train and test
    (different recordings) — this is the official ICBHI protocol.
    
    Classes:
        0: Normal (no crackles, no wheezes)
        1: Wheeze
        2: Crackles
        3: Both (wheezes + crackles)
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = "train",  # "train" or "test"
        split_file: str = None,  # Path to official_split.txt
        sample_rate: int = 16000,
        cycle_duration: float = 4.0,
        augment: bool = False,
        return_waveform: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.sample_rate = sample_rate
        self.cycle_duration = cycle_duration
        self.cycle_length = int(sample_rate * cycle_duration)
        self.return_waveform = return_waveform
        self.augment = augment
        
        # Load official split
        self.split_recordings = self._load_official_split(split_file)
        
        # Parse all annotations
        self.annotations = self._parse_annotations()
        
        # Filter to current split by recording name
        self.cycles = self.annotations[
            self.annotations["recording_id"].isin(self.split_recordings)
        ].reset_index(drop=True)
        
        unique_patients = set(r.split("_")[0] for r in self.split_recordings)
        print(f"[{split.upper()}] {len(self.cycles)} cycles from "
              f"{len(self.split_recordings)} recordings ({len(unique_patients)} patients)")
        self._print_class_distribution()
    
    def _load_official_split(self, split_file: str = None) -> set:
        """Load the official ICBHI recording-level train/test split.
        
        Format: recording_name\ttrain|test
        Example: 101_1b1_Al_sc_Meditron\ttest
        
        If no split_file provided, looks for 'official_split.txt' in data_dir.
        """
        if split_file is None:
            split_file = self.data_dir / "official_split.txt"
        split_file = Path(split_file)
        
        if not split_file.exists():
            raise FileNotFoundError(
                f"Official split file not found: {split_file}\n"
                "Place 'official_split.txt' in the dataset directory."
            )
        
        recordings = set()
        with open(split_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    recording_name, split_label = parts[0], parts[1]
                    if split_label == self.split:
                        recordings.add(recording_name)
        
        return recordings
    
    def _parse_annotations(self) -> pd.DataFrame:
        """Parse ICBHI annotation files."""
        records = []
        txt_files = list(self.data_dir.glob("*.txt"))
        
        for txt_file in txt_files:
            if "_Log" in txt_file.name:
                continue
            
            recording_id = txt_file.stem  # e.g., "101_1b1_Al_sc_Meditron"
            audio_file = self.data_dir / f"{recording_id}.wav"
            if not audio_file.exists():
                continue
            
            try:
                with open(txt_file, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) < 4:
                            continue
                        start, end, crackles, wheezes = (
                            float(parts[0]), float(parts[1]),
                            int(parts[2]), int(parts[3])
                        )
                        label = self._get_label(crackles, wheezes)
                        records.append({
                            "recording_id": recording_id,
                            "patient_id": recording_id.split("_")[0],  # Extract patient number
                            "audio_file": str(audio_file),
                            "start": start,
                            "end": end,
                            "duration": end - start,
                            "crackles": crackles,
                            "wheezes": wheezes,
                            "label": label,
                        })
            except Exception as e:
                print(f"Warning: could not parse {txt_file}: {e}")
        
        return pd.DataFrame(records)
    
    @staticmethod
    def _get_label(crackles: int, wheezes: int) -> int:
        if crackles == 0 and wheezes == 0:
            return 0  # Normal
        elif crackles == 0 and wheezes == 1:
            return 1  # Wheeze
        elif crackles == 1 and wheezes == 0:
            return 2  # Crackles
        else:
            return 3  # Both
    
    def _print_class_distribution(self):
        """Print class distribution for the current split."""
        for label, name in enumerate(["Normal", "Wheeze", "Crackles", "Both"]):
            count = (self.cycles["label"] == label).sum()
            pct = count / len(self.cycles) * 100 if len(self.cycles) > 0 else 0
            print(f"  {name}: {count} ({pct:.1f}%)")
    
    def _load_audio(self, audio_path: str, start: float, end: float) -> np.ndarray:
        """Load and preprocess audio segment."""
        duration = end - start
        
        try:
            audio, orig_sr = sf.read(audio_path, start=int(start * 44100), 
                                      stop=int(end * 44100) + 1)
        except:
            # Fallback: load with librosa
            audio, orig_sr = librosa.load(audio_path, sr=None, 
                                           offset=start, duration=duration + 0.1)
        
        # Convert to mono
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        
        # Guard against empty audio (zero-duration cycles)
        if len(audio) == 0:
            audio = np.zeros(self.cycle_length, dtype=np.float32)
            return audio
        
        # Handle different sample rates
        if orig_sr != self.sample_rate:
            audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=self.sample_rate)
        
        # Bandpass filter (lung sound range: 100-2000 Hz)
        audio = self._bandpass_filter(audio)
        
        # Normalize
        if np.abs(audio).max() > 0:
            audio = audio / np.abs(audio).max()
        
        # Pad or truncate to fixed length
        if len(audio) < self.cycle_length:
            audio = np.pad(audio, (0, self.cycle_length - len(audio)))
        else:
            audio = audio[:self.cycle_length]
        
        return audio.astype(np.float32)
    
    def _bandpass_filter(self, audio: np.ndarray, 
                          lowcut: float = 100.0, 
                          highcut: float = 2000.0) -> np.ndarray:
        """Apply bandpass filter for lung sound frequencies."""
        if len(audio) < 30:  # Too short for filter — skip
            return audio
        try:
            from scipy.signal import butter, sosfiltfilt
            nyquist = self.sample_rate / 2
            low = lowcut / nyquist
            high = highcut / nyquist
            sos = butter(4, [low, high], btype="band", output="sos")
            return sosfiltfilt(sos, audio)
        except ImportError:
            return audio
    
    def _augment_audio(self, audio: np.ndarray) -> np.ndarray:
        """Apply audio augmentations."""
        if not self.augment:
            return audio
        
        aug_type = np.random.choice(["none", "noise", "stretch", "pitch"])
        
        if aug_type == "noise":
            noise_level = np.random.uniform(0.001, 0.01)
            audio = audio + np.random.randn(len(audio)) * noise_level
        
        elif aug_type == "stretch":
            rate = np.random.uniform(0.9, 1.1)
            audio = librosa.effects.time_stretch(audio, rate=rate)
            if len(audio) > self.cycle_length:
                audio = audio[:self.cycle_length]
            elif len(audio) < self.cycle_length:
                audio = np.pad(audio, (0, self.cycle_length - len(audio)))
        
        elif aug_type == "pitch":
            n_steps = np.random.uniform(-2, 2)
            audio = librosa.effects.pitch_shift(
                audio, sr=self.sample_rate, n_steps=n_steps
            )
        
        return audio
    
    def __len__(self) -> int:
        return len(self.cycles)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.cycles.iloc[idx]
        
        # Load audio segment
        audio = self._load_audio(row["audio_file"], row["start"], row["end"])
        
        # Augment (training only)
        audio = self._augment_audio(audio)
        
        label = row["label"]
        
        if self.return_waveform:
            return torch.from_numpy(audio), torch.tensor(label, dtype=torch.long)
        else:
            # Return mel spectrogram
            mel_spec = self._waveform_to_mel(audio)
            return torch.from_numpy(mel_spec), torch.tensor(label, dtype=torch.long)
    
    def _waveform_to_mel(self, audio: np.ndarray) -> np.ndarray:
        """Convert waveform to mel spectrogram."""
        mel = librosa.feature.melspectrogram(
            y=audio, sr=self.sample_rate,
            n_mels=128, n_fft=1024,
            hop_length=160, win_length=400,
            fmin=50, fmax=2000,
        )
        log_mel = librosa.power_to_db(mel, ref=np.max)
        return log_mel.astype(np.float32)
    
    def get_class_weights(self) -> torch.Tensor:
        """Compute class weights for weighted loss."""
        counts = self.cycles["label"].value_counts().sort_index()
        weights = 1.0 / counts.values
        weights = weights / weights.sum() * len(counts)
        return torch.tensor(weights, dtype=torch.float32)


def create_dataloaders(
    data_dir: str,
    split_file: str = None,
    batch_size: int = 32,
    sample_rate: int = 16000,
    cycle_duration: float = 4.0,
    num_workers: int = 4,
    return_waveform: bool = False,
) -> Tuple[DataLoader, DataLoader, torch.Tensor]:
    """Create train and test dataloaders using the OFFICIAL recording-level split."""
    
    train_dataset = ICBHIDataset(
        data_dir=data_dir,
        split="train",
        split_file=split_file,
        sample_rate=sample_rate,
        cycle_duration=cycle_duration,
        augment=True,
        return_waveform=return_waveform,
    )
    
    test_dataset = ICBHIDataset(
        data_dir=data_dir,
        split="test",
        split_file=split_file,
        sample_rate=sample_rate,
        cycle_duration=cycle_duration,
        augment=False,
        return_waveform=return_waveform,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    class_weights = train_dataset.get_class_weights()
    
    return train_loader, test_loader, class_weights
