import os

import torch
import torch
from torch.utils.data import Dataset
import pandas as pd
import torchaudio
import yaml
from pathlib import Path
from transforms import AudioTransform

with open("../configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

DS_PATH = Path(config["directories"]["audio_data_path"])
METADATA = Path(config["directories"]["metadata"])


class UrbanDataset(Dataset):

    def __init__(self, audio_path, metadata, transform=None, device="cpu"):
        self.audio_path = audio_path
        self.metadata = pd.read_csv(metadata)
        self.device = device
        self.transform = transform.to(device) if transform else None

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        fold = f"fold{self.metadata.iloc[index, 5]}"
        path = os.path.join(self.audio_path, fold, self.metadata.iloc[index, 0])
        label = self.metadata.iloc[index, 6]
        signal, sr = torchaudio.load(path)
        if self.transform:
            signal = self.transform(signal, sr)
        return signal, label


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device {device}")
    
    transformer = AudioTransform(
        target_sr=22050, target_samples=22050, n_fft=1024, hop_length=512, n_mels=64
    )
    ds = UrbanDataset(DS_PATH, METADATA, transform=transformer)
    print(f"Total number of samples: {len(ds)}")
    signal, label = ds[0]
    print(f"Signal shape: {signal.shape}, Label: {label}")
    print(f"{signal}")
