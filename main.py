import torch
import yaml
import pandas as pd
from pathlib import Path
from torch import nn
from torch.utils.data import DataLoader, random_split
import os

from src.dataset import UrbanDataset
from src.transforms import AudioTransform
from src.model import SoundClassifier
from src.train import train, plot_training, validate

# Load config
ROOT = Path(__file__).parent
with open(ROOT / "configs" / "config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Paths
AUDIO_DIR = ROOT / config["directories"]["audio_data_path"]
METADATA  = ROOT / config["directories"]["metadata"]

# Audio config
SAMPLE_RATE  = config["audio"]["sample_rate"]
NUM_SAMPLES  = SAMPLE_RATE * config["audio"]["duration"]

# Training config
BATCH_SIZE   = config["training"]["batch_size"]
EPOCHS       = config["training"]["epochs"]
LR           = config["training"]["learning_rate"]

# Model config
NUM_CLASSES  = config["model"]["num_classes"]

NUM_WORKERS = max(1, os.cpu_count() - 1)

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"


if __name__ == "__main__":
    print(f"Using device: {device}")

    # Transform
    transform = AudioTransform(
        target_sr=SAMPLE_RATE,
        target_samples=NUM_SAMPLES,
        n_fft=config["audio"]["n_fft"],
        hop_length=config["audio"]["hop_length"],
        n_mels=config["audio"]["n_mels"]
    )

    # Dataset — fold 10 as test, rest as train/val
    metadata_df    = pd.read_csv(METADATA)
    train_val_meta = metadata_df[metadata_df["fold"] != 10]
    test_meta      = metadata_df[metadata_df["fold"] == 10]

    train_val_ds = UrbanDataset(AUDIO_DIR, train_val_meta, transform=transform, device=device)
    test_ds      = UrbanDataset(AUDIO_DIR, test_meta,      transform=transform, device=device)

    # Split train/val
    val_size   = int(len(train_val_ds) * 0.1)
    train_size = len(train_val_ds) - val_size
    train_ds, val_ds = random_split(train_val_ds, [train_size, val_size])

    # DataLoaders
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    test_dl  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    # Model
    model     = SoundClassifier(num_classes=NUM_CLASSES).to(device)
    loss_fn   = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # Train
    history = train(
        model, train_dl, val_dl, loss_fn, optimizer, device, EPOCHS,
        save_path=ROOT / "outputs" / "checkpoints" / "best_model.pth"
    )

    # Plot
    plot_training(
        history,
        save_path=ROOT / "outputs" / "figures" / "training_plot.png"
    )

    # Final test evaluation
    print("Evaluating on test set...")
    test_loss, test_acc = validate(model, test_dl, loss_fn, device)
    print(f"Test loss: {test_loss:.4f} | Test acc: {test_acc:.1f}%")