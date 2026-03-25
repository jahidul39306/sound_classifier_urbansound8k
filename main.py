import torch
import pandas as pd
from pathlib import Path
from torch import nn
from torch.utils.data import DataLoader, random_split

from src.dataset import UrbanDataset
from src.transforms import AudioTransform
from src.model import SoundClassifier
from src.train import train, plot_training, validate
from src.utils import (
    load_config,
    get_device,
    set_seed,
    get_num_workers,
    train_val_split,
)

# Load config
ROOT = Path(__file__).parent
config = load_config()

# Paths
AUDIO_DIR = ROOT / config["directories"]["audio_data_path"]
METADATA = ROOT / config["directories"]["metadata"]

# Audio config
SAMPLE_RATE = config["audio"]["sample_rate"]
NUM_SAMPLES = SAMPLE_RATE * config["audio"]["duration"]

# Training config
BATCH_SIZE = config["training"]["batch_size"]
EPOCHS = config["training"]["epochs"]
LR = config["training"]["lr"]

# Model config
NUM_CLASSES = config["model"]["num_classes"]

NUM_WORKERS = get_num_workers()

# Device
device = get_device(config["training"]["device"])


if __name__ == "__main__":
    set_seed(42)
    print(f"Using device: {device}")

    # Transform
    transform = AudioTransform(
        target_sr=SAMPLE_RATE,
        target_samples=NUM_SAMPLES,
        n_fft=config["audio"]["n_fft"],
        hop_length=config["audio"]["hop_length"],
        n_mels=config["audio"]["n_mels"],
    )

    # Dataset — fold 10 as test, rest as train/val
    metadata_df = pd.read_csv(METADATA)
    train_val_meta = metadata_df[metadata_df["fold"] != 10].reset_index(drop=True)
    test_meta = metadata_df[metadata_df["fold"] == 10].reset_index(drop=True)

    train_val_ds = UrbanDataset(
        AUDIO_DIR, train_val_meta, transform=transform
    )
    test_ds = UrbanDataset(AUDIO_DIR, test_meta, transform=transform)

    # Split train/val
    train_ds, val_ds = train_val_split(train_val_ds, val_split=0.1)

    # DataLoaders
    train_dl = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # Model
    model = SoundClassifier(num_classes=NUM_CLASSES).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # Train
    history = train(
        model,
        train_dl,
        val_dl,
        loss_fn,
        optimizer,
        device,
        EPOCHS,
        save_path=ROOT / "outputs" / "checkpoints" / "best_model.pth",
    )

    # Plot
    plot_training(history, save_path=ROOT / "outputs" / "figures" / "training_plot.png")

    # Load best model
    checkpoint_path = ROOT / "outputs" / "checkpoints" / "best_model.pth"

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    # Final test evaluation
    print("Evaluating on BEST saved model...")
    test_loss, test_acc = validate(model, test_dl, loss_fn, device)
    print(f"Test loss: {test_loss:.4f} | Test acc: {test_acc:.1f}%")
