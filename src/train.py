import torch
import torchaudio
from torch import nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import yaml
from pathlib import Path
from dataset import UrbanDataset
from model import SoundClassifier
from transforms import AudioTransform
import pandas as pd
import os


CONFIG_PATH = Path(__file__).parent.parent / "configs" / "config.yaml"
ROOT = Path(__file__).parent.parent

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

DS_PATH = ROOT / config["directories"]["audio_data_path"]
METADATA = ROOT / config["directories"]["metadata"]
SAMPLE_RATE = config["audio"]["sample_rate"]
DURATION = config["audio"]["duration"]
N_FFT = config["audio"]["n_fft"]
HOP_LENGTH = config["audio"]["hop_length"]
N_MELS = config["audio"]["n_mels"]
BATCH_SIZE = config["training"]["batch_size"]
EPOCHS = config["training"]["epochs"]
LR = config["training"]["lr"]
NUM_CLASSES = config["model"]["num_classes"]


def create_dataloader(dataset, batch_size, val_split=0.2):
    num_workers = max(1, os.cpu_count() - 1)
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    train_dl = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_dl = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    return train_dl, val_dl


def train_single_epoch(model, data_loader, loss_fn, optimizer, device):
    model.train()
    total_loss, correct = 0, 0
    loop = tqdm(data_loader, desc="Training", leave=False)
    for inputs, targets in loop:
        inputs, targets = inputs.to(device), targets.to(device)
        predictions = model(inputs)
        loss = loss_fn(predictions, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (predictions.argmax(dim=1) == targets).sum().item()
    avg_loss = total_loss / len(data_loader)
    accuracy = correct / len(data_loader.dataset) * 100
    return avg_loss, accuracy


def validate(model, data_loader, loss_fn, device):
    model.eval()
    total_loss, correct = 0, 0
    loop = tqdm(data_loader, desc="Validating", leave=False)
    with torch.no_grad():
        for inputs, targets in loop:
            inputs, targets = inputs.to(device), targets.to(device)
            predictions = model(inputs)
            total_loss += loss_fn(predictions, targets).item()
            correct += (predictions.argmax(dim=1) == targets).sum().item()
    avg_loss = total_loss / len(data_loader)
    accuracy = correct / len(data_loader.dataset) * 100
    return avg_loss, accuracy


def train(model, train_dl, val_dl, loss_fn, optimizer, device, epochs):
    epoch_bar = tqdm(range(1, epochs + 1), desc="Epochs")
    for epoch in epoch_bar:
        train_loss, train_acc = train_single_epoch(
            model, train_dl, loss_fn, optimizer, device
        )
        val_loss, val_acc = validate(model, val_dl, loss_fn, device)
        print(
            f"Epoch {epoch}/{epochs} | "
            f"Train loss: {train_loss:.4f}, acc: {train_acc:.1f}% | "
            f"Val loss: {val_loss:.4f}, acc: {val_acc:.1f}%"
        )
    print("Training complete")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    transfrom = AudioTransform(
        target_sr=SAMPLE_RATE,
        target_samples=SAMPLE_RATE * DURATION,
        n_fft=N_FFT,
        n_mels=N_MELS,
        hop_length=HOP_LENGTH,
    )
    metadata_df = pd.read_csv(METADATA)
    train_val_meta = metadata_df[metadata_df["fold"] != 10]
    ds = UrbanDataset(DS_PATH, train_val_meta, transfrom, device)
    print("Creating dataloaders... ")
    train_dl, val_dl = create_dataloader(ds, BATCH_SIZE)
    print("Dataloaders created successfully")
    model = SoundClassifier(NUM_CLASSES).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    train(model, train_dl, val_dl, loss_fn, optimizer, device, EPOCHS)
    torch.save(model.state_dict(), "../outputs/checkpoints/model.pth")
    print("Model saved to outputs/checkpoints/model.pth")
