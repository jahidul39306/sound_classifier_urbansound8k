import torch
import matplotlib.pyplot as plt
import os
from torch import nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from pathlib import Path

ROOT = Path(__file__).parent.parent


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
        loop.set_postfix(loss=f"{loss.item():.4f}")
    avg_loss = total_loss / len(data_loader)
    accuracy = correct / len(data_loader.dataset) * 100
    return avg_loss, accuracy


def validate(model, data_loader, loss_fn, device):
    model.eval()
    total_loss, correct = 0, 0
    with torch.no_grad():
        for inputs, targets in tqdm(data_loader, desc="Validating", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            predictions = model(inputs)
            total_loss += loss_fn(predictions, targets).item()
            correct += (predictions.argmax(dim=1) == targets).sum().item()
    avg_loss = total_loss / len(data_loader)
    accuracy = correct / len(data_loader.dataset) * 100
    return avg_loss, accuracy


def train(model, train_dl, val_dl, loss_fn, optimizer, device, epochs, save_path):
    best_val_loss = float("inf")
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in tqdm(range(1, epochs + 1), desc="Epochs"):
        train_loss, train_acc = train_single_epoch(
            model, train_dl, loss_fn, optimizer, device
        )
        val_loss, val_acc = validate(model, val_dl, loss_fn, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        tqdm.write(
            f"Epoch {epoch}/{epochs} | "
            f"Train loss: {train_loss:.4f}, acc: {train_acc:.1f}% | "
            f"Val loss: {val_loss:.4f}, acc: {val_acc:.1f}%"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            tqdm.write(f"  ✔ Model saved (val_loss: {val_loss:.4f})")

    print("Training complete.")
    return history


def plot_training(history, save_path):
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Loss plot
    ax1.plot(epochs, history["train_loss"], label="Train loss")
    ax1.plot(epochs, history["val_loss"], label="Val loss")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)

    # Accuracy plot
    ax2.plot(epochs, history["train_acc"], label="Train acc")
    ax2.plot(epochs, history["val_acc"], label="Val acc")
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"Training plot saved to {save_path}")
