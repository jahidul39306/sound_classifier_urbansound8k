# src/utils.py

import os
import random
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import random_split
import yaml


def load_config(config_path=None):
    if config_path is None:
        config_path = Path(__file__).parent.parent / "configs" / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_device(preference="auto"):
    if preference == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return preference


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_num_workers():
    return max(1, os.cpu_count() - 1)


def train_val_split(dataset, val_split=0.1):
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    return random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )
