import random
from typing import List, Tuple

import numpy as np
import logging
import torch
import torch.nn as nn
from omegaconf import DictConfig
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import train_test_split

from constants import ConstantsModel
from dataloader import SleepFeatureDataset
from model import SleepLSTM

device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed: int) -> None:
    """Ensure reproducibility across runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def train_one_epoch(model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_classes: int
) -> float:
    """Perform one training epoch."""
    model.train()
    total_loss = 0.0

    for features, labels, _ in train_loader:
        features, labels = features.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(features)
        outputs = outputs.view(-1, num_classes)
        labels = labels.view(-1)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss

def evaluate_model(model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    num_classes: int
) -> Tuple[str, float]:
    """Evaluate the model and return metrics."""
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for features, labels, _ in val_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            preds = torch.argmax(outputs, dim=-1)

            all_preds.extend(preds.view(-1).cpu().numpy())
            all_labels.extend(labels.view(-1).cpu().numpy())

    report = classification_report(all_labels, all_preds, labels=[0, 1, 2, 3, 5], zero_division=0)
    acc = accuracy_score(all_labels, all_preds)
    return report, acc

def run_training(cfg: DictConfig) -> None:
    """
    Run the full training and evaluation loop using config from Hydra.

    Args:
        cfg: Hydra configuration object.
    """
    set_seed(cfg.general.seed)

    dataset = SleepFeatureDataset(cfg.data_generator.feature_dir)

    train_dataset, val_dataset = train_test_split(
        dataset,
        test_size=0.1,
        random_state=cfg.general.seed,
        shuffle=True,
    )

    train_loader = DataLoader(train_dataset, batch_size=ConstantsModel.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1)

    model = SleepLSTM(cfg.data_generator.input_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.model.lr)

    for epoch in range(cfg.model.epochs):
        loss = train_one_epoch(model, train_loader, criterion, optimizer, device, cfg.general.num_classes)
        print(f"Epoch [{epoch + 1}/{cfg.model.epochs}], Loss: {loss:.4f}")

    print("\nEvaluating model...")
    report, acc = evaluate_model(model, val_loader, device, cfg.general.num_classes)
    print("Classification Report:")
    print(report)
    print(f"Validation Accuracy: {acc * 100:.2f}%")
