import os
import random
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from constants import ConstantsModel
from dataloader import SleepFeatureDataset
from model import SleepLSTM
from visu_results import plot_loss_curve, plot_roc_rem_vs_nrem


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class SleepStageTrainer:
    """
    Trainer class for sleep stage classification using an LSTM-based model.

    Attributes:
        cfg (DictConfig): Configuration loaded using Hydra.
        device (torch.device): Device on which training/evaluation runs.
        model (nn.Module): LSTM-based classifier for sleep stage prediction.
        criterion (nn.Module): Loss function (CrossEntropyLoss).
        optimizer (torch.optim.Optimizer): Optimizer (Adam).
        scheduler (StepLR): Learning rate scheduler.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        epoch_losses (List[float]): Stores loss per epoch.
        current_epoch (int): the current epoch
    """

    def __init__(self, cfg: DictConfig):
        """
        Initialize the training pipeline.

        Args:
            cfg (DictConfig): Configuration object containing model, data, and training parameters.
        """
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SleepLSTM(cfg.data_generator.input_size).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.model.lr)
        self.scheduler = StepLR(self.optimizer, step_size=15, gamma=0.5)
        self.train_loader, self.val_loader = self._prepare_data()
        self.epoch_losses = []

        self.current_epoch = 0
        ckp_path = Path(cfg.ckp_path) if cfg.ckp_path else None
        if ckp_path and ckp_path.exists():
            checkpoint = torch.load(ckp_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            self.current_epoch = checkpoint.get("epoch", 0)
            # print("self.current_epoch ", self.current_epoch)
            print(f"Resumed from epoch {self.current_epoch}")

    def _prepare_data(self) -> Tuple[DataLoader, DataLoader]:
        """
        Split the dataset into training and validation sets and return corresponding DataLoaders.

        Returns:
            Tuple[DataLoader, DataLoader]: Training and validation DataLoaders.
        """
        dataset = SleepFeatureDataset(self.cfg.data_generator.feature_dir)
        train_set, val_set = train_test_split(
            dataset, test_size=0.1, random_state=self.cfg.general.seed, shuffle=True
        )
        train_loader = DataLoader(
            train_set, batch_size=ConstantsModel.BATCH_SIZE, shuffle=True
        )
        val_loader = DataLoader(val_set, batch_size=1)
        return train_loader, val_loader

    def training(self) -> None:
        """
        Train the LSTM model using training data and save the trained model.
        """
        last_epoch = self.current_epoch
        for epoch in range(self.current_epoch, self.cfg.model.epochs):
            self.model.train()
            total_loss = 0.0
            for features, labels, _ in self.train_loader:
                features, labels = features.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(features).view(-1, self.cfg.general.num_classes)
                labels = labels.view(-1)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            self.scheduler.step()
            self.epoch_losses.append(total_loss)
            print(
                f"Epoch [{epoch + 1}/{self.cfg.model.epochs}], Loss: {total_loss:.4f}"
            )
            last_epoch = epoch + 1  # <-- update at end of each loop
        os.makedirs(self.cfg.logging.save_path, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "epoch": last_epoch,
            },
            os.path.join(self.cfg.logging.save_path, "trained_sleep_lstm.pth"),
        )
        # torch.save(self.model.state_dict(), os.path.join(self.cfg.logging.save_path, "trained_sleep_lstm.pth"))
        print("Model saved to trained_sleep_lstm.pth")
        plot_loss_curve(self.epoch_losses)

    def validate(self) -> None:
        """
        Evaluate the trained model using validation data and print classification metrics.
        """
        self.model.eval()
        all_preds, all_labels, all_probs = [], [], []

        with torch.no_grad():
            for features, labels, _ in self.val_loader:
                features, labels = features.to(self.device), labels.to(self.device)
                outputs = self.model(features)
                probs = F.softmax(outputs, dim=-1)[0]
                preds = torch.argmax(outputs, dim=-1)
                all_preds.extend(preds.view(-1).cpu().numpy())
                all_labels.extend(labels.view(-1).cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        report = classification_report(
            all_labels,
            all_preds,
            labels=list(ConstantsModel.CLASSES.keys()),
            zero_division=0,
        )
        acc = accuracy_score(all_labels, all_preds)
        plot_roc_rem_vs_nrem(np.array(all_labels), np.array(all_probs))

        print("Classification Report:")
        print("\n" + report)
        print(f"Validation Accuracy: {acc * 100:.2f}%")
