import random
from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR

from constants import ConstantsModel
from dataloader import SleepFeatureDataset
from model import SleepLSTM
from visu_results import plot_roc_rem_vs_nrem, plot_loss_curve


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

class SleepStageTrainer:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SleepLSTM(cfg.data_generator.input_size).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.model.lr)
        self.scheduler = StepLR(self.optimizer, step_size=15, gamma=0.5)
        self.train_loader, self.val_loader = self._prepare_data()
        self.epoch_losses = []

    def _prepare_data(self):
        dataset = SleepFeatureDataset(self.cfg.data_generator.feature_dir)
        train_set, val_set = train_test_split(
            dataset, test_size=0.1, random_state=self.cfg.general.seed, shuffle=True
        )
        train_loader = DataLoader(train_set, batch_size=ConstantsModel.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=1)
        return train_loader, val_loader

    def training(self):
        for epoch in range(self.cfg.model.epochs):
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
            print(f"Epoch [{epoch + 1}/{self.cfg.model.epochs}], Loss: {total_loss:.4f}")

        torch.save(self.model.state_dict(), "trained_sleep_lstm.pth")
        print("Model saved to trained_sleep_lstm.pth")
        plot_loss_curve(self.epoch_losses)

    def validate(self):
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

        report = classification_report(all_labels, all_preds, labels=list(ConstantsModel.CLASSES.keys()), zero_division=0)
        acc = accuracy_score(all_labels, all_preds)
        plot_roc_rem_vs_nrem(np.array(all_labels), np.array(all_probs))

        print("Classification Report:")
        print("\n" + report)
        print(f"Validation Accuracy: {acc * 100:.2f}%")

def run_training(cfg: DictConfig) -> None:
    set_seed(cfg.general.seed)
    trainer = SleepStageTrainer(cfg)
    trainer.training()
    trainer.validate()
