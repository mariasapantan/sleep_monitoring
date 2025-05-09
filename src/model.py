import torch
import torch.nn as nn
from torch import Tensor

class SleepLSTM(nn.Module):
    """
    A multi-layer LSTM model for sequence classification, specifically designed
    for sleep stage prediction tasks from time series input features.

    Attributes:
        lstm (nn.LSTM): The LSTM layer that processes sequential input data.
        classifier (nn.Linear): A linear layer that maps LSTM outputs to class logits.
    """
    
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2, num_classes: int = 6, dropout: float = 0.3) -> None:
        super(SleepLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        lstm_out, _ = self.lstm(x)
        output = self.classifier(lstm_out)

        return output