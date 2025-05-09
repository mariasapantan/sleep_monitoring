import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # [seq_len, d_model]
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))

        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)

        self.pe = pe.unsqueeze(0)  # [1, seq_len, d_model]

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].to(x.device)
        return x


class SleepTransformer(nn.Module):
    def __init__(self, input_dim=5, model_dim=64, num_heads=4, num_layers=2, num_classes=5, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, model_dim)
        self.pos_enc = PositionalEncoding(model_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Linear(model_dim, num_classes)

    def forward(self, x):
        """
        x: [batch_size, seq_len, input_dim]
        """
        x = self.input_proj(x)               # [B, T, model_dim]
        x = self.pos_enc(x)                  # [B, T, model_dim]
        x = self.transformer(x)              # [B, T, model_dim]
        logits = self.classifier(x)          # [B, T, num_classes]
        return logits
