import torch
from torch.utils.data import DataLoader
from dataloader import SleepDataset
from model import SleepTransformer


if __name__ == "__main__":
    dataset = SleepDataset(data_dir="data")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    model = SleepTransformer(input_dim=5, num_classes=5)  # 5 input features → 5 sleep stages (0,1,2,3,5 remapped to 0–4)

    for X, y, subject_id in dataloader:
        print(f"Testing on subject: {subject_id[0]}")
        print("Input shape:", X.shape)  # [B, T, 5]
        print("Target shape:", y.shape)  # [B, T]

        output = model(X)  # [B, T, num_classes]
        print("Output shape:", output.shape)

        # optional: check softmaxed output for one timestep
        print("Softmax at t=0:", torch.softmax(output[0, 0], dim=0))

        break
