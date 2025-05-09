from dataloader import SleepDataset
from torch.utils.data import DataLoader

if __name__ == "__main__":
    dataset = SleepDataset(data_dir="data")
    print(f"Found {len(dataset)} subjects")

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    for features, labels, subj_id in dataloader:
        print(f"\nSample from subject: {subj_id[0]}")
        print("Features shape:", features.shape)  # (1, seq_len, 5)
        print("Labels shape:", labels.shape)      # (1, seq_len)
        break  # Only one batch for test
