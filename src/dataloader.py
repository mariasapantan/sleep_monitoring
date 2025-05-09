import os
import glob
import torch
import pandas as pd
from torch.utils.data import Dataset


class SleepDataset(Dataset):
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.subject_ids = self._get_valid_subject_ids()

    def _get_valid_subject_ids(self):
        hr_ids = {f.split("_")[0] for f in os.listdir(os.path.join(self.data_dir, "heart_rate"))}
        motion_ids = {f.split("_")[0] for f in os.listdir(os.path.join(self.data_dir, "motion"))}
        steps_ids = {f.split("_")[0] for f in os.listdir(os.path.join(self.data_dir, "steps"))}
        labels_ids = {f.split("_")[0] for f in os.listdir(os.path.join(self.data_dir, "labels"))}

        common_ids = sorted(list(hr_ids & motion_ids & steps_ids & labels_ids))
        print(f"Found {len(common_ids)} subjects with complete data")
        return common_ids

    def __len__(self):
        return len(self.subject_ids)

    def __getitem__(self, idx):
        subject_id = self.subject_ids[idx]

        try:
            hr_file = os.path.join(self.data_dir, "heart_rate", f"{subject_id}_heartrate.txt")
            motion_file = os.path.join(self.data_dir, "motion", f"{subject_id}_acceleration.txt")
            steps_file = os.path.join(self.data_dir, "steps", f"{subject_id}_steps.txt")
            label_file = os.path.join(self.data_dir, "labels", f"{subject_id}_labeled_sleep.txt")

            hr_df = pd.read_csv(hr_file, sep=",", header=None, names=["time", "hr"])
            motion_df = pd.read_csv(motion_file, sep=" ", header=None, names=["time", "x", "y", "z"])
            steps_df = pd.read_csv(steps_file, sep=",", header=None, names=["time", "steps"])
            label_df = pd.read_csv(label_file, sep=" ", header=None, names=["time", "label"])

            # Round timestamps to seconds
            for df in [hr_df, motion_df, steps_df, label_df]:
                df["time"] = df["time"].round().astype(int)

            df = label_df.merge(hr_df, on="time", how="inner") \
                         .merge(motion_df, on="time", how="inner") \
                         .merge(steps_df, on="time", how="inner")

            df = df[df["label"] != -1].reset_index(drop=True)

            if len(df) == 0:
                raise ValueError(f"No valid label data for subject {subject_id}")

            features = torch.tensor(df[["hr", "x", "y", "z", "steps"]].values, dtype=torch.float32)
            labels = torch.tensor(df["label"].values, dtype=torch.long)

            return features, labels, subject_id
        except Exception as e:
            print(f"Skipping subject {subject_id}: {e}")
            return self.__getitem__((idx + 1) % len(self))

