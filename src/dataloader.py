import logging
import os

from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from constants import ConstantsDataLoader

logging.basicConfig(level=logging.INFO)

class SleepFeatureDataset(Dataset):
    """
    A custom PyTorch Dataset for loading already processed features and labels
    by sleep_classifiers/source/preprocessing/preprocessing_runner.py script

    Each sample includes time-series features (count, cosine, HR, time)
    and associated label sequences, along with the subject ID.
    """
    
    def __init__(self, feature_dir: str):
        self.feature_dir = feature_dir
        self.subject_ids = self._get_subject_ids()
 
    def _get_subject_ids(self) -> List[str]:
        return sorted([f.split('_')[0] for f in os.listdir(self.feature_dir) if f.endswith(ConstantsDataLoader.label_endfile_name)], key=int)

    def __len__(self) -> int:
        return len(self.subject_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        sid = self.subject_ids[idx]

        try:
            count_feat = np.loadtxt(os.path.join(self.feature_dir, f"{sid}" + ConstantsDataLoader.cf_endfile_name), ndmin=2)
            cosine_feat = np.loadtxt(os.path.join(self.feature_dir, f"{sid}" + ConstantsDataLoader.cos_endfile_name), ndmin=2)
            hr_feat = np.loadtxt(os.path.join(self.feature_dir, f"{sid}" + ConstantsDataLoader.hr_endfile_name), ndmin=2)
            time_feat = np.loadtxt(os.path.join(self.feature_dir, f"{sid}"  + ConstantsDataLoader.t_endfile_name), ndmin=2)
            labels = np.loadtxt(os.path.join(self.feature_dir, f"{sid}" + ConstantsDataLoader.label_endfile_name), dtype=int)

            features = np.concatenate([
                count_feat.reshape(-1, 1),
                cosine_feat.reshape(-1, 1),
                hr_feat,
                time_feat.reshape(-1, 1)
            ], axis=1)

            features = torch.tensor(features, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.long)

            return features, labels, sid
        except Exception as e:
            logging.warning(f"Skipping subject {sid} due to error: {e}")
            return self.__getitem__((idx + 1) % len(self))
        
