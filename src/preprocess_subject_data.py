import os
from glob import glob
from math import cos, pi
from typing import List, Tuple

import hydra
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from omegaconf import DictConfig
from scipy.signal import butter, filtfilt
from scipy.stats import entropy

from constants import ConstantsPreprocess
from utils import (build_cosine, build_count_features, build_hr_features,
                   build_time, get_interval, get_interval_labels,
                   get_valid_epoch_dictionary, max2epochs)


def build_activity_counts(data: pd.DataFrame) -> NDArray[np.float64]:
    """
    Processes raw accelerometer data to compute activity counts using bandpass filtering,
    binning, and epoch-wise aggregation.

    Args:
        data (pd.DataFrame): Input DataFrame containing:
            - `ConstantsPreprocess.COL_TIME`: Timestamps in seconds.
            - `ConstantsPreprocess.COL_Z`: Z-axis acceleration values.

    Returns:
        NDArray[np.float64]: A 2D NumPy array of shape (N, 2) where each row contains:
            - Time (centered timestamp of each epoch).
            - Computed activity count for that epoch.
    """
    fs = 50
    time = np.arange(np.amin(data[ConstantsPreprocess.COL_TIME]), np.amax(data[ConstantsPreprocess.COL_TIME]), 1.0 / fs)
    z_data = np.interp(time, data[ConstantsPreprocess.COL_TIME], data[ConstantsPreprocess.COL_Z])

    cf_low = 3
    cf_hi = 11
    order = 5
    w1 = cf_low / (fs / 2)
    w2 = cf_hi / (fs / 2)
    b, a = butter(order, [w1, w2], btype='bandpass')

    z_filt = np.abs(filtfilt(b, a, z_data))

    top_edge = 5
    bottom_edge = 0
    number_of_bins = 128
    bin_edges = np.linspace(bottom_edge, top_edge, number_of_bins + 1)
    binned = np.digitize(z_filt, bin_edges)

    epoch = 15
    counts = max2epochs(binned, fs, epoch)
    counts = (counts - 18) * 3.07
    counts[counts < 0] = 0

    time_counts = np.linspace(np.min(data[ConstantsPreprocess.COL_TIME]), np.max(data[ConstantsPreprocess.COL_TIME]), len(counts))
    time_counts = time_counts[:, np.newaxis]
    counts = counts[:, np.newaxis]

    output = np.hstack((time_counts, counts))
    return output

def preclean_labels(data: pd.DataFrame) -> List[Tuple[float, int]]:
    """
    Constructs a list of epoch tuples containing start times and corresponding epoch indices.

    Each row in the input DataFrame is assumed to represent a sequential sleep epoch of fixed duration.
    The function computes the timestamp for the start of each epoch based on the first timestamp
    and a constant epoch duration.

    Args:
        data (pd.DataFrame): DataFrame containing a time column named `ConstantsPreprocess.COL_TIME`,
            with the first row representing the start of the first epoch.

    Returns:
        List[Tuple[float, int]]: A list of tuples where each tuple contains:
            - The start timestamp (in seconds) of the epoch.
            - The epoch index (1-based).
    """
    start_time = data.iloc[0][ConstantsPreprocess.COL_TIME]
    epochs = [
        (start_time + i * ConstantsPreprocess.EPOCH_DURATION, i + 1)
        for i in range(len(data))
    ]
    return epochs

def filter_by_time(df: pd.DataFrame, start_time: float, end_time: float) -> pd.DataFrame:
    """
    Filters a DataFrame to include only rows where the time column is within a specified interval.

    Args:
        df (pd.DataFrame): The input DataFrame containing a time column identified by `ConstantsPreprocess.COL_TIME`.
        start_time (float): Start of the time interval (in seconds).
        end_time (float): End of the time interval (in seconds).

    Returns:
        pd.DataFrame: A new DataFrame containing only rows with timestamps between `start_time` and `end_time`,
                      with the index reset.
    """
    return df[(df[ConstantsPreprocess.COL_TIME] > start_time) & (df[ConstantsPreprocess.COL_TIME] < end_time)].reset_index(drop=True)

def build_labels(
    array: pd.DataFrame,
    valid_epochs: List[Tuple[float, ...]]
) -> NDArray[np.float64]:
    """
    Builds a label array by interpolating label values at the timestamps of valid epochs.

    Args:
        array (pd.DataFrame): A DataFrame containing:
            - `ConstantsPreprocess.COL_TIME`: Timestamps in seconds.
            - `ConstantsPreprocess.COL_LABEL`: Corresponding label values (numeric).
        valid_epochs (List[Tuple[float, ...]]): A list of tuples where the first element of each tuple
            is the timestamp for which to interpolate a label.

    Returns:
        NDArray[np.float64]: A NumPy array of interpolated label values for each epoch.
    """
    return np.array([
        np.interp(epoch[0], array[ConstantsPreprocess.COL_TIME], array[ConstantsPreprocess.COL_LABEL])
        for epoch in valid_epochs
    ])

def preprocess_subject(subject_id: str, data_dir: str) -> Tuple[
    np.ndarray,  # feature_labels
    np.ndarray,  # cosine_features
    np.ndarray,  # time_feature
    np.ndarray,  # count_features
    np.ndarray   # hr_feature
]:
    """
    Preprocesses data for a single subject by loading, aligning, and extracting time-series features.

    This function performs the following steps:
    1. Loads heart rate, motion, and label data from text files.
    2. Cleans, sorts, and time-aligns the data.
    3. Interpolates and filters each data stream to keep only valid overlapping segments.
    4. Constructs a list of valid epochs with matching data across all sources.
    5. Extracts multiple feature types: label targets, cosine-based circadian features,
       time-since-start features, motion-based activity count features, and heart rate variability.

    Args:
        subject_id (str): Unique identifier of the subject (used to locate files).
        data_dir (str): Base directory path containing the subfolders `heart_rate`, `motion`, and `labels`.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            - feature_labels: Array of interpolated ground truth sleep stage labels.
            - cosine_features: Array of cosine-based circadian rhythm features.
            - time_feature: Array of time-in-hours features (relative to the first epoch).
            - count_features: Array of activity count features from accelerometer data.
            - hr_feature: Array of heart rate variability features (standard deviation per epoch).
    """
    print("Processing subject ", subject_id)

    hr = pd.read_csv(f"{data_dir}/heart_rate/{subject_id}_heartrate.txt", sep=",", 
                     names=[ConstantsPreprocess.COL_TIME, ConstantsPreprocess.COL_HR])
    motion = pd.read_csv(f"{data_dir}/motion/{subject_id}_acceleration.txt", sep=" ", 
                         names=[ConstantsPreprocess.COL_TIME, ConstantsPreprocess.COL_X, 
                                ConstantsPreprocess.COL_Y, ConstantsPreprocess.COL_Z])
    labels = pd.read_csv(f"{data_dir}/labels/{subject_id}_labeled_sleep.txt", sep=" ", 
                         names=[ConstantsPreprocess.COL_TIME, ConstantsPreprocess.COL_LABEL])
    
    motion = motion.drop_duplicates()
    motion = motion.sort_values(by=ConstantsPreprocess.COL_TIME)
    motion[ConstantsPreprocess.COL_TIME] = motion[ConstantsPreprocess.COL_TIME].apply(lambda v: round(float(v), 6))

    hr = hr.drop_duplicates()
    hr = hr.sort_values(by=ConstantsPreprocess.COL_TIME)
    
    labels[ConstantsPreprocess.COL_EPOCHS] = preclean_labels(labels)
    
    start_time = max(get_interval(motion[ConstantsPreprocess.COL_TIME])[0], get_interval(hr[ConstantsPreprocess.COL_TIME])[0], get_interval_labels(labels)[0])
    end_time = min(get_interval(motion[ConstantsPreprocess.COL_TIME])[1], get_interval(hr[ConstantsPreprocess.COL_TIME])[1], get_interval_labels(labels)[1])
    
    labels = filter_by_time(labels, start_time, end_time)
    motion = filter_by_time(motion, start_time, end_time)
    hr = filter_by_time(hr, start_time, end_time)
    
    act_counts = build_activity_counts(motion)
    
    motion_epoch_dictionary = get_valid_epoch_dictionary(motion[ConstantsPreprocess.COL_TIME], labels[ConstantsPreprocess.COL_EPOCHS][0][0])

    hr_epoch_dictionary = get_valid_epoch_dictionary(hr[ConstantsPreprocess.COL_TIME], labels[ConstantsPreprocess.COL_EPOCHS][0][0])
    
    valid_epochs = []
    for j,r in labels.iterrows():
        e = r['epochs']

        if e[0] in motion_epoch_dictionary and e[0] in hr_epoch_dictionary\
            and r['label'] != -1:
            valid_epochs.append(e)
 
    feature_labels = build_labels(labels, valid_epochs)
    cosine_features = build_cosine(valid_epochs)
    time_feature = build_time(valid_epochs)
    count_features = build_count_features(act_counts, valid_epochs)
    hr_feature = build_hr_features(hr, valid_epochs)
    
    return feature_labels, cosine_features, time_feature, count_features, hr_feature
    
@hydra.main(config_path="../configs", config_name="config_hydra", version_base=None)
def main(cfg: DictConfig):
    """
    Main function wrapped with Hydra for config management for training.

    Args:
        cfg (DictConfig): Configuration object loaded from YAML file.
    """

    # Run training with Hydra config
    subjects = [os.path.basename(f).split("_")[0] for f in glob(f"{cfg.data_preprocess.input_dir}/heart_rate/*_heartrate.txt")]

    os.makedirs(cfg.data_preprocess.output_dir, exist_ok=True)
    for sid in subjects:
        feature_labels, cosine_features, time_feature, count_features, hr_feature = preprocess_subject(sid, cfg.data_preprocess.input_dir)
        np.savetxt(cfg.data_preprocess.output_dir + str(sid) + "_psg_labels.out", feature_labels, fmt='%f')
        np.savetxt(cfg.data_preprocess.output_dir + str(sid) + "_count_feature.out", count_features, fmt='%f')
        np.savetxt(cfg.data_preprocess.output_dir + str(sid) + "_hr_feature.out", hr_feature, fmt='%f')
        np.savetxt(cfg.data_preprocess.output_dir + str(sid) + "_cosine_feature.out", cosine_features, fmt='%f')
        np.savetxt(cfg.data_preprocess.output_dir + str(sid) + "_time_feature.out", time_feature, fmt='%f')


if __name__ == "__main__":
    main()

