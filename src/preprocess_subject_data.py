import os
import numpy as np
import pandas as pd
from glob import glob
from scipy.stats import entropy
from math import cos, pi
from scipy.signal import butter, filtfilt
import hydra
from omegaconf import DictConfig

from constants import ConstantsPreprocess 
from utils import max2epochs, get_interval, get_interval_labels, get_valid_epoch_dictionary, build_cosine, build_time, build_count_features, build_hr_features

def build_activity_counts(data):
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

def preclean_labels(data):
    start_time = data.iloc[0][ConstantsPreprocess.COL_TIME]
    epochs = [
        (start_time + i * ConstantsPreprocess.EPOCH_DURATION, i + 1)
        for i in range(len(data))
    ]
    return epochs

def filter_by_time(df, start_time, end_time):
    return df[(df[ConstantsPreprocess.COL_TIME] > start_time) & (df[ConstantsPreprocess.COL_TIME] < end_time)].reset_index(drop=True)

def build_labels(array, valid_epochs):
    return np.array([
        np.interp(epoch[0], array[ConstantsPreprocess.COL_TIME], array[ConstantsPreprocess.COL_LABEL])
        for epoch in valid_epochs
    ])

def preprocess_subject(subject_id, data_dir):
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

