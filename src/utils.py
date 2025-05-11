import numpy as np

from constants import ConstantsPreprocess

def max2epochs(data, fs, epoch):
    data = data.flatten()

    seconds = int(np.floor(np.shape(data)[0] / fs))
    data = np.abs(data)
    data = data[0:int(seconds * fs)]

    data = data.reshape(fs, seconds, order='F').copy()

    data = data.max(0)
    data = data.flatten()
    N = np.shape(data)[0]
    num_epochs = int(np.floor(N / epoch))
    data = data[0:(num_epochs * epoch)]

    data = data.reshape(epoch, num_epochs, order='F').copy()
    epoch_data = np.sum(data, axis=0)
    epoch_data = epoch_data.flatten()

    return epoch_data
    
def get_interval_labels(data):
    min_timestamp = 1e15
    max_timestamp = -1
    for i, row in data.iterrows():
        if row['epochs'][0] < min_timestamp:
            min_timestamp = row['epochs'][0]
        if row['epochs'][0] > max_timestamp:
            max_timestamp = row['epochs'][0]
  
    return min_timestamp, max_timestamp

def get_interval(data):
    return np.amin(data), np.amax(data)

def get_valid_epoch_dictionary(timestamps, start_time):
    epoch_dictionary = {}
    for ind in range(np.shape(timestamps)[0]):
        time = timestamps[ind]
        floored_timestamp = time - np.mod(time - start_time, 30)

        epoch_dictionary[floored_timestamp] = True

    return epoch_dictionary

def cosine_proxy(time):
    sleep_drive_cosine_shift = 5
    return -1 * np.cos((time - sleep_drive_cosine_shift * 3600) *2 * np.pi / (3600*24))

def build_cosine(valid_epochs):
    first_timestamp = valid_epochs[0][0]
    return np.array([
        cosine_proxy(epoch[0] - first_timestamp)
        for epoch in valid_epochs
    ])
    
def build_time(valid_epochs):
    features = []
    first_timestamp = valid_epochs[0][0]
    for epoch in valid_epochs:
        value = epoch[0] - first_timestamp
        value = value / 3600.0  # Changing units to hours improves performance
        features.append(value)
    return np.array(features)
    
def interpolate(activity_count_collection):
    activity_count_collection = activity_count_collection.T
    timestamps = np.round(activity_count_collection[0].flatten(), 6)
    np.set_printoptions(suppress=True, precision=6)
    activity_count_values = np.round(activity_count_collection[1].flatten(), 6)
    interpolated_timestamps = np.arange(np.amin(timestamps),
                                        np.amax(timestamps), 1)
    interpolated_counts = np.interp(interpolated_timestamps, timestamps, activity_count_values)
    return interpolated_timestamps, interpolated_counts

def build_count_features(count_features, valid_epochs):

    feature_count =[]
    interpolated_timestamps, interpolated_counts = interpolate(count_features)
    for epoch in valid_epochs:
        indices_in_range = get_window(interpolated_timestamps, epoch, (10*30-15))
        activity_counts_in_range = interpolated_counts[indices_in_range]

        feature = get_feature(activity_counts_in_range)
        feature_count.append(feature)
        
    return feature_count

def get_window(timestamps, epoch, window_size):
    start_time = epoch[0]- window_size
    end_time = epoch[0] + 30 + window_size
    timestamps_ravel = timestamps.ravel()
    indices_in_range = np.unravel_index(np.where((timestamps_ravel > start_time) & (timestamps_ravel < end_time)),
                                        timestamps.shape)
    return indices_in_range[0][0]

def get_feature(count_values):
    convolution = smooth_gauss(count_values.flatten(), np.shape(count_values.flatten())[0])
    return np.array([convolution])

def smooth_gauss(y, box_pts):
    box = np.ones(box_pts) / box_pts
    mu = int(box_pts / 2.0)
    sigma = 50  # seconds

    for ind in range(0, box_pts):
        box[ind] = np.exp(-1 / 2 * (((ind - mu) / sigma) ** 2))

    box = box / np.sum(box)
    sum_value = 0
    for ind in range(0, box_pts):
        sum_value += box[ind] * y[ind]

    return sum_value

def build_hr_features(hr, valid_epochs):
    heart_rate_features = []
    interpolated_timestamps, interpolated_hr = interpolate_and_normalize(hr)
    
    for epoch in valid_epochs:
        indices_in_range = get_window(interpolated_timestamps, epoch, (10 * 30 - 15))
        heart_rate_values_in_range = interpolated_hr[indices_in_range]

        feature = np.std(heart_rate_values_in_range)

        heart_rate_features.append(feature)

    return np.array(heart_rate_features)

def interpolate_and_normalize(heart_rate_collection):
    timestamps = heart_rate_collection[ConstantsPreprocess.COL_TIME].to_numpy().flatten()
    heart_rate_values = heart_rate_collection[ConstantsPreprocess.COL_HR].to_numpy().flatten()
    interpolated_timestamps = np.arange(np.amin(timestamps),
                                        np.amax(timestamps), 1)
    interpolated_hr = np.interp(interpolated_timestamps, timestamps, heart_rate_values)

    interpolated_hr = convolve_with_dog(interpolated_hr, (10 * 30 - 15))

    scalar = np.percentile(np.abs(interpolated_hr), 90)
    interpolated_hr = interpolated_hr / scalar

    return interpolated_timestamps, interpolated_hr

def convolve_with_dog(y, box_pts):
    y = y - np.mean(y)
    box = np.ones(box_pts) / box_pts

    mu1 = int(box_pts / 2.0)
    sigma1 = 120

    mu2 = int(box_pts / 2.0)
    sigma2 = 600

    scalar = 0.75

    for ind in range(0, box_pts):
        box[ind] = np.exp(-1 / 2 * (((ind - mu1) / sigma1) ** 2)) - scalar * np.exp(
            -1 / 2 * (((ind - mu2) / sigma2) ** 2))

    y = np.insert(y, 0, np.flip(y[0:int(box_pts / 2)]))  # Pad by repeating boundary conditions
    y = np.insert(y, len(y) - 1, np.flip(y[int(-box_pts / 2):]))
    y_smooth = np.convolve(y, box, mode='valid')

    return y_smooth