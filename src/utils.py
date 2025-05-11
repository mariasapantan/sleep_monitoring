from typing import List, Tuple, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from constants import ConstantsPreprocess


def max2epochs(data: NDArray[np.float64], fs: int, epoch: int) -> NDArray[np.float64]:
    """_summary_
    Processes a 1D signal array by computing the maximum absolute value in fixed-duration epochs.
    Args:
        data (NDArray[np.float64]): Input signal data (can be 1D or flattenable to 1D).
        fs (int): Sampling frequency
        epoch (int): Epoch length in seconds for aggregation.

    Returns:
        NDArray[np.float64]: 1D array of summed max values per epoch.
    """
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
    
def get_interval_labels(data: pd.DataFrame) -> Tuple[float, float]:
    """
    Compute the minimum and maximum timestamp from the first value of the 'epochs' list in each row.

    Args:
        data (pd.DataFrame): DataFrame where each row contains a list-like object in the 'epochs' column.

    Returns:
        Tuple[float, float]: The minimum and maximum epoch timestamps.
    """
    first_epochs = data['epochs'].apply(lambda x: x[0] if x else None).dropna()
    min_timestamp = first_epochs.min()
    max_timestamp = first_epochs.max()
    return min_timestamp, max_timestamp

def get_interval(data: NDArray[np.float64]) -> Tuple[float, float]:
    """
    Compute the minimum and maximum values in a NumPy array.

    Args:
        data (NDArray[np.float64]): A NumPy array of floating-point numbers.

    Returns:
        Tuple[float, float]: A tuple containing the minimum and maximum values in the array.
    """
    return np.amin(data), np.amax(data)

def get_valid_epoch_dictionary(timestamps, start_time):
    epoch_dictionary = {}
    for ind in range(np.shape(timestamps)[0]):
        time = timestamps[ind]
        floored_timestamp = time - np.mod(time - start_time, 30)

        epoch_dictionary[floored_timestamp] = True

    return epoch_dictionary

def cosine_proxy(time: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]:
    """
    Computes a cosine-based proxy for sleep drive, shifted by 5 hours.

    Args:
        time (float or NDArray[np.float64]): Time in seconds. Can be a scalar or a NumPy array.

    Returns:
        float or NDArray[np.float64]: The computed cosine proxy value(s).
    """
    sleep_drive_cosine_shift = 5
    return -1 * np.cos((time - sleep_drive_cosine_shift * 3600) *2 * np.pi / (3600*24))

def build_cosine(valid_epochs: List[Tuple[float, ...]]) -> NDArray[np.float64]:
    """
    Builds a NumPy array of cosine proxy values based on the first timestamp.

    Args:
        valid_epochs (List[Tuple[float, ...]]): A list of tuples, each containing a timestamp
            as the first element (e.g., epochs with format like (timestamp, ...)).

    Returns:
        NDArray[np.float64]: A NumPy array of cosine proxy values aligned to the first timestamp.
    """
    first_timestamp = valid_epochs[0][0]
    return np.array([
        cosine_proxy(epoch[0] - first_timestamp)
        for epoch in valid_epochs
    ])
    
def build_time(valid_epochs: List[Tuple[float, ...]]) -> NDArray[np.float64]:
    """
    Builds a NumPy array of time features in hours, relative to the first timestamp.

    Args:
        valid_epochs (List[Tuple[float, ...]]): A list of tuples where each tuple's first element
            is a timestamp in seconds (e.g., [(timestamp1, ...), (timestamp2, ...), ...]).

    Returns:
        NDArray[np.float64]: A NumPy array of time values in hours, shifted so that the first timestamp is 0.
    """
    features = []
    first_timestamp = valid_epochs[0][0]
    for epoch in valid_epochs:
        value = epoch[0] - first_timestamp
        value = value / 3600.0  
        features.append(value)
    return np.array(features)
    
def interpolate(activity_count_collection: NDArray[np.float64]) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Interpolates activity count data to generate uniformly spaced timestamps with corresponding values.

    Args:
        activity_count_collection (NDArray[np.float64]):
            A 2D NumPy array of shape (2, N) or (N, 2) containing:
              - timestamps in seconds (first row or column)
              - activity counts (second row or column)

    Returns:
        Tuple[NDArray[np.float64], NDArray[np.float64]]:
            - interpolated_timestamps: Linearly spaced timestamps at 1-second resolution.
            - interpolated_counts: Interpolated activity counts at those timestamps.
    """
    activity_count_collection = activity_count_collection.T
    timestamps = np.round(activity_count_collection[0].flatten(), 6)
    np.set_printoptions(suppress=True, precision=6)
    activity_count_values = np.round(activity_count_collection[1].flatten(), 6)
    interpolated_timestamps = np.arange(np.amin(timestamps),
                                        np.amax(timestamps), 1)
    interpolated_counts = np.interp(interpolated_timestamps, timestamps, activity_count_values)
    return interpolated_timestamps, interpolated_counts

def build_count_features(
    count_features: NDArray[np.float64],
    valid_epochs: List[Tuple[float, ...]]
) -> List[float]:
    """
    Builds a list of extracted activity count features for each valid epoch.

    Args:
        count_features (NDArray[np.float64]): A 2D NumPy array of shape (2, N) or (N, 2),
            where one axis contains timestamps and the other contains corresponding activity counts.
        valid_epochs (List[Tuple[float, ...]]): A list of tuples, each containing at least a timestamp
            as the first element (used to extract time windows around each epoch).

    Returns:
        List[float]: A list of extracted features (one per epoch) based on activity count data.
    """

    feature_count =[]
    interpolated_timestamps, interpolated_counts = interpolate(count_features)
    for epoch in valid_epochs:
        indices_in_range = get_window(interpolated_timestamps, epoch, (10*30-15))
        activity_counts_in_range = interpolated_counts[indices_in_range]

        feature = get_feature(activity_counts_in_range)
        feature_count.append(feature)
        
    return feature_count

def get_window(
    timestamps: NDArray[np.float64],
    epoch: Tuple[float, ...],
    window_size: float
) -> Union[int, NDArray[np.int_]]:
    """
    Finds the index or indices of timestamps within a specified time window around a given epoch.

    Args:
        timestamps (NDArray[np.float64]): 1D or 2D NumPy array of timestamps (in seconds).
        epoch (Tuple[float, ...]): A tuple where the first element is the epoch's central timestamp.
        window_size (float): The size of the time window in seconds, extending before and after the epoch.

    Returns:
        Union[int, NDArray[np.int_]]: Index or indices of the timestamps that fall within the specified window.
                                      If only one index is found, it returns an int.
    """
    start_time = epoch[0]- window_size
    end_time = epoch[0] + 30 + window_size
    timestamps_ravel = timestamps.ravel()
    indices_in_range = np.unravel_index(np.where((timestamps_ravel > start_time) & (timestamps_ravel < end_time)),
                                        timestamps.shape)
    return indices_in_range[0][0]

def get_feature(count_values: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Computes a feature representation from activity count values using Gaussian smoothing.

    Args:
        count_values (NDArray[np.float64]): A 1D array of raw activity count values for an epoch or time window.

    Returns:
        NDArray[np.float64]: A 1D array containing the smoothed feature values.
    """
    convolution = smooth_gauss(count_values.flatten(), np.shape(count_values.flatten())[0])
    return np.array([convolution])

def smooth_gauss(y: NDArray[np.float64], box_pts: int) -> float:
    """
    Applies Gaussian-weighted smoothing to a 1D array of values using a fixed-width Gaussian kernel.

    Args:
        y (NDArray[np.float64]): Input 1D array of values to smooth. Must be at least `box_pts` in length.
        box_pts (int): Number of points in the smoothing window (kernel size).

    Returns:
        float: The smoothed value computed by applying the Gaussian kernel over the input values.
    """
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

def build_hr_features(
    hr: NDArray[np.float64],
    valid_epochs: List[Tuple[float, ...]]
) -> NDArray[np.float64]:
    """
    Builds heart rate variability features for each valid epoch by computing the standard deviation
    of normalized heart rate values within a defined time window.

    Args:
        hr (NDArray[np.float64]): A 2D NumPy array with heart rate data, shaped (2, N) or (N, 2),
            where one axis contains timestamps and the other contains heart rate values.
        valid_epochs (List[Tuple[float, ...]]): A list of tuples where the first element is the
            timestamp of the epoch (in seconds).

    Returns:
        NDArray[np.float64]: A 1D NumPy array of standard deviation features (one per epoch).
    """
    heart_rate_features = []
    interpolated_timestamps, interpolated_hr = interpolate_and_normalize(hr)
    
    for epoch in valid_epochs:
        indices_in_range = get_window(interpolated_timestamps, epoch, (10 * 30 - 15))
        heart_rate_values_in_range = interpolated_hr[indices_in_range]

        feature = np.std(heart_rate_values_in_range)

        heart_rate_features.append(feature)

    return np.array(heart_rate_features)

def interpolate_and_normalize(heart_rate_collection: pd.DataFrame) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Interpolates and normalizes heart rate data to produce uniformly spaced, scaled values.

    The function performs the following steps:
    1. Extracts timestamps and heart rate values from the input DataFrame.
    2. Interpolates heart rate values to a uniform 1-second resolution.
    3. Applies a Difference of Gaussians (DoG) convolution to smooth the data.
    4. Normalizes the result using the 90th percentile of the absolute values.

    Args:
        heart_rate_collection (pd.DataFrame): DataFrame containing time and heart rate values.
            Expects columns `ConstantsPreprocess.COL_TIME` and `ConstantsPreprocess.COL_HR`.

    Returns:
        Tuple[NDArray[np.float64], NDArray[np.float64]]:
            - Interpolated timestamps at 1-second resolution.
            - Normalized and convolved heart rate values.
    """
    timestamps = heart_rate_collection[ConstantsPreprocess.COL_TIME].to_numpy().flatten()
    heart_rate_values = heart_rate_collection[ConstantsPreprocess.COL_HR].to_numpy().flatten()
    interpolated_timestamps = np.arange(np.amin(timestamps),
                                        np.amax(timestamps), 1)
    interpolated_hr = np.interp(interpolated_timestamps, timestamps, heart_rate_values)

    interpolated_hr = convolve_with_dog(interpolated_hr, (10 * 30 - 15))

    scalar = np.percentile(np.abs(interpolated_hr), 90)
    interpolated_hr = interpolated_hr / scalar

    return interpolated_timestamps, interpolated_hr

def convolve_with_dog(y: NDArray[np.float64], box_pts: int) -> NDArray[np.float64]:
    """
    Applies a Difference of Gaussians (DoG) convolution to the input signal.

    The DoG filter is constructed by subtracting a wide Gaussian (sigma2) from a narrow one (sigma1),
    optionally scaled, to highlight transitions in the signal. The signal is padded using flipped
    boundary values before convolution to preserve edge information.

    Args:
        y (NDArray[np.float64]): 1D input signal array.
        box_pts (int): Number of points in the convolution kernel (window size).

    Returns:
        NDArray[np.float64]: The smoothed signal after applying DoG convolution.
    """
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