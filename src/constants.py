class ConstantsDataLoader:
    """
    Constants defining standard filename suffixes used when loading precomputed features and labels.

    Attributes:
        label_endfile_name (str): Suffix for label files (e.g., sleep stage annotations).
        cf_endfile_name (str): Suffix for count feature files (e.g., activity counts).
        cos_endfile_name (str): Suffix for cosine feature files (e.g., circadian rhythm proxies).
        hr_endfile_name (str): Suffix for heart rate feature files.
        t_endfile_name (str): Suffix for time feature files (e.g., hours since start).
    """
    
    label_endfile_name = '_psg_labels.out'
    cf_endfile_name = '_count_feature.out'
    cos_endfile_name = '_cosine_feature.out'
    hr_endfile_name = '_hr_feature.out'
    t_endfile_name = '_time_feature.out'
    
class ConstantsModel:
    """
    Constants used in the sleep stage classification model.

    Attributes:
        BATCH_SIZE (int): Batch size used for training and inference.
        INPUT_SIZE (int): Number of input features to the model.
        NUM_CLASSES (int): Total number of output classes (including unused class 4).
        CLASSES (dict): Mapping of class indices to sleep stage labels:
            - 0: Wake
            - 1: N1 (Light sleep)
            - 2: N2 (Light sleep)
            - 3: N3 (Deep sleep)
            - 5: REM (Rapid Eye Movement; dreaming stage with brain activity similar to wakefulness)

    Notes:
        Class 4 is intentionally skipped or unused.
        The NREM stages include N1, N2, and N3.
    """
    BATCH_SIZE = 1
    INPUT_SIZE = 4
    NUM_CLASSES = 6
    CLASSES = {0: 'wake', 1: 'N1', 2:'N2', 3:'N3', 5: 'REM'}

class ConstantsPreprocess:
    """
    A container class for constants used during the preprocessing of physiological and behavioral data.

    Attributes:
        COL_TIME (str): Column name for timestamps.
        COL_HR (str): Column name for heart rate values.
        COL_X (str): Column name for X-axis acceleration.
        COL_Y (str): Column name for Y-axis acceleration.
        COL_Z (str): Column name for Z-axis acceleration.
        COL_LABEL (str): Column name for sleep stage labels.
        COL_EPOCHS (str): Column name for precomputed epoch tuples.
        EPOCH_DURATION (int): Duration of each epoch in seconds (default: 30s).
        SEC_PER_HOUR (int): Number of seconds in an hour (3600).
        SECONDS_PER_DAY (int): Number of seconds in a day (86400).
    """
    COL_TIME = "time"
    COL_HR = "hr"
    COL_X = "x"
    COL_Y = "y"
    COL_Z = "z"
    COL_LABEL = "label"
    COL_EPOCHS = "epochs"
    EPOCH_DURATION = 30
    SEC_PER_HOUR = 3600
    SECONDS_PER_DAY = 3600 *24