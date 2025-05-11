class ConstantsDataLoader:
    
    label_endfile_name = '_psg_labels.out'
    cf_endfile_name = '_count_feature.out'
    cos_endfile_name = '_cosine_feature.out'
    hr_endfile_name = '_hr_feature.out'
    t_endfile_name = '_time_feature.out'
    
class ConstantsModel:
    """Constants for the model
         CLASSES = {0: 'wake', 1: 'N1', 2:'N2', 3:'N3', 5: 'REM'}
         0: wake
         1: Light sleep
         2: Light sleep
         3: Deep Sleep
         5: Rapid Eye Movement (Dreaming stage; brain activity similar to wakefulness)
         NREM = N1, N2, N3
    """
    BATCH_SIZE = 1
    INPUT_SIZE = 4
    NUM_CLASSES = 6
    CLASSES = {0: 'wake', 1: 'N1', 2:'N2', 3:'N3', 5: 'REM'}

class ConstantsPreprocess:
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