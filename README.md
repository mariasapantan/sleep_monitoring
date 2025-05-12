# sleep_monitoring

This code is a modified version of [1]. It's a sleep classifier with an LSTM network. The data preprocessing step follow the same steps as in the reference, but refactored. As results, it outputs the latest trained epoch, the ROC curve for REM vs NREM classes and a classification report with precision, f1-score and recall for every class.


# Data

We used the dataset collected using Apple Watch from here: https://physionet.org/content/sleep-accel/1.0.0/ .
You need to download the data before using this project.


# Intallation

Clone the repository:

<pre>git clone https://github.com/mariasapantan/sleep_monitoring.git cd sleep_monitoring python3 -m venv .venv pip install poetry source .venv/bin/activate</pre>

After this, you need to install all the project dependecies/requirements using poetry:

<pre>bash poetry install</pre>


# Data preprocessing

For preprocessing the data, you need to have the downloaded data directory in sleep_monitoring directory. The data directory has heart_rate, labels and motion directories inside, which contain raw data.

You need to run the preprocessing step only once and they will be stored in features directory by default. You can modify the save path in configs/config.yaml 

<pre>

data_preprocess:
    input_dir: "data"
    output_dir: "features/"

</pre>

<pre> python src/preprocess_subject.py </pre>


# Model training

For training the model, you can update the config.yaml file for model hyperparameters.
This code is for training and validation/inference during the same run or separately:

For training and validation, the run_mode has to be 'both':

<pre>python src/main.py </pre> 

or it is possible to specify this directy in the command:

<pre>python src/main.py run_mode=both </pre> 


For training only, the run_mode has to be 'train':

<pre>python src/main.py  </pre>

or it is possible to specify this directy in the command:

<pre>bash python src/main.py run_mode=train </pre> 


For validation/inference only, the run_mode has to be 'validate':

<pre>bash python src/main.py </pre>

or it is possible to specify this directy in the command:

<pre>bash python src/main.py run_mode=validate </pre> 

In the config.yaml file, there is a seed value, only for keeping the random split for training&validation under control, when only the validation part is used. 

# References
[1] Olivia Walch , Yitong Huang , Daniel Forger , Cathy Goldstein, Sleep stage prediction with raw acceleration and photoplethysmography heart rate data derived from a consumer wearable device, [https://doi.org/10.1093/sleep/zsz180](https://academic.oup.com/sleep/article/42/12/zsz180/5549536)

[2] Walch, O. (2019). Motion and heart rate from a wrist-worn wearable and labeled sleep from polysomnography (version 1.0.0). PhysioNet. https://doi.org/10.13026/hmhs-py35.

