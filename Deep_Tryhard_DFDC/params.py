### MLFLOW configuration - - - - - - - - - - - - - - - - - - -

MLFLOW_URI = "https://mlflow.lewagon.co/"
EXPERIMENT_NAME = "france paris benragot model + version"

### DATA & MODEL LOCATIONS  - - - - - - - - - - - - - - - - - - -

PATH_TO_LOCAL_MODEL = 'model.joblib'

### GCP configuration - - - - - - - - - - - - - - - - - - -

# /!\ you should fill these according to your account

### GCP Project - - - - - - - - - - - - - - - - - - - - - -

PROJECT_ID_data_set = 'deep-tryhard-dfdc'

### GCP Storage - - - - - - - - - - - - - - - - - - - - - -

BUCKET_NAME = 'bachata-kfc-bucket'

##### Data  - - - - - - - - - - - - - - - - - - - - - - - -

# train data file location
# /!\ here you need to decide if you are going to train using the provided and uploaded data/train_1k.csv sample file
# or if you want to use the full dataset (you need need to upload it first of course)
BUCKET_TRAIN_DATA_PATH = 'raw_data/metadata.csv'

##### Training  - - - - - - - - - - - - - - - - - - - - - -

# not required here

##### Model - - - - - - - - - - - - - - - - - - - - - - - -

# model folder name (will contain the folders for all trained model versions)
MODEL_NAME = 'taxifare'

# model version folder name (where the trained model.joblib file will be stored)
MODEL_VERSION = 'v2'

### GCP AI Platform - - - - - - - - - - - - - - - - - - - -

# not required here

### - - - - - - - - - - - - - - - - - - - - - - - - - - - -

### FACE DETECTION PARAMS (TO UPDATE)

VIDEO_PATH = 'BUCKET'

PATH = 'BUCKET'

MODEL_MOBILENETV2 = 'BUCKET'

DATA_FOLDER = '/mnt/d/deepfake-detection-challenge'

TRAIN_SAMPLE_FOLDER = 'train_sample_videos'

TEST_FOLDER = 'test_videos'

VIDEOS_DIR = '/mnt/d/deepfake-detection-challenge/train_sample_videos/'

SAMPLE_REAL = f'{VIDEOS_DIR}/abarnvbtwb.mp4'

SAMPLE_FAKE = f'{VIDEOS_DIR}/eepezmygaq.mp4'

path='/mnt/d/deepfake-detection-challenge/test_videos/test/'
