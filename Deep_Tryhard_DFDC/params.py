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

MODEL_MOBILENETV2 = 'yolo_model/facedetection-mobilenetv2-size224-alpha0.75.h5'

TRAIN_FOLDER_FAKE = 'Data/fake'

TRAIN_FOLDER_REAL = 'Data/real'

TEST_FOLDER = 'test_videos'

SAMPLE_REAL = f'{TRAIN_FOLDER_FAKE}/abarnvbtwb.mp4'

SAMPLE_FAKE = f'{TRAIN_FOLDER_FAKE}/eepezmygaq.mp4'

# title of folder as follow : images_{number of videos}vid_{number of theorical frames}tf /fake or real
PATH_WRITE_FAKE ='Data/images_50vid_100tf/fake'

PATH_WRITE_REAL = 'Data/images_50vid_100tf/real'
