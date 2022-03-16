### DATA & MODEL LOCATIONS  - - - - - - - - - - - - - - - - - - -

PATH_TO_LOCAL_MODEL = 'model.joblib'

### GCP configuration - - - - - - - - - - - - - - - - - - -

# /!\ you should fill these according to your account
SERVICE_KEY = '/home/benjamin/code/benragot/gcp/deep-tryhard-dfdc-dd1c7261c311.json'

### GCP Project - - - - - - - - - - - - - - - - - - - - - -

PROJECT_ID_data_set = 'deep-tryhard-dfdc'

### GCP Storage - - - - - - - - - - - - - - - - - - - - - -

BUCKET_NAME = 'bachata-kfc-bucket'

##### Data  - - - - - - - - - - - - - - - - - - - - - - - -

# train data file location
# /!\ here you need to decide if you are going to train using the provided and uploaded data/train_1k.csv sample file
# or if you want to use the full dataset (you need need to upload it first of course)
BUCKET_TRAIN_DATA_PATH = 'raw_data/metadata.csv'

### FACE DETECTION PARAMS (TO UPDATE)

import os

MODEL_MOBILENETV2 = os.path.join(
    os.path.dirname(__file__),
    'model_Yolov2',
    'facedetection-mobilenetv2-size224-alpha0.75.h5')

TRAIN_FOLDER_FAKE = 'Data/fake'

TRAIN_FOLDER_REAL = 'Data/real'

TEST_FOLDER = 'test_videos'

SAMPLE_REAL = f'{TRAIN_FOLDER_FAKE}/abarnvbtwb.mp4'

SAMPLE_FAKE = f'{TRAIN_FOLDER_FAKE}/eepezmygaq.mp4'

# title of folder as follow : images_{number of videos}vid_{number of theorical frames}tf /fake or real
PATH_WRITE_FAKE ='Data/images_50vid_100tf/fake'

PATH_WRITE_REAL = 'Data/images_50vid_100tf/real'
