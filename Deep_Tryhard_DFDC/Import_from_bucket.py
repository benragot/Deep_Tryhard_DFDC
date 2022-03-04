'''Package to import files from a google bucket'''

import pandas as pd
import os
from google.cloud import storage
from termcolor import colored
from tqdm import tqdm

BUCKET='bachata-kfc-bucket'
STORAGE_LOCATION = 'Full_dataset/'
FOLDER = 'Data/'
LIST_OF_VIDEOS = 'list_of_videos_to_select.csv'

def import_files_from_bucket(path_to_list_of_videos:str,
                            bucket=BUCKET,
                            storage_location=STORAGE_LOCATION,
                            folder = FOLDER):

    '''downloads all files from a given bucket to a specific folder.
    The output folder must include subfolders 'fake' and 'real'
    The function uses the csv list of video to determine which videos are fake or real'''

    df = pd.read_csv('list_of_videos_to_select.csv')
    df = df.set_index('video_id')
    df = df.sample(frac=1,random_state=0)

    for video in df.index:
        client = storage.Client().from_service_account_json('service_key.json').bucket(bucket)
        blob = client.blob(storage_location + video)
        blob.download_to_filename(folder+df.loc[video,'fake or real']+'/'+video)
        print(colored(f"=> {storage_location + video} downloaded from bucket {bucket} to local path",
                      "green"))
