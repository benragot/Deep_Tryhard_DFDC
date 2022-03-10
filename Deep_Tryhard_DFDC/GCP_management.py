'''
This module includes functions to manage files (uploads and downloads) on Google Cloud Storage.
You have to check params.py to change the direct path of you SERVICE_KEY
'''
import os
from google.cloud import storage
from termcolor import colored
from Deep_Tryhard_DFDC.parammmms import BUCKET_NAME, MODEL_NAME, MODEL_VERSION, SERVICE_KEY

def storage_upload_file(name,path,storage_location,rm=False):
    '''
    Uploads a file to a given bucket in the params.py (the kfc-bucket),
    thanks to its name and path. It will be stored in bucket/{storage_location}.
    rm options is to be used as True if you also want to remove the uploaded file from
    your system.
    Example :
    storage_upload_file(name='metadata.csv',
                        path='/home/benjamin/code/benragot/Deep_Tryhard_DFDC/raw_data/',
                        storage_location="tests_buckets/metadata/",
                        rm=False)
    '''
    client = storage.Client().from_service_account_json(SERVICE_KEY).bucket(BUCKET_NAME)
    blob = client.blob(storage_location + name)
    blob.upload_from_filename(path + name)
    print(colored(f"=> {path + name} uploaded to bucket {BUCKET_NAME} inside {storage_location}",
                  "green"))
    if rm:
        os.remove(path + name)

def storage_download_file(name, storage_location, bucket=BUCKET_NAME):
    '''
    Uploads a file to a given bucket in the params.py (the kfc-bucket),
    thanks to its name and path. It will be stored in bucket/{storage_location}.
    Example :
    storage_download_file('metadata.csv', 'tests_buckets/metadata/', bucket=BUCKET_NAME)
    '''
    client = storage.Client().from_service_account_json(SERVICE_KEY).bucket(bucket)
    blob = client.blob(storage_location + name)
    blob.download_to_filename(name)
    print(colored(f"=> {storage_location + name} downloaded from bucket {bucket} to local path",
                  "green"))

if __name__ == "__main__":
    print(f'BUCKET_NAME  {BUCKET_NAME}, MODEL_NAME  {MODEL_NAME}, MODEL_VERSION  {MODEL_VERSION}')
    storage_upload_file(name='metadata.csv',
                        path='/home/benjamin/code/benragot/Deep_Tryhard_DFDC/raw_data/',
                        storage_location="tests_buckets/metadata/",
                        rm=False)
    storage_download_file('metadata.csv', 'tests_buckets/metadata/', bucket=BUCKET_NAME)
