'''This predict module offers function that work with the API'''

from google.cloud import storage
import joblib
import os

def download_model(model_name,
                   model_directory="models",
                   bucket='bachata-kfc-bucket',
                   rm=True):
    #AJOUTER LE MODEL AU BUCKET GOOGLE

    client = storage.Client().bucket(bucket)

    storage_location = '{}/{}'.format(
        model_directory,
        model_name)
    blob = client.blob(storage_location)
    blob.download_to_filename('model.joblib')
    print("=> pipeline downloaded from storage")
    model = joblib.load('model.joblib')
    if rm:
        os.remove('model.joblib')
    return model


def get_model(path_to_joblib):
    pipeline = joblib.load(path_to_joblib)
    return pipeline

if __name__ == '__main__':
    # tests
    pass
