'''
This module allows one to process a downloaded video in order
to classify it as DeepFake or Real.
'''
import numpy as np
import pandas as pd
from Deep_Tryhard_DFDC.Face_detection import FaceDetector
from Deep_Tryhard_DFDC.Import_model import get_model
from Deep_Tryhard_DFDC.utils import *
from Deep_Tryhard_DFDC.entity_selection import load_mobilenetv2_224_075_detector
import cv2
import matplotlib.pyplot as plt

#stuff done here to save time of inference :

#loading the benjamin model to predict if a face is a deepfake
deepfake_detection_model = get_model('models/model_T2PV_KERsize_3_MP_4_Denses_128_16_drop_10/model_T2PV_KERsize_3_MP_4_Denses_128_16_drop_10.joblib')

def selecting_10_faces_from_video(video_path):
    '''
    This function takes a video path, processes the video and then returns 10 faces as numpy arrays.
    It has been tested on several videos on the dataset.
    '''
    # Load the mobilenet model tha can detect faces
    mobilenetv2 = load_mobilenetv2_224_075_detector('../models/facedetection-mobilenetv2-size224-alpha0.75.h5')
    # Initialize FaceDetector
    detector = FaceDetector(model = mobilenetv2)
    capture_image = cv2.VideoCapture(video_path)
    #iterating to save pictures
    count = 0
    face_list = []
    while capture_image.isOpened() and count<271:
        ret, frame = capture_image.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes = detector.detect(frame)
            # lets's draw boxes, just multiply each predicted [0, 1] relative coordinate to image side in pixels respectively
            for box in boxes:
                lx = int(round(box[0] * frame.shape[1]))
                ly = int(round(box[1] * frame.shape[0]))
                rx = int(round(box[2] * frame.shape[1]))
                ry = int(round(box[3] * frame.shape[0]))
                face_list.append(frame[ly:ry, lx:rx])
            count += 30 # i.e. at 30 fps, this advances 2 second
            capture_image.set(cv2.CAP_PROP_POS_FRAMES, count)
        else:
            capture_image.release()
            break
    capture_image.release()
    #resizing faces, changing shape to (224,224,3)
    face_list = [cv2.resize(face, (224,224)).astype(np.int16) for face in face_list]
    return face_list

def probabilities_foreach_face(face_list):
    '''
    This function takes a face list and apply a model on them
    to get the probability that each one is a DeepFake.
    '''
    #loading the model that predicts if a face is a deepfake
    deepfake_detection_model = get_model('../models/model_T2PV_KERsize_3_MP_4_Denses_128_16_drop_10/model_T2PV_KERsize_3_MP_4_Denses_128_16_drop_10.joblib')

    #adding 1 dimension, changing shape of faces to (None, 224,224,3) to be processed by the TF model.
    face_list = [np.expand_dims(face, 0) for face in face_list]

    #calculating the proba that each face is a deepfake and changing the format
    predict_proba_per_face = [1 - deepfake_detection_model.predict(face)[0,0] for face in face_list]

    return predict_proba_per_face

def classify_from_probabilities(probabilities):
    '''
    This function classify a list of probabilities as DeepFake (=1) or Real (=0)
    '''
    threshold = 0.45
    if np.mean(probabilities) > threshold:
        return 1
    return 0
