# Packages
import os
import math
import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook
import cv2
from matplotlib.patches import Rectangle
from tensorflow.keras.layers import Conv2D, Input, ZeroPadding2D, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
import tensorflow as tf
from Deep_Tryhard_DFDC.params import MODEL_MOBILENETV2
from Deep_Tryhard_DFDC.params import *
from Deep_Tryhard_DFDC.utils import *
from Deep_Tryhard_DFDC.entity_selection import *
tf.compat.v1.disable_eager_execution()


### FACE DETECTION CLASS ###

class FaceDetector():
    """
    That's API you can easily use to detect faces

    __init__ parameters:
    -------------------------------
    model - model to infer
    shots - list of aspect ratios that images could be (described earlier)
    image_size - model's input size (hardcoded for mobilenetv2)
    grids - model's output size (hardcoded for mobilenetv2)
    union_threshold - threshold for union of predicted boxes within multiple shots
    iou_threshold - IOU threshold for non maximum suppression used to merge YOLO detected boxes for one shot,
                    you do need to change this because there are one face per image as I can see from the samples
    prob_threshold - probability threshold for YOLO algorithm, you can balance beetween precision and recall using this threshold

    detect parameters:
    -------------------------------
    frame - (1920, 1080, 3) or (1080, 1920, 3) RGB Image
    returns: list of 4 element tuples (left corner x, left corner y, right corner x, right corner y) of detected boxes within [0, 1] range (see box draw code below)
    """
    def __init__(self, model=MODEL_MOBILENETV2, shots=[SHOTS['10-16/9'], SHOTS_T['10-9/16']], image_size=224, grids=7, iou_threshold=0.2, union_threshold=0.1):
        self.model = model
        self.shots = shots
        self.image_size = image_size
        self.grids = grids
        self.iou_threshold = iou_threshold
        self.union_threshold = union_threshold
        self.prob_threshold = 0.8


    def detect(self, frame, threshold = 0.8):
        original_frame_shape = frame.shape
        self.prob_threshold = threshold
        aspect_ratio = None
        for shot in self.shots:
            if abs(frame.shape[1] / frame.shape[0] - shot["aspect_ratio"]) < 1e-9:
                aspect_ratio = shot["aspect_ratio"]
                shots = shot

        assert aspect_ratio is not None

        c = min(frame.shape[0], frame.shape[1] / aspect_ratio)
        slice_h_shift = r((frame.shape[0] - c) / 2)
        slice_w_shift = r((frame.shape[1] - c * aspect_ratio) / 2)
        if slice_w_shift != 0 and slice_h_shift == 0:
            frame = frame[:, slice_w_shift:-slice_w_shift]
        elif slice_w_shift == 0 and slice_h_shift != 0:
            frame = frame[slice_h_shift:-slice_h_shift, :]

        frames = []
        for s in shots["shots"]:
            frames.append(cv2.resize(frame[r(s[1] * frame.shape[0]):r((s[1] + s[3]) * frame.shape[0]), r(s[0] * frame.shape[1]):r((s[0] + s[2]) * frame.shape[1])], (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST))
        frames = np.array(frames)

        predictions = self.model.predict(frames, batch_size=len(frames), verbose=0)

        boxes = []
        prob = []
        shots = shots['shots']
        for i in range(len(shots)):
            slice_boxes = []
            slice_prob = []
            for j in range(predictions.shape[1]):
                for k in range(predictions.shape[2]):
                    p = sigmoid(predictions[i][j][k][4])
                    if not(p is None) and p > self.prob_threshold:
                        px = sigmoid(predictions[i][j][k][0])
                        py = sigmoid(predictions[i][j][k][1])
                        pw = min(math.exp(predictions[i][j][k][2] / self.grids), self.grids)
                        ph = min(math.exp(predictions[i][j][k][3] / self.grids), self.grids)
                        if not(px is None) and not(py is None) and not(pw is None) and not(ph is None) and pw > 1e-9 and ph > 1e-9:
                            cx = (px + j) / self.grids
                            cy = (py + k) / self.grids
                            wx = pw / self.grids
                            wy = ph / self.grids
                            if wx <= shots[i][4] and wy <= shots[i][4]:
                                lx = min(max(cx - wx / 2, 0), 1)
                                ly = min(max(cy - wy / 2, 0), 1)
                                rx = min(max(cx + wx / 2, 0), 1)
                                ry = min(max(cy + wy / 2, 0), 1)

                                lx *= shots[i][2]
                                ly *= shots[i][3]
                                rx *= shots[i][2]
                                ry *= shots[i][3]

                                lx += shots[i][0]
                                ly += shots[i][1]
                                rx += shots[i][0]
                                ry += shots[i][1]

                                slice_boxes.append([lx, ly, rx, ry])
                                slice_prob.append(p)

            slice_boxes = np.array(slice_boxes)
            slice_prob = np.array(slice_prob)

            slice_boxes = non_max_suppression(slice_boxes, slice_prob, self.iou_threshold)

            for sb in slice_boxes:
                boxes.append(sb)


        boxes = np.array(boxes)
        boxes = union_suppression(boxes, self.union_threshold)

        for i in range(len(boxes)):
            boxes[i][0] /= original_frame_shape[1] / frame.shape[1]
            boxes[i][1] /= original_frame_shape[0] / frame.shape[0]
            boxes[i][2] /= original_frame_shape[1] / frame.shape[1]
            boxes[i][3] /= original_frame_shape[0] / frame.shape[0]

            boxes[i][0] += slice_w_shift / original_frame_shape[1]
            boxes[i][1] += slice_h_shift / original_frame_shape[0]
            boxes[i][2] += slice_w_shift / original_frame_shape[1]
            boxes[i][3] += slice_h_shift / original_frame_shape[0]

        return list(boxes)


### PERFORM A FULL TRANSFORMATION FROM VIDEO TO CROPPED DETECTED FACES ###
def from_video_to_faces(video_path, writing_path, framerate):
    '''
    __init__ parameters:
    -------------------------------
    video_path - path directly to the video - string
    writing_path - path where the cropped detected face will be saved - string
    framerate - delay you want between each frame /!\ First frame is always selected. /!\
    For exemple --- framerate = 30 means you try to detect a face every 30 frames

    __process__:
    -------------------------------
    1. perform a video capture from the video
    2. read the image
    3. apply the yolo model
    4. saved the detected face

    return : cropped pictured saved to the specified path
    '''
    # detector = FaceDetector() à ajouter dans le package
    count = 0
    capture_image = cv2.VideoCapture(video_path)
    while capture_image.isOpened() and count<271: # A changer, mettre 300 - frammerate
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
                cv2.imwrite(os.path.join(writing_path , os.path.basename(video_path).replace(".mp4", "")+str(count)+'.jpg'), frame[ly:ry, lx:rx])
            count += framerate # i.e. at 30 fps, this advances one second
            capture_image.set(cv2.CAP_PROP_POS_FRAMES, count)

        else:
            capture_image.release()
            break
    capture_image.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":

    # Initialize list of videos
    fake_train_sample_video = os.listdir('Data/fake')[0:1000]
    real_train_sample_video = os.listdir('Data/real')[0:1000]

    # Load model
    mobilenetv2 = load_mobilenetv2_224_075_detector(MODEL_MOBILENETV2)

    # Initialize FaceDetector
    detector = FaceDetector()

    # Initialize logs
    fail=0
    success=0
    total=0
    log=[]

    # Loop over the selected list
    for video_file in fake_train_sample_video:
        try:
            from_video_to_faces(os.path.join(TRAIN_FOLDER_REAL, video_file), writing_path=PATH_WRITE_REAL, framerate=30)
            success+=1
            total+=1
        except:
            log.append(video_file)
            fail+=1
            total+=1
            pass

    # Save to specific
    text_file = open("log_success_1000.txt", "w")
    n = text_file.write('\n'.join(log))
    text_file.close()

    print(f'Il y a {fail} vidéos fail sur un total de {total} vidéos soit {(fail/total)*100}% de fail')
    print(f'Il y a {success} vidéos qui fonctionnent sur un total de {total} vidéos soit {(success/total)*100}% de réussies')

    # Fast resize
    fast_resize(path=PATH_WRITE_REAL, width=224, height=224)
