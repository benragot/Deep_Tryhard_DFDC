import os
import random

def move_test_files(data_file,path = '', fake = 'fake',real = 'real', x_fakes = 0.2, x_reals = 0.2):
    '''
    Function to split our train data into x% of test data.

    data_file : name of the data_train folder
    path : path to folder {data_file}, None when calling the function at the root
    fake : the name of the fake subfolder
    real : the name of the real subfolder
    x_fakes / x_reals : ratio of fakes and ratio of reals in case we want more fakes

    8==============D

    '''
    fullpath = os.path.join(path,data_file)

    # assert(data_file in os.listdir(path))
    isExist = os.path.exists(fullpath + '_test')
    if not isExist:
        os.mkdir(fullpath + '_test') # Please don't end train folder name with train or TV
    fakes_test_path = os.path.join(fullpath + '_test','fake_test')
    reals_test_path = os.path.join(fullpath + '_test','real_test')
    isExist2 = os.path.exists(fakes_test_path)
    if not isExist2:
        os.makedirs(fakes_test_path)
    isExist3 = os.path.exists(reals_test_path)
    if not isExist3:
        os.makedirs(reals_test_path)
    fakes_path = os.path.join(fullpath,fake)
    reals_path = os.path.join(fullpath,real)
    fake_list = os.listdir(fakes_path)
    real_list = os.listdir(reals_path)

    for i in range(int(len(fake_list)*x_fakes)):
        ffile = fake_list.pop(random.randint(0,len(fake_list)-1 -i))
        os.replace(os.path.join(fakes_path,ffile),os.path.join(fakes_test_path,ffile))

    for i in range(int(len(real_list)*x_reals)):

        rfile = real_list.pop(random.randint(0,len(real_list)-1 -i))
        os.replace(os.path.join(reals_path,rfile),os.path.join(reals_test_path,rfile))

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
from Deep_Tryhard_DFDC.params import *
from PIL import Image
tf.compat.v1.disable_eager_execution()


### LOADING FUNCTION OF THE DETECTION FACE MODEL ###

def load_mobilenetv2_224_075_detector(path):
    """
    MobileNetV2 is a convolutional neural network architecture that seeks to perform well on mobile devices.
    It is based on an inverted residual structure where the residual connections are between the bottleneck layers.

    Layer description:
    -------------------------------
    The intermediate expansion layer uses lightweight depthwise convolutions to filter features
    as a source of non-linearity. As a whole, the architecture of MobileNetV2 contains the initial
    fully convolution layer with 32 filters, followed by 19 residual bottleneck layers.
    """
    input_tensor = Input(shape=(224, 224, 3))
    output_tensor = MobileNetV2(weights=None, include_top=False, input_tensor=input_tensor, alpha=0.75).output
    output_tensor = ZeroPadding2D()(output_tensor)
    output_tensor = Conv2D(kernel_size=(3, 3), filters=5)(output_tensor)
    model = Model(inputs=input_tensor, outputs=output_tensor)
    model.load_weights(path)
    return model


### PRE PROCESSING ###

# Converts A:B aspect rate to B:A
def transpose_shots(shots):
    """
    List of aspect ratios that images could be
    """
    return [(shot[1], shot[0], shot[3], shot[2], shot[4]) for shot in shots]

#That constant describe pieces for 16:9 images
SHOTS = {
    # fast less accurate
    '2-16/9' : {
        'aspect_ratio' : 16/9,
        'shots' : [
             (0, 0, 9/16, 1, 1),
             (7/16, 0, 9/16, 1, 1)
        ]
    },
    # slower more accurate
    '10-16/9' : {
        'aspect_ratio' : 16/9,
        'shots' : [
             (0, 0, 9/16, 1, 1),
             (7/16, 0, 9/16, 1, 1),
             (0, 0, 5/16, 5/9, 0.5),
             (0, 4/9, 5/16, 5/9, 0.5),
             (11/48, 0, 5/16, 5/9, 0.5),
             (11/48, 4/9, 5/16, 5/9, 0.5),
             (22/48, 0, 5/16, 5/9, 0.5),
             (22/48, 4/9, 5/16, 5/9, 0.5),
             (11/16, 0, 5/16, 5/9, 0.5),
             (11/16, 4/9, 5/16, 5/9, 0.5),
        ]
    }
}

# 9:16 respectively
SHOTS_T = {
    '2-9/16' : {
        'aspect_ratio' : 9/16,
        'shots' : transpose_shots(SHOTS['2-16/9']['shots'])
    },
    '10-9/16' : {
        'aspect_ratio' : 9/16,
        'shots' : transpose_shots(SHOTS['10-16/9']['shots'])
    }
}

def r(x):
    return int(round(x))

def sigmoid(x):
    return 1 / (np.exp(-x) + 1)


### FUNCTION TO SELECT ENTITIES IN THE FRAME ###

def non_max_suppression(boxes, p, iou_threshold):
    """
    Non Maximum Suppression (NMS) is a technique used in numerous
    computer vision tasks.

    It is a class of algorithms to select one entity (e.g., bounding boxes)
    out of many overlapping entities. We can choose the selection criteria
    to arrive at the desired results. The criteria are most commonly some form
    of probability number and some form of overlap measure.

    In our case overlap measure is union.

     __init__ parameters:
    -------------------------------
    boxes - detected boxes
    p - probability
    iou_threshold - IOU threshold for non maximum suppression used to merge YOLO detected boxes for one shot,
                    you do need to change this because there are one face per image as I can see from the samples

    return : true boxes

    """

    if len(boxes) == 0:
        return np.array([])

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    indexes = np.argsort(p)
    true_boxes_indexes = []

    while len(indexes) > 0:
        true_boxes_indexes.append(indexes[-1])

        intersection = np.maximum(np.minimum(x2[indexes[:-1]], x2[indexes[-1]]) - np.maximum(x1[indexes[:-1]], x1[indexes[-1]]), 0) * np.maximum(np.minimum(y2[indexes[:-1]], y2[indexes[-1]]) - np.maximum(y1[indexes[:-1]], y1[indexes[-1]]), 0)
        iou = intersection / ((x2[indexes[:-1]] - x1[indexes[:-1]]) * (y2[indexes[:-1]] - y1[indexes[:-1]]) + (x2[indexes[-1]] - x1[indexes[-1]]) * (y2[indexes[-1]] - y1[indexes[-1]]) - intersection)

        indexes = np.delete(indexes, -1)
        indexes = np.delete(indexes, np.where(iou >= iou_threshold)[0])

    return boxes[true_boxes_indexes]


def union_suppression(boxes, threshold):
    if len(boxes) == 0:
        return np.array([])

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    indexes = np.argsort((x2 - x1) * (y2 - y1))
    result_boxes = []

    while len(indexes) > 0:
        intersection = np.maximum(np.minimum(x2[indexes[:-1]], x2[indexes[-1]]) - np.maximum(x1[indexes[:-1]], x1[indexes[-1]]), 0) * np.maximum(np.minimum(y2[indexes[:-1]], y2[indexes[-1]]) - np.maximum(y1[indexes[:-1]], y1[indexes[-1]]), 0)
        min_s = np.minimum((x2[indexes[:-1]] - x1[indexes[:-1]]) * (y2[indexes[:-1]] - y1[indexes[:-1]]), (x2[indexes[-1]] - x1[indexes[-1]]) * (y2[indexes[-1]] - y1[indexes[-1]]))
        ioms = intersection / (min_s + 1e-9)
        neighbours = np.where(ioms >= threshold)[0]
        if len(neighbours) > 0:
            result_boxes.append([min(np.min(x1[indexes[neighbours]]), x1[indexes[-1]]), min(np.min(y1[indexes[neighbours]]), y1[indexes[-1]]), max(np.max(x2[indexes[neighbours]]), x2[indexes[-1]]), max(np.max(y2[indexes[neighbours]]), y2[indexes[-1]])])
        else:
            result_boxes.append([x1[indexes[-1]], y1[indexes[-1]], x2[indexes[-1]], y2[indexes[-1]]])

        indexes = np.delete(indexes, -1)
        indexes = np.delete(indexes, neighbours)

    return result_boxes

def get_boxes_points(boxes, frame_shape):
    result = []
    for box in boxes:
        lx = int(round(box[0] * frame_shape[1]))
        ly = int(round(box[1] * frame_shape[0]))
        rx = int(round(box[2] * frame_shape[1]))
        ry = int(round(box[3] * frame_shape[0]))
        result.append((lx, ly, rx, ry))
    return result


### HELPER FUNCTION TO RESIZE AND SCALE ###

def fast_resize(path, width, height):
    lst_imgs = os.listdir(path)
    for image in lst_imgs:
        im = Image.open(os.path.join(path, image))
        new_image = im.resize((width, height))
        new_image.save(os.path.join(path, image))
        print(new_image)

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

def scale_boxes(boxes, scale_w, scale_h):
    sb = []
    for b in boxes:
        sb.append((b[0] * scale_w, b[1] * scale_h, b[2] * scale_w, b[3] * scale_h))
    return sb
