from turtle import down
import streamlit as st
import numpy as np
import pandas as pd
from Deep_Tryhard_DFDC.Face_detection import FaceDetector
from Deep_Tryhard_DFDC.Import_model import get_model
from Deep_Tryhard_DFDC.utils import *
from Deep_Tryhard_DFDC.entity_selection import load_mobilenetv2_224_075_detector
import cv2
import tempfile
import matplotlib.pyplot as plt

st.title("Deepfake detection")

uploaded_file = st.file_uploader("Gimme some video",type=["mp4","mov","mkv","wmv"])
tfile = tempfile.NamedTemporaryFile(delete=False)

tfile.write(uploaded_file.read())
capture_image = cv2.VideoCapture(tfile.name)

# Load model
mobilenetv2 = load_mobilenetv2_224_075_detector('facedetection-mobilenetv2-size224-alpha0.75.h5')

# Initialize FaceDetector
detector = FaceDetector(model = mobilenetv2)

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
        count += 30 # i.e. at 30 fps, this advances one second
        capture_image.set(cv2.CAP_PROP_POS_FRAMES, count)

    else:
        capture_image.release()
        break
capture_image.release()
cv2.destroyAllWindows()

#displaying faces
#for face in face_list:

st.title('Selecting 10 frames from the video and identifying faces')

fig, ax = plt.subplots(1, 1, figsize=(3,3))
plt.axis('off')
ax.imshow(face_list[0])
st.pyplot(fig)

deepfake_detection_model = get_model('model_simple_32_neurons_dense.joblib')
face_for_pred = np.expand_dims(face_list[0], axis = 0)

st.write(deepfake_detection_model.predict(face_list[0]))
