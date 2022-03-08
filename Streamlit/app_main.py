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
import matplotlib.patches as patches
import imageio
import datetime

#stuff done here to save time of inference :
# Load model
mobilenetv2 = load_mobilenetv2_224_075_detector('facedetection-mobilenetv2-size224-alpha0.75.h5')

# Initialize FaceDetector
detector = FaceDetector(model = mobilenetv2)

#loading the benjamin model to predict if a face is a deepfake
deepfake_detection_model = get_model('model_simple_32_neurons_dense.joblib')

def subplotting_10_faces_with_proba(uploaded_file):
    start_time = datetime.datetime.now()
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    capture_image = cv2.VideoCapture(tfile.name)

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

    #displaying faces
    #for face in face_list:

    #resizing faces, changing shape to (224,224,3)
    face_list = [cv2.resize(face, (224,224)) for face in face_list]

    #adding 1 dimension, changing shape of faces to (None, 224,224,3)
    face_list = [np.expand_dims(face, 0) for face in face_list]

    #calculating the proba that each face is a deepfake and changing the format
    predict_proba_per_face = [deepfake_detection_model.predict(face)[0,0] for face in face_list]
    predict_proba_per_face = [str(int(round(proba*100)))+'%' for proba in predict_proba_per_face]

    fig, ax = plt.subplots(2, 5, figsize=(12,4))
    plt.axis('off')
    for i in range(2):
        for j in range(5):
            ax[i,j].imshow(np.reshape(face_list[i*5+j],(224,224,3)))
            ax[i,j].set_title(predict_proba_per_face[i*5+j])
            ax[i,j].axis('off')
    st.title('Selecting 10 faces from the video and identifying faces')
    st.pyplot(fig)

    end_time = datetime.datetime.now()
    display_time = True
    if display_time:
        st.text(f"Total time of inference : {(end_time - start_time)}")
def gifing_faces_with_proba(uploaded_file):
    start_time = datetime.datetime.now()
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    capture_image = cv2.VideoCapture(tfile.name)

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
            count += 5 # i.e. at 30 fps, this advances one second
            capture_image.set(cv2.CAP_PROP_POS_FRAMES, count)

        else:
            capture_image.release()
            break
    capture_image.release()

    #resizing faces, changing shape to (224,224,3)
    face_list = [cv2.resize(face, (224,224)) for face in face_list]

    #adding 1 dimension, changing shape of faces to (None, 224,224,3)
    face_list = [np.expand_dims(face, 0) for face in face_list]

    #calculating the proba that each face is a deepfake and changing the format
    predict_proba_per_face = [deepfake_detection_model.predict(face)[0,0] for face in face_list]
    predict_proba_per_face = [str(int(round(proba*100)))+'%' for proba in predict_proba_per_face]

    for i in range(len(face_list)):
        plt.close()
        plt.imshow(np.reshape(face_list[i],(224,224,3)))
        plt.title(predict_proba_per_face[i])
        plt.axis('off')
        plt.savefig(f'Streamlit/{i}.jpg')
        #todo see graphs of evol of proba
    images = []
    nb_images_per_face = 2
    for i in range(len(face_list)):
        MY_ARRAY = imageio.imread(f'Streamlit/{i}.jpg')
        for __ in range(nb_images_per_face):
            images.append(MY_ARRAY)
    imageio.mimsave('Streamlit/dumb.gif', images)
    st.title('GIF from faces ! ')
    st.image('Streamlit/dumb.gif')
    end_time = datetime.datetime.now()
    display_time = True
    if display_time:
        st.text(f"Total time of inference : {(end_time - start_time).seconds}.")

def gifing_images_with_bounding_box(video):
    #monitoring the inference time !
    start_time = datetime.datetime.now()
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    capture_image = cv2.VideoCapture(tfile.name)

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
            count += 5 # i.e. at 30 fps, this advances one second
            capture_image.set(cv2.CAP_PROP_POS_FRAMES, count)

        else:
            capture_image.release()
            break
    capture_image.release()

    #resizing faces, changing shape to (224,224,3)
    face_list = [cv2.resize(face, (224,224)) for face in face_list]

    #adding 1 dimension, changing shape of faces to (None, 224,224,3)
    face_list = [np.expand_dims(face, 0) for face in face_list]

    #calculating the proba that each face is a deepfake and changing the format
    predict_proba_per_face = [deepfake_detection_model.predict(face)[0,0] for face in face_list]
    predict_proba_per_face = [str(int(round(proba*100)))+'%' for proba in predict_proba_per_face]

    for i in range(len(face_list)):
        plt.close()
        plt.imshow(np.reshape(face_list[i],(224,224,3)))
        plt.title(predict_proba_per_face[i])
        plt.axis('off')
        plt.savefig(f'Streamlit/{i}.jpg')
        #todo see graphs of evol of proba
    images = []
    nb_images_per_face = 2
    for i in range(len(face_list)):
        MY_ARRAY = imageio.imread(f'Streamlit/{i}.jpg')
        for __ in range(nb_images_per_face):
            images.append(MY_ARRAY)
    imageio.mimsave('Streamlit/dumb.gif', images)
    st.title('GIF from faces ! ')
    st.image('Streamlit/dumb.gif')
    end_time = datetime.datetime.now()
    display_time = True
    if display_time:
        st.text(f"Total time of inference : {(end_time - start_time).seconds}.")

def gif_with_face_detection(video):
    #monitoring the inference time !
    start_time = datetime.datetime.now()
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    capture_image = cv2.VideoCapture(tfile.name)

    #iterating to save pictures
    count = 0
    frame_list = []
    face_list = []
    box_coord = []
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
                frame_list.append(frame)
                box_coord.append([ly, ry, lx, rx])
            count += 3 # we are advance of 'count' frames.
            capture_image.set(cv2.CAP_PROP_POS_FRAMES, count)

        else:
            capture_image.release()
            break
    capture_image.release()

    #resizing faces, changing shape to (224,224,3)
    face_list = [cv2.resize(face, (224,224)) for face in face_list]

    #adding 1 dimension, changing shape of faces to (None, 224,224,3) to be able to predict on them
    face_list = [np.expand_dims(face, 0) for face in face_list]

    #calculating the proba that each face is a deepfake and changing the format
    predict_proba_per_face = [deepfake_detection_model.predict(face)[0,0] for face in face_list]
    predict_proba_per_face = [str(int(round(proba*100)))+'%' for proba in predict_proba_per_face]

    for i in range(len(frame_list)):
        plt.close()
        fig, ax = plt.subplots(figsize=(8,4))
        ax.imshow(frame_list[i])
        ax.add_patch(patches.Rectangle((int(box_coord[i][2]), int(box_coord[i][0])),
                                       (int(box_coord[i][3]) - int(box_coord[i][2])),
                                       (int(box_coord[i][1]) - int(box_coord[i][0])),
                                       linewidth=2,
                                       edgecolor='r',
                                       facecolor='none',
                                       label = 'Label'))
        centerx = int(box_coord[i][2]) - 20 #the minus 20 is to make sur we don't plot the proba on the bounding box
        centery = int(box_coord[i][0]) - 20
        plt.text(centerx,centery,predict_proba_per_face[i],c='r',fontsize=20)
        #ax.title.set_text(predict_proba_per_face[i])
        ax.axis('off')
        fig.savefig(f'Streamlit/{i}.jpg')
        #todo see graphs of evol of proba
    #let's create the gif !
    images = []
    #nb_images_per_face is how many time we are going to repeat each frame in the gif.
    nb_images_per_face = 1
    for i in range(len(frame_list)):
        MY_ARRAY = imageio.imread(f'Streamlit/{i}.jpg')
        for __ in range(nb_images_per_face): #repeating to add nb_images_per_face images
            images.append(MY_ARRAY)
    #saving the gif.
    imageio.mimsave('Streamlit/dumb.gif', images)
    st.title('GIF from faces ! ')
    st.image('Streamlit/dumb.gif')
    end_time = datetime.datetime.now()
    display_time = True
    if display_time:
        st.text(f"Total time of inference : {(end_time - start_time).seconds} seconds.")

#actual code of the page :

st.title("DeepFake Detection Challenge")

#Uploading the file :
uploaded_file = st.file_uploader("Gimme some video",type=["mp4","mov","mkv","wmv"])
#if there is a file then we do something.
if uploaded_file:
    #todo play la video
    video = uploaded_file
    st.video(video)
    if st.button('Click me to subplot 10 faces and their associated probabilities of being fake !'):
        # print is visible in the server output, not in the page
        subplotting_10_faces_with_proba(video)
    else:
        pass

    if st.button('Click me to get a cute gif !'):
        # print is visible in the server output, not in the pacenterxge
        gifing_faces_with_proba(video)
    else:
        pass

    if st.button('Click me to get a bounedinegue bauxe ! !'):
        # print is visible in the server output, not in the page
        gif_with_face_detection(video)
    else:
        pass
