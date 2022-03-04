'''this module takes a picture with a face as input and returns an 'aligned face'
It is based on eyes only.
'''

import cv2
from cv2 import detail_AffineBasedEstimator
import numpy as np
import matplotlib.pyplot as plt
import os

def align_faces_in_folder(folder_input:str,
                          folder_output:str,
                          pic_extension = '.jpg',
                          xml_path='Align_face_xml/'):
    '''Takes a folder containing pictures of faces and aligns all the faces, creating new pictures, in the folder_output'''

    #checking that the folder to save ends with /
    if folder_input[-1]!='/':
        raise Exception("folder_input must end with a /")
    if folder_output[-1]!='/':
        raise Exception("folder_output must end with a /")

    #generating a list of picture in the folder_input
    file_list= os.listdir(folder_input)
    pic_list=[file for file in file_list if file.endswith(pic_extension)]

    if not pic_list:
        raise Exception('No picture found in input folder')

    for pic in pic_list:
        align_one_face(img_path=folder_input+pic,
                       folder_to_save=folder_output,
                       xml_path= xml_path)


def align_one_face(img_path:str, folder_to_save:str, xml_path='Align_face_xml/'):
    '''
    This function takes an image with a face in input, in the form of the path: folder/subfolder/picture.jpg
    it returns a face in output, aligned vertically and saves it in a folder_to_save in the form : folder_to_save/subfolder_to_save/ '''

    #checking that the folder to save ends with /
    if folder_to_save[-1]!='/':
        raise Exception("folder_output must end with a /")


    img_name = img_path.split('/')[-1]
    img_name, img_ext = img_name.split('.')

    if not img_ext:
        raise Exception('No picture extension found.\
                        img_path must include the extension of the image, example: folder/subfolder/picture.jpg')

    # Loads the image
    img = cv2.imread(img_path)

    # Creating face_cascade and eye_cascade objects
    face_cascade=cv2.CascadeClassifier(xml_path+"haarcascade_frontalface_default.xml")
    eye_cascade=cv2.CascadeClassifier(xml_path+"haarcascade_eye.xml")

    # Converting the image into grayscale
    gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Creating variable faces
    faces= face_cascade.detectMultiScale(gray, 1.1, 4,)

    # Defining and drawing the rectangle around the face
    x , y,  w,  h = faces[0]
    roi_gray=gray[y:(y+h), x:(x+w)]
    roi_color=img[y:(y+h), x:(x+w)]

    # Creating variable eyes
    eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 4)
    index=0
    # Creating for loop in order to divide one eye from another
    for (ex , ey,  ew,  eh) in eyes:
        if index == 0:
            eye_1 = (ex, ey, ew, eh)
        elif index == 1:
            eye_2 = (ex, ey, ew, eh)
        index = index + 1

    if eye_1[0] < eye_2[0]:
        left_eye = eye_1
        right_eye = eye_2
    else:
        left_eye = eye_2
        right_eye = eye_1

    # Calculating coordinates of a central points of the rectangles
    left_eye_center = (int(left_eye[0] + (left_eye[2] / 2)), int(left_eye[1] + (left_eye[3] / 2)))
    left_eye_x = left_eye_center[0]
    left_eye_y = left_eye_center[1]

    right_eye_center = (int(right_eye[0] + (right_eye[2]/2)), int(right_eye[1] + (right_eye[3]/2)))
    right_eye_x = right_eye_center[0]
    right_eye_y = right_eye_center[1]

    if left_eye_y > right_eye_y:
        A = (right_eye_x, left_eye_y)
        # Integer -1 indicates that the image will rotate in the clockwise direction
        direction = -1
    else:
        A = (left_eye_x, right_eye_y)
        # Integer 1 indicates that image will rotate in the counter clockwise
        # direction
        direction = 1


    delta_x = right_eye_x - left_eye_x
    delta_y = right_eye_y - left_eye_y
    #Delta are the distance between the eyes, vertically and horyzontally
    angle=np.arctan(delta_y/delta_x)
    angle = (angle * 180) / np.pi

    # Width and height of the image
    h, w = img.shape[:2]
    # Calculating a center point of the image
    # Integer division "//"" ensures that we receive whole numbers
    center = (w // 2, h // 2)
    # Defining a matrix M and calling
    # cv2.getRotationMatrix2D method
    M = cv2.getRotationMatrix2D(center, (angle), 1.0)
    # Applying the rotation to our image using the
    # cv2.warpAffine method
    img_rotated = cv2.warpAffine(img, M, (w, h))

    has_black = True
    while has_black :
        top_left_corner_is_black = all(img_rotated[0,0,:] == [0,0,0])
        bottom_left_corner_is_black = all(img_rotated[-1,0,:] ==[0,0,0])
        has_black = top_left_corner_is_black or bottom_left_corner_is_black
        img_rotated=img_rotated[1:-1, 1:-1,:]

    #saves the final image
    cv2.imwrite(folder_to_save+img_name+'_aligned.'+img_ext, img_rotated)

if __name__ == '__main__':
    align_one_face('Align_face_xml/emily.jpg', 'Align_face_xml/')
