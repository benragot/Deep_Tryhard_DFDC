import os
import random
import cv2
from PIL import Image

def move_test_files(data_file,path = '', fake = 'fake',real = 'real', x_fakes = 0.2, x_reals = 0.2):
    '''
    Function to split our train data into x% of test data.

    data_file : name of the data_train folder
    path : path to folder {data_file}, None when calling the function at the root
    fake : the name of the fake subfolder
    real : the name of the real subfolder
    x_fakes / x_reals : ratio of fakes and ratio of reals in case we want more fakes


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
