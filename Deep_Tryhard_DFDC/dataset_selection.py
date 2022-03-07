import re
import os
import numpy as np
from PIL import Image



def tryint(s):
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    l.sort(key=alphanum_key)


def get_diff(images_list):
    temp = np.array([images_list[i].astype('int16')-images_list[i+1].astype('int16') for i in range(len(images_list)-1)])
    values = abs(temp)
    summed_values = [values[i,:,:].sum() for i in range(values.shape[0])]
    return summed_values


def make_dataset_bonus(videos_path, img_path,frames_triplets = 1):
    '''
    Function to select best frames from a video.
    videos_path : path to fake (or real) videos
    img_path : path where fake (or real) videos are

    '''
    train_sample_video = os.listdir(videos_path)
    for vid in range(len(train_sample_video)):
        list_ = [i for i in os.listdir(img_path) if train_sample_video[vid][:-4] in i ]
        sort_nicely(list_)
        newlist_ = [np.array(Image.open(os.path.join(img_path, i))) for i in list_]
        npvalues = np.array(get_diff(newlist_))
        if len(npvalues) > 3:
            range_values = [npvalues[i-1]+npvalues[i]+npvalues[i+1] for i in range(1,len(npvalues)-1)]
            sorted_values = range_values.copy()
            sorted_values.sort()
            new_img = []
            tobekep = []
            for trips in range(frames_triplets):
                best_part = int(range_values.index(sorted_values[-(trips+1)]))
                tobekep.extend(list_[best_part:best_part+3])
                r = re.compile(f"{train_sample_video[vid][:-4]}.*")
                newww = list(filter(r.match, os.listdir(img_path)))
                new_img.extend(newww)
            tobedeleted = []
            for _ in range(len(new_img)):
                if new_img[_] not in tobekep:
                    tobedeleted.append(new_img[_])
            tobedeleted = set(tobedeleted)
            for killed in tobedeleted:
                os.remove(os.path.join(img_path,killed))


def make_dataset(videos_path, img_path):
    '''
    Function to select best frames from a video.
    videos_path : path to fake (or real) videos
    img_path : path where fake (or real) videos are

    '''
    train_sample_video = os.listdir(videos_path)
    for vid in range(len(train_sample_video)):
        list_ = [i for i in os.listdir(img_path) if train_sample_video[vid][:-4] in i ]
        sort_nicely(list_)
        newlist_ = [np.array(Image.open(os.path.join(img_path, i))) for i in list_]
        npvalues = np.array(get_diff(newlist_))
        if len(npvalues) > 3:
            range_values = [npvalues[i-1]+npvalues[i]+npvalues[i+1] for i in range(1,len(npvalues)-1)]
            sorted_values = range_values.copy()
            sorted_values.sort()
            best_part = int(range_values.index(sorted_values[-1]))
            tobekep = list_[best_part:best_part+3]
            r = re.compile(f"{train_sample_video[vid][:-4]}.*")
            newww = list(filter(r.match, os.listdir(img_path)))
            tobedeleted = []
            for _ in range(len(newww)):
                if newww[_] not in tobekep:
                    tobedeleted.append(newww[_])
            for killed in tobedeleted:
                os.remove(os.path.join(img_path,killed))
