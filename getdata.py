import glob
import os
#import uuid

import numpy as np
from PIL import Image
import scipy.io as sio

save_dir = 'C:/유민형/개인 연구/BESimGAN/Dataset/real9/'

import random

a = range(427317)
b = random.sample(a, len(a))

def butchered_mp_normalized_matlab_helper(mat_file_path, idx, num):
    """
    Normalized data is provided in matlab files in MPIIGaze Dataset and these are tricky to load with Python.
    This function was made with guessing and checking. Very frustrating.
    :param mat_file_path: Full path to MPIIGaze Dataset matlab file.
    :return: np array of images.
    """
    x = sio.loadmat(mat_file_path)
    y = x.get('data')
    z = y[0, 0]

    left_imgs = z['left']['image'][0, 0]
    right_imgs = z['right']['image'][0, 0]
    
    cnt = idx
    for img in np.concatenate((left_imgs, right_imgs)):
        name = num.pop() + 1
        Image.fromarray(img).resize((55, 35), resample=Image.ANTIALIAS).save(os.path.join(save_dir, '%d.png' % name))
        cnt += 1

    return cnt

if __name__ == '__main__':
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    idx = 1
    for filename in glob.iglob('C:/유민형/개인 연구/BESimGAN/Dataset/eye-gaze/normalized/**/*.mat', recursive=True):
        print(filename)
        idx = butchered_mp_normalized_matlab_helper(filename, idx, b)
        print(idx)
