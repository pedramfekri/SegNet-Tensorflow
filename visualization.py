import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import os
import pandas as pd
import pathlib


list_train = pd.read_csv('/home/pedram/caffe-segnet-cudnn-v/models/SegNet-Tutorial/CamVid/train-tf.txt', header=None)
img_dir_train = '/home/pedram/caffe-segnet-cudnn-v/models/SegNet-Tutorial/CamVid/train/'
mask_dir_train = '/home/pedram/caffe-segnet-cudnn-v/models/SegNet-Tutorial/CamVid/trainannot/'

im = np.zeros([360, 480, 3])

original_mask = cv2.imread(mask_dir_train + list_train.iloc[200, 0])
# print(original_mask[100:130, 100:130, 0])
for i in range(original_mask.shape[0]):
    for j in range(original_mask.shape[1]):
        if original_mask[i, j, 0] == 1:
            im[i, j, 0] = 128
            im[i, j, 1] = 128
            im[i, j, 2] = 128
            # print(1)
        elif original_mask[i, j, 0] == 2:
            im[i, j, 0] = 128
            im[i, j, 1] = 0
            im[i, j, 2] = 0
            # print(2)
        elif original_mask[i, j, 0] == 3:
            im[i, j, 0] = 192
            im[i, j, 1] = 192
            im[i, j, 2] = 128
            # print(3)
        elif original_mask[i, j, 0] == 4:
            im[i, j, 0] = 255
            im[i, j, 1] = 69
            im[i, j, 2] = 0
            # print(4)
        elif original_mask[i, j, 0] == 5:
            im[i, j, 0] = 128
            im[i, j, 1] = 64
            im[i, j, 2] = 128
            # print(5)
        elif original_mask[i, j, 0] == 6:
            im[i, j, 0] = 60
            im[i, j, 1] = 40
            im[i, j, 2] = 222
            # print(6)
        elif original_mask[i, j, 0] == 7:
            im[i, j, 0] = 128
            im[i, j, 1] = 128
            im[i, j, 2] = 0
            # print(7)
        elif original_mask[i, j, 0] == 8:
            im[i, j, 0] = 192
            im[i, j, 1] = 128
            im[i, j, 2] = 128
            # print(8)
        elif original_mask[i, j, 0] == 9:
            im[i, j, 0] = 64
            im[i, j, 1] = 64
            im[i, j, 2] = 128
            # print(9)
        elif original_mask[i, j, 0] == 10:
            im[i, j, 0] = 64
            im[i, j, 1] = 0
            im[i, j, 2] = 128
            # print(10)
        elif original_mask[i, j, 0] == 11:
            im[i, j, 0] = 64
            im[i, j, 1] = 64
            im[i, j, 2] = 0
            # print(11)
        else:
            im[i, j, 0] = 0
            im[i, j, 1] = 0
            im[i, j, 2] = 0
            print(original_mask[i, j, 0])


im = im / 255
plt.imshow(im)
plt.show()



