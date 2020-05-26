from __future__ import absolute_import, division, print_function, unicode_literals
import os
import time

import tensorflow as tf
tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0],True)
from tensorflow import keras
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.python.saved_model import tag_constants
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import pathlib
from full_model import autoencoder
from data_preparation import data_gen_small
from tensorflow.keras.preprocessing.image import img_to_array

list_train = pd.read_csv('/home/pedram/caffe-segnet-cudnn-v/models/SegNet-Tutorial/CamVid/train-tf.txt', header=None)
img_dir_train = '/home/pedram/caffe-segnet-cudnn-v/models/SegNet-Tutorial/CamVid/train/'
mask_dir_train = '/home/pedram/caffe-segnet-cudnn-v/models/SegNet-Tutorial/CamVid/trainannot/'
dims = (256, 256)
img_path = img_dir_train + list_train.iloc[200, 0]
original_img = cv2.imread(img_path)[:, :, ::-1]
resized_img = cv2.resize(original_img, dims)
array_img = img_to_array(resized_img) / 255
array_img = np.expand_dims(array_img, axis=0)

saved_model_loaded = tf.saved_model.load('/home/pedram/tensorflow-segnet/full_model_TFTRT_FP32', tags=[tag_constants.SERVING])
signature_keys = list(saved_model_loaded.signatures.keys())
print(signature_keys)

infer = saved_model_loaded.signatures['serving_default']
print(infer.structured_outputs)

x = tf.constant(array_img)
out = infer(x)


# print(out[0, 0, :])
# # out = np.reshape(out[0, :, :], (256, 256, 12))
# out = out[0, :, :].reshape(256, 256, 12)
# print(out.shape)
# print(np.argmax(out[0, 0, :]))
#
# color = [[128,128,128]
#         ,[128,0,0]
#         ,[192,192,128]
#         ,[255,69,0]
#         ,[128,64,128]
#         ,[60,40,222]
#         ,[128,128,0]
#         ,[192,128,128]
#         ,[64,64,128]
#         ,[64,0,128]
#         ,[64,64,0]
#         ,[0,128,192]
#         ,[0,0,0]]
#
# im = np.zeros([256, 256, 3])
#
# for i in range(im.shape[0]):
#     for j in range(im.shape[1]):
#         im[i, j, :] = color[np.argmax(out[i, j, :])]
#
# im = im / 255
# plt.imshow(im)
# plt.show()