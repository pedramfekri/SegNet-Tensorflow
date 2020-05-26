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

model = autoencoder()

model.load_weights('/home/pedram/tensorflow-segnet/200.hdf5')
model.save('/home/pedram/tensorflow-segnet/full_model')

print('Converting to TF-TRT FP32...')
conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(precision_mode=trt.TrtPrecisionMode.FP32,
                                                               max_workspace_size_bytes=8000000000)

converter = trt.TrtGraphConverterV2(input_saved_model_dir='/home/pedram/tensorflow-segnet/full_model',
                                    conversion_params=conversion_params)
converter.convert()
converter.save(output_saved_model_dir='/home/pedram/tensorflow-segnet/full_model_TFTRT_FP32')
print('Done Converting to TF-TRT FP32')

