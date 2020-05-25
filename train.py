import tensorflow as tf
tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0],True)
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import pathlib
from full_model import autoencoder
from data_preparation import data_gen_small

list_train = pd.read_csv('/home/pedram/caffe-segnet-cudnn-v/models/SegNet-Tutorial/CamVid/train-tf.txt', header=None)
img_dir_train = '/home/pedram/caffe-segnet-cudnn-v/models/SegNet-Tutorial/CamVid/train/'
mask_dir_train = '/home/pedram/caffe-segnet-cudnn-v/models/SegNet-Tutorial/CamVid/trainannot/'

list_test = pd.read_csv('/home/pedram/caffe-segnet-cudnn-v/models/SegNet-Tutorial/CamVid/test-tf.txt', header=None)
img_dir_test = '/home/pedram/caffe-segnet-cudnn-v/models/SegNet-Tutorial/CamVid/test/'
mask_dir_test = '/home/pedram/caffe-segnet-cudnn-v/models/SegNet-Tutorial/CamVid/testannot/'

list_val = pd.read_csv('/home/pedram/caffe-segnet-cudnn-v/models/SegNet-Tutorial/CamVid/val-tf.txt', header=None)
img_dir_val = '/home/pedram/caffe-segnet-cudnn-v/models/SegNet-Tutorial/CamVid/val/'
mask_dir_val = '/home/pedram/caffe-segnet-cudnn-v/models/SegNet-Tutorial/CamVid/valannot/'

input_shape = (256, 256)

n_labels = 12

batch = 10

train_gen = data_gen_small(img_dir_train,
                           mask_dir_train,
                           list_train,
                           batch,
                           input_shape,
                           n_labels,
                           )

val_gen = data_gen_small(img_dir_val,
                         mask_dir_val,
                         list_val,
                         batch,
                         input_shape,
                         n_labels,
                        )

model = autoencoder()
optimizer = tf.keras.optimizers.Adadelta()
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

epoch_steps = 200
n_epochs = 200
val_steps = 30

model.fit_generator(
                   train_gen,
                   steps_per_epoch=epoch_steps,
                   epochs=n_epochs,
                   validation_data=val_gen,
                   validation_steps=val_steps,
               )

model.save_weights('/home/pedram/tensorflow-segnet/' + str(n_epochs) + ".hdf5")
print("sava weight done..")
