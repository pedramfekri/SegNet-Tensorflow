import tensorflow as tf
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import pathlib


def encoder():
    kernel = 3
    filter_size = 64
    pad = 1
    pool_size = 2
    encoder_input = tf.keras.layers.Input(shape=(360, 480, 3), name='input_encoder')

    x = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(encoder_input)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding="valid")(x)
    x = tf.keras.layers.BatchNormalization(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)

    x = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(encoder_input)
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding="valid")(x)
    x = tf.keras.layers.BatchNormalization(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)

    x = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(encoder_input)
    x = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding="valid")(x)
    x = tf.keras.layers.BatchNormalization(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)

    x = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(encoder_input)
    x = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding="valid")(x)
    x = tf.keras.layers.BatchNormalization(x)
    encoder_output = tf.keras.layers.Activation('relu')(x)
    encoder_model = tf.keras.Model(encoder_input, encoder_output, name='encoder')
    return encoder_model


def decoder():
    decoder_input = tf.keras.layers.Input(shape=(512,), name='input_decoder')

    x = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(decoder_input)
    x = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding="valid")(x)
    x = tf.keras.layers.BatchNormalization(x)
    x = tf.keras.layers.UpSampling2D(size=(2, 2))(x)

