import tensorflow as tf
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import pathlib
from full_model import autoencoder

model = autoencoder()

optimizer = tf.keras.optimizers.Adadelta()

model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])


## set the necessary list
#    train_list = pd.read_csv(args.train_list, header=None)
#    val_list = pd.read_csv(args.val_list, header=None)
#
#    # set the necessary directories
#    trainimg_dir = args.trainimg_dir
#    trainmsk_dir = args.trainmsk_dir
#    valimg_dir = args.valimg_dir
#    valmsk_dir = args.valmsk_dir
#
#    train_gen = data_gen_small(
#        trainimg_dir,
#        trainmsk_dir,
#        train_list,
#        args.batch_size,
#        [args.input_shape[0], args.input_shape[1]],
#        args.n_labels,
#    )
#    val_gen = data_gen_small(
#        valimg_dir,
#        valmsk_dir,
#        val_list,
#        args.batch_size,
#        [args.input_shape[0], args.input_shape[1]],
#        args.n_labels,
#    )
#
#    model = segnet(
#        args.input_shape, args.n_labels, args.kernel, args.pool_size, args.output_mode
#    )
#    print(model.summary())
#
#    model.compile(loss=args.loss, optimizer=args.optimizer, metrics=["accuracy"])
#    model.fit_generator(
#        train_gen,
#        steps_per_epoch=args.epoch_steps,
#        epochs=args.n_epochs,
#        validation_data=val_gen,
#        validation_steps=args.val_steps,
#    )