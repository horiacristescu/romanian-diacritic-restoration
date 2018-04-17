#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from lol import *
from debug import *

import re, random, sys, io
import numpy as np
from time import time

import madoka

import keras
from keras.callbacks import Callback, ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Activation, Bidirectional, Input, Concatenate
from keras.layers import Embedding, Reshape, Permute, Conv1D, TimeDistributed, Dropout
from keras.layers import LSTM, GRU, LeakyReLU, BatchNormalization, CuDNNLSTM, Lambda
from keras.layers import GlobalAveragePooling1D, multiply
from keras.layers.crf import CRF
from keras import backend as K
from keras.optimizers import RMSprop, Adam
from keras.utils.data_utils import get_file

from preprocessing import *

batch_size = 64
nb_valid = 5000
max_word_hash = 1000000

class LearningRateTracker(Callback):
    def on_epoch_begin(self, epoch, logs=None):
        lr = self.model.optimizer.lr
        # If you want to apply decay.
        decay = self.model.optimizer.decay
        iterations = self.model.optimizer.iterations
        lr_with_decay = lr / (1. + decay * K.cast(iterations, K.dtype(decay)))
        print("Learning rate:", K.eval(lr_with_decay))

def train():

    bg = BatchGenerator()
    x_chars_valid, x_words_valid, word_char_tensor_valid, y_chars_valid = bg.batch_generator(max_word=max_word_hash, forValidation=True)

    print("Validation on:", x_chars_valid.shape[0], "examples")

    print('Build model...')

    def model_def():
        input_char = Input(shape=(None, ))
        input_word = Input(shape=(None, ))
        input_map  = Input(shape=(None, None))

        char_embed = Embedding(char_vocab_size, 50)(input_char)

        char_pipe = Conv1D(128, 31, name="conv_size_31", activation='relu', padding='same')(char_embed)
        char_pipe = Conv1D(128, 21, name="conv_size_21", activation='relu', padding='same')(char_pipe)
        char_pipe = Conv1D(128, 15, name="conv_size_15", activation='relu', padding='same')(char_pipe)

        word_pipe = Embedding(max_word_hash, 50, name='embed_word')(input_word)
        word_pipe = Bidirectional(LSTM(50, return_sequences=True))(word_pipe) # (None, 27, 100)

        # map word space to char space
        input_map_p = Lambda(lambda x: K.permute_dimensions(x, (0,2,1)), name='transpose_map')(input_map)
        word_pipe = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[2,1]), name='project_words_chars')([input_map_p, word_pipe])

        pipe = keras.layers.concatenate([word_pipe, char_pipe, char_embed], axis=-1)
        pipe = Conv1D(128, 11, name="conv_size_11", activation='relu', padding='same')(pipe)
        pipe = Conv1D(128, 7,  name="conv_size_7",  activation='relu', padding='same')(pipe)
        pipe = Conv1D(128, 3,  name="conv_size_3",  activation='relu', padding='same')(pipe)

        output = TimeDistributed(Dense(4, activation='softmax'))(pipe)

        model = Model([input_char, input_word, input_map], output)
        optimizer = Adam(lr=0.001)
        model.compile(
            loss='categorical_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])
        return model

    model = model_def()
    model.load_weights("diacritice.lstm.keras.model")
    K.set_value(model.optimizer.lr, 0.00001)

    #DBG()

    print(model.summary())

    def batch_generator():
        while True:
            x_chars, x_words, word_char_tensor, y_chars = bg.batch_generator(max_word=max_word_hash)
            yield [x_chars, x_words, word_char_tensor], y_chars

    checkpoint = ModelCheckpoint(
        "diacritice.lstm.keras.model", 
        monitor='val_acc', 
        verbose=1, 
        save_best_only=True, 
        mode='max')

    learning_rate_reduction = ReduceLROnPlateau(
        monitor='val_acc',
        patience=5,
        verbose=1,
        factor=0.5,
        min_lr=0.000001)

    csv_logger = CSVLogger('training.log')

    tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

    lr_tracker = LearningRateTracker()        

    model.fit_generator(batch_generator(),
              validation_data=([x_chars_valid, x_words_valid, word_char_tensor_valid], y_chars_valid),
              steps_per_epoch=1000,
              epochs=100,
              callbacks=[checkpoint, csv_logger, tensorboard, lr_tracker, learning_rate_reduction])

DBG()
