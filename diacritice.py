#!/usr/bin/env python
# -*- coding: utf-8 -*-

#DBG()

# input_layer = Input(shape=(maxlen, len(allowed_characters)))
# l1, state_h, state_c = LSTM(129, return_state=True)(input_layer)
# final_layer = Dense(len(allowed_characters))(state_h)
# model = Model(input_layer, final_layer)
# lstm = Bidirectional(LSTM(128, return_state=True))
# lstm_outputs, forward_h, forward_c, backward_h, backward_c = lstm(inputs)
# state_c = Concatenate()([forward_c, backward_c])


from __future__ import print_function
from lol import *
from debug import *

import keras
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Bidirectional, Input, Concatenate
from keras.layers import Embedding, Reshape, Permute, Conv1D, TimeDistributed
from keras.layers import LSTM, LeakyReLU, BatchNormalization, CuDNNGRU
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
import io

allowed_characters = u' \n-_abcdefghijklmnopqrstuvwxyz0123456789ăâîșț'
char_indices = dict((c, i) for i, c in enumerate(allowed_characters))
indices_char = dict((i, c) for i, c in enumerate(allowed_characters))
char_vocab_size = len(allowed_characters)

translate_dia = {
    u"ț": 1,
    u"Ț": 1,
    u"ș": 1,
    u"Ș": 1,
    u"â": 2,
    u"Â": 2,
    u"î": 1,
    u"Î": 1,
    u"ă": 1,
    u"Ă": 1,
}

translate_flat = {
    u"ă": "a",
    u"Ă": "A",
    u"â": "a",
    u"Â": "A",
    u"ț": "t",
    u"Ț": "T",
    u"î": "i",
    u"Î": "I",
    u"ș": "s",
    u"Ș": "S",
}

def dia_tag(kw):
    kw = kw.replace(u"ș", u"ş")
    sig = re.sub(u"[^ăĂâÂțȚîÎşșȘ]", "", kw)
    return flatten(sig[-2:])

def prep_data():
    with open("/mnt/data/data/corpus.wikipedia.diacritice.txt", "r") as f:
        text = f.read().replace(" #", "").decode("utf-8")
        text = text.replace(u"ã", u"ă").replace(u"Ş", u"Ș").replace(u"ş", u"ș").replace(u"Ţ", u"Ț").replace(u"ţ", u"ț")
        text = text.lower()
    print('corpus length:', len(text))
    print('total chars:', len(allowed_characters))
    text_in  = [ char_indices[translate_flat[c]] if c in translate_flat else char_indices[c] for c in text ]
    text_out = [ translate_dia[c] if c in translate_dia else 0 for c in text ]
    text_in_np = np.array(text_in, dtype=np.uint8)
    text_out_np = np.array(text_out, dtype=np.uint8)
    np.save("data.input.npy", text_in_np)
    np.save("data.output.npy", text_out_np)
    print("Saved data.input.npy, data.output.npy")

def vectorize_sentence(sent):
    sent_x = np.zeros((1, len(sent), len(allowed_characters)), dtype=np.bool)
    for t, char in enumerate(sent):
        sent_x[0, t, char_indices[char]] = 1
    return sent_x

def batch_generator():
    pass

nb_valid = 100000
maxlen = 21
output_classes = 3
batch_size = 512

X = np.load("data.input.npy", mmap_mode='r')
Y = np.load("data.output.npy", mmap_mode='r')

def batch_generator(X, Y, line_size=maxlen, batch_size=batch_size, char_vocab_size=char_vocab_size, output_classes=output_classes, nb_valid=nb_valid):
    nr_items = X.shape[0]
    pos = 0
    while True:
        X_batch = []
        Y_batch = []
        for i in range(batch_size):
            pos = nb_valid + np.random.randint(nr_items-line_size-nb_valid)
            X_batch.append( keras.utils.to_categorical(X[pos:pos+line_size], num_classes=char_vocab_size) )
            Y_batch.append( keras.utils.to_categorical(Y[pos+line_size/2], num_classes=output_classes) )
        yield np.array(X_batch), np.array(Y_batch)

def valid_prepropcessor(X, Y, line_size=maxlen, batch_size=batch_size, char_vocab_size=char_vocab_size, output_classes=output_classes, nb_valid=nb_valid):
    nr_items = X.shape[0]
    pos = 0
    X_batch = []
    Y_batch = []
    for pos in range(nb_valid-line_size):
        X_batch.append( keras.utils.to_categorical(X[pos:pos+line_size], num_classes=char_vocab_size) )
        Y_batch.append( keras.utils.to_categorical(Y[pos+line_size/2], num_classes=output_classes) )
    return np.array(X_batch), np.array(Y_batch)

X_valid, Y_valid = valid_prepropcessor(X, Y)

print('Build model...')

model = Sequential()
model.add(Bidirectional(CuDNNGRU(128), input_shape=(maxlen, char_vocab_size)))
model.add(Dense(3, activation='softmax'))

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy'])

print(model.summary())

gen = batch_generator(X, Y)

checkpoint = ModelCheckpoint("diacritice.lstm.keras.model", monitor='val_acc', verbose=1, save_best_only=True, mode='max')

learning_rate_reduction = ReduceLROnPlateau(
    monitor='val_acc',
    patience=2,
    verbose=1,
    factor=0.25,
    min_lr=0.00001)

model.fit_generator(gen,
          validation_data=(X_valid, Y_valid),
          steps_per_epoch=5000,
          epochs=50,
          callbacks=[checkpoint, learning_rate_reduction])

DBG()
