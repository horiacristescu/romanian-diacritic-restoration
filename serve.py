#!/usr/bin/env ipython2
# -*- coding: utf-8 -*- 

import re

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from collections import defaultdict

from klein import Klein
from twisted.web.static import File

from debug import *
import numpy as np

import keras
from keras.callbacks import Callback, ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard
from keras.models import Model, load_model
from keras.layers import Dense, Activation, Bidirectional, Input
from keras.layers import Embedding, Conv1D, TimeDistributed, Dropout
from keras.layers import LSTM, GRU, BatchNormalization, Lambda
from keras import backend as K
from keras.optimizers import Adam

from preprocessing import *

output_classes = 4
max_word_hash = 1000000

def model_def():

    # char level input (char ids)
    input_char = Input(shape=(None, ))
    
    # word level input (word ids)
    input_word = Input(shape=(None, ))
    
    # word x char translation map (array)
    input_map  = Input(shape=(None, None))

    # embed chars
    char_embed = Embedding(char_vocab_size, 50)(input_char)

    # run through 3 layers of CNN
    char_pipe = Conv1D(128, 31, name="conv_size_31", activation='relu', padding='same')(char_embed)
    char_pipe = Conv1D(128, 21, name="conv_size_21", activation='relu', padding='same')(char_pipe)
    char_pipe = Conv1D(128, 15, name="conv_size_15", activation='relu', padding='same')(char_pipe)

    # pass words through LSTM
    word_pipe = Embedding(max_word_hash, 50, name='embed_word')(input_word)
    word_pipe = Bidirectional(LSTM(50, return_sequences=True))(word_pipe) # (None, 27, 100)

    # map word space to char space
    input_map_p = Lambda(lambda x: K.permute_dimensions(x, (0,2,1)), name='transpose_map')(input_map)
    word_pipe = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[2,1]), name='project_words_chars')([input_map_p, word_pipe])

    # concatenate word, char level and char input
    pipe = keras.layers.concatenate([word_pipe, char_pipe, char_embed], axis=-1)

    # three more layers of CNN
    pipe = Conv1D(128, 11, name="conv_size_11", activation='relu', padding='same')(pipe)
    pipe = Conv1D(128, 7,  name="conv_size_7",  activation='relu', padding='same')(pipe)
    pipe = Conv1D(128, 3,  name="conv_size_3",  activation='relu', padding='same')(pipe)

    # reduce output to 4 channels per char
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

def predict(model, text):
    # fix some alternative diacritics
    text = text.replace(u"ş", u"ș")
    text = text.replace(u"ţ", u"ț")
    text = text.replace(u"Ş", u"Ș")
    text = text.replace(u"Ţ", u"Ț")

    text0 = text
    # convert to plain ascii
    text = flatten(text0).lower()
    # eliminate chars outside allowed set
    text_ = []
    for i in range(len(text)):
        if text[i] not in char_indices:
            text_.append(" ")
        else:
            text_.append(text[i])    
    text = "".join(text_)

    # generate an one-example batch
    batch = [ text ]
    bg = BatchGenerator(batch=batch)
    x_chars, x_words, word_char_tensor, _ = bg.batch_generator(max_word=max_word_hash)

    # make prediction
    Y_pred = model.predict([x_chars, x_words, word_char_tensor]).argmax(axis=-1)[0]

    # apply results on text
    rez = []
    for ch, mod in zip(flatten(text0), Y_pred):
        ch_mod = ch + str(int(mod))
        if ch_mod in repair_dia_table:
            ch = repair_dia_table[ch_mod]
        rez.append(ch)    
    Y_pred_text = "".join(rez)

    # validate with dictionary
    text_pred = Y_pred_text
    for m in re.finditer(ur"[a-zțţșşâăãîîâA-ZȚŢȘŞÂĂÃÎÎÂ0-9_-]+", Y_pred_text):
        kw = Y_pred_text[m.start():m.end()].lower()
        kw0 = text0[m.start():m.end()].lower()
        kw_ = flatten(kw)
        if kw_ in kw_flat_dia:
            if kw not in kw_flat_dia[kw_]:
                kw_vars = kw_flat_dia[kw_]
                if flatten(kw0) not in kw_vars:
                    kw_repl = kw_vars.keys()[0]
                    print("fixed=", text_pred[m.start():m.end()], kw_repl)
                    text_pred = text_pred.replace(text_pred[m.start():m.end()], kw_repl)
                else:
                    print("revert=", text_pred[m.start():m.end()], kw0)
                    text_pred = text_pred.replace(text_pred[m.start():m.end()], kw0)
            else:
                pass
                #print("found=", text_pred[m.start():m.end()], kw0)
    Y_pred_text = text_pred

    # construct output, highlight diacritics
    rez = []
    for ch0, ch2 in zip(text0, Y_pred_text):
        ch = ch0
        if ch0.lower() != ch2.lower() and ch2 != " ":
            ch = "<span class='mod'>"+ch2+"</span>"
        rez.append(ch)
    rez_str = "".join(rez).strip()
    return rez_str

#DBG()

app = Klein()

@app.route("/ajax")
def generate_ajax(request):
    txt = request.content.read().decode("utf-8")
    print("GOT TXT=", txt, type(txt))
    request.setHeader('Content-Type', 'text/html; charset=utf-8')
    request.write(predict(model, txt).encode("utf-8"))	

@app.route("/", branch=True)
def generate_index(request):
    return File("./app")

if __name__ == '__main__':
	print " * Web API started"
	app.run(host='0.0.0.0', port=5080)
