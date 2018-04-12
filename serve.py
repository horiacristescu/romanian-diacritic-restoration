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
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Activation, Bidirectional, Input, Concatenate
from keras.layers import Embedding, Reshape, Permute, Conv1D, TimeDistributed, Dropout
from keras.layers import LSTM, GRU, LeakyReLU, BatchNormalization, CuDNNLSTM, Lambda
from keras.layers import GlobalAveragePooling1D, multiply
from keras.optimizers import RMSprop, Adam

allowed_characters = u' \n\t?!\.,:=+-_abcdefghijklmnopqrstuvwxyz'
char_indices = dict((c, i) for i, c in enumerate(allowed_characters))
indices_char = dict((i, c) for i, c in enumerate(allowed_characters))
char_vocab_size = len(allowed_characters)
window_size = 50
batch_size = 128
output_classes = 3
nb_valid = 100000
max_ngram_hash = 30000
max_word_hash = 500000

chars_of_interest = {
    u"ț": True,
    u"ţ": True,
    u"ș": True,
    u"ş": True,
    u"ă": True,
    u"ã": True,
    u"î": True,
    u"â": True,
    u"t": True,
    u"s": True,
    u"a": True,
    u"i": True,
}

translate_dia = {
    u"ț": 1,
    u"ţ": 1,
    u"ș": 1,
    u"ş": 1,
    u"î": 1,
    u"â": 1,
    u"ă": 2,
    u"ã": 2,
}

repair_dia_table = {
    "t1": u"ț",
    "s1": u"ș",
    "a1": u"â",
    "i1": u"î",
    "a2": u"ă",
    "T1": u"Ț",
    "S1": u"Ș",
    "A1": u"Â",
    "I1": u"Î",
    "A2": u"Ă",
    "t0": u"t",
    "s0": u"s",
    "a0": u"a",
    "i0": u"i",
    "T0": u"T",
    "S0": u"S",
    "A0": u"A",
    "I0": u"I",
}

translate_flat = {
    u"ț": "t",
    u"ţ": "t",
    u"ș": "s",
    u"ş": "s",
    u"ă": "a",
    u"ã": "a",
    u"î": "i",
    u"â": "a",
    u"Ț": "T",
    u"Ţ": "T",
    u"Ș": "S",
    u"Ş": "S",
    u"Ă": "A",
    u"Ã": "A",
    u"Î": "I",
    u"Â": "A",
}

def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer.lr
    return lr

def model_lstm_word_char():
    input_char = Input(shape=(None, ))
    input_word = Input(shape=(None, ))
    embed_char = Embedding(char_vocab_size, 100, name='embed_char')(input_char)
    embed_word = Embedding(max_word_hash,    25, name='embed_word')(input_word)
    concat = keras.layers.concatenate([embed_char, embed_word], axis=-1)
    lstm_h = Bidirectional(LSTM(64, return_sequences=True))(concat)
    concat2 = keras.layers.concatenate([lstm_h, embed_char], axis=-1)
    lstm_h2 = LSTM(64, return_sequences=True)(concat2)
    concat3 = keras.layers.concatenate([lstm_h2, embed_char], axis=-1)
    output = TimeDistributed(Dense(4, activation='softmax'))(concat3)
    model = Model([input_char, input_word], output)
    optimizer = Adam(lr=0.001)
    lr_metric = get_lr_metric(optimizer)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy'])
    return model

def model_bilstm_simplu():
    input_char = Input(shape=(None, ))
    input_word = Input(shape=(None, ))
    embed_char = Embedding(char_vocab_size, 200, name='embed_char')(input_char)
    embed_word = Embedding(max_word_hash,    25, name='embed_word')(input_word)
    concat = keras.layers.concatenate([embed_char, embed_word], axis=-1)
    lstm_h1 = Bidirectional(LSTM(128, return_sequences=True))(concat)
    concat2 = keras.layers.concatenate([lstm_h1, concat], axis=-1)
    lstm_h2 = Bidirectional(LSTM(128, return_sequences=True))(concat2)
    concat3 = keras.layers.concatenate([lstm_h2, concat2], axis=-1)
    output = TimeDistributed(Dense(4, activation='softmax'))(concat3)
    model = Model([input_char, input_word], output)
    optimizer = Adam(lr=0.001)
    lr_metric = get_lr_metric(optimizer)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy'])
    return model

model = model_bilstm_simplu()
model.load_weights("diacritice.lstm.keras.model")

def flatten(txt):
    if type(txt) == str:
        txt = txt.decode('utf-8')
    return "".join([ translate_flat[c] if c in translate_flat else c for c in txt ])


def load_kw_flat_dia():
    txt = open("list-of-words.ok.txt", "r").read().decode("utf-8")
    kw_flat_dia = defaultdict(dict)
    for kw in txt.split("\n"):
        kw_ = flatten(kw)
        kw_flat_dia[kw_][kw] = True
    return kw_flat_dia

kw_flat_dia = load_kw_flat_dia()

def map_words_to_chars(text, max_hash=4294967000):
    vtxt = np.zeros(len(text), dtype=np.uint32)
    nr = 0
    for w in re.finditer(r"\b([a-zA-Z0-9_-]+)\b", text):
        for i in range(w.start(), w.end()):
            vtxt[i] = hash(w.group()) % max_hash
            nr += 1
            if nr % 10000000 == 0:
                print("mapping", nr)
    return vtxt

def predict(model, text):
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

    n = len(text)
    X_batch_words = np.array([ map_words_to_chars(text) % max_word_hash ])
    X_batch_chars = np.array([ [ char_indices.get(ch, 0) for ch in text ] ])

    # make prediction
    Y_pred = model.predict([X_batch_chars, X_batch_words]).argmax(axis=-1)[0]

    rez = []
    for ch, mod in zip(text, Y_pred):
        ch_mod = ch + str(int(mod))
        if ch_mod in repair_dia_table:
            ch = repair_dia_table[ch_mod]
        rez.append(ch)    
    Y_pred_text = "".join(rez)

    text_pred = Y_pred_text
    for m in re.finditer(ur"[a-zțţșşâăãîîâ0-9_-]+", Y_pred_text):
        kw = Y_pred_text[m.start():m.end()]
        kw0 = text0[m.start():m.end()]
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

    #DBG()

    # construct output
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
