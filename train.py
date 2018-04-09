#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from lol import *
from debug import *

import re, random, sys, io
import numpy as np

import keras
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Activation, Bidirectional, Input, Concatenate
from keras.layers import Embedding, Reshape, Permute, Conv1D, TimeDistributed, Dropout
from keras.layers import LSTM, GRU, LeakyReLU, BatchNormalization, CuDNNLSTM, Lambda
from keras.layers import GlobalAveragePooling1D, multiply
from keras import backend as K
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file

'''

only focus on windows of (s,t,a,i) 
add word embeddings path

'''


allowed_characters = u' \n\t?!\.,:=+-_abcdefghijklmnopqrstuvwxyz'
char_indices = dict((c, i) for i, c in enumerate(allowed_characters))
indices_char = dict((i, c) for i, c in enumerate(allowed_characters))
char_vocab_size = len(allowed_characters)

word_chars = set([ char_indices[c] for c in u"abcdefghijklmnopqrstuvwxyz" ])

window_size = 100
mid_window = window_size/2
batch_size = 256
output_classes = 4
nb_valid = 500000
max_word_hash = 1000000
window_size_wordlevel = 7

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
}

def flatten(txt):
    if type(txt) == str:
        txt = txt.decode('utf-8')
    return "".join([ translate_flat[c] if c in translate_flat else c for c in txt ])

def dia_tag(kw):
    kw = kw.replace(u"ș", u"ş")
    sig = re.sub(u"[^ăĂâÂțȚîÎşșȘ]", "", kw)
    return flatten(sig[-2:])

def count_char_ngrams():
    ngram_count = defaultdict(int)
    fname = "/mnt/data/diacritice/opencrawl.diacritics.filtered.1GB.txt"
    nr = 0
    with open(fname, "r") as f:
        for line in f:
            line = line.decode("utf-8").lower()
            line = flatten(line)
            for ng in zip(line, line[1:], line[2:]):
                ngram_count["".join(ng)] += 1
            nr += 1
            if nr % 10000 == 0:
                print(nr)
            if nr > 10000000:
                break    
    DBG()

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

def prep_data():

    # fname = "/mnt/data/diacritice/opencrawl.diacritics.filtered.1GB.txt"
    fname = "/mnt/data/data/corpus.wikipedia.diacritice.txt"
    print("Loading text corpus", fname)
    with open(fname, "r") as f:
        text = f.read().decode("utf-8").lower()

    #text = text[:100000000]

    n = len(text)

    print('Corpus length:', n)
    print('Total chars:', len(allowed_characters))

    print("Allocating text_in = np.zeros(n, dtype=np.uint16)")
    text_in = np.zeros(n, dtype=np.uint8)
    print("Allocating text_out = np.zeros(n, dtype=np.uint8)")
    text_out = np.zeros(n, dtype=np.uint8)

    nrl = 0
    last_ln = 0
    for i in range(1,n-1):
        ch = text[i]

        if i % 1000000 == 0:
            print("parsing", i) 

        # flatten text
        if ch in translate_flat:
           ch_flat = translate_flat[ch]
        else:
           ch_flat = ch
        ch_i = char_indices.get(ch_flat, 0)

        if ch in translate_dia:
            # picks the diacritic for this char
            dia = translate_dia[ch]
        else:
            if ch not in chars_of_interest:
                dia = 3 # ignore, no diacritics possible for this char
            else:
                dia = 0 # no diacritics sould be applied, leave flat

        text_in[i] = ch_i
        text_out[i] = dia

    print("Hashing words into np.array(*, dtype=uint32)")
    text_in_words = map_words_to_chars(flatten(text))

    print("Saving data.input.*.npy, data.output.*.npy")
    np.save("data.input.char.wiki1.npy", text_in)
    np.save("data.input.word.wiki1.npy", text_in_words)
    np.save("data.output.wiki1.npy", text_out)

def vectorize_sentence(sent):
    sent_x = np.zeros((1, len(sent), len(allowed_characters)), dtype=np.bool)
    for t, char in enumerate(sent):
        ch_i = 0
        if char in char_indices:
            ch_i =  char_indices[char]
        sent_x[0, t, ch_i] = 1
    return sent_x

dia_stats = defaultdict(int)

def scan_word_window(X, position, window=3):
    i = position
    n = len(X)
    rez = []
    while (i > 0) and (X[i] in word_chars):
        i -= 1
    i += 1    
    i2 = i    
    for nrw in range(window+1):
        if i >= n:
            rez.insert(0, "")
            continue
        while (i < n) and (X[i] not in word_chars):
            i += 1
        i0 = i    
        while (i < n) and (X[i] in word_chars):
            i += 1
        i1 = i
        word = "".join([ indices_char[xi] for xi in X[i0:i1] ])
        rez.append(word)   
    i = i2 - 1
    for nrw in range(window):
        if i == 0:
            rez.insert(0, "")
            continue
        while (i > 0) and (X[i] not in word_chars):
            i -= 1
        i0 = i    
        while (i > 0) and (X[i] in word_chars):
            i -= 1
        i1 = i  
        word = "".join([ indices_char[xi] for xi in X[i1+1:i0+1] ])
        rez.insert(0, word)
    return rez

def batch_generator(X_char, X_word, Y):
    nr_items = X_char.shape[0]
    X_batch_char = []
    X_batch_word = []
    Y_batch = []
    while True:
        for pos in range(nb_valid, nr_items-nb_valid, window_size):
            X_batch_char.append( X_char[pos:pos+window_size] )
            X_batch_word.append( X_word[pos:pos+window_size] % max_word_hash )
            Y_batch.append( keras.utils.to_categorical(Y[pos:pos+window_size], num_classes=output_classes) )
            if len(X_batch_char) >= batch_size:
                yield [ np.array(X_batch_char), np.array(X_batch_word) ], np.array(Y_batch)
                X_batch_char = []
                X_batch_word = []
                Y_batch = []


def validation_data_prepropcessor(X_char, X_word, Y):
    pos = 0
    X_batch_char = []
    X_batch_word = []
    Y_batch = []
    for pos in range(0, nb_valid, window_size):
        X_batch_char.append( X_char[pos:pos+window_size] )
        X_batch_word.append( X_word[pos:pos+window_size] % max_word_hash )
        Y_batch.append( keras.utils.to_categorical(Y[pos:pos+window_size], num_classes=output_classes) )
    return np.array(X_batch_char), np.array(X_batch_word), np.array(Y_batch)

def train():

    X_char = np.load("data.input.char.wiki1.npy", mmap_mode='r')
    X_word = np.load("data.input.word.wiki1.npy", mmap_mode='r')
    Y = np.load("data.output.wiki1.npy", mmap_mode='r')

    X_char_valid, X_word_valid, Y_valid = validation_data_prepropcessor(X_char, X_word, Y)
    print("Validation on:", X_char_valid.shape[0], "examples")

    print('Build model...')

    def model_lstm_stanga_mijloc_dreapta():
        input_char = Input(shape=(None, ))
        input_word = Input(shape=(None, ))
        embed_char = Embedding(char_vocab_size, 50)(input_char)
        embed_word = Embedding(max_word_hash, 50)(input_word)
        concat = keras.layers.concatenate([embed_char, embed_word], axis=-1)
        lstm_h = Bidirectional(LSTM(128, return_sequences=True))(concat)
        output = TimeDistributed(Dense(4, activation='softmax'))(lstm_h)
        model = Model([input_char, input_word], output)
        return model

    #model = model_lstm_stanga_mijloc_dreapta()
    model = load_model("diacritice.lstm.keras.model")
    #DBG()

    model.compile(
        loss='categorical_crossentropy',
        optimizer='RMSprop',
        metrics=['accuracy'])

    print(model.summary())

    gen = batch_generator(X_char, X_word, Y)

    checkpoint = ModelCheckpoint("diacritice.lstm.keras.model", monitor='val_acc', verbose=1, save_best_only=True, mode='max')

    learning_rate_reduction = ReduceLROnPlateau(
        monitor='val_acc',
        patience=2,
        verbose=1,
        factor=0.2,
        min_lr=0.000001)

    model.fit_generator(gen,
              validation_data=([X_char_valid, X_word_valid], Y_valid),
              steps_per_epoch=1000,
              epochs=100,
              callbacks=[learning_rate_reduction, checkpoint])


DBG()
