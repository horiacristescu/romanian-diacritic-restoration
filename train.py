#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from lol import *
from debug import *

import re, random, sys, io
import numpy as np

import madoka

import keras
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Activation, Bidirectional, Input, Concatenate
from keras.layers import Embedding, Reshape, Permute, Conv1D, TimeDistributed, Dropout
from keras.layers import LSTM, GRU, LeakyReLU, BatchNormalization, CuDNNLSTM, Lambda
from keras.layers import GlobalAveragePooling1D, multiply
from keras.layers.crf import CRF
from keras import backend as K
from keras.optimizers import RMSprop, Adam
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

window_size = 50
batch_size = 512
output_classes = 4
nb_valid = 500000
max_word_hash = 500000
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
}

def flatten(txt):
    if type(txt) == str:
        txt = txt.decode('utf-8')
    return "".join([ translate_flat[c] if c in translate_flat else c for c in txt ])

def dia_tag(kw):
    kw = kw.replace(u"ș", u"ş")
    sig = re.sub(u"[^ăĂâÂțȚîÎşșȘ]", "", kw)
    return flatten(sig[-2:])

def load_kw_flat_dia():
    txt = open("list-of-words.ok.txt", "r").read().decode("utf-8")
    kw_flat_dia = defaultdict(dict)
    for kw in txt.split("\n"):
        kw_ = flatten(kw)
        kw_flat_dia[kw_][kw] = True
    return kw_flat_dia

def count_ngrams_for_diacritic_varians():
    kw_dia_variants = dict()
    for kw_ in kw_flat_dia:
        for kw in kw_flat_dia[kw_]:
            kw_dia_variants[kw] = len(kw_flat_dia[kw_])

    nr = 0
    ngram_count = defaultdict(int)
    with open("filtered.text.ok.txt", "r") as f:
        for line in f:
            nr += 1
            if nr % 10000 == 0:
                print(nr, "nrgams=", len(ngram_count))
            line = line.strip().decode("utf-8")
            line = re.sub("  +", " ", line)
            vline = [ kw for kw in line.split(" ") if kw != "" ]
            for i in range(len(vline)):
                if kw_dia_variants.get(vline[i],0)>1:
                    ngram_count[vline[i]] += 1
            for i in range(len(vline)-1):
                if kw_dia_variants.get(vline[i],0)>1 or kw_dia_variants.get(vline[i+1],0)>1:
                    ngram_count[vline[i]+" "+vline[i+1]] += 1
            for i in range(len(vline)-2):
                if kw_dia_variants.get(vline[i],0)>1 or kw_dia_variants.get(vline[i+1],0)>1 or kw_dia_variants.get(vline[i+2],0)>1:
                    ngram_count[vline[i]+" "+vline[i+1]+" "+vline[i+2]] += 1

    sketch = madoka.Sketch(width=4*len(ngram_count))
    nr = 0
    for ng in ngram_count: 
        try:
            ng_ = ng.encode("utf-8")
        except:
            continue
        sketch[ng_] = ngram_count[ng]
        nr += 1
        if nr % 1000000 == 0:
            print(nr)
    print("saving ngrams.madoka")
    sketch.save("ngrams.madoka")
    DBG()

kw_flat_dia = load_kw_flat_dia()
ngrams_count = madoka.Sketch()
ngrams_count.load("ngrams.madoka")

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

    #fname = "/mnt/data/diacritice/opencrawl.diacritics.filtered.1GB.txt"
    #fname = "/mnt/data/data/corpus.wikipedia.diacritice.txt"
    fname = "filtered.text.ok.txt"
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
    np.save("data.input.char.filtered1.npy", text_in)
    np.save("data.input.word.filtered1.npy", text_in_words)
    np.save("data.output.filtered1.npy", text_out)

def vectorize_sentence(sent):
    sent_x = np.zeros((1, len(sent), len(allowed_characters)), dtype=np.bool)
    for t, char in enumerate(sent):
        ch_i = 0
        if char in char_indices:
            ch_i =  char_indices[char]
        sent_x[0, t, ch_i] = 1
    return sent_x

dia_stats = defaultdict(int)

def collect_errors():

    X_char = np.load("data.input.char.opencrawl1.npy", mmap_mode='r')
    X_word = np.load("data.input.word.opencrawl1.npy", mmap_mode='r')
    Y = np.load("data.output.opencrawl1.npy", mmap_mode='r')
    # X_char = np.load("data.input.char.wiki1.npy", mmap_mode='r')
    # X_word = np.load("data.input.word.wiki1.npy", mmap_mode='r')
    # Y = np.load("data.output.wiki1.npy", mmap_mode='r')

    model = load_model("diacritice.lstm.keras.model")
    print(model.summary())

    ferrw = open("errors.words.opencrawl.log.txt", "a+")
    ferrl = open("errors.lines.opencrawl.log.txt", "a+")

    n = X_char.shape[0]
    example_size = 150
    batch_size = 1000
    window_size = 24

    nr_out = 0
    nr_ln = 0

    for batch_nr in range(n / batch_size - 1):
        batch_st = batch_nr*batch_size*example_size
        batch_end = (batch_nr+1)*batch_size*example_size

        X_char_batch = X_char[batch_st : batch_end]
        X_text = np.array([ indices_char[c] for c in X_char_batch ])
        X_char_batch = X_char_batch.reshape((batch_size, example_size))
        X_text = X_text.reshape((batch_size, example_size))

        X_word_batch = X_word[batch_st : batch_end]
        X_word_batch = X_word_batch % max_word_hash
        X_word_batch = X_word_batch.reshape((batch_size, example_size))

        Y_target_batch = Y[batch_st : batch_end]
        Y_target_batch = Y_target_batch.reshape((batch_size, example_size))

        Y_pred_batch = model.predict([X_char_batch, X_word_batch]).argmax(axis=-1)

        for i in range(Y_pred_batch.shape[0]):
            X_target = rehydrate_text(X_char_batch[i], Y_target_batch[i])
            X_pred = rehydrate_text(X_char_batch[i], Y_pred_batch[i])

            scor_target, scor_pred, scor_match, scor_flat = 0, 0, 0, 0
            for ch_t, ch_p in zip(X_target, X_pred):
                dia_t = ch_t in translate_dia
                dia_p = ch_p in translate_dia
                if dia_t and not dia_p:
                    scor_target += 1
                elif dia_p and not dia_t:
                    scor_pred += 1
                elif dia_p and dia_t:
                    scor_match += 1
                else:
                    scor_flat += 1

            line = "target=%d|pred=%d|match=%d|flat=%d|%s|%s" % (scor_target, scor_pred, scor_match, scor_flat, X_pred, X_target) 
            line = line.encode("utf-8")
            line = line.replace("\n", " ")
            print(line, file=ferrl)
            nr_ln += 1
            if nr_ln % 1000 == 0:
                print("nr=", nr_ln, line)

            for j, target, pred in zip(range(batch_size), X_target, X_pred)[5:batch_size-5]:
                if target != pred:
                    window_start = j - window_size
                    window_end = j + window_size + 1
                    if window_start<0 or window_end > batch_size:
                        continue 
                    example_target = "".join(X_target[window_start:window_end])
                    example_pred = "".join(X_pred[window_start:window_end])
                    nr_out += 1
                    example_target_s = example_target.split()
                    example_pred_s = example_pred.split()
                    nex = len(example_target_s)
                    for k, (kw_targ, kw_pred) in enumerate(zip(example_target_s, example_pred_s)):
                        if kw_targ != kw_pred:
                            #ngram_st = max(k-1,0)
                            #ngram_end = min(k+2,nex)
                            ngram_st = k
                            ngram_end = k+1
                            ngram_targ = " ".join(example_target_s[ngram_st:ngram_end])
                            ngram_pred = " ".join(example_pred_s[ngram_st:ngram_end])
                            line = "%s|%s" % (ngram_targ, ngram_pred)
                            print (line.encode("utf-8"), file=ferrw)
                            #print (line.encode("utf-8"))

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

    X_char = np.load("data.input.char.filtered1.npy", mmap_mode='r')
    X_word = np.load("data.input.word.filtered1.npy", mmap_mode='r')
    Y = np.load("data.output.filtered1.npy", mmap_mode='r')

    print("X_char=", X_char.shape)

    X_char_valid, X_word_valid, Y_valid = validation_data_prepropcessor(X_char, X_word, Y)
    print("Validation on:", X_char_valid.shape[0], "examples")

    print('Build model...')

    def model_lstm_crf():
        input_char = Input(shape=(None, ))
        input_word = Input(shape=(None, ))
        embed_char = Embedding(char_vocab_size, 100)(input_char)
        embed_word = Embedding(max_word_hash, 20)(input_word)
        concat = keras.layers.concatenate([embed_char, embed_word], axis=-1)
        lstm_h = Bidirectional(LSTM(128, return_sequences=True))(concat)
        concat2 = keras.layers.concatenate([lstm_h, embed_char], axis=-1)
        crf_layer = CRF(4, sparse_target=True)
        output = crf_layer(concat2)
        model = Model([input_char, input_word], output)
        model.compile(
            loss=crf_layer.loss_function,
            optimizer=Adam(clipvalue=5.0),
            metrics=[crf_layer.accuracy])
        return model

    def model_2x2_lstm_word_char():
        input_char = Input(shape=(None, ))
        input_word = Input(shape=(None, ))
        embed_char = Embedding(char_vocab_size, 50)(input_char)
        embed_word = Embedding(max_word_hash, 50)(input_word)
        concat = keras.layers.concatenate([embed_char, embed_word], axis=-1)
        lstm_h = Bidirectional(LSTM(128, return_sequences=True))(concat)
        concat2 = keras.layers.concatenate([lstm_h, embed_char], axis=-1)
        lstm_h2 = Bidirectional(LSTM(128, return_sequences=True))(concat2)
        output = TimeDistributed(Dense(4, activation='softmax'))(lstm_h2)
        model = Model([input_char, input_word], output)
        model.compile(
            loss='categorical_crossentropy',
            optimizer=Adam(clipvalue=5.0),
            metrics=['accuracy'])
        return model

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
    #model.load_weights("diacritice.lstm.keras.model")
    # DBG()

    print(model.summary())

    gen = batch_generator(X_char, X_word, Y)

    checkpoint = ModelCheckpoint(
        "diacritice.lstm.keras.model", 
        monitor='val_acc', 
        verbose=1, 
        save_best_only=True, 
        mode='max')

    learning_rate_reduction = ReduceLROnPlateau(
        monitor='val_acc',
        patience=2,
        verbose=1,
        factor=0.2,
        min_lr=0.00001)

    model.fit_generator(gen,
              validation_data=([X_char_valid, X_word_valid], Y_valid),
              steps_per_epoch=1000,
              epochs=100,
              callbacks=[checkpoint]) #learning_rate_reduction

def rehydrate_text(X, Y):
    rez = []
    for x, y in zip(X.flatten(), Y.flatten()):
        ch = indices_char[x]
        ch_mod = flatten(ch) + str(int(y))
        if ch_mod in repair_dia_table:
            ch = repair_dia_table[ch_mod]
        rez.append(ch)
    return "".join(rez)



DBG()
