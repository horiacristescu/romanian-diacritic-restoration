#!/usr/bin/env ipython2
# -*- coding: utf-8 -*- 

from __future__ import print_function
from collections import defaultdict
from debug import *
import re
import numpy as np
import os.path
import random
import keras

output_classes = 4

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

allowed_characters = u' \n\t\'\"?!\.\\/,:=<>+-_0123456789abcdefghijklmnopqrstuvwxyz'
char_indices = dict((c, i) for i, c in enumerate(allowed_characters))
indices_char = dict((i, c) for i, c in enumerate(allowed_characters))
char_vocab_size = len(allowed_characters)

def flatten(txt):
    # remove diacritics
    if type(txt) == str:
        txt = txt.decode('utf-8')
    return "".join([ translate_flat[c] if c in translate_flat else c for c in txt ])

def pad_line(text, n=None):
    if n == None:
        return text
    if len(text) > n:
        if text[n-1] != " " and text[n] != " ":
            # delete last word, don't want incomplete words
            text = re.sub(r"[^\s]+$", "", text[:n])
        else:
            text = text[:n]
    return text + " "*(n-len(text))

def load_dictionary():
    # loads a list of correctly spelled words
    kw_index = dict()
    index_kw = dict()
    kw_flat_dia = defaultdict(dict)
    txt = open("dictionary.txt", "r").read().decode("utf-8")
    kw_flat_dia = defaultdict(dict)
    for i, kw in enumerate(txt.split("\n")):
        kw_index[kw] = i
        index_kw[i] = kw
        kw_ = flatten(kw)
        kw_flat_dia[kw_][kw] = True
    return kw_index, index_kw, kw_flat_dia

kw_index, index_kw, kw_flat_dia = load_dictionary()

class BatchGenerator:
    
    def __init__(self, batch=[], fname = "/mnt/data/diacritice/opencrawl.diacritics.filtered.txt", nr_valid=5000, limit=None):
        self.fname = fname
        self.nr_valid = nr_valid
        self.line_offs = []
        self.line_len = []
        self.batch = []
        if len(batch)>0:
            # in-memory batch
            self.nr_valid = 0
            self.batch = [] + batch
        elif os.path.exists(fname+".offs.npy") and os.path.exists(fname+".len.npy"):
            # load cached offsets
            print("fast loading", self.fname)
            self.line_offs = np.load(fname+".offs.npy").tolist()
            self.line_len = np.load(fname+".len.npy").tolist()
        else:
            # index training file
            print("indexing", self.fname)
            print("loading", fname)
            with open(fname, "r") as f:
                nr = 0
                while True:
                    self.line_offs.append(f.tell())
                    line = f.readline()
                    if not line:
                        break
                    self.line_len.append(len(line)-1)
                    nr += 1
                    if nr % 100000 == 0:
                        print(nr)
                    if limit != None and nr >= limit:
                        break
            np.save(fname+".offs.npy", self.line_offs)        
            np.save(fname+".len.npy", self.line_len)
        self.line_order = []
        self.line_pos = 0

    def generate_text_batch(self, batch_size=32):
        if len(self.batch) > 0:
            # for in-memory batches (prediction)
            return self.batch
        # shuffle lines, emit text batch
        if len(self.line_order) < self.line_pos + batch_size:
            print("Shuffling", len(self.line_offs), "lines from", self.fname)
            self.line_order = range(len(self.line_offs))[self.nr_valid:]
            random.shuffle(self.line_order)
            self.line_pos = 0
        # read batch from input file
        batch = []
        with open(self.fname, "r") as f:
            for i in range(batch_size):
                ln_id = self.line_order[self.line_pos + i]
                f.seek(self.line_offs[ln_id])
                line = f.read(self.line_len[ln_id])
                batch.append(line.decode("utf-8"))
        self.line_pos += batch_size
        return batch

    def generate_validation_batch(self, batch_size=32):
        # special batch, no shuffle, size=self.nr_valid
        batch = []
        with open(self.fname, "r") as f:
            for ln_id in range(self.nr_valid):
                f.seek(self.line_offs[ln_id])
                line = f.read(self.line_len[ln_id])
                batch.append(line.decode("utf-8"))
        return batch

    def featurize_text_to_words_tensor(self, text, max_hash=4294967000):
        text = flatten(text)
        # extract words from text
        words = []
        words_r = []
        for w in re.finditer(r"\b([a-zA-Z0-9_-]+)\b", text):
            kw_i = hash(w.group()) % max_hash
            words.append(kw_i)
            words_r.append((w.start(), w.end()))
        # create a tensor words x chars 
        word_char_tensor = np.zeros((len(words), len(text)))
        for i in range(len(words)):
            for j in range(words_r[i][0], words_r[i][1]):
                word_char_tensor[i,j] = 1.
        return words, word_char_tensor

    def featurize_txt_to_chars(self, text):
        text = text.lower()
        if type(text) == str:
            text = text.decode('utf-8')
        X = []
        Y = []
        for i in range(len(text)):
            ch = text[i]
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
            X.append(ch_i)
            Y.append(dia)
        return X, Y

    def batch_generator(self, forValidation=False, batch_size=32, max_word=500000, returnPlainText=False):
        if forValidation:
            batch = self.generate_validation_batch()
            batch_size = len(batch)
        else:
            batch = self.generate_text_batch(batch_size=batch_size)
            batch_size = len(batch)
        # pad lines to same length
        lens = sorted([ len(l) for l in batch ])
        max_len = lens[ int(len(lens)*0.9) ]
        batch = [ pad_line(l, n=max_len) for l in batch  ]
        if returnPlainText:
            return batch
        # binarize
        x_chars_b = []
        x_words_b = []
        y_chars_b = []
        word_char_tensor_b = []
        for line in batch:
            x_chars, y_chars = self.featurize_txt_to_chars(line)
            x_words, word_char_tensor = self.featurize_text_to_words_tensor(line, max_hash=max_word)
            x_chars_b.append(x_chars)
            x_words_b.append(x_words)
            word_char_tensor_b.append(word_char_tensor)
            y_chars_b.append(y_chars)
        # pad x_words to same length
        if batch_size > 1:
            max_line_words = max([ len(l) for l in x_words_b ])
            for i in range(batch_size):
                dif = max_line_words - len(x_words_b[i])
                if dif > 0:
                    x_words_b[i] = x_words_b[i] + [0] * dif
                    word_char_tensor_b[i] = np.pad(word_char_tensor_b[i], pad_width=((0,dif),(0,0)), mode='constant')
        y_chars_b = keras.utils.to_categorical(np.array(y_chars_b), num_classes=output_classes)
        return np.array(x_chars_b), np.array(x_words_b), np.array(word_char_tensor_b), y_chars_b

