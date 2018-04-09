#!/usr/bin/env ipython2
# -*- coding: utf-8 -*- 

import re

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from klein import Klein
from twisted.web.static import File

from debug import *

import numpy as np
import keras
from keras.models import load_model

allowed_characters = u' \n\t?!\.,:=+-_abcdefghijklmnopqrstuvwxyz'
char_indices = dict((c, i) for i, c in enumerate(allowed_characters))
indices_char = dict((i, c) for i, c in enumerate(allowed_characters))
char_vocab_size = len(allowed_characters)
window_size = 41
mid_window = window_size/2
batch_size = 128
output_classes = 3
nb_valid = 100000
max_ngram_hash = 30000
max_word_hash = 1000000

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

model = load_model("diacritice.lstm.keras.model")

def flatten(txt):
    if type(txt) == str:
        txt = txt.decode('utf-8')
    return "".join([ translate_flat[c] if c in translate_flat else c for c in txt ])

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
    mid_window = window_size/2    
    just_aist = set([ char_indices[c] for c in [u"a", u"i", u"s", u"t"] ])
    text = "&nbsp;" + text
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

    # construct output
    rez = []
    for ch, mod in zip(text0, Y_pred):
        # print(ch, mod)
        ch2 = ch
        ch_mod = ch + str(int(mod))
        if ch_mod in repair_dia_table:
            ch2 = "<span class='mod'>"+repair_dia_table[ch_mod]+"</span>"
        # print(ch, mod, ch2)
        rez.append(ch2)
    rez_str = "".join(rez).strip()
    rez_str = re.sub(u"^&nbsp;", "", rez_str)
    return rez_str

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
