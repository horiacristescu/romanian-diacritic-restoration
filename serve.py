#!/usr/bin/env ipython2
# -*- coding: utf-8 -*- 

import re

from klein import Klein
from twisted.web.static import File

from debug import *

import numpy as np
import keras
from keras.models import load_model

allowed_characters = u' \n-_abcdefghijklmnopqrstuvwxyz0123456789ăâîșț'
char_indices = dict((c, i) for i, c in enumerate(allowed_characters))
indices_char = dict((i, c) for i, c in enumerate(allowed_characters))
char_vocab_size = len(allowed_characters)
window_size = 21
batch_size = 512
output_classes = 3
nb_valid = 100000

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

repair_dia_table = {
    "t1": u"ț",
    "T1": u"Ț",
    "s1": u"ș",
    "S1": u"Ș",
    "a1": u"â",
    "A1": u"Â",
    "i1": u"î",
    "I1": u"Î",
    "a2": u"ă",
    "A2": u"Ă",
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

model = load_model("models/diacritice.lstm.keras.model")

def flatten(txt):
    if type(txt) == str:
        txt = txt.decode('utf-8')
    return "".join([ translate_flat[c] if c in translate_flat else c for c in txt ])

def predict(model, text):
    text0 = text
    text = flatten(text0).lower()
    text = re.sub("[^a-z_-]", " ", text)
    text = " "*(window_size/2) + text + " "*(window_size/2)
    text_in  = [ char_indices[translate_flat[c]] if c in translate_flat else char_indices[c] for c in text ]
    X_batch = []
    for pos in range(0, len(text_in) - window_size + 1):
        X_batch.append( keras.utils.to_categorical(text_in[pos:pos+window_size], num_classes=char_vocab_size) )
    X_batch = np.array(X_batch)
    Y_pred = model.predict_classes(X_batch)
    rez = []
    for ch, mod in zip(text0, Y_pred):
        ch2 = ch
        ch_mod = ch + str(int(mod))
        if ch_mod in repair_dia_table:
            ch2 = "<span class='mod'>"+repair_dia_table[ch_mod]+"</span>"
        # print(ch, mod, ch2)
        rez.append(ch2)
    return "".join(rez)

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
