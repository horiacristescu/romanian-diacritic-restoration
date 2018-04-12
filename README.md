# Romanian Diacritic Restoration With Neural Nets

## Live Demo: http://pagini.ro/

Author [Horia Cristescu](mailto:horia.cristescu@gmail.com). I can work remote and am available for hiring in Bucharest.

## Why?

Writing with diacritics on an English keyboard can be hard. Every day millions of people write comments, articles and emails without diacritics. A corpus study based on OpenCrawl revealed that only 81% of the online text in Romanian has diacritics.

That is why I was motivated in picking this problem. On the other hand it's a "nice problem" because there is a lot of training data available. I used the Romanian Wikipedia and OpenCrawl.
 
## How?

I took a large text corpus and removed the diacritics, then trained a neural network to predict the diacritics back. I used recurrent neural networks (LSTMs) to learn to reconstruct. After the neural net makes a prediction, I run a check for obvious mistakes with a large dictionary.

## Features:

To train the network I used both character and word level features. The obvious problem is how to align them inside a neural net. I chose to multiply the word embeddings for each letter, thus obtaining an more complex embedding for characters that takes into account the whole word. 

I lowercased the text and removed all characters except letters, digits and a few punctuation marks. Later, when the model makes predictions, I lowercase the input text and then recover the case on the prediction, including the out-of-set characters.

To compute word embeddings I chose to hash words into the range 0..500,000 and then run the word ids through a similarly sized Embedding layer of width 25. The char embeddings are based on an Embedding table as well, this time width 200. The char and word embeddings are learned jointly (end-to-end).

The output should be the correctly 'diacritised' word, but instead the model predicts only the diacritic sign itself. I mapped "no diacritics" to 0, "ț", "ș" and "î" to 1 and "ă" to 2. Out of set chars are mapped to 3. This way I limited the size of the softmax layer and sped up training.

## Architecture

<img src="diacritic_restoration_lstm.png?raw=true" width="508">

The model is based on LSTMs. I tried many combinations, from single LSTM and two-layer LSTM to bi-LSTM and even multiple bi-LSTMS's stacked on top of each other. The output is run through a TimeDistributed(Dense(4)) layer. I used skip connections to send the char data to each LSTM layer.

```python
def build_model():
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
```

## Training

The final accuracy before dictionary word check is 99.8%.

I used batches of 256, examples of 150 chars and 100 training epochs with Adam (initial lr = 0.001). The model reaches 99.3% accuracy in the first epoch. But then it takes a long time to reach 99.75% after which it can't improve anymore. No matter how I changed the architecture, this limit stands. It only changes if I train on different data. At this point the model makes about 1 error in 500 characters. Some of those errors would have been hard to predict even for humans given only the flattened text.

For validation I set apart 500k of text. The training and text scores converged remarcably well at the end of training.
  
## What didn't work so well:

I tried char based LSTM without word level information, but got 0.5% lower accuracy. I tried predicting only the diacritic of the center character in the example, but this gives similar accuracy with predicting the whole example at once.

## Other methods:

Other approaches are usually based on ngram-models. I tried to count word ngrams up to size 3 in a corpus of 1Gb of cleaned up text. The ngram model solves a large portion of the diacritics well but nowhere near the neural model, it was too brittle. Counting larger ngrams would have been hard and the tables very sparse. In reality it is too hard to find ngrams in the wild for all possible word combinations.

## Website

I used Klein as backend and jQuery with plain HTML/CSS for the front end. The theme is based on Bootstrap.

## Other Romanian diacritic restoration services:
- http://diacritice.ai/
- http://plagiarisma.net/ro/spellcheck.php
- http://diacritice.opa.ro/
- http://www.diacritice.com/

