# Romanian Diacritic Restoration With Neural Nets

## Live Demo: http://pagini.ro/

Author [Horia Cristescu](mailto:horia.cristescu@gmail.com). I can work remote and am available for hiring in Bucharest.

## Why?

Writing in Romanian with diacritics on an English keyboard can be hard. Every day millions of people write comments, articles and emails without diacritics. A corpus study based on OpenCrawl revealed that only 81% of the online text in Romanian has diacritics.

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

The model is based on CNNs and LSTMs. We have two paths - character level and word level. My intuition for using separate word and char level paths is to learn both long range structure and morphology. For the character path, we use embeddings and three layers of CNN. The word path goes through embedding and biLSTM. We merge the two paths by projecting words to characters, based on a projection matrix which is received as an additional input. Then we have three more CNN layers and output predictions.

```python
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
```

## Training

The final accuracy before dictionary word check is 99.82%.

I used batches of 256, examples of 150 chars and 100 training epochs with Adam (initial lr = 0.001). The model reaches 99.3% accuracy in the first epoch. But then it takes a long time to reach 99.75% after which it can't improve anymore. No matter how I changed the architecture, this limit stands. It only changes if I train on different data. At this point the model makes about 1 error in 500 characters. Some of those errors would have been hard to predict even for humans given only the flattened text.

For validation I set apart 500k of text. The training and text scores converged remarcably well at the end of training.

The trained model is available upon request, being too large to host on Github.

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

