# Romanian Diacritic Restoration With Neural Nets

##Why?

Writing with diacritics on an English keyboard can be hard. Every day millions of people write comments, articles and emails without diacritics. A corpus study based on OpenCrawl revealed that only 81% of the online text in Romanian has diacritics.

That is why I was motivated in picking this problem. It's a "nice problem" because there is a lot of training data available. I used the Romanian Wikipedia and OpenCrawl.
 
##How?

We take a large text corpus and remove the diacritics, then train a neural network to predict the diacritics back. We're using recurrent neural networks (LSTMs) to learn to reconstruct. After the neural net does its thing, we check results with a large dictionary to check for obvious mistakes (such as words badly placed diacritics).

- take a large text corpus and learn to restors its diacritics (romanian Wikipedia)
 - using LSTMs
 - predicting sequence from sequence
 - learning happens in Embedding layer and LSTM
 - other methods: ngrams language models - data sparsity problem, no lateral learning, large model size

##Features:

To train the network we use both character and word level features. The obvious problem is how to align them inside a neural net. I chose to mutiply the word embeddings for each letter, thus obtaining an more complex embeddings for characters that takes into account the whole word. 

We lowercase the text and remove all characters except letters, digits and a few punctuation marks. Later, when the model will do prediction we will lowercase the input text and then recover the case on the prediction, including the out-of-set characters.

To compute word embeddings I chose to hash words into the range 0..500,000 and then run the word ids through a similarly sized Embedding layer of width 25. The char embeddings are based on an Embedding table as well, this time width 100. The char and word embeddings learned jointly (end-to-end).

The output should be the correctly diactitised word, but instead we remember just the diacritic. We map no diacritics to 0, "ț", "ș" and "î" to 1 and "ă" to 2. Out of set chars are mapped to 3. This way we limit the size of the softmax layer and speed up training.

##Architecture

The model is based on LSTMs. I tried many combinations, from single LSTM and two-layer LSTM to bi-LSTM and even multiple bi-LSTMS's stacked on top of each other. The output is run through a TimeDistributed(Dense(4)) layer. We use skip connections to send the char data to each LSTM layer.

##Training

I used batches of 256, examples of 150 chars and 100 training epochs with Adam (initial lr = 0.001).

The model reaches 99.3% accuracy in the first epoch. But then it takes a long time to reach 99.75% after which it can't improve anymore. No matter how I changed the architecture, this limit remains. It only changes if I train on different data. At this point the model makes about 1 error in 400 characters. Some of those errors would have been hard to predict even for humans given only the flattened text.
 
##Validation

We set apart 500k of text. The training and text scores converge remarcably well at the end of training.
 
What didn't work so well:
I tried char based LSTM without word level information, but get 0.5% lower accuracy. I tried predicting only the diacritic of the center character in the example, but this gives similar accuracy with predicting the whole example at once.

##Prior work:

Other approaches are usually based on ngram-models. We tried to count word ngrams up to size 3 in a corpus of 1Gb of cleaned up text. The ngram model solves a large portion of the diacritics well but nowhere near the neural model, it was too brittle. Counting even larger ngrams would have been hard and the tables very sparse. In reality it is too hard to find ngrams for all possible word combinations.

##Other diacritic restoration services:
- diacritice.ai
