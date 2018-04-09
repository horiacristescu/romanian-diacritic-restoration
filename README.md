# Romanian Diacritics Restoration With Neural Nets

Why?
 - many errors on web
 - people and sometimes newspapers don't bother
 - "nice problem" - lots of data available, hard to do with regular programming
 
How?
 - take a large text corpus and learn to restors its diacritics (romanian Wikipedia)
 - using LSTMs
 - predicting sequence from sequence
 - learning happens in Embedding layer and LSTM

Features:
 - both char and word level
    - only use a restricted set of chars for prediction (letters, digits, punctuation and space)
    - at word level, we hash words into a range of 1,000,000
 - char and word codes are replaced with Embeddings which are jointly learned for the task
    
 - representing the target as 
    - 0 - no diacritic
    - 1 - for "ț, ș, î"
    - 2 - for "ă"
    - 3 - ignored chars  

Training
 - takes a few minutes to reach 98% accuracy
 - and many hours to reach 99.75% 
 - still makes some errors, about 1 in 400 chars
    - but the traing data still has errors
    - there are situations where additional context is needed but unavailable
 
What didn't work so well:
 - tried char level alone -> lacking long range correlations and can apply diacritics where they shouldn't be
 - tried to predict each character separately
 
