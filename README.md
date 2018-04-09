# Romanian Diacritics Restoration With Neural Nets

Why?
 - many errors on web
 - people and sometimes newspapers don't bother
 
How?
 - take a large text corpus and learn to restors its diacritics (romanian Wikipedia)
 - using LSTMs
 - predicting sequence from sequence
 - learning happens in Embedding layer and LSTM

Features:
 - both char and word level
 - representing the target as 
    - 0 - no diacritic
    - 1 - for "ț, ș, î"
    - 2 - for "ă"
    - 3 - ignored chars  

 
What didn't work so well:
 - tried char level alone -> lacking long range correlations and can apply diacritics where they shouldn't be
 - tried to predict each character separately
 
