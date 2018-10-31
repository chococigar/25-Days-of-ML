# 100 Days Of ML - LOG

## Day 1 : September 6th, 2018
 
**Today's Progress** : Attempted Siraj's Hidden Markov model lyrics generator, just considering frequency (not grammar / semantics)

**Thoughts** : Multi-lingual training is difficult, as they have different rules (especially Korean, where syllables are composed of 3 consonants and vowels). Might need to attempt Word2Vec to perform better.

**Link of Work:**   [Commit](https://github.com/chococigar/100-Days-of-ML/tree/master/Day1_hidden_markov_generator)

## Day 2 : September 7th, 2018
 
**Today's Progress** : Got staretd with W2V on local, realized t-SNE is slow so set up env for colab

**Thoughts** : I need a gpu

**Link of Work:**   [Commit](https://github.com/chococigar/100-Days-of-ML/tree/master/Day2_word2vec)

## Day 3 : September 10th, 2018 

_(technically including 9th)_
 
**Today's Progress** : Tokenized text / analyzed morph / calculated word similarities with nltk, gensim, konlpy on colab. Spent some time encoding types and colab settings. 

**Thoughts** : Realized a lot more data is needed, w2v is the way, K-pop lyrics is difficult as english phrases and korean phrases are so stochastically knit together.

**Link of Work:**   [Commit](https://github.com/chococigar/100-Days-of-ML/tree/master/Day3_NLP)


## Day 4 : September 12th, 2018 
 
**Today's Progress** : Read/studied various textmining techniques including sent2vec, doc2vec, etc. Studied references mentioned in Prof Sam Bowman's glue. Sought for help on analyzing multilingual BTS texts from colleagues.

**Thoughts** : 
My mission is to detect which of the following types the fed lyrics are 🙄 : 
1) "나는 apple 을 먹었다" : Uses English word in Korean phrase under Korean grammatical rule
2) "I ate a 사과" : Uses Korean word in English phrase under English grammatical rule

## Day 5 : Oct 9th, 2018
**Today's Progress** : Using Music21 and Keras, generated kpop songs via an LSTM model.

## Day 12 : Oct 31th, 2018
**Today's Progress** : Working with large files on colab, while the files are loading head-started with RL project. Maybe next time, do the required uploads and installations the day before. Files in colab.

