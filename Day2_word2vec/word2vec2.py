#-*- coding: utf-8 -*-


from gensim.models import Word2Vec
# define training data
sentences = [['this', 'is', 'the', 'first', 'sentence', 'for', 'word2vec'],
			['this', 'is', 'the', 'second', 'sentence'],
			['yet', 'another', 'sentence'],
			['one', 'more', 'sentence'],
			['and', 'the', 'final', 'sentence']]
# train model
model = Word2Vec(sentences, min_count=1)
# summarize the loaded model
print("the model is : ", model)
# summarize vocabulary
words = list(model.wv.vocab)
print("the words is : ", words)
# access vector for one word
print("the model[sentence] is : ", model['sentence'])
# save model
model.save('model.bin')
# load model
new_model = Word2Vec.load('model.bin')
print("the new_model is : ", new_model)

X = model[model.wv.vocab]

pca = PCA(n_components=2)
result = pca.fit_transform(X)

pyplot.scatter(result[:, 0], result[:, 1])

words = list(model.wv.vocab)
for i, word in enumerate(words):
	pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))

"""
from gensim.test.utils import common_texts
from gensim.models import Word2Vec
import gensim.downloader as api

model = Word2Vec(common_texts, size=100, window=5, min_count=1, workers=4)
word_vectors = model.wv

word_vectors = api.load("glove-wiki-gigaword-100")  # load pre-trained word-vectors from gensim-data
result = word_vectors.most_similar(positive=['woman', 'king'], negative=['man'])
print("{}: {:.4f}".format(*result[0]))

"""