
from gensim.models.doc2vec import LabeledSentence

#model = Doc2Vec(sentences)



## Reference : http://stackoverflow.com/questions/22129943/how-to-calculate-the-sentence-similarity-using-word2vec-model-of-gensim-with-pyt


model = Word2Vec(sentences, size=100, window=5, min_count=5, workers=4)