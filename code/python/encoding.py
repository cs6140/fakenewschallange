from gensim import models
import numpy as np

model = models.Word2Vec.load_word2vec_format('/media/sree/venus/code/word2vec/GoogleNews-vectors-negative300.bin', binary=True)

def tovector(words):
    vector_array = []
    for w in words:
        try:
            vector_array.append(model[w])
        except:
            continue
    vector_array = np.array(vector_array)
    v = vector_array.sum(axis=0)
    return v / np.sqrt((v ** 2).sum())



def get_vectorizer_model():
    """
    """
    return model
    