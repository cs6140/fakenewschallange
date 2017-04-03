from gensim.models.keyedvectors import KeyedVectors

import numpy as np




model = KeyedVectors.load_word2vec_format('C:/Users/AKHIL NAIR/Documents/Project CS6140/GoogleNews-vectors-negative300.bin.gz', binary=True)


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
    