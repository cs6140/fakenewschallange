import sys
sys.path.append("../")


from gensim import models

import pandas as pd
import numpy as np
import scipy as sp

import featureengineering as fe


filename = "../../../data/sample.csv"
data = pd.read_csv(filename, sep=',')


data['header_features'] = data.Headline.apply(lambda x : fe.process(x))
data['content_features'] = data.articleBody.apply(lambda x : fe.process(x))

## change this to load the word2vec model from your system
model = models.Word2Vec.load_word2vec_format('/media/sree/venus/code/word2vec/GoogleNews-vectors-negative300.bin', binary=True)

def sent2vec(words):
    vector_array = []
    for w in words:
        try:
            vector_array.append(model[w])
        except:
            continue
    vector_array = np.array(vector_array)
    v = vector_array.sum(axis=0)
    return v / np.sqrt((v ** 2).sum())



## create the header vector    
header_vectors = np.zeros((data.shape[0], 300))
for i, q in enumerate(data.header_features.values):
    header_vectors[i, :] = sent2vec(q)

## create the content vector    
content_vectors  = np.zeros((data.shape[0], 300))
for i, q in enumerate(data.content_features.values):
    content_vectors[i, :] = sent2vec(q)


header_series = pd.Series(header_vectors.tolist())
data['header_vector'] = header_series.values
    
content_series = pd.Series(content_vectors.tolist())
data['content_vector'] = content_series.values


def cosine(u, v):
    """
    
    Arguments:
    - `u`:
    - `v`:
    """
    dist = 0.0
    try:
        dist = sp.spatial.distance.cosine(u,v)
    except:
        print("Error")
    return dist



data['cosine_distance'] = data[['header_vector','content_vector']].apply(lambda x: cosine(*x), axis=1)


data['header_vectors'] = data.header_features.apply(lambda x : sent2vec(x))
data['content_vectors'] = data.header_features.apply(lambda x : sent2vec(x))





for stance_level in np.unique(data.Stance):
    filtered_rows = data[(data.Stance == stance_level)]

    print("Statistics for group : " + stance_level)

    ## range of cosines
    group_max_cosine = np.max(filtered_rows.cosine_distance)  
    group_min_cosine = np.min(filtered_rows.cosine_distance)

    print("Max cosine for range : " , group_max_cosine)
    print("Min cosine for range : " , group_min_cosine)
