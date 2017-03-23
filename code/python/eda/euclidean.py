import sys, os
sys.path.append("../")


from gensim import models

import pandas as pd
import numpy as np
import scipy as sp

import featureengineering as pp


filename = "../../../data/sample.csv"
data = pd.read_csv(filename, sep=',')


data['header_features'] = data.Headline.apply(lambda x : pp.process(x))
data['content_features'] = data.articleBody.apply(lambda x : pp.process(x))

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


def euclidean(u, v):
    """
    
    Arguments:
    - `u`:
    - `v`:
    """
    dist = 0.0
    try:
        dist = sp.spatial.distance.euclidean(u,v)
    except:
        print("Error")
    return dist



data['euclidean_distance'] = data[['header_vector','content_vector']].apply(lambda x: euclidean(*x), axis=1)


data['header_vectors'] = data.header_features.apply(lambda x : sent2vec(x))
data['content_vectors'] = data.header_features.apply(lambda x : sent2vec(x))




def toCSV(stance, values):
    metric = "euclidean_"
    f = open(metric + stance + "_values.csv", "w")

    try:
        os.remove(f.name)
    except OSError:
        pass

    f = open(metric + stance + "_values.csv", "w")        
    for i in values:
        f.write(str(i) + "\n")


range_of_values = {}

for stance_level in np.unique(data.Stance):
    filtered_rows = data[(data.Stance == stance_level)]

    print("Statistics for group : " + stance_level)

    ## range of euclideans
    group_max_euclidean = np.max(filtered_rows.euclidean_distance)  
    group_min_euclidean = np.min(filtered_rows.euclidean_distance)

    print("Max euclidean for range : " , group_max_euclidean)
    print("Min euclidean for range : " , group_min_euclidean)

    range_of_values[stance_level] = filtered_rows.euclidean_distance.tolist()

    value_list = filtered_rows.euclidean_distance.tolist()
    value_list.sort()
    toCSV(stance_level, value_list)
