import sys
sys.path.append("../")

import pandas as pd
import featureengineering as fe

import scipy as sp
import numpy as np

from gensim.models import doc2vec




## read the  joined csv file
filename = "../../../data/sample.csv"
data = pd.read_csv(filename, sep=',')


## generate necessary token features for dnews heading and news body
data['header_features'] = data.Headline.apply(lambda x : fe.process(x))
data['content_features'] = data.articleBody.apply(lambda x : fe.process(x))


model = doc2vec.Doc2Vec.load('../../../models/doc2vec.model')

data['header_vector'] = data.header_features.apply(lambda x : model.infer_vector(x))
data['content_vector'] = data.content_features.apply(lambda x : model.infer_vector(x))

data['euclidean_distance'] = data[['header_vector','content_vector']].apply(lambda x: sp.spatial.distance.euclidean(*x), axis=1)


### we have calculated the cosine distance between header and body


### Exploring range of values of euclidean distances obtained
max_euclidean = np.max(data.euclidean_distance)  ## 1.9628843084637788
min_euclidean = np.min(data.euclidean_distance)  ## 0.02141936019460422


for stance_level in np.unique(data.Stance):
    filtered_rows = data[(data.Stance == stance_level)]

    print("Statistics for group : " + stance_level)

    ## range of euclideans
    group_max_euclidean = np.max(filtered_rows.euclidean_distance)  
    group_min_euclidean = np.min(filtered_rows.euclidean_distance)

    print("Max euclidean for range : " , group_max_euclidean)
    print("Min euclidean for range : " , group_min_euclidean)
    