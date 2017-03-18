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

data['cosine_distance'] = data[['header_vector','content_vector']].apply(lambda x: sp.spatial.distance.cosine(*x), axis=1)


### we have calculated the cosine distance between header and body


### Exploring range of values of cosine distances obtained
max_cosine = np.max(data.cosine_distance)  ## 1.9628843084637788
min_cosine = np.min(data.cosine_distance)  ## 0.02141936019460422


for stance_level in np.unique(data.Stance):
    filtered_rows = data[(data.Stance == stance_level)]

    print("Statistics for group : " + stance_level)

    import pdb
    pdb.set_trace()
    
    ## range of cosines
    group_max_cosine = np.max(filtered_rows.cosine_distance)  
    group_min_cosine = np.min(filtered_rows.cosine_distance)

    print("Max cosine for range : " , group_max_cosine)
    print("Min cosine for range : " , group_min_cosine)
    