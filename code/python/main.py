import pandas as pd

import preprocessing as pp
# import encoding
import features
import classifier
import numpy as np


## read the  joined csv file
filename = "../../data/train.csv"
data = pd.read_csv(filename, sep=',')

#data = data.sample(frac=0.05)

## generate necessary token features for dnews heading and news body
data['header_features'] = data.Headline.apply(lambda x : pp.process(x))
data['content_features'] = data.articleBody.apply(lambda x : pp.process(x))

data['overlapping'] = data[['header_features','content_features']].apply(lambda x: features.overlapping(*x), axis=1)

## XGBoost classifier
gbm = classifier.classify_XGB(data)
print("XGBoost classifier built...")

# ## generate the similarity features (ordinal)
# header_vectors = np.zeros((data.shape[0], 300))
# for i, q in enumerate(data.header_features.values):
#     header_vectors[i, :] = encoding.tovector(q)

# ## create the content vector    
# content_vectors  = np.zeros((data.shape[0], 300))
# for i, q in enumerate(data.content_features.values):
#     content_vectors[i, :] = encoding.tovector(q)


# header_series = pd.Series(header_vectors.tolist())
# data['header_vector'] = header_series.values
    
# content_series = pd.Series(content_vectors.tolist())
# data['content_vector'] = content_series.values

# import pdb
# pdb.set_trace()

# model = encoding.get_vectorizer_model()

# data['cosine_distance'] = data[['header_vector','content_vector']].apply(lambda x: features.cosine(*x), axis=1)
# data['euclidean_distance'] = data[['header_vector','content_vector']].apply(lambda x: features.euclidean(*x), axis=1)
# data['wordmover_distance'] = data.apply(lambda x: model.wmdistance(x['header_features'], x['content_features']), axis=1)
# data['minkowski_distance'] = data[['header_vector','content_vector']].apply(lambda x: features.minkowski(*x), axis=1)
# data['canberra_distance'] = data[['header_vector','content_vector']].apply(lambda x: features.canberra(*x), axis=1)