import pandas as pd
pd.set_option('max_rows', 7)
pd.set_option('expand_frame_repr', False)
import numpy as np
from sklearn.model_selection import train_test_split

import preprocessing as pp
#import encoding
import classifier
import features

bodies = "../../data/train_bodies.csv"
stances = "../../data/train_stances.csv"

content = pd.read_csv(bodies, sep=",")
headlines = pd.read_csv(stances, sep=",")

## generate necessary token features for dnews heading and news body
content['content_tokens'] = content.articleBody.apply(lambda x : pp.process(x))
headlines['headline_tokens'] = headlines.Headline.apply(lambda x: pp.process(x))


# ## Begin sentence embedding
# header_vectors = np.zeros((headlines.shape[0], 300))
# for i, q in enumerate(headlines.headline_tokens.values):
#     header_vectors[i, :] = encoding.tovector(q)

# ## create the content vector    
# content_vectors  = np.zeros((content.shape[0], 300))
# for i, q in enumerate(content.content_tokens.values):
#     content_vectors[i, :] = encoding.tovector(q)


# header_series = pd.Series(header_vectors.tolist())
# headlines['headline_vector'] = header_series.values
    
# content_series = pd.Series(content_vectors.tolist())
# content['content_vector'] = content_series.values


data = pd.merge(content, headlines, how="left", on="Body ID")

data['char_length_body']=data['articleBody'].str.len()
data['char_length_headline']=data['Headline'].str.len()


#Feature 1 - Words overlapping between headline and content
data['overlapping'] = data[['headline_tokens','content_tokens']].apply(lambda x: features.overlapping(*x), axis=1)
data['phrase_reoccurance'] = data[['headline_tokens','content_tokens']].apply(lambda x: features.freqency_features(*x), axis=1)

## stupid code - boo !
reoccurance_cols = ["reoccur1", "reoccur2", "reoccur3", "reoccur4", "reoccur5", "reoccur6"]
for i in range(0,6) :
    print(i)
    data[reoccurance_cols[i]] = data['phrase_reoccurance'].apply(lambda x: x[i])

## visualization of variation with Stance value
## data[['phrase_reoccurance','Stance']][1:10]

import visualization as viz
#Calling summary statistics from visualization.py
viz.summaryStatistics(data)

#Calling plots from visualization.py
viz.plot_overlapping(data)
viz.plot_HLS(data)
viz.plot_CLS(data)
viz.plot_headlineLength(data)

train, test = train_test_split(data, test_size = 0.2)

## XGBoost classifier
gbm = classifier.train_XGB(train)
print("XGBoost classifier built...")


## XGBoost only accepts numerical fields - So I'm gonna remove the rest from test data
## we need to confirm this
_test = test[["overlapping", "reoccur1", "reoccur2", "reoccur3", "reoccur4", "reoccur5", "reoccur6"]]
_predictions = gbm.predict(_test)

predictions = pd.Series(_predictions.tolist())
test["predicted"] = predictions.values


## Accuracy calculation
test["is_correct_prediction"] = test["Stance"] == test["predicted"]
correctly_predicted_rows = test[test['is_correct_prediction'] == True]

print("Accuracy : ", float(len(correctly_predicted_rows))/len(test))

