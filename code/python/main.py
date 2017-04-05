import pandas as pd
pd.set_option('max_rows', 7)
pd.set_option('expand_frame_repr', False)
import numpy as np
from sklearn.model_selection import train_test_split
#from fuzzy import fuzz

import preprocessing as pp
import encoding
import classifier
import features
import visualization as viz

bodies = "../../data/train_bodies.csv"
stances = "../../data/train_stances.csv"

content = pd.read_csv(bodies, sep=",")
headlines = pd.read_csv(stances, sep=",")


## generate necessary token features for dnews heading and news body
content['content_tokens'] = content.articleBody.apply(lambda x : pp.process(x))
headlines['headline_tokens'] = headlines.Headline.apply(lambda x: pp.process(x))


# ## Begin sentence embedding
header_vectors = np.zeros((headlines.shape[0], 300))
for i, q in enumerate(headlines.headline_tokens.values):
    header_vectors[i, :] = encoding.tovector(q)

# ## create the content vector
content_vectors  = np.zeros((content.shape[0], 300))
for i, q in enumerate(content.content_tokens.values):
    content_vectors[i, :] = encoding.tovector(q)


header_series = pd.Series(header_vectors.tolist())
headlines['headline_vector'] = header_series.values

content_series = pd.Series(content_vectors.tolist())
content['content_vector'] = content_series.values


data = pd.merge(content, headlines, how="left", on="Body ID")

data['char_length_body']=data['articleBody'].str.len()
data['char_length_headline']=data['Headline'].str.len()


#Feature 1 - Words overlapping between headline and content
data['overlapping'] = data[['headline_tokens','content_tokens']].apply(lambda x: features.overlapping(*x), axis=1)
data['phrase_reoccurance'] = data[['headline_tokens','content_tokens']].apply(lambda x: features.freqency_features(*x), axis=1)

## stupid code - boo !
reoccurance_cols = ["reoccur1", "reoccur2", "reoccur3", "reoccur4", "reoccur5", "reoccur6"]
for i in range(0,6) :
    data[reoccurance_cols[i]] = data['phrase_reoccurance'].apply(lambda x: x[i])


## Cosine similarity between word vectors
# data['cosine'] = data[['headline_vector','content_vector']].apply(lambda x: features.cosine(*x), axis=1)
# #data['wmdistance'] = data[['headline_tokens','content_tokens']].apply(lambda x: features.wmdistance(*x), axis=1)
data['euclidean'] = data[['headline_vector','content_vector']].apply(lambda x: features.euclidean(*x), axis=1)


# 80/20 Train-Test Split keeping splits consistent for future runs
train, test = train_test_split(data, test_size = 0.2,random_state= 55)

# ----------------------------------------------- Training Data Exploration/Visulation --------------------------------- #

#viz.summaryStatistics(train)
viz.plot_overlapping(train)
viz.plot_HLS(train)
viz.plot_CLS(train)
viz.pairPlot(train)
viz.feature_bodyLength(train)
viz.countPlot_headline_article(train)
# ---------------------------------------------------------------------------------------------------------------------#
## XGBoost classifier
gbm = classifier.train_XGB(train)
print("XGBoost classifier built...")


## XGBoost only accepts numerical fields - So I'm gonna remove the rest from test data
## we need to confirm this
_test = test[["overlapping", "reoccur1", "reoccur2", "reoccur3","reoccur4", "reoccur5", "reoccur6","euclidean"]]#,"cosine"#,"wmdistance", "euclidean"]]
_predictions = gbm.predict(_test)

predictions = pd.Series(_predictions.tolist())
test["predicted_XGB"] = predictions.values


## Accuracy calculation
test["is_correct_prediction_XGB"] = test["Stance"] == test["predicted_XGB"]
correctly_predicted_rows = test[test['is_correct_prediction_XGB'] == True]

print("Accuracy : ", float(len(correctly_predicted_rows))/len(test))


clf = classifier.train_SVM(train)
print("SVM Classifier")

_test = test[["overlapping", "reoccur1", "reoccur2", "reoccur3", "reoccur4", "reoccur5", "reoccur6","euclidean"]]

_predictions = clf.predict(_test)

predictions = pd.Series(_predictions.tolist())
test["predicted_SVM"] = predictions.values

test["is_correct_prediction_SVM"] = test["Stance"] == test["predicted_SVM"]
correctly_predicted_rows = test[test['is_correct_prediction_SVM'] == True]

print("Accuracy : ", float(len(correctly_predicted_rows))/len(test))

# ---------------------------------------- Random Forest Classifier -----------------------------------------------------------#

print("Random Forest classifier building...")
rfc = classifier.randomForest(train,test)
print("Random Forest classifier built ...")

# ----------------------------------------- Cross Tabulation --------------------------------------------------------------------#

print("\n Cross Tabulation for XGBOOST ",)
print (pd.crosstab(test.Stance, test.predicted_XGB))

print("\n Cross Tabulation for RANDOM FOREST ")
print (pd.crosstab(test.Stance, test.predicted_RF))

print("\n Cross Tabulation for SVM  ")
print (pd.crosstab(test.Stance, test.predicted_SVM))

# ----------------------------------------  Test Data Visualization / Plots  ------------------------------------------------------------#

#viz.summaryStatistics(test)

# Bar Plot for comparing counts of  Actual Stances vs Predicted Stances in Test Data on Random Forest model
viz.countPlot(test)

# Compare Countplots of Random Forest, XGBoost, SVM on test set
viz.compare_countPlots(test)

# Swarm Plot for comparing counts of  Actual Stances vs Predicted Stances in Test Data on Random Forest model
#viz.swarmPlot(test)





