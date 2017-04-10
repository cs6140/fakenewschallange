import pandas as pd

pd.set_option('max_rows', 7)
pd.set_option('expand_frame_repr', False)
import numpy as np
from sklearn.model_selection import train_test_split
# from fuzzy import fuzz

import preprocessing as pp
import encoding
import classifier
import features
import visualization as viz
from matplotlib.mlab import PCA
import numpy as np

import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale

bodies = "../../data/train_bodies.csv"
stances = "../../data/train_stances.csv"

content = pd.read_csv(bodies, sep=",")
headlines = pd.read_csv(stances, sep=",")

## generate necessary token features for dnews heading and news body
content['content_tokens'] = content.articleBody.apply(lambda x: pp.process(x))
headlines['headline_tokens'] = headlines.Headline.apply(lambda x: pp.process(x))

# ## Begin sentence embedding
header_vectors = np.zeros((headlines.shape[0], 300))
for i, q in enumerate(headlines.headline_tokens.values):
    header_vectors[i, :] = encoding.tovector(q)

# ## create the content vector
content_vectors = np.zeros((content.shape[0], 300))
for i, q in enumerate(content.content_tokens.values):
    content_vectors[i, :] = encoding.tovector(q)

header_series = pd.Series(header_vectors.tolist())
headlines['headline_vector'] = header_series.values

content_series = pd.Series(content_vectors.tolist())
content['content_vector'] = content_series.values

data = pd.merge(content, headlines, how="left", on="Body ID")

data['char_length_body'] = data['articleBody'].str.len()
data['char_length_headline'] = data['Headline'].str.len()
# print(headlines.shape)
# print(header_vectors.shape)
# print(content.shape)
# print(content_vectors.shape)
# print(data[['content_vector']].shape)
# headline_content = data[['headline_vector']]-data[['content_vector']];
# print(headline_content.shape)
# print(headlines[1])
# print(header_vectors[1])
# print(data[1])

# Feature 1 - Words overlapping between headline and content
data['overlapping'] = data[['headline_tokens', 'content_tokens']].apply(lambda x: features.overlapping(*x), axis=1)
data['phrase_reoccurance'] = data[['headline_tokens', 'content_tokens']].apply(lambda x: features.freqency_features(*x),
                                                                               axis=1)

## stupid code - boo !
reoccurance_cols = ["reoccur1", "reoccur2", "reoccur3", "reoccur4", "reoccur5", "reoccur6"]
for i in range(0, 6):
    data[reoccurance_cols[i]] = data['phrase_reoccurance'].apply(lambda x: x[i])

## Cosine similarity between word vectors
# data['cosine'] = data[['headline_vector','content_vector']].apply(lambda x: features.cosine(*x), axis=1)
# #data['wmdistance'] = data[['headline_tokens','content_tokens']].apply(lambda x: features.wmdistance(*x), axis=1)
data['euclidean'] = data[['headline_vector', 'content_vector']].apply(lambda x: features.euclidean(*x), axis=1)

# difference between headline_vector and content_vector
data['difference'] = data[['headline_vector', 'content_vector']].apply(lambda x: features.difference1(*x), axis=1)

# print(data['euclidean'].shape)
# print(data['difference'])
# print(data['difference'].as_matrix().shape)
# print(len(data['difference'])[0])



# # 80/20 Train-Test Split keeping splits consistent for future runs
# train, test = train_test_split(data, test_size = 0.2,random_state= 55)

# PCA


ddd = np.zeros((len(data['content_vector']), 300))
for i, q in enumerate(data['content_vector']):
    for j, p in enumerate(q):
        if np.isnan(p) == True:
            q[j] = 0
    ddd[i] = q

# label = np.zeros((len(data['Stance']), 2))
# for i, q in enumerate(data['Stance']):
#    for j, p in enumerate(q):
#        if np.isnan(p) == True:
#            q[j] = 0
#    label[i] = q

# print(q.shape)
# if np.isnan(0) == False:
#     print(100)

print(ddd.shape)

ddd_scaled = scale(ddd)

print(ddd_scaled.shape)

# a= np.array(data[])
pca = PCA(n_components=300)
results = pca.fit(ddd_scaled)

print(pca.explained_variance_ratio_)

a = pca.fit_transform(ddd_scaled)

aa = pd.Series(a[:, 0].tolist())
data["PCA1"] = aa.values
aa = pd.Series(a[:, 1].tolist())
data["PCA2"] = aa.values
aa = pd.Series(a[:, 2].tolist())
data["PCA3"] = aa.values
aa = pd.Series(a[:, 3].tolist())
data["PCA4"] = aa.values
aa = pd.Series(a[:, 4].tolist())
data["PCA5"] = aa.values
aa = pd.Series(a[:, 5].tolist())
data["PCA6"] = aa.values
aa = pd.Series(a[:, 6].tolist())
data["PCA7"] = aa.values
aa = pd.Series(a[:, 7].tolist())
data["PCA8"] = aa.values

# aa = pd.Series(a[:,299].tolist())
# data["PCA1"] = aa.values
# aa = pd.Series(a[:,298].tolist())
# data["PCA2"] = aa.values
# aa = pd.Series(a[:,297].tolist())
# data["PCA3"] = aa.values
# aa = pd.Series(a[:,296].tolist())
# data["PCA4"] = aa.values
# aa = pd.Series(a[:,295].tolist())
# data["PCA5"] = aa.values
# aa = pd.Series(a[:,294].tolist())
# data["PCA6"] = aa.values
# aa = pd.Series(a[:,293].tolist())
# data["PCA7"] = aa.values
# aa = pd.Series(a[:,292].tolist())
# data["PCA8"] = aa.values


print(a)

# 80/20 Train-Test Split keeping splits consistent for future runs
train, test = train_test_split(data, test_size=0.2, random_state=55)

# this will return an array of variance percentages for each component
# print(results.fracs.shape)
# print(results.fracs[0:3])

# this will return a 2d array of the data projected into PCA space
# print(results.Y.shape)
# print(results.Y[0:3])

clf = classifier.train_SVM_PCA(train)
print("SVM Classifier PCA")

_test = test[["PCA1", "PCA2", "PCA3", "PCA4", "PCA5", "PCA6", "PCA7", "PCA8"]]
# _test = test[["PCA1", "PCA2", "PCA3", "PCA4","PCA5", "PCA6", "PCA7"]]

_predictions = clf.predict(_test)

predictions = pd.Series(_predictions.tolist())
test["predicted_SVM_PCA"] = predictions.values

test["is_correct_prediction_SVM_PCA"] = test["Stance"] == test["predicted_SVM_PCA"]
correctly_predicted_rows = test[test['is_correct_prediction_SVM_PCA'] == True]

print("Accuracy : ", float(len(correctly_predicted_rows)) / len(test))

# ----------------------------------------------- Training Data Exploration/Visulation --------------------------------- #

# viz.summaryStatistics(train)
# viz.plot_overlapping(train)
# viz.plot_HLS(train)
# viz.plot_CLS(train)
# viz.pairPlot(train)
# viz.feature_bodyLength(train)
# viz.countPlot_headline_article(train)
# ---------------------------------------------------------------------------------------------------------------------#
## XGBoost classifier
# gbm = classifier.train_XGB(train)
# print("XGBoost classifier built...")


## XGBoost only accepts numerical fields - So I'm gonna remove the rest from test data
## we need to confirm this
# _test = test[["overlapping", "reoccur1", "reoccur2", "reoccur3","reoccur4", "reoccur5", "reoccur6","euclidean"]]#,"cosine"#,"wmdistance", "euclidean"]]
# _predictions = gbm.predict(_test)
#
# predictions = pd.Series(_predictions.tolist())
# test["predicted_XGB"] = predictions.values
#
#
# ## Accuracy calculation
# test["is_correct_prediction_XGB"] = test["Stance"] == test["predicted_XGB"]
# correctly_predicted_rows = test[test['is_correct_prediction_XGB'] == True]
#
# print("Accuracy : ", float(len(correctly_predicted_rows))/len(test))test


clf = classifier.train_SVM(train)
print("SVM Classifier")

_test = test[["overlapping", "reoccur1", "reoccur2", "reoccur3", "reoccur4", "reoccur5", "reoccur6", "euclidean"]]

_predictions = clf.predict(_test)

predictions = pd.Series(_predictions.tolist())
test["predicted_SVM"] = predictions.values

test["is_correct_prediction_SVM"] = test["Stance"] == test["predicted_SVM"]
correctly_predicted_rows = test[test['is_correct_prediction_SVM'] == True]

print("Accuracy : ", float(len(correctly_predicted_rows)) / len(test))

# ---------------------------------------- Random Forest Classifier PCA -----------------------------------------------------------#

print("Random Forest classifier building...")
rfc = classifier.randomForest_PCA2(train, test)
print("Random Forest classifier built ...")

# ---------------------------------------- Random Forest Classifier -----------------------------------------------------------#

print("Random Forest classifier building...")
rfc = classifier.randomForest(train, test)
print("Random Forest classifier built ...")

# ----------------------------------------- Cross Tabulation --------------------------------------------------------------------#

# print("\n Cross Tabulation for XGBOOST ",)
# print (pd.crosstab(test.Stance, test.predicted_XGB,margins=True))

print("\n Cross Tabulation for RANDOM FOREST ")
print(pd.crosstab(test.Stance, test.predicted_RF, margins=True))

print("\n Cross Tabulation for SVM  ")
print(pd.crosstab(test.Stance, test.predicted_SVM, margins=True))

print("\n Cross Tabulation for SVM PCA ")
print(pd.crosstab(test.Stance, test.predicted_SVM_PCA, margins=True))

print("\n Cross Tabulation for RANDOM FOREST PCA ")
print(pd.crosstab(test.Stance, test.predicted_RF_PCA, margins=True))

# ----------------------------------------  Test Data Visualization / Plots  ------------------------------------------------------------#

# viz.summaryStatistics(test)

# Bar Plot for comparing counts of  Actual Stances vs Predicted Stances in Test Data on Random Forest model
viz.countPlot(test)

# Compare Countplots of Random Forest, XGBoost, SVM on test set
viz.compare_countPlots(test)

# Swarm Plot for comparing counts of  Actual Stances vs Predicted Stances in Test Data on Random Forest model
# viz.swarmPlot(test)





