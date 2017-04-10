import os

mingw_path = 'C:\\Program Files\\mingw-w64\\x86_64-6.3.0-posix-seh-rt_v5-rev2\\mingw64\\bin'

os.environ['PATH'] = mingw_path + ';' + os.environ['PATH']
# import xgboost as xgb
import sklearn.svm
from sklearn.ensemble import RandomForestClassifier
import pandas as pd


def train_XGB(data):
    """
    
    Arguments:
    - `data`:
    """
    predictors = ["overlapping","reoccur1", "reoccur2", "reoccur3","reoccur4", "reoccur5", "reoccur6","euclidean"]#,"cosine"]#,"wmdistance", "euclidean"]
    response = data.Stance

    gbm = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05).fit(data[predictors], response)
    
    return gbm

def train_SVM(data):

    predictors = ["overlapping","reoccur1", "reoccur2", "reoccur3", "reoccur4", "reoccur5", "reoccur6","euclidean"]
    response = data.Stance

    clf = sklearn.svm.LinearSVC().fit(data[predictors],response)

    return clf

def randomForest (train,test):
    """

    Arguments:
    - `data`:
    """

    predictors = ["overlapping","reoccur1", "reoccur2", "reoccur3","reoccur4", "reoccur5", "reoccur6", "euclidean"]#,"cosine"]#,"wmdistance", "euclidean"]
    response = train.Stance

    _test = test[["overlapping", "reoccur1", "reoccur2", "reoccur3","reoccur4", "reoccur5", "reoccur6","euclidean"]]#,"cosine"]] #,"wmdistance", "euclidean"]]

    clf = RandomForestClassifier(n_jobs=2)
    clf.fit(train[predictors],response)
    _predictions = clf.predict(_test)

    predictions = pd.Series(_predictions.tolist())
    test["predicted_RF"] = predictions.values

    test["is_correct_prediction_RF"] = test["Stance"] == test["predicted_RF"]
    correctly_predicted_rows = test[test['is_correct_prediction_RF'] == True]

    print("Accuracy for Random Forest : ", float(len(correctly_predicted_rows))/len(test))
    # print(" Cross Tab for Random Forest ")
    # print (pd.crosstab(test.Stance, test.predicted_RF))


def train_SVM_PCA(data):

    predictors = ["PCA1", "PCA2", "PCA3", "PCA4","PCA5", "PCA6", "PCA7", "PCA8"]
    # predictors = ["PCA1", "PCA2", "PCA3", "PCA4","PCA5", "PCA6", "PCA7"]
    response = data.Stance

    clf = sklearn.svm.LinearSVC().fit(data[predictors],response)

    return clf

def randomForest_PCA (train,test):
    """

    Arguments:
    - `data`:
    """

    predictors = ["PCA1", "PCA2", "PCA3", "PCA4","PCA5", "PCA6", "PCA7", "PCA8"]
    response = train.Stance

    _test = test[["PCA1", "PCA2", "PCA3", "PCA4","PCA5", "PCA6", "PCA7", "PCA8"]]

    clf = RandomForestClassifier(n_jobs=2)
    clf.fit(train[predictors],response)
    _predictions = clf.predict(_test)

    predictions = pd.Series(_predictions.tolist())
    test["predicted_RF_PCA"] = predictions.values

    test["is_correct_prediction_RF_PCA"] = test["Stance"] == test["predicted_RF_PCA"]
    correctly_predicted_rows = test[test['is_correct_prediction_RF_PCA'] == True]

    print("Accuracy for Random Forest : ", float(len(correctly_predicted_rows))/len(test))
    # print(" Cross Tab for Random Forest ")
    # print (pd.crosstab(test.Stance, test.predicted_RF))

def randomForest_PCA2 (train,test):
    """

    Arguments:
    - `data`:
    """

    predictors = ["PCA1", "PCA2", "PCA3", "PCA4","PCA5", "overlapping","reoccur1", "reoccur2"]
    response = train.Stance

    _test = test[["PCA1", "PCA2", "PCA3", "PCA4","PCA5", "overlapping","reoccur1", "reoccur2"]]

    clf = RandomForestClassifier(n_jobs=2)
    clf.fit(train[predictors],response)
    _predictions = clf.predict(_test)

    predictions = pd.Series(_predictions.tolist())
    test["predicted_RF_PCA"] = predictions.values

    test["is_correct_prediction_RF_PCA"] = test["Stance"] == test["predicted_RF_PCA"]
    correctly_predicted_rows = test[test['is_correct_prediction_RF_PCA'] == True]

    print("Accuracy for Random Forest : ", float(len(correctly_predicted_rows))/len(test))
    # print(" Cross Tab for Random Forest ")
    # print (pd.crosstab(test.Stance, test.predicted_RF))

    
def randomForest_small (train,test):
    """

    Arguments:
    - `data`:
    """

    predictors = ["overlapping","reoccur1", "reoccur2"]
    response = train.Stance

    _test = test[["overlapping","reoccur1", "reoccur2"]]

    clf = RandomForestClassifier(n_jobs=2)
    clf.fit(train[predictors],response)
    _predictions = clf.predict(_test)

    predictions = pd.Series(_predictions.tolist())
    test["predicted_RF_PCA"] = predictions.values

    test["is_correct_prediction_RF_PCA"] = test["Stance"] == test["predicted_RF_PCA"]
    correctly_predicted_rows = test[test['is_correct_prediction_RF_PCA'] == True]

    print("Accuracy for Random Forest : ", float(len(correctly_predicted_rows))/len(test))
    # print(" Cross Tab for Random Forest ")
    # print (pd.crosstab(test.Stance, test.predicted_RF))


def randomForest_PCA3 (train,test):
    """

    Arguments:
    - `data`:
    """

    predictors = ["PCA1", "PCA2", "PCA3", "PCA4","PCA5", "PCA6", "PCA7", "PCA8", "overlapping","reoccur1", "reoccur2"]
    response = train.Stance

    _test = test[["PCA1", "PCA2", "PCA3", "PCA4","PCA5", "PCA6", "PCA7", "PCA8", "overlapping","reoccur1", "reoccur2"]]

    clf = RandomForestClassifier(n_jobs=2)
    clf.fit(train[predictors],response)
    _predictions = clf.predict(_test)

    predictions = pd.Series(_predictions.tolist())
    test["predicted_RF_PCA"] = predictions.values

    test["is_correct_prediction_RF_PCA"] = test["Stance"] == test["predicted_RF_PCA"]
    correctly_predicted_rows = test[test['is_correct_prediction_RF_PCA'] == True]

    print("Accuracy for Random Forest : ", float(len(correctly_predicted_rows))/len(test))
    # print(" Cross Tab for Random Forest ")
    # print (pd.crosstab(test.Stance, test.predicted_RF))
