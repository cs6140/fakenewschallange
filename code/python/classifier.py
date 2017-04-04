import os

mingw_path = 'C:\\Program Files\\mingw-w64\\x86_64-6.3.0-posix-seh-rt_v5-rev2\\mingw64\\bin'

os.environ['PATH'] = mingw_path + ';' + os.environ['PATH']
import xgboost as xgb
import sklearn.svm
from sklearn.ensemble import RandomForestClassifier
import pandas as pd


def train_XGB(data):
    """
    
    Arguments:
    - `data`:
    """
    predictors = ["overlapping","reoccur1", "reoccur2", "reoccur3", "reoccur4", "reoccur5", "reoccur6","cosine","wmdistance", "euclidean"]
    response = data.Stance



    gbm = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05).fit(data[predictors], response)
    
    return gbm

def train_SVM(data):

     predictors = ["overlapping","reoccur1", "reoccur2", "reoccur3", "reoccur4", "reoccur5", "reoccur6","cosine"]
     response = data.Stance

     clf = sklearn.svm.LinearSVC().fit(data[predictors],response)

     return clf

def randomForest (train,test):
    """

    Arguments:
    - `data`:
    """

    predictors = ["overlapping","reoccur1", "reoccur2", "reoccur3", "reoccur4", "reoccur5", "reoccur6","cosine","wmdistance", "euclidean"]
    response = train.Stance

    _test = test[["overlapping", "reoccur1", "reoccur2", "reoccur3", "reoccur4", "reoccur5", "reoccur6","cosine","wmdistance", "euclidean"]]

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
