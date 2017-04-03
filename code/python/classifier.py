import os

mingw_path = 'C:\\Program Files\\mingw-w64\\x86_64-6.3.0-posix-seh-rt_v5-rev2\\mingw64\\bin'

os.environ['PATH'] = mingw_path + ';' + os.environ['PATH']
import xgboost as xgb
import sklearn.svm


def train_XGB(data):
    """
    
    Arguments:
    - `data`:
    """
    predictors = ["overlapping","reoccur1", "reoccur2", "reoccur3", "reoccur4", "reoccur5", "reoccur6","cosine"]
    response = data.Stance
    
    gbm = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05).fit(data[predictors], response)
    
    return gbm

def train_SVM(data):

     predictors = ["overlapping","reoccur1", "reoccur2", "reoccur3", "reoccur4", "reoccur5", "reoccur6","cosine"]
     response = data.Stance

     clf = sklearn.svm.LinearSVC().fit(data[predictors],response)

     return clf