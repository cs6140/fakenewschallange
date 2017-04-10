import os

mingw_path = 'C:\\Program Files\\mingw-w64\\x86_64-6.3.0-posix-seh-rt_v5-rev2\\mingw64\\bin'

os.environ['PATH'] = mingw_path + ';' + os.environ['PATH']
import xgboost as xgb
import sklearn.svm
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

from sklearn import linear_model, datasets



from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV




def train_XGB(data, test):
    """
    
    Arguments:
    - `data`:
    """
    predictors = ["overlapping","reoccur1", "reoccur2", "reoccur3","reoccur4", "reoccur5", "reoccur6","euclidean","refuting_feature_count","char_length_headline","char_length_body"]#,"cosine"]#,"wmdistance", "euclidean"]
    response = data.Stance


    # n_estimators = range(50,500, 50)
    # depth = range(1,10)
    #
    # param_grid = dict(n_estimators=n_estimators, max_depth = depth)
    #
    # kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
    #
    # model = xgb.XGBClassifier()
    #
    # grid_search = GridSearchCV(model, param_grid, scoring="neg_log_loss", n_jobs=-1, cv=kfold)
    # grid_result = grid_search.fit(data[predictors], response)
    #
    # print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))



    gbm = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05).fit(data[predictors], response)

    _test = test[["overlapping", "reoccur1", "reoccur2", "reoccur3","reoccur4", "reoccur5", "reoccur6","euclidean","refuting_feature_count","char_length_headline","char_length_body"]]#,"cosine"#,"wmdistance", "euclidean"]]
    _predictions = gbm.predict(_test)

    predictions = pd.Series(_predictions.tolist())
    test["predicted_XGB"] = predictions.values


    ## Accuracy calculation
    test["is_correct_prediction_XGB"] = test["Stance"] == test["predicted_XGB"]
    correctly_predicted_rows = test[test['is_correct_prediction_XGB'] == True]

    print("Accuracy : ", float(len(correctly_predicted_rows))/len(test))


    
def train_SVM(data):

     predictors = ["overlapping","reoccur1", "reoccur2", "reoccur3", "reoccur4", "reoccur5", "reoccur6","euclidean","refuting_feature_count","char_length_headline","char_length_body"]
     response = data.Stance

     clf = sklearn.svm.LinearSVC().fit(data[predictors],response)

     return clf



def train_logistic(data):
    """
    
    Arguments:
    - `data`:
    """
    predictors = ["overlapping","reoccur1", "reoccur2", "reoccur3", "reoccur4", "reoccur5", "reoccur6","euclidean""refuting_feature_count","char_length_headline","char_length_body"]
    response = data.Stance

    logreg = linear_model.LogisticRegression(C=1e5)
    logistic_classifier = logreg.fit(data[predictors],response)
    
    


     
     

def randomForest (train,test):
    """

    Arguments:
    - `data`:
    """

    predictors = ["overlapping","reoccur1", "reoccur2", "reoccur3","reoccur4", "reoccur5", "euclidean","refuting_feature_count","char_length_headline","char_length_body"]#,"cosine"]#,"wmdistance", "euclidean"]
    response = train.Stance

    _test = test[["overlapping", "reoccur1", "reoccur2", "reoccur3","reoccur4", "reoccur5","euclidean","refuting_feature_count","char_length_headline","char_length_body"]]#,"cosine"]] #,"wmdistance", "euclidean"]]

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
