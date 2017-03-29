import xgboost as xgb



def train_XGB(data):
    """
    
    Arguments:
    - `data`:
    """
    predictors = ['overlapping']
    response = data.Stance
    
    gbm = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05).fit(data[predictors], response)
    
    return gbm



