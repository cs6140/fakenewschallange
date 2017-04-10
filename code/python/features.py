import encoding
import scipy as sp
from collections import Counter
import numpy as np
# <<<<<<< HEAD
# import sklearn.feature_extraction.text
# =======
# import encoding
#
# >>>>>>> 7fafaaef1c972a0de2ee95d6e1a281f22e6f7331

model = encoding.get_vectorizer_model()

def overlapping(headline, body):
    """
    
       Ratio of intersection upon union of tokens
    
    Arguments:
    - `headline`:
    - `body`:
    """
    return len(set(headline).intersection(body)) / float(len(set(headline).union(body)))




def generate_ngrams(input_list, n):
    """
    generate n grams

    pass 2 to get bigrams
    pass 3 to get bigrams
    
    Arguments:
    - `tokens`:
    """
    return zip(*[input_list[i:] for i in range(n)])
    

def count_grams(headline, body, n):
    """
    
    Arguments:
    - `headline`:
    - `body`:
    - `n`:
    """
    headline_ngrams = list(generate_ngrams(headline, n))
    body_ngrams = list(generate_ngrams(body, n))

    reoccuring_grams = [gram for gram in headline_ngrams if gram in body_ngrams]
    counts = Counter(reoccuring_grams)
    return sum(counts.values())
    


    
def freqency_features(headline, body):
    """
    
    Arguments:
    - `headline`:
    - `body`:
    """
    return [count_grams(headline, body, i) for i in range(1,7)]
  

    
def cosine(u, v):
    """
    
    Arguments:
    - `u`:
    - `v`:
    """
    dist = 0.0
    try:
        dist = sp.spatial.distance.cosine(u,v)
    except:
        print("Error...Returning 0.0")
    return dist




def euclidean(u, v):
    """
    
    Arguments:
    - `u`:
    - `v`:
    """
    dist = 0.0
    try:
        dist = sp.spatial.distance.euclidean(u,v)
    except:
        print("Error...Returning 0.0")
    return dist
    

    
def wmdistance(u, v):
    """
    
    Arguments:
    - `u`:
    - `v`:
    """
    dist = 0.0
    try:
        dist = model.wmdistance(u,v)
    except:
        print("Error...Returning 0.0")
    return dist


    

def minkowski(u, v):
    """
    
    Arguments:
    - `u`:
    - `v`:
    """
    dist = 0.0
    try:
        dist = sp.spatial.distance.minkowski(u,v)
    except:
        print("Error...Returning 0.0")
    return dist
    

    
def canberra(u, v):
    """
    
    Arguments:
    - `u`:
    - `v`:
    """
    dist = 0.0
    try:
        dist = sp.spatial.distance.canberra(u,v)
    except:
        print("Error...Returning 0.0")
    return dist


def refuting_features(headline, body):
    _refuting_words = [
        'fake',
        'fraud',
        'hoax',
        'false',
        'deny', 'denies',
        # 'refute',
        'not',
        'despite',
        'nope',
        'doubt', 'doubts',
        'bogus',
        'debunk',
        'pranks',
        'retract'
    ]

    X = []
    features = [1 if word in headline else 0 for word in _refuting_words]
    X.append(features)
    return X
    # for i, (headline, body):
    #
    #     features = [1 if word in headline else 0 for word in _refuting_words]
    #     X.append(features)

    
def refuting_features_count(headline, body):
    _refuting_words = [
        'fake','doctored'
        'fraud',
        'hoax',
        'false',
        'deny', 'denies',
        'not',
        'despite',
        'nope',
        'doubt', 'doubts',
        'bogus',
        'debunk','debunked'
        'pranks',
        'retract',
        'wrong','wrongly'
    ]
    count=0
    for token in headline:
        if token in _refuting_words:
            count += 100
    return count