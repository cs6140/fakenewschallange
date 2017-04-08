import scipy as sp
from collections import Counter
import sklearn.feature_extraction.text
import encoding

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