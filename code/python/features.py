import scipy as sp






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
        dist = sp.spatial.distance.euclidean(u,v)
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