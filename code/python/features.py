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




