import scipy as sp






def cosine(u, v):
    """
    
    Arguments:
    - `u`:
    - `v`:
    """
    dist = 0.0
    import pdb;
    pdb.set_trace()
    try:
        dist = sp.spatial.distance.cosine(u,v)
    except:
        print("Error")
    return dist




