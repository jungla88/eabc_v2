# -*- coding: utf-8 -*-

import numpy as np
import scipy.spatial.distance as scDist

def medoid(cluster,dissimilarity):
    
    pairwise_dist = dissimilarity.pdist(cluster)
    
    if scDist.is_valid_y(pairwise_dist):
        pairwise_dist = scDist.squareform(pairwise_dist)
        
    SoD = np.sum(pairwise_dist, axis = 0)
    
    minSoDIdx = np.argmin(SoD)
    
    return cluster[minSoDIdx],SoD[minSoDIdx]
        
        
        