# -*- coding: utf-8 -*-

from eabc.representatives import Representative
import scipy.spatial.distance as scDist
import numpy as np


class Medoid(Representative):
    
    def __init__(self,initRepr = None):

        super(Medoid,self).__init__(initReprElem=initRepr)   
        
    def evaluate(self,cluster,Dissimilarity):
        
        pairwise_dist = Dissimilarity.pdist(cluster)
    
        if scDist.is_valid_y(pairwise_dist):
            pairwise_dist = scDist.squareform(pairwise_dist)
        
        self._DistanceMatrix = pairwise_dist
        
        #FIXME: diss matrix for graphs is not symmetric
        SoD = np.sum(pairwise_dist, axis = 0)    
        minSoDIdx = np.argmin(SoD)
        
        self._representativeElem = cluster[minSoDIdx]    
        self._SOD = SoD[minSoDIdx]                