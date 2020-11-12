# -*- coding: utf-8 -*-

from eabc.dissimilarities import Dissimilarity
from scipy.spatial.distance import sqeuclidean, pdist,squareform
class scipyMetrics(Dissimilarity):
    
    def __init__(self, metricTypeStr, w = None):
        
        self._weight = w
        
        self._metricType = metricTypeStr
        
    def __call__(self, u, v):
        
        d = sqeuclidean(u,v,self._weight)
        
        return d
    
    def pdist(self, set1):
        
        return squareform(pdist(set1, self._metricType))
        
    
        