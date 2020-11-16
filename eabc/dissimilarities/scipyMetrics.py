# -*- coding: utf-8 -*-

from eabc.dissimilarities import Dissimilarity
import scipy.spatial.distance as ssd
from numpy import sqrt
class scipyMetrics(Dissimilarity):
    
    def __init__(self, metricTypeStr, w = None):
        
        self._weight = w
        
        self._metricType = metricTypeStr
        
        if self._metricType == 'euclidean':
            self.diss = ssd.euclidean
        
    def __call__(self, u, v):
        
        d = self.diss(u,v,self._weight)
        
        #TODO: possibili problemi con numero di feature e weight
        return d/sqrt(len(u))
    
    def pdist(self, set1):
        
        return ssd.squareform(ssd.pdist(set1, self._metricType))
        
    
        