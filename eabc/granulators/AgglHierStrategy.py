# -*- coding: utf-8 -*-

from eabc.granulators import Granulator
from eabc.extras import GapStatistic
from sklearn.cluster import AgglomerativeClustering
import numpy as np

###Just a wrapper for the evaluator
class AgglWrapper:
    
    def __init__(self,affinity="sqeuclidean",linkage = "average"):
        
        self.method = AgglomerativeClustering()        
        self.affinity= affinity
        self.linkage = linkage
    
    def __call__(self,X,k,precomputed=False):
        
        self.method.n_clusters=k
        self.method.affinity = "precomputed" if precomputed else self.affinity        
        self.method.linkage = self.linkage
        
        labels = self.method.fit_predict(X)
        
        return labels
###

class HierchicalAggl(Granulator):
    
    def __init__(self,DistanceFunction,clusterRepresentative):
        
        self._distanceFunction = DistanceFunction
        self._representation = clusterRepresentative #An object for evaluate the representative
        
        #TODO: pdist
        self._clusteringMethod = AgglWrapper(affinity=self._distanceFunction.pdist)
        
        self._gapStatEvaluator = GapStatistic(self._clusteringMethod,self._distanceFunction)
        
        #debug
        self.repr = None
        
        super(HierchicalAggl,self).__init__()
        
    def granulate(self,data):
        
        gapk = self._gapStatEvaluator.evaluateGap(data)
        bestK = np.argmax(gapk)
        
        #TODO:naive
        clustersLabels = self._gapStatEvaluator.solutions[bestK+1]
        
        #FIXME: wrong result
        reprElems = [self._representation(data[l == clustersLabels], self._distanceFunction) for l in range(bestK+1)]
        
        self.repr = reprElems
        #avg distance
            
        