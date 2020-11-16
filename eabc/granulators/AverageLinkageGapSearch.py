# -*- coding: utf-8 -*-

from eabc.granulators import Granulator
from eabc.granulators import Granule
from eabc.datasets import vectorDataset
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
        
        super(HierchicalAggl,self).__init__()
        
    def granulate(self,Dataset):
        
        if not isinstance(Dataset,vectorDataset):
            raise TypeError
        
        gapk = self._gapStatEvaluator.evaluateGap(Dataset.data)
        #TODO:naive method for search gap
        bestK = np.argmax(gapk)+1
        
        clustersLabels = self._gapStatEvaluator.solutions[bestK]
       
        reprElems = [self._representation(Dataset.data[l == clustersLabels], self._distanceFunction) for l in range(bestK)]
                
        #Evaluation - Lower is better
        normalizeCard = [1-(clustersLabels.tolist().count(l)/len(Dataset.data)) 
                         if clustersLabels.tolist().count(l)>1 else 1 for l in range(bestK)]
        normalizeComp = [reprElems[l][1]/(clustersLabels.tolist().count(l)-1) 
                         if clustersLabels.tolist().count(l)>1 else 1 for l in range(bestK)]
        
        for i,repres in enumerate(reprElems):
            F = super(HierchicalAggl,self)._evaluateF(normalizeComp[i],normalizeCard[i])
            newGr = Granule(repres[0],self._distanceFunction,F,normalizeCard[i],normalizeComp[i])
            super(HierchicalAggl,self)._addSymbol(newGr)
                
            
        
        