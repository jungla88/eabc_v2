# -*- coding: utf-8 -*-

from eabc.granulators import Granulator
from eabc.granulators import Granule
from eabc.extras import BSAS
from eabc.extras.BsasThetaSearch import LinearSearch

class BsasGralgLike(Granulator):
    
    #TEST: Trying normalization of F value fo min-max approach
    def __init__(self,DistanceFunction,clusterRepresentative,tStep=0.1,Qmax=100):
        
        self._distanceFunction = DistanceFunction
        self._representation = clusterRepresentative #An object for evaluate the representative
        
        self._Qmax= Qmax
        
        self._method=BSAS(self._representation,self._distanceFunction, Q= self._Qmax)
        self._tStep = tStep
 
        super().__init__()
        
    @property
    def BsasQmax(self):
        return self._Qmax
    @BsasQmax.setter
    def BsasQmax(self,val):
        if val > 0:
            self._Qmax = val
            self._method.Q = self._Qmax 
        else:
            raise ValueError
    
        
    def granulate(self,Dataset):
        
        partitions = LinearSearch(Dataset.data,self._method,self._tStep)
                    
        #TODO: interface for switching to this method
        #All symbols
        for key in partitions.keys():
            clustersLabels, reprElems = partitions[key][0],partitions[key][1]
            # singleton or universe clusters        
            clustersLabels,reprElems = super(BsasGralgLike,self)._removeSingularity(clustersLabels,reprElems,Dataset)

            nClust = len(reprElems)
            #Evaluation - Lower is better
            normalizeCard = [1-(len(clustersLabels[l])/len(Dataset.data)) 
                              if len(clustersLabels)>1 else 1 for l in range(nClust)]
            normalizeComp = [reprElems[l]._SOD/(len(clustersLabels[l])-1) 
                              if len(clustersLabels[l])>1 else 1 for l in range(nClust)]
          
            for i,repres in enumerate(reprElems):
                F = super(BsasGralgLike,self)._evaluateF(normalizeComp[i],normalizeCard[i])
                newGr = Granule(repres._representativeElem,self._distanceFunction,F,normalizeCard[i],normalizeComp[i])
                super(BsasGralgLike,self)._addSymbol(newGr)