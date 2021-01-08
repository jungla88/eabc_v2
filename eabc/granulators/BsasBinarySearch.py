# -*- coding: utf-8 -*-

from eabc.granulators import Granulator
from eabc.granulators import Granule
from eabc.extras import BSAS
from eabc.extras.BinarySearch import BinarySearch

import numpy as np



class BsasBinarySearch(Granulator):
    
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
        
        partitions = BinarySearch(Dataset.data,self._method,self._tStep)
        
        #Select partition based on persistence
        gap = 0
        bestP = None
        for i in range(len(partitions)-1):                        
            theta = sorted(list(partitions.keys()))[i]
            thetaNext = sorted(list(partitions.keys()))[i+1]            
            gapTemp = thetaNext - theta            
            if gapTemp > gap:
                bestP = i
        
        theta = sorted(list(partitions.keys()))[bestP]
        
        clustersLabels, reprElems = partitions[theta][0],partitions[theta][1]

        #TODO: interface for switching to this method
        #All symbols
        # for key in partitions.keys():
        #     clustersLabels, reprElems = partitions[key][0],partitions[key][1]
        #     nClust = len(reprElems)
        #     #Evaluation - Lower is better
        #     normalizeCard = [1-(len(clustersLabels[l])/len(Dataset.data)) 
        #                       if len(clustersLabels)>1 else 1 for l in range(nClust)]
        #     normalizeComp = [reprElems[l]._SOD/(len(clustersLabels[l])-1) 
        #                       if len(clustersLabels[l])>1 else 1 for l in range(nClust)]
            
        #     for i,repres in enumerate(reprElems):
        #         F = super(BsasBinarySearch,self)._evaluateF(normalizeComp[i],normalizeCard[i])
        #         newGr = Granule(repres._representativeElem,self._distanceFunction,F,normalizeCard[i],normalizeComp[i])
        #         super(BsasBinarySearch,self)._addSymbol(newGr)
        
        # singleton or universe clusters 
        
        try:
            clustersLabels,reprElems = zip(*filter(lambda x: not(len(x[0])==1 or
                                               len(x[0])/len(Dataset.data)==1), zip(clustersLabels,reprElems))) 
        #No elements to unpack
        except ValueError:
            clustersLabels = []
            reprElems = []
            
        nClust = len(reprElems)
        #Evaluation - Lower is better
        normalizeCard = [1-(len(clustersLabels[l])/len(Dataset.data)) for l in range(nClust)]
        normalizeComp = [reprElems[l]._SOD/(len(clustersLabels[l])-1) for l in range(nClust)]
        
        for i,repres in enumerate(reprElems):
            F = super(BsasBinarySearch,self)._evaluateF(normalizeComp[i],normalizeCard[i])
            newGr = Granule(repres._representativeElem,self._distanceFunction,F,normalizeCard[i],normalizeComp[i])
            super(BsasBinarySearch,self)._addSymbol(newGr)
    