# -*- coding: utf-8 -*-

from eabc.granulators import Granulator
from eabc.granulators import Granule
from eabc.extras import BSAS
from eabc.extras.BinarySearch import BinarySearch

import numpy as np



class BsasBinarySearch(Granulator):
    
    def __init__(self,DistanceFunction,clusterRepresentative,tStep):
        
        self._distanceFunction = DistanceFunction
        self._representation = clusterRepresentative #An object for evaluate the representative
        
        self.method=BSAS(self._representation,self._distanceFunction,)
        self.tStep = tStep
 
        super(BsasBinarySearch,self).__init__()
        
    def granulate(self,Dataset):
        
        partitions = BinarySearch(Dataset.data,self.method,self.tStep)
        
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
        nClust = len(reprElems)
    
        #Evaluation - Lower is better
        normalizeCard = [1-(len(clustersLabels[l])/len(Dataset.data)) 
                          if len(clustersLabels)>1 else 1 for l in range(nClust)]
        normalizeComp = [reprElems[l]._SOD/(len(clustersLabels[l])-1) 
                          if len(clustersLabels[l])>1 else 1 for l in range(nClust)]
        
        for i,repres in enumerate(reprElems):
            F = super(BsasBinarySearch,self)._evaluateF(normalizeComp[i],normalizeCard[i])
            newGr = Granule(repres._representativeElem,self._distanceFunction,F,normalizeCard[i],normalizeComp[i])
            super(BsasBinarySearch,self)._addSymbol(newGr)
        
        return partitions 
