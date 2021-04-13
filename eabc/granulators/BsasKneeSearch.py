# -*- coding: utf-8 -*-

from eabc.granulators import Granulator
from eabc.granulators import Granule
from eabc.extras import BSAS
from eabc.extras.BsasThetaSearch import LinearSearch

from kneed import KneeLocator

class BsasKneeSearch(Granulator):
    
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
        
        #Select partition based on persistence
        # gap = 0
        # bestP = None
        # for i in range(len(partitions)-1):                        
        #     theta = sorted(list(partitions.keys()))[i]
        #     thetaNext = sorted(list(partitions.keys()))[i+1]            
        #     gapTemp = thetaNext - theta            
        #     if gapTemp > gap:
        #         bestP = i
        #theta = sorted(list(partitions.keys()))[bestP]
        
        x = sorted([t for t in partitions.keys()])
        y = sorted([len(cluster[1]) for cluster in partitions.values()],reverse = True)
        print(x,y)
        kl = KneeLocator(x,y,curve='convex',direction = 'decreasing',S = 2)
        theta= kl.knee
        if kl.knee:
            theta = kl.knee
        else:
            #No symbol produced if knee is not found
            return
            
        
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
        clustersLabels,reprElems = super(BsasKneeSearch,self)._removeSingularity(clustersLabels,reprElems,Dataset)
            
        nClust = len(reprElems)
        #Evaluation - Lower is better
        normalizeCard = [1-(len(clustersLabels[l])/len(Dataset.data)) for l in range(nClust)]
        normalizeComp = [reprElems[l]._SOD/(len(clustersLabels[l])-1) for l in range(nClust)]

        for i,repres in enumerate(reprElems):
            
            F = super(BsasKneeSearch,self)._evaluateF(normalizeComp[i],normalizeCard[i])
            newGr = Granule(repres._representativeElem,self._distanceFunction,F,normalizeCard[i],normalizeComp[i])
            super(BsasKneeSearch,self)._addSymbol(newGr)