# -*- coding: utf-8 -*-

from eabc.embeddings import Embedder
import numpy
import copy
from joblib import Parallel, delayed

##Helper functions for Joblib
def PgetVector(elemDecomposition, alphabetSet,Dissimilarity):

    histogram = numpy.zeros((len(alphabetSet,)))
    for i,sym in enumerate(alphabetSet): 
        
        Diss = Dissimilarity if Dissimilarity else sym.dissimilarity    
        j=0
        for x in elemDecomposition:
                if Diss(sym.representative,x)<=sym.matchThr:
                    j+=1
                    
        e = copy.deepcopy(j)
        histogram[i] = e
    
    return histogram    
        
        
def JobLibHelper(x,IDs,datasetDecomposed,alphabetSet,Dissimilarity):
  
    boolIDs = IDs == x
    substructIndices = [idx for idx,ids in enumerate(boolIDs) if ids==True]        
    substrDecomposition = datasetDecomposed[substructIndices]
    h = PgetVector(substrDecomposition.data,alphabetSet,Dissimilarity)            
    return (h,x)
##


class SymbolicHistogram(Embedder):
    
    def __init__(self,Dissimilarity = None, isSymbolDiss = False,isParallel=True):
        
        self._isSymbolDiss = isSymbolDiss
        self._isParallel = isParallel
        super().__init__(Dissimilarity)        

    @property
    def isSymbolDiss(self):
        return self._isSymbolDiss
    
    def getVector(self, elemDecomposition, alphabetSet):
       
        histogram = numpy.zeros((len(alphabetSet,)))
        for i,sym in enumerate(alphabetSet): 
            Dissimilarity = self._Dissimilarity if not self._isSymbolDiss else sym.dissimilarity
            
            j=0
            for x in elemDecomposition:
                if Dissimilarity(sym.representative,x)<=sym.matchThr:
                    j+=1
                    
            e = copy.deepcopy(j)
            histogram[i] = e
        
        return histogram
            
    def getSet(self,datasetDecomposed,alphabetSet):
        
        self.__sanityCheck()
        
        IDs= numpy.array(datasetDecomposed.indices)
        #IDs are sorted in ascending order 
        uniqueIDs = numpy.unique(IDs)

        embeddedSet, embeddedIDs =[],[]        
        if self._isParallel:
            Dissimilarity = self._Dissimilarity if self._isSymbolDiss==False else None
            #Might be not necessary return the ids
            output=Parallel(n_jobs=-1)(delayed(JobLibHelper)(x,IDs,datasetDecomposed,alphabetSet,Dissimilarity) for x in uniqueIDs)
            for i in range(len(output)):
                # embeddedSet[i]=output[i][0]
                # embeddedIDs[i]=output[i][1]         
                embeddedSet.append(output[i][0])
                embeddedIDs.append(output[i][1])
            #For debug    
            assert(len(embeddedSet)==len(embeddedIDs))

        else:
            
            for x in uniqueIDs:
                
                boolIDs = IDs == x
                substructIndices = [idx for idx,ids in enumerate(boolIDs) if ids==True]
                
                substrDecomposition = datasetDecomposed[substructIndices]
                embeddedSet.append(self.getVector(substrDecomposition.data,alphabetSet))
                embeddedIDs.append(x)
                
        self._embeddedIDs = embeddedIDs
        self._embeddedSet = embeddedSet
        
    def __sanityCheck(self):
        
        if not self._isSymbolDiss and not self._Dissimilarity:
            raise ValueError("Dissimilarity is not available")
        elif self._isSymbolDiss and self._Dissimilarity:
            print("Warning: Using symbol dissimilarity while global dissimilarity is available")

        
        
        
        