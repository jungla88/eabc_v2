# -*- coding: utf-8 -*-

from eabc.embeddings import Embedder
import numpy
import copy

class SymbolicHistogram(Embedder):
    
    def __init__(self,Dissimilarity = None, isSymbolDiss = False):
        
        self._isSymbolDiss = isSymbolDiss
        
        super().__init__(Dissimilarity)        

    @property
    def isSymbolDiss(self):
        return self._isSymbolDiss
    
    def getVector(self, elemDecomposition, alphabetSet):

        self.__sanityCheck()
        
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
        
        IDs= numpy.array(datasetDecomposed.indices)
        uniqueIDs = numpy.unique(IDs)
        
        embeddedSet, embeddedIDs =[],[]# []*len(uniqueIDs),[]*len(uniqueIDs)
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
            
            
            
            
        
        
        
        