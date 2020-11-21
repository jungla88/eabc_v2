# -*- coding: utf-8 -*-

class Embedder:
    
    def __init__(self,Dissimilarity=None):
        
        self._Dissimilarity = Dissimilarity
        
        self._embeddedIDs = None
        self._embeddedSet = None
        
    
    @property
    def embeddedIDs(self):
        return self._embeddedIDs

    @property
    def embeddedSet(self):
        return self._embeddedSet
    
    def getVector(self):
        raise NotImplementedError
        
    def getSet(self):
        raise NotImplementedError