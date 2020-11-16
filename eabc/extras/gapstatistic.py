#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.spatial.distance import pdist

class GapStatistic:
    
    def __init__(self,
                 clusteringMethod,
                 distanceFunction,
                 nFeatures= None,
                 precomputed = False,
                 minVal=None,
                 maxVal = None,
                 n_refs = 10,
                 kMax = 10):
        
        #check precompute 
        if precomputed:
            assert(nFeatures)
            assert(minVal.all())
            assert(maxVal.all())
           
        assert(clusteringMethod)
        
        self._n_refs = n_refs
        self.clusteringMethod = clusteringMethod
        self.distanceFunction = distanceFunction 
        self._kMax = kMax
        self.__precomputed = precomputed
        self._minValCols=minVal
        self._maxValCols=maxVal
        self.__numFeatures = nFeatures

        self._clustSolutions = None
            
        #data assumed be N pattern by F num_features matrix
        # self.pre_data = data
        # self.__numObservations = X.shape[0]
        # self.__numFeatures = features

        #Setup references distribution
        self._referenceSets = None
        #else
        #setup PCA
        #...
        
        self._logWdata=None
        self._ExpectedlogW_nrefs=None
        
    @property
    def solutions(self):
        return self._clustSolutions
    
    @property
    def referenceSets(self):
        return self._referenceSets
    
    @property
    def logWdata(self):
        return self._logWdata
    
    @property
    def E_logWrefs(self):
        return self._ExpectedlogW_nrefs
    
    @property
    def kMax(self):
        return self._kMax
    @kMax.setter
    def kMax(self,val):
        if val > 0:
            self._kMax = val 
            
    @property
    def n_refs(self):
        return self._n_refs
    @n_refs.setter
    def n_refs(self,val):
        if val>0:
            self._n_refs = val
            
    @property
    def minVal(self):
        return self._minValCols
    
    @property
    def maxVal(self):
        return self._maxValCols
    
    def evaluateGap(self,data):

        if not self.__precomputed:
            self.__setupUniformRefs(data)
            self.__numFeatures = data.shape[1]
        
        numObservations= data.shape[0] 
        
        self._clustSolutions = []
        
        #Data init
        logW_ref = np.zeros((self._n_refs,self._kMax))
        logW_data = np.zeros((1,self._kMax))       
        self._referenceSets = [np.zeros((numObservations,self.__numFeatures)) for _ in range(self._n_refs)]
        #Reference loops
        for i in range(self._n_refs):
            self._referenceSets[i] = self.__generateReferences(numObservations)
            
            for k in range(self._kMax):
                labels = self.clusteringMethod(self._referenceSets[i],k+1, precomputed = False)                
                clusters = [self._referenceSets[i][labels==l] for l in range(k+1)]
                clusters_sumofdist = [sum(pdist(cluster,metric = self.distanceFunction)/len(cluster)) 
                                      for cluster in clusters]                
                logW_ref[i][k] = np.log(sum(clusters_sumofdist))                
         
        
        self._ExpectedlogW_nrefs = np.mean(logW_ref, axis = 0, keepdims = True)
        
        #Data loop
        for k in range(self._kMax):
            labels = self.clusteringMethod(data, k+1, precomputed = self.__precomputed)
            self._clustSolutions.append(labels)
            if self.__precomputed:
                clusters_distMat = [data[l==labels][:,l==labels] for l in range(k+1)]
                sumofdist = [0.5*np.sum(cluster,axis=(0,1))/len(cluster) for cluster in clusters_distMat]
            else:
                clusters = [data[labels==l] for l in range(k+1)]
                sumofdist = [sum(pdist(cluster,metric = self.distanceFunction)/(len(cluster)))
                             for cluster in clusters]
                
            logW_data[0][k] = np.log(sum(sumofdist))

        self._logWdata = logW_data
        
        gap_k = self._ExpectedlogW_nrefs - self._logWdata
        
        return gap_k
                
    def __generateReferences(self,numObservations):
        
        return np.random.uniform(low= self._minValCols,high=self._maxValCols,size=(numObservations,self.__numFeatures))

    
    def __setupUniformRefs(self,data):
        
        self._minValCols = np.min(data, axis = 0, keepdims = True)
        self._maxValCols = np.max(data, axis = 0, keepdims = True)
        
       # self._referenceSets = [np.zeros((data.shape[0],data.shape[1])) for _ in range(self.n_refs)]