# -*- coding: utf-8 -*-

from eabc.representatives import Representative
import scipy.spatial.distance as scpDist
import numpy as np
import eabc.dissimilarities

from eabc.representatives import Representative
import scipy.spatial.distance as scpDist
import numpy as np
import eabc.dissimilarities

class Medoid(Representative):

    #Initialize on as class attributes.
    #These can be change BEFORE create the object e.g:
    #<<<Medoid._isApprox = False
    #<<<medoid= Medoid()
    #<<<medoid._isApprox
    #False
    _PoolSize = 20
    _isApprox = True
    
    #Seeding the rng always with the same seed prevents inconsistency properties between equal medoids due to randomness in update
    _rng = np.random.default_rng(0)
    
    def __init__(self,initRepr = None):#,approximateUpdate = True, PoolSize = 20):

        super(Medoid,self).__init__(initReprElem=initRepr)   
        
        #Debug
        self._cluster = [initRepr] if initRepr else []
        self._ids = None
        self._minSodIdx = None
        
            
    def evaluate(self,cluster,Dissimilarity):
        
        #We have a distance matrix for the cluster then we update the minSOD                    
        if self._DistanceMatrix is not None:
            self._update(cluster,Dissimilarity)
        
        #First time we see this cluster. It could be:
        # - Just initialize with a minSOD, no distance matrix available nor minSOD
        # - Populated with data but not evaluated yet, no distance matrix available nor minSDO
        else:
            self._evaluate(cluster,Dissimilarity)
            
        return 0
            

    def __fullEvalMat(self,cluster, Dissimilarity):

            # Evaluate all distances even if we have a diss matrix?
            M = Dissimilarity.pdist(cluster)
            if scpDist.is_valid_y(M):
                M = scpDist.squareform(M)            
            #debug
            assert(M.shape[0] == M.shape[1])
            
            return M
            
    def _evaluate(self,cluster,Dissimilarity):
        
        if len(cluster) <= self._PoolSize or self._isApprox==False:
            self._cluster = cluster            
            M = self.__fullEvalMat(cluster, Dissimilarity)

        
        else:
            #Workaround for offline evaluation
            #Extract poolSize pattern from cluster and evaluate the minSOD
            ids = self._rng.choice(len(cluster),self._PoolSize,replace = False)
            self._cluster = [cluster[i] for i in ids]
            M = self.__fullEvalMat(self._cluster,Dissimilarity)
            
        SoD = np.sum(M, axis = 0)

        self._minSodIdx = np.argmin(SoD)        
        self._representativeElem = self._cluster[self._minSodIdx]
        self._SOD = SoD[self._minSodIdx]
        self._DistanceMatrix = M            
            
        
        
    def _update(self, cluster, Dissimilarity):
                
        M = self._DistanceMatrix

        #generate updated cluster ids
        _ids = np.asarray(range(len(cluster)))
        
        #If cluster size is smaller than poolSize or approximation is not required
        #we keep evaluating all distances          
        if len(cluster) <= self._PoolSize or self._isApprox==False:
            M = self.__fullEvalMat(cluster, Dissimilarity)
            self._cluster = cluster                                
        
        else:            
            #randomly choose two pattern from the pool
            id_p1, id_p2 = self._rng.choice(self._PoolSize,2,replace=False)
            d1 = self._DistanceMatrix[self._minSodIdx,id_p1]
            d2 = self._DistanceMatrix[self._minSodIdx,id_p2]                             
            #Farthest pattern from medoid will be changed 
            id_p = id_p1 if d1>=d2 else id_p2
            
            #set of cluster ids without the pattern id to remove
            #diffIds = np.setdiff1d(self._ids,id_p)
            diffIds = np.setdiff1d(range(self._PoolSize),id_p)                
            #New Id to be introduced
            #FIXME: we are assuming an update with a single pattern introduced
            #newP might be unuseful since the new pattern is always the last in cluster arg
            #What if we want update with more data?
            newP = np.setdiff1d(_ids,self._ids)[0]
            
            #Change old pattern in the container with the newest 
            self._cluster[id_p] = cluster[newP]
            
            v_h = np.zeros((self._PoolSize,))
            v_v = np.zeros((self._PoolSize,))
            
            #FIXME: not needed for dissimilarities with symmetric property
            for u in diffIds:
                v_h[u]=Dissimilarity(cluster[newP], self._cluster[u])
                v_v[u]=Dissimilarity(self._cluster[u],cluster[newP])

            v = 0.5*(np.asarray(v_h) + np.asarray(v_v))
            ###
            M[id_p,:] = v
            M[:,id_p] = v
                
        SoD = np.sum(M, axis = 0)

        self._minSodIdx = np.argmin(SoD)        
        self._representativeElem = self._cluster[self._minSodIdx]
        self._SOD = SoD[self._minSodIdx]
        self._ids = _ids
        self._DistanceMatrix = M
        
        return 0