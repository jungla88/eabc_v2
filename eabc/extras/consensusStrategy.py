#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 16:19:33 2021

@author: luca
"""
from itertools import groupby,combinations
import scipy.spatial.distance as ssdist
import numpy as np

class consensusStrategy:
    
    def __init__(self,problemName,thr=0.1):
        
        self._problemName = problemName
        self._threshold = thr

    #Naive
    def applyConsensus(self,agents,symbols):


        for commonMetric,group in groupby(agents,lambda x: x[1:7]):
            
            g = tuple(group)
            
            pairAgentIterator = combinations(g,2)
            
            for agent1,agent2 in pairAgentIterator:
        
                id1 = agent1.ID
                id2 = agent2.ID
                symbols1 = [sym for sym in symbols if sym.owner == id1 ]
                symbols2 = [sym for sym in symbols if sym.owner == id2 ]
                
                if symbols1 and symbols2:
                    #Grub the core dissimilarity. In same group, the parameters must be the same
                    #TODO: check correct parameters
                    diss = symbols1[0].dissimilarity
                    
                    repr1 = [sym.representative for sym in symbols1]
                    repr2 = [sym.representative for sym in symbols2]
                    
                    M = diss.pdist2(repr1,repr2)
                        
                    #column indices of minimum values on rows
                    minIndices = np.argmin(M,axis = 0)
                    #their distance values
                    minDistances = np.choose(minIndices,M)
                    
                    #find which rows satisfy the threshold
                    symbols_j = np.where(minDistances<= self._threshold)[0]
                    #
                    symbols_i = minIndices[symbols_j]
                    
                    #First column indicate the row of the matrix, the second the column
                    symbolspair_indices = np.vstack((symbols_i,symbols_j)).T
                    
                    for row in symbolspair_indices:
                        self._reward(symbols1[row[0]],symbols2[row[1]])
                    
                
        if self._problemName=="GREC":
            pass
        
        return 0
    
    
    def _reward(self,sym1,sym2):
        
        sym1.quality *= 2 
        sym2.quality *=2
        
        return 0