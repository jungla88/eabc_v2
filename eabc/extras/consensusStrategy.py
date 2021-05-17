#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 16:19:33 2021

@author: luca
"""
from itertools import groupby,combinations
import scipy.spatial.distance as ssdist
import numpy as np
from scipy.spatial.distance import squareform,is_valid_y

class consensusStrategy:
    
#    def __init__(self,problemName, GEDretriever, thr=0.1, constantReward = 5):
    def __init__(self,GEDretriever, thr=0.1, constantReward = 5):        
        
#        self._problemName = problemName
        self._threshold = thr
        self._constantReward = constantReward
        self._GEDretr = GEDretriever
        
#     def applyConsensus(self,agents,symbols):

#         for commonMetric,group in groupby(agents,self._GEDretr):
# #        for commonMetric,group in groupby(agents,lambda x: x[1:7]):
            
#             g = tuple(group)
            
#             pairAgentIterator = combinations(g,2)
            
#             for agent1,agent2 in pairAgentIterator:
        
#                 id1 = agent1.ID
#                 id2 = agent2.ID
#                 symbols1 = [sym for sym in symbols if sym.owner == id1 ]
#                 symbols2 = [sym for sym in symbols if sym.owner == id2 ]
                
#                 if symbols1 and symbols2:
#                     #Grub the core dissimilarity. In same group, the parameters must be the same
#                     #TODO: check correct parameters
#                     diss = symbols1[0].dissimilarity
                    
#                     repr1 = [sym.representative for sym in symbols1]
#                     repr2 = [sym.representative for sym in symbols2]
                    
#                     M = diss.pdist2(repr1,repr2)
#                     #this contains two column array defining the pairs of symbol that satisfy the constrain
#                     symbolpairs = np.where(M<= self._threshold)
                    
#                     for i,j in zip(symbolpairs[0],symbolpairs[1]):
#                         self._reward(symbols1[i],symbols2[j])
                        
#         return 0

    def applyConsensus(self,symbols):

        for commonMetric,group in groupby(symbols,self._GEDretr):
            
            symbols_group = tuple(group)
            diss = symbols_group[0].dissimilarity
            if symbols_group:
                            
                repr_elem = [sym.representative for sym in symbols_group]
                
                M = diss.pdist(repr_elem)
                
                if is_valid_y(M):
                    M = squareform(M)
                    
                #this contains two column array defining the pairs of symbol that satisfy the constrain
                symbolpairs = np.where((M<= self._threshold) & (M > 0))
                
                for i,j in zip(symbolpairs[0],symbolpairs[1]):
                    self._reward(symbols_group[i],symbols_group[j])
                        
        return 0    
    
    def _reward(self,sym1,sym2):
            
        sym1.quality += self._constantReward
        sym2.quality += self._constantReward
        
        return 0