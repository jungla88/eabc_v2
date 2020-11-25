#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 18:13:23 2019

@author: luca
"""

"""
Custom Graph Edit Distance known as node Best Match First.

The function can be initialized with two arguments:
1) A node dissimilarity function
2) An edge dissimilarity function

BMF() return normalized dissimilarity between two graphs.

Implemented as described in:
Bianchi, Filippo Maria, et al. "Granular computing techniques for classification and semantic characterization of structured data." Cognitive Computation 8.3 (2016), pp. 442-461.

Normalisation has been taken from:
Baldini, Luca et al. "Stochastic information granules extraction for graph embedding and classification." In Proceedings of the 11th International Joint Conference on Computational Intelligence, vol. 1 (2019), pp. 391-402.
"""

from copy import deepcopy
from eabc.dissimilarities import Dissimilarity
import numpy as np
import itertools
from scipy import special

class BMF(Dissimilarity):

    def __init__(self, nodeDiss, edgeDiss):

        """ User Defined Node/Edges dissimilarity """
        self._nodeDiss = nodeDiss
        self._edgeDiss = edgeDiss

        """Default cost Parameters """
        self._nodesParam = {'sub': 1.0, 'del': 1.0, 'ins': 1.0}
        self._edgesParam = {'sub': 1.0, 'del': 1.0, 'ins': 1.0}


    @property
    def nodeInsWeight(self):
        return self._nodesParam['ins']
    @nodeInsWeight.setter
    def nodeInsWeight(self,val):
        if 0<=val<=1:
            self._nodesParam['ins']=val
            
    @property
    def nodeSubWeight(self):
        return self._nodesParam['sub']
    @nodeSubWeight.setter
    def nodeSubWeight(self,val):
        if 0<=val<=1:
            self._nodesParam['sub']=val

    @property
    def nodeDelWeight(self):
        return self._nodesParam['del']
    @nodeDelWeight.setter
    def nodeDelWeight(self,val):
        if 0<=val<=1:
            self._nodesParam['del']=val            

    @property
    def edgeInsWeight(self):
        return self._edgesParam['ins']
    @edgeInsWeight.setter
    def edgeInsWeight(self,val):
        if 0<=val<=1:
            self._edgesParam['ins']=val
            
    @property
    def edgeSubWeight(self):
        return self._edgesParam['sub']
    @edgeSubWeight.setter
    def edgeSubWeight(self,val):
        if 0<=val<=1:
            self._edgesParam['sub']=val

    @property
    def edgeDelWeight(self):
        return self._edgesParam['del']
    @edgeDelWeight.setter
    def edgeDelWeight(self,val):
        if 0<=val<=1:
            self._edgesParam['del']=val
    
    
    def __call__(self, g1, g2):

        """ node Best Match First """

        totVertex_DelCost = 0.0
        totVertex_InsCost = 0.0
        totVertex_SubCost = 0.0

        o1 = g1.order()
        o2 = g2.order()

        hash_table = set()  # Best match are evaluated in a single loop
        assignments = {}

        i = 0

        N1 = sorted(g1.nodes())       # store sorted nodes, so we call sorted()
        N2 = sorted(g2.nodes())       # only twice rather than 'o1 + 1' times
        for g1_n in N1:
        
            if(i >= o2):
                break

            minDiss = float("inf")

            for g2_n in N2:

                if g2_n not in hash_table:
                    tmpDiss = self._nodeDiss(g1.nodes[g1_n], g2.nodes[g2_n])
                    if tmpDiss < minDiss:
                        assigned_id = deepcopy(g2_n)
                        minDiss = tmpDiss
                        assignments[g1_n] = assigned_id

            hash_table.add(assigned_id)

            totVertex_SubCost += minDiss

            i += 1

        if(o1 > o2):
            totVertex_InsCost = abs(o1 - o2)
        else:
            totVertex_DelCost = abs(o2 - o1)

        vertexDiss = self._nodesParam['sub'] * totVertex_SubCost + self._nodesParam['ins'] * totVertex_InsCost + self._nodesParam['del'] * totVertex_DelCost

        """ Edge Induced Matches """

        totEdge_SubCost = 0.0
        totEdge_InsCost = 0.0
        totEdge_DelCost = 0.0
        edgeInsertionCount = 0
        edgeDeletionCount = 0

        edgesIndex1 = 0
        for matchedNodes1 in assignments.items():

            edgesIndex2 = 0
            edge_g1_exist = False
            edge_g2_exist = False

            u_g1 = matchedNodes1[0]
            u_g2 = matchedNodes1[1]

            for matchedNodes2 in assignments.items():

                if matchedNodes1 != matchedNodes2 and edgesIndex2 <= edgesIndex1:

                    v_g1 = matchedNodes2[0]
                    v_g2 = matchedNodes2[1]

                    edge_g1_exist = g1.has_edge(u_g1, v_g1)
                    edge_g2_exist = g2.has_edge(u_g2, v_g2)

                    if edge_g1_exist and edge_g2_exist:
                        totEdge_SubCost += self._edgeDiss(g1.edges[(u_g1, v_g1)], g2.edges[(u_g2, v_g2)])                        
                    elif edge_g1_exist:
                        edgeInsertionCount += 1
                    elif edge_g2_exist:
                        edgeDeletionCount += 1

                edgesIndex2 += 1

            edgesIndex1 += 1

        edgeDiss = self._edgesParam['sub'] * totEdge_SubCost + self._edgesParam['ins'] * edgeInsertionCount + self._edgesParam['del'] * edgeDeletionCount


        #Normalization assume node/edge dissimilarities are normalised [0,1] as well
        normaliseFactor_vertex = max(o1, o2)
        normaliseFactor_edge = 0.5 * (min(o1, o2) * (min(o1, o2) - 1))

        vertexDiss_norm = vertexDiss / normaliseFactor_vertex
        edgeDiss_norm = edgeDiss if normaliseFactor_edge == 0 else edgeDiss / normaliseFactor_edge

        return 0.5 * (vertexDiss_norm + edgeDiss_norm)

    #NON-symmetric dissimilarity matrix or forced to be taking mean value of d_ij and d_ji
    def pdist(self, set1, forceSym = True):

        if forceSym:
            #Return Numpy like condensed array 
            M = np.zeros(shape= (int(special.comb(len(set1),2)),))            
            ij = itertools.combinations(range(len(set1)),r=2)             
  
            for idx,(i,j) in enumerate(ij):
                x_l = self.__call__(set1[i],set1[j])
                x_r = self.__call__(set1[j],set1[i])    
                M[idx] = 0.5*(x_l+x_r) 
#                    next_idx = (j-1)*(len(set1)-1)+(i-1)
        else:
            #Return the dissimilarity matrix 
            #Naive way.
            M = np.zeros(shape= (len(set1),len(set1)))
            for i,g1 in enumerate(set1):
                row = [self.__call__(g1,g2) for g2 in set1]
                M[i]= row;
        
        return M
        