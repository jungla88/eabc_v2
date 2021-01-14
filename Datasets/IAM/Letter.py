#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 11:45:22 2020

@author: luca
"""

from ast import literal_eval as lit_ev
import numpy
import copy

class LETTERdiss:
    
    def __init__(self,vertexWeight = numpy.sqrt(2)):

        self._VertexDissWeights=vertexWeight
        
    def nodeDissimilarity(self,a, b):
        return numpy.linalg.norm(a['coords'] - b['coords']) /  self._VertexDissWeights    
    
    def edgeDissimilarity(self,a, b):
        return 0.0

def parser(g):

    for node in g.nodes():
        list_labels = list(map(lambda x: float(lit_ev(x)), g.nodes[node].values()))
        g.nodes[node]['coords'] = numpy.reshape(numpy.asarray(list_labels), (1, 2))
        {g.nodes[node].pop(k) for k in ['y', 'x']}
        

def normalize(strAttribute,*argv):
    """ Normalization routine for Letter 1, 2 and 3.
    Normalization includes scaling [x, y] coordinates according to the maximum value found in the overall dataset.
    The input sets will be modified in-place.
    """

    # Find max value in training set by stacking [x, y] pairs, then flattening and taking the max
    

    MAXval = []    
    for targetSet in argv:    
        MAX_Set = numpy.zeros((1, 2))
        for g in targetSet:
        # extract graph
#            thisGraph = targetSet[k]
            # append node labels
            for n in g.nodes():
                MAX_Set = numpy.vstack((MAX_Set, g.nodes[n][strAttribute]))
        MAX_Set = copy.deepcopy(MAX_Set.ravel().max())
        MAXval.append(MAX_Set)

    # Find overall max value
    MAX = max(MAXval)
    
    
    for targetSet in argv:
        for g in targetSet:
            for n in g.nodes():
                g.nodes[n][strAttribute] = g.nodes[n][strAttribute]/MAX
                
    print("Letter Normalization value = {}".format(MAX))
                
    return [None,None]