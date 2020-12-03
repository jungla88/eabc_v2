#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 11:45:22 2020

@author: luca
"""

from numpy import asarray, reshape, linalg, sqrt
from ast import literal_eval as lit_ev


class LETTERdiss:
    
    def __init__(self,vertexWeight = 2):

        self._VertexDissWeights=vertexWeight
        
    def nodeDissimilarity(self,a, b):
        return linalg.norm(a['coords'] - b['coords']) /  self._VertexDissWeights
    #w=sqrt(a['coords'].shape[1])
    
    
    def edgeDissimilarity(self,a, b):
        return 0.0

def parser(g):

    for node in g.nodes():
        list_labels = list(map(lambda x: float(lit_ev(x)), g.nodes[node].values()))
        g.nodes[node]['coords'] = reshape(asarray(list_labels), (1, 2))
        {g.nodes[node].pop(k) for k in ['y', 'x']}