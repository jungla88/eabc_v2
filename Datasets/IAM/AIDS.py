#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 10:37:45 2020

@author: luca
"""

from ast import literal_eval as lit_ev
import numpy

class AIDSdiss:
    
    def __init__(self,vertexWeight):
        self._VertexDissWeight=vertexWeight
#        self._EdgeDissWeights=1.0
    
    def nodeDissimilarity(self,a, b):
        alpha=0.1
        beta=0.2
        gamma=0.7 # NON SI SA DOVE LI HANNO TROVATI
        D=0
    
        if(a['charge']!= b['charge']):
            D+=alpha
        if(a['chem']!= b['chem']):
            D+=beta
        
        D+=numpy.linalg.norm(a['coords'] - b['coords'])*gamma /self._VertexDissWeight
        return D 
    
    
    def edgeDissimilarity(self,a, b):
        return 0.0


def parser(g):

    for node in g.nodes():
        g.nodes[node]['charge']=int(lit_ev(g.nodes[node]['charge']))
        g.nodes[node]['chem']=int(lit_ev(g.nodes[node]['chem']))
        g.nodes[node]['symbol']=lit_ev(g.nodes[node]['symbol']).strip()
        real_2Dcoords = [float(lit_ev(g.nodes[node]['x'])), float(lit_ev(g.nodes[node]['y']))]        
        g.nodes[node]['coords'] = numpy.reshape(numpy.asarray(real_2Dcoords), (2,))
        {g.nodes[node].pop(k) for k in ['y', 'x']}
        

def normalize(*argv):
    """ Normalization routine for AIDS.
    Normalization consists in finding a normalization constant by exploiting node attributes across the entire dataset.
    The input sets remain untouched.

    Input:

    Output:
    - aidsw: a normalization factor that will be used in nodeDissimilarity. """

    # Init main structure
    Set_stack = numpy.zeros((1, 2))

    for targetSet in argv:    
        for g in targetSet:
        # extract graph
            for n in g.nodes():
                Set_stack = numpy.vstack((Set_stack, g.nodes[n]['coords']))

    # Find column-wise min and max values
    MIN_SetX, MAX_SetX = Set_stack[:, 0].min(), Set_stack[:, 0].max()
    MIN_SetY, MAX_SetY = Set_stack[:, 1].min(), Set_stack[:, 1].max()

    # Eval normalization factor
    aidsw = numpy.sqrt((MAX_SetX - MIN_SetX)**2 + (MAX_SetY - MIN_SetY)**2)
    return [aidsw,0]
