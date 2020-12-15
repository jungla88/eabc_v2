#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 17:43:29 2020

@author: luca
"""


from ast import literal_eval as lit_ev
import numpy


class GRECdiss:
    
    def __init__(self,vertexWeight=1,edgeWeight=1):

        self._VertexDissWeight=vertexWeight
        self._EdgeDissWeight=edgeWeight
        
        self._vParam1=0.9353
        self._eParam1=0.1423
        self._eParam2=0.125
        self._eParam3=0.323
        self._eParam4=0.1069
    
    
    @property
    def v1(self):
        return self._vParam1
    @v1.setter
    def v1(self,v):
        if 0 <= v <= 1:
            self._vParam1 = v
        else:
            return ValueError

    @property
    def e1(self):
        return self._eParam1
    @e1.setter
    def e1(self,v):
        if 0 <= v <= 1:
            self._eParam1 = v
        else:
            return ValueError

    @property
    def e2(self):
        return self._eParam2
    @e2.setter
    def e2(self,v):
        if 0 <= v <= 1:
            self._eParam2 = v
        else:
            return ValueError
        
        
    @property
    def e3(self):
        return self._eParam3
    @e3.setter
    def e3(self,v):
        if 0 <= v <= 1:
            self._eParam3 = v
        else:
            return ValueError

    @property
    def e4(self):
        return self._eParam4
    @e4.setter
    def e4(self,v):
        if 0 <= v <= 1:
            self._eParam4 = v
        else:
            return ValueError

    def nodeDissimilarity(self,a, b):
        D=0
        if(a['type']!=b['type']):
            D+=self._vParam1
        D+=(1-self._vParam1)*numpy.linalg.norm(a['coords'] - b['coords']) /  self._VertexDissWeight

        return D
    
    
    def edgeDissimilarity(self,a, b):
        D=0
        if(a['frequency']==b['frequency']):
            if(a['frequency']==1):
                if(a['type0']=="line" and b['type0']=="line"):
                    D=self._eParam1*(numpy.abs( a['angle0'] - b['angle0']) +1.575 )/3.15
                elif (a['type0']=="arc" and b['type0']=="arc"):
                    D=self._eParam2*numpy.abs(a['angle0']-b['angle0'])/self._EdgeDissWeight
                else:
                    D=self._eParam3
            else:
                if(a['type0']=="line" and b['type0']=="line"):
                    D=self._eParam1*(numpy.abs(a['angle0'] - b['angle0']) + 1.575)/(2*3.15) + self._eParam2*abs(a['angle1'] - b['angle1'])/(2*self._EdgeDissWeight)
                elif (a['type0']=="arc" and b['type0']=="arc"):
                    D=self._eParam1*(numpy.abs(a['angle1'] - b['angle1']) + 1.575)/(2*3.15) + self._eParam2*abs(a['angle0'] - b['angle0'])/(2*self._EdgeDissWeight)
                else:
                    D=self._eParam3    
        else:
            D=self._eParam4
        
        return D
                    
        
def parser(g):
    # GREC has particular attributes on edges which are not common to all graphs in dataset.
    # In order to manage this issue, when an attribute is not found, it will be added to the graph 
    # with empty strings or 0's depending on its type. 
    # Since NetworkX edges/nodes are dict structure a try/except workaround is employed, then when
    # missing key (KeyError) error occurs, the parser manages the exception adding the key to the edge labels 
    
    # Vertex Attributes:
    # -type(string)
    # -coords (x,y real values)
    # Edge attributes:
    # -frequency (real)
    # -type0,1 (strings)
    # -angle0,1 (real)

    for node in g.nodes():
        g.nodes[node]['type']=g.nodes[node]['type']
        real_2Dcoords = [float(lit_ev(g.nodes[node]['x'])), float(lit_ev(g.nodes[node]['y']))]        
        g.nodes[node]['coords'] = numpy.reshape(numpy.asarray(real_2Dcoords), (2,))
        {g.nodes[node].pop(k) for k in ['y', 'x']}
        
    for edge in g.edges():
        u=edge[0]
        v=edge[1]
        try:
            g.edges[u,v]['angle0']=float(lit_ev(g.edges[u,v]['angle0']))
        except KeyError:
            g.edges[u,v]['angle0']=0
        try:
            g.edges[u,v]['angle1']=float(lit_ev(g.edges[u,v]['angle1']))
        except KeyError:
            g.edges[u,v]['angle1']=0
        try:
            g.edges[u,v]['type0']=g.edges[u,v]['type0']
        except KeyError:
            g.edges[u,v]['type0']=""
        try:
            g.edges[u,v]['type1']=g.edges[u,v]['type1']
        except KeyError:
            g.edges[u,v]['type1']=""                    
            
        g.edges[u,v]['frequency']=float(lit_ev(g.edges[u,v]['frequency']))
        

def normalize(*argv):
    """ Normalization routine for GREC.
    Normalization consists in finding normalization constants by exploiting node/edge attributes across the entire dataset.
    The input sets remain untouched.

    Output:
    - vertexW: a normalization factor that will be used in nodeDissimilarity
    - edgeW: a normalization factor that will be used in edgeDissimilarity. """

    Set_stackCoords = numpy.zeros((1, 2))
    Set_stackAngles = numpy.zeros((1,))

#    sets = [trSet, vsSet, tsSet]

#    i = 0
#    for thisSet in sets:
    for targetSet in argv:
        for thisGraph in targetSet:
        # for k in sorted(thisSet.keys()):
        #     # extract graph
        #     thisGraph = thisSet[k][0]
            # Parsing node
            for n in thisGraph.nodes():
                Set_stackCoords = numpy.vstack((Set_stackCoords, thisGraph.nodes[n]['coords']))
            # Parsing edges
            for e in thisGraph.edges():
                if(thisGraph.edges[e]['frequency'] == 1):
                    if(thisGraph.edges[e]['type0'] == "arc"):
                        Set_stackAngles = numpy.vstack((Set_stackAngles, thisGraph.edges[e]['angle0']))
                else:
                    if(thisGraph.edges[e]['type0'] == "arc"):
                        Set_stackAngles = numpy.vstack((Set_stackAngles, thisGraph.edges[e]['angle0']))
                    else:
                        Set_stackAngles = numpy.vstack((Set_stackAngles, thisGraph.edges[e]['angle1']))
#        i += 1

    MINx, MAXx = Set_stackCoords[:, 0].min(), Set_stackCoords[:, 0].max()
    MINy, MAXy = Set_stackCoords[:, 1].min(), Set_stackCoords[:, 1].max()

    MINa, MAXa = Set_stackAngles.min(), Set_stackAngles.max()

    vertexW = numpy.sqrt((MAXx - MINx)**2 + (MAXy - MINy)**2)
    edgeW = numpy.abs(MAXa - MINa)

    return [vertexW, edgeW]
