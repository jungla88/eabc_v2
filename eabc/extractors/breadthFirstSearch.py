#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 18:14:16 2020

@author: luca
"""
import copy
import networkx as nx

from littleballoffur import BreadthFirstSearchSampler
from littleballoffur import RandomNodeSampler
from eabc.data import Graph_nx
import numpy.random

#Interface for Graph_nx Data
#Accept a Graph_nx type and return a Graph_nx type 
class extr_strategy:
    
    def __init__(self, max_order = 5, seed = None, restart=False):
        self.sampler = BreadthFirstSearchSampler()
        self.max_order = max_order
        
        self._rng = numpy.random.default_rng(seed)

            
        
    def __call__(self, data):
        
        upperBoundOrder = self.max_order if data.x.order() > self.max_order else data.x.order()
        #TODO: else case always exclude highest value from being extracted
        order = self._rng.random.randint(1, upperBoundOrder)  #high value excluded for randint         

        if order == 1:
            self.sampler = RandomNodeSampler(number_of_nodes=1)
        else:
            self.sampler.number_of_nodes = order
        
        subgraph = self.sampler.sample(data.x)        
        sg = copy.deepcopy(data.x)
        
        nRList = []
        for node in sg.nodes():
            if node not in subgraph.nodes():
                nRList.append(node)
        for node in nRList:
            sg.remove_node(node)
            
        subgraphData = Graph_nx()
        subgraphData.x = sg
        subgraphData.y = data.y
        
        return subgraphData
        