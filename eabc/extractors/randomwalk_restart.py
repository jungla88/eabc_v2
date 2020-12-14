# -*- coding: utf-8 -*-

from littleballoffur import RandomWalkWithRestartSampler
from littleballoffur import RandomWalkSampler
from eabc.data import Graph_nx
import numpy.random

#Interface for Graph_nx Data
#Accept a Graph_nx type and return a Graph_nx type 
class extr_strategy:
    
    def __init__(self, max_order = 5, seed = None, restart=False):
        self.sampler = RandomWalkSampler() if restart==False else RandomWalkWithRestartSampler() 
        self.max_order = max_order
        
        if seed:
            self._seed = seed;
            numpy.random.seed(seed)
            
        
    def __call__(self, data, start_node=None):
        
        upperBoundOrder = self.max_order if data.x.order() > self.max_order else data.x.order()
        #TODO: else case always exclude highest value from being extracted
        order = numpy.random.randint(1, upperBoundOrder)  #high value excluded for randint         
        self.sampler.number_of_nodes = order
        
        subgraph = self.sampler.sample(data.x,start_node)
        subgraphData = Graph_nx()
        subgraphData.x = subgraph
        subgraphData.y = data.y
        
        return subgraphData
        