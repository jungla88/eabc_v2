# -*- coding: utf-8 -*-
from littleballoffur import RandomWalkWithRestartSampler
from littleballoffur import RandomWalkSampler

from eabc.data import Graph_nx
import numpy
import networkx as nx

r"""
Graph extractor interface for Graph_nx Data with Little Ball of Fur library
Resolve assumption on connectivity,indexing and length required by LBF lib

Accept a Graph_nx type and return a Graph_nx type 

"""

class extr_strategy:
    def __init__(self, order = 5, seed = None, restart=False):        

        self.sampler = RandomWalkSampler() if restart==False else RandomWalkWithRestartSampler() 

        self._order = order
        
        if seed:
            self._seed = seed;
            numpy.random.seed(seed)
            

    @property
    def order(self):
        return self._order
    @order.setter
    def order(self,val):
        self._order = val
    
    def __call__(self, data, start_node=None):

        
        G = data.x
        #For littleballoffur assumption connectivity
        #Take the connected component where the starting node resides if provided,
        #Otherwise selected a node at ramdom and return the connected componets in which it resides.
        if not nx.is_connected(data.x):
            nodeComponent = start_node if start_node else numpy.random.choice(G.nodes())
            #Get the connected components
            G = G.subgraph(nx.node_connected_component(G,nodeComponent)).copy()

        #For littleballoffur assumption indexing
        #Mapping and relabeling nodes is [0,#node in connected components]
        ForwardMapping = {k:n for n,k in enumerate(G.nodes())}
        ReverseMapping = {v:k for k,v in ForwardMapping.items()}
        G = nx.relabel_nodes(G, ForwardMapping,copy = True)
        
        node = ForwardMapping[start_node] if start_node else start_node

        #For littleballoffur assumption on graph order be less the number of nodes in the subgraph
        self.sampler.number_of_nodes = self._order if self._order<=len(G) else len(G) 
        
        #Extracting on the relabeled subgraph
        subgraph = self.sampler.sample(G,node)
        
        #Relabel nodes as original graph
        subgraph = nx.relabel_nodes(subgraph,ReverseMapping,copy=True)

        subgraphData = Graph_nx()
        subgraphData.x = subgraph
        subgraphData.y = data.y
        
        return subgraphData
        