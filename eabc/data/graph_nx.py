# -*- coding: utf-8 -*-

from eabc.data import Data
import networkx as nx

class Graph_nx(Data):
    
    def __init__(self):
        super(Graph_nx,self).__init__()

    @Data.x.setter 
    def x(self, value):
        if isinstance(value, nx.classes.graph.Graph):
            self._x = value
        else:
            raise TypeError("Networkx undirected graph for graph_nx object data type")

    @Data.y.setter 
    def y(self, value):
        self._y = value

    def __len__(self):
        return len(self._x)
    
    def __copy__(self):
        class_ = self.__class__
        newDataGraph_nx = class_.__new__(class_)
        newDataGraph_nx.__dict__.update(self.__dict__)
        newDataGraph_nx.x = self.x.copy()
        
        return newDataGraph_nx