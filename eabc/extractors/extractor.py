#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 13:36:01 2020

@author: luca
"""
import numpy
import copy
class Extractor:
    
    def __init__(self, strategy, seed = None):
        
        self.extr_strategy = strategy
        self._seed = seed;
        if self._seed:
            numpy.random.seed(seed)

            
    
    def extract(self,pattern,start_node=None):
        substruct = self.extr_strategy(pattern,start_node)
        return substruct
        
    #It should not be used in embedding since the returning graph set is 
    #randomly selected from different graphs 
    def randomExtractDataset(self, dataset, W, maxOrder=5):       
        
        substruct_Dataset = dataset.fresh_dpcopy()
        
        for _ in range(W):
            idx = numpy.random.randint(0,len(dataset))            
            self.extr_strategy.order = numpy.random.randint(1,maxOrder + 1)
            g = self.extract(dataset[idx])

            #Add the key of the graph related to dataset list index 
            substruct_Dataset.add_keyVal(dataset.to_key(idx),g)
            
        return substruct_Dataset
    
    def decomposeGraphDataset(self,dataset,maxOrder):
        
        substruct_Dataset = dataset.fresh_dpcopy()
        
        for g,i in zip(dataset, dataset.indices):
                
            #TODO: Do I really need to get the graph?            
            visitedNode = set()
            for node in g.x.nodes():

                if node not in visitedNode:
                    
                    o = maxOrder if maxOrder<= len(g.x) else len(g.x)
                    
                    for order in range(1,o+1):                
                        self.extr_strategy.order = order
                        sg = self.extract(g,node)
                        substruct_Dataset.add_keyVal(i,sg)
                        #Update the reached node
                        visitedNode.update(list(sg.x.nodes()))
                                            
        return substruct_Dataset
                        
                
        
            
        
        
        