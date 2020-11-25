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

            
    
    def extract(self,pattern):
        substruct = self.extr_strategy(pattern)
        return substruct
        
    #It should not be used in embedding since the returning graph set is 
    #randomly selected from different graphs 
    def randomExtractDataset(self, dataset, W):       
        
        substruct_Dataset = dataset.fresh_dpcopy()
        
        for i in range(W):
            idx = numpy.random.randint(0,len(dataset))
            g = self.extract(dataset[idx])
            #Add the key of the graph related to dataset list index 
            substruct_Dataset.add_keyVal(dataset.to_key(idx),g)
            
        return substruct_Dataset
            
        
        
        