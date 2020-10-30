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
        
    def randomExtractDataset(self, dataset, W):
        
        substruct_set, idx = [None] * W , [None] * W
        
        #TODO: deepcopy necessary. Copy memo and avoid data and indices?
        substruct_Dataset = copy.deepcopy(dataset)
        del substruct_Dataset.data
        del substruct_Dataset.indices
        
        for i in range(W):
            idx = numpy.random.randint(0,len(dataset))
            substruct_Dataset.add_keyVal(idx, self.extract(dataset[idx]))            
            #substruct_set.data(self.extract(dataset[idx]), idx )
            
        return substruct_Dataset
            
        
        
        