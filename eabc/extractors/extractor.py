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
        
    #FIXME: It should not be used in embedding. 
    #substruct ID may not be the same of the structure from which it is extracted
    
    def randomExtractDataset(self, dataset, W):       
        substruct_set, idx = [None] * W , [None] * W
        
        #TODO: deepcopy necessary. Copy memo and avoid data and indices?
        # substruct_Dataset = copy.deepcopy(dataset)
        # del substruct_Dataset.data
        # del substruct_Dataset.indices
        
        substruct_Dataset = dataset.fresh_dpcopy()
        
        for i in range(W):
            idx = numpy.random.randint(0,len(dataset))
            #debug
            g = self.extract(dataset[idx])
            if g:
                substruct_Dataset.add_keyVal(idx, g)
            #--
            #TODO:verificare add_keyVal(dataset.tokey(idx),self.extract...)

            #substruct_Dataset.add_keyVal(idx, self.extract(dataset[idx]))            
            
        return substruct_Dataset
            
        
        
        