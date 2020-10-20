#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 13:36:01 2020

@author: luca
"""
import numpy
import copy
class Extractor:
    
    def __init__(self, strategy, seed = 0):
        
        self.extr_strategy = strategy
        self.seed = seed;
    
    def extract(self,pattern):
        substruct = self.extr_strategy(pattern)
        return substruct
        
    def randomExtractDataset(self, dataset, W):
        
        substruct_set, idx = [None] * W , [None] * W
        
        substruct_Dataset = copy.copy(dataset)
        # TODO: can not set data 
        # substruct_Dataset.data =[]
        # substruct_Dataset.indices = []
        
        for i in range(W):
            idx = numpy.random.randint(0,len(dataset))            
            substruct_set.data(self.extract(dataset[idx]), idx )
            
        return substruct_Dataset
            
        
        
        