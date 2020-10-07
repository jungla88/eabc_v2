#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 16:12:31 2020

@author: luca
"""

from Extractor import Extractor

class vectorExtractor(Extractor):
    
    def __init__(self,dataset, num_samples):
        
        self.dataset = dataset
        self.num_samples = num_samples
        
        
        
        