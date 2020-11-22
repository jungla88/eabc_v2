#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from eabc.data import Data

class Representative:
    
    def __init__(self,initReprElem = None):

        self._representativeElem = initReprElem
        self._DistanceMatrix = None
        self._SOD = None
    
    def evaluate(self, cluster):        
        raise NotImplementedError
        
    def update(self,cluster):
        raise NotImplementedError
        
    def __isEabcData(self,data):
        return True if isinstance(data,Data) else False
            
        
        
    