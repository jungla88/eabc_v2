#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

class Representative:
    
    def __init__(self,initReprElem = None):

        self._representativeElem = initReprElem
        self._DistanceMatrix = None
        self._SOD = None
    
    def evaluate(self, cluster):        
        raise NotImplementedError
        
    def update(self,cluster):
        raise NotImplementedError
        
    