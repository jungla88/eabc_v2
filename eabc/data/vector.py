#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 11:34:15 2020

@author: luca
"""

from eabc.data import Data
from numpy import ndarray,array

class Vector(Data):
    
    def __init__(self):
        super(Vector,self).__init__()

    @Data.x.setter 
    def x(self, value):
        if isinstance(value, ndarray):
            self._x = value
        #if value is int,float etc...?    
        elif isinstance(value,list) or isinstance(value,tuple):
            self._x = array(value)
        else:
            raise TypeError("Only numpy.ndarray, list or tuple accepted for Vector object data type")

    @Data.y.setter 
    def y(self, value):
        self._y = value

    def __len__(self):
        return len(self._x)