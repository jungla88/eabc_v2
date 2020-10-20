#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Base class for Data.
Abstract the concept of pattern.

-self.x: the data structure for the examples
-self.y: the data structure for the attributes

@author: luca
"""

class Data:
    
    def __init__(self):
        self._x = None
        self._y = None
               
    @property
    def x(self):
        return self._x
    @x.setter
    def x(self, value):
        raise NotImplementedError()
    @x.deleter
    def x(self):
        del self._x
        
        
    @property
    def y(self):
        return self._y
    @y.setter
    def y(self, value):
        raise NotImplementedError()
    @y.deleter
    def y(self):
        del self._y
        
    