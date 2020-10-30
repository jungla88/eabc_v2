# -*- coding: utf-8 -*-

from eabc.data import Vector
import numpy.ma
#-data must be a numpy.array
#-mask must be either boolean or 0,1 vector. Other values are valid entries but mask will be meaningless
class extr_strategy:
    
    def __init__(self, mask=None, seed = None):
        
        self.mask = mask
        
    def __call__(self, data):
        
        if self.mask:
            x = numpy.ma.array(data.x,mask = self.mask)
            sel_feat = x.data[~x.mask]            
            
            substruct = Vector()
            substruct.x = sel_feat 
            substruct.y = data.y
        
        return substruct
        
        