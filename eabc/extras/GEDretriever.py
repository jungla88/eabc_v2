#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 11:27:11 2021

@author: luca
"""

def getParams(agents,datasetName):

    p = agents[1:7]    
    if datasetName == "GREC":
        p = p + agents[8:13]
    
    return p
    
    