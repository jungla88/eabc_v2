#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 11:27:11 2021

@author: luca
"""

def getAgentGEDParams(agent,datasetName):

    p = agent[1:7]    
    if datasetName == "GREC":
        p = p + agent[8:13]
    
    return p
    

def getParams(symbol,datasetName):
    
    diss = symbol.dissimilarity

    wNSub = diss.nodeSubWeight
    wNIns = diss.nodeInsWeight
    wNDel = diss.nodeDelWeight
    wESub = diss.edgeSubWeight
    wEIns = diss.edgeInsWeight
    wEDel = diss.edgeDelWeight
    
    p = [wNSub,wNIns,wNDel,wESub,wEIns,wEDel]
    
    if datasetName == "GREC":
        
        
        vParam1  = diss.nodeEdgeObj.v1 
        eParam1  = diss.nodeEdgeObj.e1
        eParam2  = diss.nodeEdgeObj.e2
        eParam3  = diss.nodeEdgeObj.e3 
        eParam4  = diss.nodeEdgeObj.e4
    
        addParams = [vParam1,eParam1,eParam2,eParam3,eParam4]
        
        p = p + addParams
    
    return p