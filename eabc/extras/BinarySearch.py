# -*- coding: utf-8 -*-

from copy import copy

def BinarySearch(dataset, clusteringProcedure,tStep):
    r""" Ensemble of partition with a Recursive Binary search strategy.
    Input:
    - tm: lower bound
    - tM: upper bound
    - tStep: resolution step at which the search must be stopped.
    Output:
    - partition: a dict dict.keys = theta - dict.values = [clustersLabels, representatives]"""
    
    partition = {} 
    #Initialize min
    clusteringProcedure.theta = 0
#    partition[0]= [clusteringProcedure.process(dataset)]
    partition[0]= clusteringProcedure.process(dataset)
    numClustMin = len(partition[0][1])
    
    #Initialize max
    clusteringProcedure.theta = 1
#    partition[1]= [clusteringProcedure.process(dataset)]
    partition[1]= clusteringProcedure.process(dataset)
    numClustMax = len(partition[1][1])
    
    recursiveBSP(0,1,numClustMin,numClustMax,dataset,tStep,partition,clusteringProcedure)
    
    return partition
    
#TODO: Return correct partition only for latest if condition. The clustering procedure
# tries to perform operations even if there equal partitions 
def recursiveBSP(tm,tM,NC1,NC2,data,tStep,partitionDict,clusteringProcedure):
            
    dt = copy(tM - tm)/2
    
    if (NC1!=NC2
        and (dt>= tStep)
        and (tM-(tm+dt)>=tStep)):
        
        newTheta = dt+tm
        clusteringProcedure.theta = newTheta
        res = clusteringProcedure.process(data)
        numClust = copy(len(res[1]))
        
        if numClust!=NC1 and numClust!=NC2:        
            partitionDict[newTheta]=res

            recursiveBSP(tm,tm+dt,NC1,numClust,data,tStep,partitionDict,clusteringProcedure)
            recursiveBSP(tm+dt,tM,numClust,NC2,data,tStep,partitionDict,clusteringProcedure)
        
        
    
    
    