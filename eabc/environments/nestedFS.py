#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 11:44:14 2020

@author: luca
"""

from eabc.representatives import Medoid
from eabc.dissimilarities import BMF
from eabc.granulators import BsasBinarySearch

import numpy as np
from deap import tools
import random

class eabc_Nested:
    
    def __init__(self,DissimilarityClass,problemName,DissNormFactors = None):
        
        self._problemName = problemName
        self._dissimilarityClass = DissimilarityClass
        
        if self._problemName == 'GREC':
            assert(all(DissNormFactors))
            self._VertexDissWeight=DissNormFactors[0]
            self._EdgeDissWeight=DissNormFactors[1]
        if self._problemName == 'AIDS':
            assert(DissNormFactors[0])
            self._VertexDissWeight = DissNormFactors[0]
        
    @property
    def problemName(self):
        return self._problemName
    
    def customXover(self,ind1,ind2,indpb):
        """
        Parameters
        ----------
        ind1 : Deap invidual parent 1
        ind2 : Deap invidual parent 2
        indpb : float
            Defined the probability that a semantically equal slices of genetic code are recombined 

        Returns
        -------
        ind1 : Deap invidual
        ind2 : Deap invidual
            Recombined Individuals
        
        Semantically equal slice are recombined together. If the slice is a single attribute cxUniform is used,
        CxTwoPoint deap crossover is used in the other case. Each slice has a given indpb probability to be recombined.

        """
        #Q
        g_q1,g_q2 = tools.cxUniform([ind1[0]], [ind2[0]], indpb = indpb)
        #GED
        g_ged1,g_ged2 = tools.cxTwoPoint(ind1[1:7], ind2[1:7])
        #Tau
        g_tau1,g_tau2 = tools.cxUniform([ind1[7]], [ind2[7]], indpb = indpb)
    
        #Set if crossovered
        ind1[0]=g_q1[0]
        ind2[0]=g_q2[0]
        #two point crossover indivividuals are always modified. We edit this slice of genetic code only if condition is valid
        if random.random()<indpb:
            for i,(g1,g2) in enumerate(zip(g_ged1,g_ged2),start = 1):
                ind1[i]=g1
            for i in range(1,7):
                ind2[i]=g2
        #
        ind1[7]=g_tau1[0]
        ind2[7]=g_tau2[0]
    
        if self._problemName == 'GREC':
            
            g_add1, g_add2 =  tools.cxTwoPoint(ind1[8:13], ind2[8:13])
            #Same for ged attributes
            if random.random()<indpb:
                for i,(g1,g2) in enumerate(zip(g_add1,g_add2),start = 8):
                    ind1[i]=g1
                for i in range(1,7):
                    ind2[i]=g2
            
        return ind1,ind2
    
    def customMutation(self,ind,sigma,indpb):
        
        mu = [ind[i] for i in range(len(ind))]
        return tools.mutGaussian(ind, mu, sigma, indpb)
        
    def fitness(self,args):    
        
        individual,granulationBucket = args
        Q= individual[0]
        wNSub= individual[1]
        wNIns= individual[2]
        wNDel= individual[3]
        wESub= individual[4]
        wEIns= individual[5]
        wEDel= individual[6]
        tau = individual[7]

        if self._problemName == 'GREC':
                    
            vParam1=individual[8]
            eParam1=individual[9]
            eParam2=individual[10]
            eParam3=individual[11]
            eParam4=individual[12]

            diss = self._dissimilarityClass(vertexWeight= self._VertexDissWeight,edgeWeight = self._EdgeDissWeight)
        
            diss.v1 = vParam1
            diss.e1 = eParam1
            diss.e2 = eParam2
            diss.e3 = eParam3
            diss.e4 = eParam4
        
        elif self._problemName == 'AIDS':
            diss = self._dissimilarityClass(vertexWeight= self._VertexDissWeight)
        else:
            diss = self._dissimilarityClass()
        
        Repr=Medoid
    
        graphDist=BMF(diss.nodeDissimilarity,diss.edgeDissimilarity)
        graphDist.nodeSubWeight=wNSub
        graphDist.nodeInsWeight=wNIns
        graphDist.nodeDelWeight=wNDel
        graphDist.edgeSubWeight=wESub
        graphDist.edgeInsWeight=wEIns
        graphDist.edgeDelWeight=wEDel
        
        granulationStrategy = BsasBinarySearch(graphDist,Repr,0.1)
        granulationStrategy.BsasQmax = Q
      
        granulationStrategy.symbol_thr = tau
        
        granulationStrategy.granulate(granulationBucket)
        f_sym = np.array([symbol.Fvalue for symbol in granulationStrategy.symbols])
    #    f = np.average(f_sym) if f_sym.size!=0 else np.nan
    
        #f_sym is better when lower. The problem is casted for maximasation
        f = 1-np.average(f_sym) if f_sym.size!=0 else np.nan     
        
        fitness = f if not np.isnan(f) else 0
        
        return (fitness,), granulationStrategy.symbols
    
    @staticmethod
    def checkBounds(QMAX,scaleFactor):
        def decorator(func):
            def wrapper(*args, **kargs):
                offspring = func(*args, **kargs)
                for child in offspring:
                    child[0] = round(child[0])
                    if child[0] < 1:
                       child[0] = 1
                    elif child[0]>QMAX/scaleFactor:
                        child[0] = round(QMAX/scaleFactor)
                    for i in range(1,len(child)):
                        if child[i] > 1:
                            child[i] = 1
                        elif child[i] <= 0:
                            child[i] = np.finfo(float).eps
                return offspring
            return wrapper
        return decorator
    
    def gene_bound(self,QMAX):
        ranges=[np.random.randint(1, QMAX), #BSAS q value bound 
                np.random.uniform(0, 1), #GED node wcosts
                np.random.uniform(0, 1),
                np.random.uniform(0, 1),
                np.random.uniform(0, 1), #GED edge wcosts
                np.random.uniform(0, 1),
                np.random.uniform(0, 1),
                np.random.uniform(np.finfo(float).eps, 1)] #Symbol Threshold
        
        if self._problemName ==  'GREC':
            additional  = [np.random.uniform(0, 1) for _ in range(5)]
            ranges = ranges + additional
            
        return ranges
