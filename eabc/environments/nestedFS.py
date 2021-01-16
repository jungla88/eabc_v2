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
        
        Semantically equal slice are recombined together with indpb probability. If the slice is a single attribute cxUniform is used,
        CxTwoPoint deap crossover is used in the other case. 

        """
        #Q
        g_q1,g_q2 = tools.cxUniform([ind1[0]], [ind2[0]], indpb = indpb)
        #GED
        g_ged1,g_ged2 = tools.cxTwoPoint(ind1[1:7], ind2[1:7])
        #Tau
        g_tau1,g_tau2 = tools.cxUniform([ind1[7]], [ind2[7]], indpb = indpb)
        #Test weight comp card
        g_eta1,g_eta2 = tools.cxUniform([ind1[8]], [ind2[8]], indpb = indpb)
        
    
        #Set if crossovered
        ind1[0]=g_q1[0]
        ind2[0]=g_q2[0]
        #two point crossover individuals are always modified. We edit this slice of genetic code only if condition is valid
        if random.random()<indpb:
            for i,(g1,g2) in enumerate(zip(g_ged1,g_ged2),start = 1):
                ind1[i]=g1
                ind2[i]=g2
            # for i in range(1,7):
            #     ind2[i]=g2
        #
        ind1[7]=g_tau1[0]
        ind2[7]=g_tau2[0]
        #Test weight comp card
        ind1[8]=g_eta1[0]
        ind2[8]=g_eta2[0]        
    
        if self._problemName == 'GREC':
            
#            g_add1, g_add2 =  tools.cxTwoPoint(ind1[8:13], ind2[8:13])
            g_add1, g_add2 =  tools.cxTwoPoint(ind1[9:14], ind2[9:14])
            #Same for ged attributes
            if random.random()<indpb:
#                for i,(g1,g2) in enumerate(zip(g_add1,g_add2),start = 8):
                for i,(g1,g2) in enumerate(zip(g_add1,g_add2),start = 9):
                    ind1[i]=g1
                    ind2[i]=g2
            
        return ind1,ind2
    
    def customMutation(self,ind,mu,indpb):
        """

        Parameters
        ----------
        ind : Deap individual
        mu : scalar value for gaussian mean value
        indpb : scalar value in [0,1] for individual gene probability of mutation

        Returns
        -------
        Mutated individual
        
        Description
        ------
        Each individual gene in in is mutated with dea mutGaussian function.
        Gaussian mean should be null, whereas the sigma is defined as a sequence for each guassian as required by deap mutGaussian
        """
#        mu = [ind[i] for i in range(len(ind))]
        sigmaQ = [10]
        sigma01 = [0.2 for _ in range(len(ind))]
        #Assuming genes  [Q, 01bounded, ... ]
        sigma = sigmaQ+sigma01
        return tools.mutGaussian(ind, mu, sigma, indpb)
        
    def fitness(self,args):    
        """
        

        Parameters
        ----------
        args : 2-length list with sequences of individual and bucket

        Returns
        -------
        fitness : a tuple-like fitness value
        symbols : a list of symbols synthesized by the agent.
        
        Description:
        -------
        
        According to individual genes, a Dissimilarity and a Granulator are instantiated with suitable parameters.
        Genetic optimization tries to maximise the average 1-F value of the symbol synthesized since lower value of F are better.

        """
        
        individual,granulationBucket = args
        Q= individual[0]
        wNSub= individual[1]
        wNIns= individual[2]
        wNDel= individual[3]
        wESub= individual[4]
        wEIns= individual[5]
        wEDel= individual[6]
        tau = individual[7]
        #Test eta
        eta = individual[8]

        if self._problemName == 'GREC':
                    
            # vParam1=individual[8]
            # eParam1=individual[9]
            # eParam2=individual[10]
            # eParam3=individual[11]
            # eParam4=individual[12]

            vParam1=individual[9]
            eParam1=individual[10]
            eParam2=individual[11]
            eParam3=individual[12]
            eParam4=individual[13]

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
        granulationStrategy.eta = eta
        
        granulationStrategy.granulate(granulationBucket)
        f_sym = np.array([symbol.Fvalue for symbol in granulationStrategy.symbols])
    #    f = np.average(f_sym) if f_sym.size!=0 else np.nan
    
        #f_sym is better when lower. The problem is casted for maximasation
        f = 1-np.average(f_sym) if f_sym.size!=0 else np.nan     
        
        fitness = f if not np.isnan(f) else 0
        
        ID = individual.ID
        symbols = granulationStrategy.symbols
        for symbol in symbols:
            symbol.owner = ID
            
        individual.alphabetSize = len(symbols)
        #Debug
        # individual.alphabets.append(symbols)
        
        return (fitness,), symbols
    
    @staticmethod
    def checkBounds(QMAX):
        """
        Parameters
        ----------
        QMAX : Integer
            Bound for gene.

        Returns
        -------
        None.
        
        Description
        -------
        First gene is bounded in [1,Qmax] and [0,1] otherwise.
        """
        def decorator(func):
            def wrapper(*args, **kargs):
                offspring = func(*args, **kargs)
                for child in offspring:
                    child[0] = round(child[0])
                    if child[0] < 1:
                       child[0] = 1
                    elif child[0]>QMAX:
                        child[0] = QMAX
                    for i in range(1,len(child)):
                        if child[i] > 1:
                            child[i] = 1
                        elif child[i] <= 0:
                            child[i] = np.finfo(float).eps
                return offspring
            return wrapper
        return decorator
    

    def gene_bound(self,QMAX):
        """
        

        Parameters
        ----------
        QMAX : Integer
            Qmax value for BSAS granulator.

        Returns
        -------
        ranges : List-like mixed integer and float values 
            Initial values for breeded individual

        """
        ranges=[np.random.randint(1, QMAX), #BSAS q value bound 
                np.random.uniform(0, 1), #GED node wcosts
                np.random.uniform(0, 1),
                np.random.uniform(0, 1),
                np.random.uniform(0, 1), #GED edge wcosts
                np.random.uniform(0, 1),
                np.random.uniform(0, 1),
                np.random.uniform(np.finfo(float).eps, 1), #Symbol Threshold
                np.random.uniform(np.finfo(float).eps, 1)] #Test eta
        
        if self._problemName ==  'GREC':
            additional  = [np.random.uniform(0, 1) for _ in range(5)]
            ranges = ranges + additional
            
        return ranges

    @staticmethod
    def initAgents(agentCreator,n):
        c = []
        for i in range(n):
            a = agentCreator()
            a.ID = i
            c.append(a)
        
        return c
    
    @staticmethod
    def varOr(population, toolbox, lambda_, idHistory,cxpb, mutpb):
        """Part of an evolutionary algorithm applying only the variation part
        (crossover, mutation **or** reproduction). The modified individuals have
        their fitness invalidated. The individuals are cloned so returned
        population is independent of the input population.
    
        :param population: A list of individuals to vary.
        :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                        operators.
        :param lambda\_: The number of children to produce
        :param cxpb: The probability of mating two individuals.
        :param mutpb: The probability of mutating an individual.
        :returns: The final population.
    
        The variation goes as follow. On each of the *lambda_* iteration, it
        selects one of the three operations; crossover, mutation or reproduction.
        In the case of a crossover, two individuals are selected at random from
        the parental population :math:`P_\mathrm{p}`, those individuals are cloned
        using the :meth:`toolbox.clone` method and then mated using the
        :meth:`toolbox.mate` method. Only the first child is appended to the
        offspring population :math:`P_\mathrm{o}`, the second child is discarded.
        In the case of a mutation, one individual is selected at random from
        :math:`P_\mathrm{p}`, it is cloned and then mutated using using the
        :meth:`toolbox.mutate` method. The resulting mutant is appended to
        :math:`P_\mathrm{o}`. In the case of a reproduction, one individual is
        selected at random from :math:`P_\mathrm{p}`, cloned and appended to
        :math:`P_\mathrm{o}`.
    
        This variation is named *Or* because an offspring will never result from
        both operations crossover and mutation. The sum of both probabilities
        shall be in :math:`[0, 1]`, the reproduction probability is
        1 - *cxpb* - *mutpb*.
        """
        assert (cxpb + mutpb) <= 1.0, (
            "The sum of the crossover and mutation probabilities must be smaller "
            "or equal to 1.0.")
    
        offspring = []
        for _ in range(lambda_):
            op_choice = random.random()
            if op_choice < cxpb:            # Apply crossover
                ind1, ind2 = list(map(toolbox.clone, random.sample(population, 2)))
                ind1, ind2 = toolbox.mate(ind1, ind2)
                del ind1.fitness.values
                offspring.append(ind1)
                ## New ID for recombined ind
                ind1.ID = max(idHistory)+1
                idHistory.append(ind1.ID)
                ##
            elif op_choice < cxpb + mutpb:  # Apply mutation
                ind = toolbox.clone(random.choice(population))
                ind, = toolbox.mutate(ind)
                del ind.fitness.values
                ## New ID for mutated ind
                ind.ID = max(idHistory)+1
                idHistory.append(ind.ID)
                ###   
                offspring.append(ind)
            else:                           # Apply reproduction
                #offspring.append(random.choice(population))
                randomSpawned = toolbox.agent()
                randomSpawned.ID = max(idHistory)+1
                idHistory.append(randomSpawned.ID)
                offspring.append(randomSpawned)
                
        return offspring