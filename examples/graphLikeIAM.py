#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 11:37:48 2020

@author: luca
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 10:58:02 2020

@author: luca
"""


from deap import base, creator,tools
from deap.algorithms import varOr
import numpy as np
import random
import math


from sklearn.neighbors import KNeighborsClassifier as KNN

from Datasets.IAM import IamDotLoader,Letter

from eabc.datasets import graph_nxDataset
from eabc.dissimilarities import BMF
from eabc.extractors import Extractor
from eabc.extractors import breadthFirstSearch
from eabc.extractors import randomwalk_restart
from eabc.granulators import BsasBinarySearch
from eabc.representatives import Medoid
from eabc.embeddings import SymbolicHistogram
from eabc.extras.normalizeLetter import normalize

import networkx as nx
import functools
import multiprocessing


def IAMreadergraph(path):
    
    delimiters = "_", "."      
    
    Loader = IamDotLoader.DotLoader(Letter.parser,delimiters=delimiters)
    graphDict = Loader.load(path)
    
    graphs,classes=[],[]
    for g,label in graphDict.values():
        graphs.append(g)
        classes.append(label)
    
    return graphs, classes 


QMAX = 100
N_GEN = 100
CXPROB = 0.2
MUTPROB = 0.2
MU= 20
LAMBDA=20

def customXover(ind1,ind2):
    
    g_01,g_02 = tools.cxUniform([ind1[0]], [ind2[0]], CXPROB)
    g1,g2 = tools.cxTwoPoint(ind1[1:], ind2[1:])
    
    ind1[0]=g_01[0]
    ind2[0]=g_02[0]
    
    for i in range(1,len(ind1)):
        ind1[i]=g1[i-1]
    for i in range(1,len(ind2)):
        ind2[i]=g2[i-1]
    
    return ind1,ind2
    
    
def fitness(individual,granulationBucket,trEmbeddBucket, vsEmbeddBucket,TRindices,VSindices,TRlabels,VSlabels):    
       
    Q= individual[0]
    wNSub= individual[1]
    wNIns= individual[2]
    wNDel= individual[3]
    wESub= individual[4]
    wEIns= individual[5]
    wEDel= individual[6]
    tau = individual[7]
    eta = individual[8]
    
    Repr=Medoid
    
    #Setting GED   
    diss = Letter.LETTERdiss()
         
    graphDist=BMF(diss.nodeDissimilarity,diss.edgeDissimilarity)
    graphDist.nodeSubWeight=wNSub
    graphDist.nodeInsWeight=wNIns
    graphDist.nodeDelWeight=wNDel
    graphDist.edgeSubWeight=wESub
    graphDist.edgeInsWeight=wEIns
    graphDist.edgeDelWeight=wEDel
    
    #Setting granulation strategy
    granulationStrategy = BsasBinarySearch(graphDist,Repr,0.1)
    granulationStrategy.BsasQmax = Q
    granulationStrategy.eta=eta
    granulationStrategy.symbol_thr = tau
    #Setup embedder
    embeddingStrategy = SymbolicHistogram(Dissimilarity=graphDist,isSymbolDiss=False,isParallel=False)
    
    #Start granulation
    granulationStrategy.granulate(granulationBucket)
    #retrieving alphabet
    alphabet = granulationStrategy.symbols
    
    if alphabet:
        
        #Embedded with current symbols
        embeddingStrategy.getSet(trEmbeddBucket, alphabet)
        TRembeddingMatrix = np.asarray(embeddingStrategy._embeddedSet)
        TRpatternID = embeddingStrategy._embeddedIDs

        ##Debug
        # embeddingStrategy.getSetDebug(trEmbeddBucket, alphabet,TRindices)
        # TRembeddingMatrix = np.asarray(embeddingStrategy._embeddedSet)
        # TRpatternID = embeddingStrategy._embeddedIDs
        # print(np.all(np.asarray(TRlabels) == np.asarray(embeddingStrategy._embeddedClass)))
        ##
        
        embeddingStrategy.getSet(vsEmbeddBucket, alphabet)
        VSembeddingMatrix = np.asarray(embeddingStrategy._embeddedSet)
        VSpatternID = embeddingStrategy._embeddedIDs        
        # embeddingStrategy.getSetDebug(vsEmbeddBucket, alphabet,VSindices)
        # VSembeddingMatrix = np.asarray(embeddingStrategy._embeddedSet)
        # VSpatternID = embeddingStrategy._embeddedIDs        
        # print(np.all(np.asarray(VSlabels) == np.asarray(embeddingStrategy._embeddedClass)))
              
        #Resorting matrix for consistency with dataset        
        TRorderID = np.asarray([TRpatternID.index(x) for x in TRindices])        
        VSorderID = np.asarray([VSpatternID.index(x) for x in VSindices])        
        TRMat = TRembeddingMatrix[TRorderID,:]
        VSMat = VSembeddingMatrix[VSorderID,:]        
        
        #DEBUG
        # x = np.all(TRMat==TRembeddingMatrix2)
        # y = np.all(VSMat==VSembeddingMatrix2)
        # print(x,y)               
        ##                
        
        classifier = KNN()
        classifier.fit(TRMat,TRlabels)
        predictedVSLabels = classifier.predict(VSMat)
        
        # classifier.fit(TRembeddingMatrix,TRlabels)
        # predictedVSLabels = classifier.predict(VSembeddingMatrix)        
        accuracyVS = sum(predictedVSLabels==VSlabels)/len(VSlabels)
        
        
        print("Accuracy VS = {} - {}".format(accuracyVS,len(alphabet)))
        
        #Minimisation problem
        indFit = 0.9*(1-accuracyVS) + 0.1*(len(alphabet)/len(granulationBucket))

        
    else:
        
        print("Empty alphabet. Penalizing fitness with worst fitness")
        indFit = 1

    fitness = indFit
    
    return fitness,


def checkBounds():
    def decorator(func):
        def wrapper(*args, **kargs):
            offspring = func(*args, **kargs)
            for child in offspring:
                if child[0] < 1:
                   child[0] = 1
                if child[0]>QMAX:
                    child[0] = QMAX
                for i in range(1,len(child)):
                    if child[i] > 1:
                        child[i] = 1
                    if child[i] <= 0:
                        child[i] = np.finfo(float).eps
            return offspring
        return wrapper
    return decorator

def gene_bound():
    ranges=[np.random.randint(1, QMAX), #BSAS q value bound 
            np.random.uniform(0, 1), #GED node wcosts
            np.random.uniform(0, 1),
            np.random.uniform(0, 1),
            np.random.uniform(0, 1), #GED edge wcosts
            np.random.uniform(0, 1),
            np.random.uniform(0, 1),
            np.random.uniform(np.finfo(float).eps, 1), #Symbol Threshold
            np.random.uniform(0, 1)] #eta

    return ranges



creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

# pool = multiprocessing.Pool()
# toolbox.register("map", pool.map)

toolbox.register("attr_genes", gene_bound)
toolbox.register("individual", tools.initIterate,
                creator.Individual, toolbox.attr_genes)
toolbox.register("population", tools.initRepeat, list, toolbox.individual,n=100)

toolbox.register("evaluate", fitness,
                 granulationBucket=[],
                 trEmbeddBucket=[],
                 vsEmbeddBucket=[],
                 TRindices=[],
                 VSindices=[],
                 TRlabels=[],
                 VSlabels=[])
#toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mate", customXover)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

toolbox.decorate("mate", checkBounds())
toolbox.decorate("mutate", checkBounds())

def main():

# """
# Reproducing GRALG optimization with mu+lambda strategy of evolution
# """
####################
    random.seed(64)
    verbose = True
   
    population = toolbox.population(n=MU)
    cxpb=CXPROB
    mutpb=MUTPROB
    ngen=N_GEN 
    mu = MU
    lambda_ = LAMBDA

    halloffame = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    

    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])
###################

    print("Loading...")
    data1 = graph_nxDataset("/home/luca/Documenti/Progetti/E-ABC_v2/eabc_v2/Datasets/IAM/Letter3/Training/", "LetterH", reader = IAMreadergraph)
    data2 = graph_nxDataset("/home/luca/Documenti/Progetti/E-ABC_v2/eabc_v2/Datasets/IAM/Letter3/Validation/", "LetterH", reader = IAMreadergraph)    
    # data1 = data1.shuffle()
    # data2 = data2.shuffle()
    #Removed not connected graph and null graph!
    cleanData=[]
    for dataset in [data1,data2]:
        for g,idx,label in zip(dataset.data,dataset.indices,dataset.labels):
            if not nx.is_empty(g):
                if nx.is_connected(g):
                    cleanData.append((g,idx,label)) 

    cleanData = np.asarray(cleanData,dtype=object)
    normalize('coords',cleanData[:750,0],cleanData[750:,0])
    
    #Slightly different from dataset used in pygralg
    dataTR = graph_nxDataset([cleanData[:750,0],cleanData[:750,2]],"LetterH",idx = cleanData[:750,1])
    dataVS = graph_nxDataset([cleanData[750:,0],cleanData[750:,2]],"LetterH", idx = cleanData[750:,1])    
    del data1
    del cleanData

    print("Setup...")
    
    extract_func = randomwalk_restart.extr_strategy(max_order=6)
#    extract_func = breadthFirstSearch.extr_strategy(max_order=6)
#    subgraph_extr = Extractor(extract_func)
    subgraph_extr = Extractor(extract_func)

    expTRSet = dataTR.fresh_dpcopy()
    for i,x in enumerate(dataTR):
        k=0
        while(k<50):
            for j in range(1,6):
                subgraph_extr.max_order=j
                expTRSet.add_keyVal(dataTR.to_key(i),subgraph_extr.extract(x))
            k+=6
    expVSSet = dataVS.fresh_dpcopy()
    for i,x in enumerate(dataVS):
        k=0
        while(k<50):
            for j in range(1,6):
                subgraph_extr.max_order=j
                expVSSet.add_keyVal(dataVS.to_key(i),subgraph_extr.extract(x))
            k+=6
    # Evaluate the individuals with an invalid fitness
    print("Initializing population...")
    subgraphs= subgraph_extr.randomExtractDataset(dataTR, 1260)
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    
    fitnesses = toolbox.map(
        functools.partial(toolbox.evaluate,
                      granulationBucket= subgraphs,
                      trEmbeddBucket=expTRSet,
                      vsEmbeddBucket=expVSSet,
                      TRindices=dataTR.indices,
                      VSindices=dataVS.indices,
                      TRlabels=dataTR.labels,
                      VSlabels=dataVS.labels), invalid_ind)

    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)
        
    # Begin the generational process
    for gen in range(1, ngen + 1):
        # Vary the population
        offspring = varOr(population, toolbox, lambda_, cxpb, mutpb)

        #Evaluate invalid of modified individual
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(
            functools.partial(toolbox.evaluate,
                      granulationBucket= subgraphs,
                      trEmbeddBucket=expTRSet,
                      vsEmbeddBucket=expVSSet,
                      TRindices=dataTR.indices,
                      VSindices=dataVS.indices,
                      TRlabels=dataTR.labels,
                      VSlabels=dataVS.labels), invalid_ind)
        
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Select the next generation population
        population[:] = toolbox.select(population + offspring, mu)

        # Update the statistics with the new population
        record = stats.compile(population) if stats is not None else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)        

    return population, logbook

if __name__ == "__main__":
    pop, log = main()