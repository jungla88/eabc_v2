
from deap import base, creator,tools
from deap.algorithms import varAnd
import numpy as np
import random
import math


from Datasets.tudataset import datasets,reader
from eabc.datasets import graph_nxDataset
#from eabc.dissimilarities import BMF
from eabc.dissimilarities import newBMF
from eabc.extractors import Extractor
from eabc.extractors import randomwalk_restart
from eabc.granulators import BsasBinarySearch
#from eabc.representatives import Medoid
from eabc.representatives import newMedoid
from eabc.embeddings import SymbolicHistogram 
import networkx as nx
import numpy as np

from scoop import futures
from functools import partial


def nodeDissimilarity(a, b):
        D = 0
        if(a['labels'] != b['labels']):
            D = 1
        return D

def edgeDissimilarity(a, b):
        D = 0
        if(a['labels'] != b['labels']):
            D = 1
        return D
    
def readergraph(path):
    graphs_nx = reader.tud_to_networkx("Mutagenicity")
    classes = [g.graph['classes'] for g in graphs_nx]
    return graphs_nx, classes 



QMAX = 100
POP_SIZE =20
N_GEN = 100
CXPROB = 0.2
MUTPROB = 0.2

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
    
    
def fitness(args):    
    
    individual,granulationBucket = args
    Q= individual[0]
    wNSub= individual[1]
    wNIns= individual[2]
    wNDel= individual[3]
    wESub= individual[4]
    wEIns= individual[5]
    wEDel= individual[6]
    tau = individual[7]
    
    Repr=newMedoid
        
    graphDist=newBMF(nodeDissimilarity,edgeDissimilarity)
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
    f = np.average(f_sym)         
    
    fitness = f if not np.isnan(f) else math.inf
    
    return fitness,


def checkBounds():
    def decorator(func):
        def wrapper(*args, **kargs):
            offspring = func(*args, **kargs)
            for child in offspring:
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

def gene_bound():
    ranges=[np.random.randint(1, QMAX), #BSAS q value bound 
            np.random.uniform(0, 1), #GED node wcosts
            np.random.uniform(0, 1),
            np.random.uniform(0, 1),
            np.random.uniform(0, 1), #GED edge wcosts
            np.random.uniform(0, 1),
            np.random.uniform(0, 1),
            np.random.uniform(np.finfo(float).eps, 1)] #Symbol Threshold

    return ranges



creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

toolbox.register("map", futures.map)

toolbox.register("attr_genes", gene_bound)
toolbox.register("individual", tools.initIterate,
                creator.Individual, toolbox.attr_genes)
toolbox.register("population", tools.initRepeat, list, toolbox.individual,n=100)

toolbox.register("evaluate", fitness)
#toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mate", customXover)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

toolbox.decorate("mate", checkBounds())
toolbox.decorate("mutate", checkBounds())

def main():
    
# def eaSimple(population, toolbox, cxpb, mutpb, ngen, stats=None,
#              halloffame=None, verbose=__debug__):
####################
    random.seed(64)
    verbose = True
   
    population = toolbox.population(n=POP_SIZE)
    cxpb=CXPROB
    mutpb=MUTPROB
    ngen=N_GEN 

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
    data1 = graph_nxDataset("/home/luca/Documenti/Progetti/E-ABC_v2/eabc_v2/Datasets/tudataset/Mutagenicity", "Mutagenicity", readergraph)
    #not connected graph
    cleanData = [(g,idx,label) for g,idx,label in zip(data1.data,data1.indices,data1.labels) if nx.is_connected(g)]
    cleanData = np.asarray(cleanData,dtype=object)
    data1 = graph_nxDataset([cleanData[:,0],cleanData[:,2]],"Mutagenicity")

    extract_func = randomwalk_restart.extr_strategy(max_order=6)
    subgraph_extr = Extractor(extract_func)


    # Evaluate the individuals with an invalid fitness
    print("Initializing population...")
    subgraphs = [subgraph_extr.randomExtractDataset(data1, 1000) for _ in population]
#    mapWithConst = partial(toolbox.evaluate,granulationBucket=subgraphs)
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    
    fitnesses = toolbox.map(toolbox.evaluate, zip(invalid_ind,subgraphs))    

    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    # Begin the generational process
    print("Start evolution")
    for gen in range(1, ngen + 1):
        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))

        # Vary the pool of individuals
        offspring = varAnd(offspring, toolbox, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness

        subgraphs = [subgraph_extr.randomExtractDataset(data1, 1000) for _ in population]
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        #mapWithConst = partial(toolbox.evaluate,granulationBucket=subgraphs)        
        fitnesses = toolbox.map(toolbox.evaluate, zip(invalid_ind,subgraphs))    

        
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Replace the current population by the offspring
        population[:] = offspring

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

    return population, logbook

if __name__ == "__main__":
    main()