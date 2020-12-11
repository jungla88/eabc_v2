
from deap import base, creator,tools
from deap.algorithms import varOr
import numpy as np
import random
import math
import networkx as nx
import multiprocessing


from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.metrics import confusion_matrix
from scipy.optimize import differential_evolution

from Datasets.IAM import IamDotLoader,Letter
from eabc.datasets import graph_nxDataset
from eabc.dissimilarities import BMF
from eabc.extractors import Extractor
from eabc.extractors import randomwalk_restart
from eabc.granulators import BsasBinarySearch
from eabc.representatives import Medoid
from eabc.embeddings import SymbolicHistogram
from eabc.extras.normalizeLetter import normalize
from eabc.extras.featureSelDE import FSsetup_DE,FSfitness_DE 


QMAX = 500
N_GEN = 100
CXPROB = 0.33
MUTPROB = 0.33
MU= 20
LAMBDA=20


def IAMreadergraph(path):
    
    delimiters = "_", "."      
    
    Loader = IamDotLoader.DotLoader(Letter.parser,delimiters=delimiters)
    graphDict = Loader.load(path)
    
    graphs,classes=[],[]
    for g,label in graphDict.values():
        graphs.append(g)
        classes.append(label)
    
    return graphs, classes 

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
    
    Repr=Medoid
        
    diss = Letter.LETTERdiss()

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
    f = 1-np.average(f_sym) if f_sym.size!=0 else np.nan     
    
    fitness = f if not np.isnan(f) else 0
    
    return (fitness,), granulationStrategy.symbols


def checkBounds():
    def decorator(func):
        def wrapper(*args, **kargs):
            offspring = func(*args, **kargs)
            for child in offspring:
                if child[0] < 1:
                   child[0] = 1
                elif child[0]>QMAX/15: # number of letter classes
                    child[0] = QMAX/15 
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



# creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
# creator.create("Individual", list, fitness=creator.FitnessMin)

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

pool = multiprocessing.Pool()
toolbox.register("map", pool.map)

toolbox.register("attr_genes", gene_bound)
toolbox.register("individual", tools.initIterate,
                creator.Individual, toolbox.attr_genes)
toolbox.register("population", tools.initRepeat, list, toolbox.individual,n=100)

toolbox.register("evaluate", fitness)
#toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mate", customXover)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=5)

toolbox.decorate("mate", checkBounds())
toolbox.decorate("mutate", checkBounds())

def main():
    
# def eaSimple(population, toolbox, cxpb, mutpb, ngen, stats=None,
#              halloffame=None, verbose=__debug__):
####################
    random.seed(64)
    verbose = True
   
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
    data1 = graph_nxDataset("/home/LabRizzi/Documents/Alessio_Martino/eabc_v2/Datasets/IAM/Letter3/Training/", "LetterH", reader = IAMreadergraph)
    data2 = graph_nxDataset("/home/LabRizzi/Documents/Alessio_Martino/eabc_v2/Datasets/IAM/Letter3/Validation/", "LetterH", reader = IAMreadergraph)    
    data1 = data1.shuffle()
    data2 = data2.shuffle()
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
    dataVS = graph_nxDataset([cleanData[750:1500,0],cleanData[750:1500,2]],"LetterH", idx = cleanData[750:1500,1])    
    del data1
    del data2
    del cleanData


    print("Setup...")
    
    extract_func = randomwalk_restart.extr_strategy(max_order=6)
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
            expVSSet.add_keyVal(dataVS.to_key(i),subgraph_extr.extract(x))

    # Evaluate the individuals with an invalid fitness

    DEBUG_FIXSUBGRAPH = True
    print("Initializing populations...")
    if DEBUG_FIXSUBGRAPH:
        print("DEBUG SUBGRAPH STOCHASTIC TRUE")
    classes= dataTR.unique_labels()
    #Initialize a dict of swarms - {key:label - value:deap popolution}
    population = {thisClass:toolbox.population(n=MU) for thisClass in classes}

    if DEBUG_FIXSUBGRAPH:
        subgraphsByclass = {thisClass:[] for thisClass in classes}
        
    for swarmClass in classes:

        thisClassPatternIDs = np.where(np.asarray(dataTR.labels)==swarmClass)[0]
        classAwareTR = dataTR[thisClassPatternIDs.tolist()]
        ##
        if DEBUG_FIXSUBGRAPH:
            subgraphsByclass[swarmClass] = subgraph_extr.randomExtractDataset(classAwareTR, 200)
            subgraphs = [subgraphsByclass[swarmClass] for _ in population[swarmClass]]
        else:
            subgraphs = [subgraph_extr.randomExtractDataset(classAwareTR, 20) for _ in population[swarmClass]]
        ##

        invalid_ind = [ind for ind in population[swarmClass] if not ind.fitness.valid]
        fitnesses,symbols = zip(*toolbox.map(toolbox.evaluate, zip(invalid_ind,subgraphs)))

        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

    # if halloffame is not None:
    #     halloffame.update(population)

    # record = stats.compile(population) if stats else {}
    # logbook.record(gen=0, nevals=len(invalid_ind), **record)
    # if verbose:
    #     print(logbook.stream)
        
    # Begin the generational process   
    ClassAlphabets={thisClass:[] for thisClass in classes}
    for gen in range(1, ngen + 1):
        
            print("Generation: {}".format(gen))
            
            for swarmClass in classes:
                
                print("############")
                # Vary the population
                offspring = varOr(population[swarmClass], toolbox, lambda_, cxpb, mutpb)
        
                #Selecting data for this swarm               
                thisClassPatternIDs = np.where(np.asarray(dataTR.labels)==swarmClass)[0]
                classAwareTR = dataTR[thisClassPatternIDs.tolist()]
                
                #Select both old and offspring for evaluation
                pop = population[swarmClass] + offspring
                invalid_ind = pop
                #Select pop number of buckets to be assigned to agents
                if DEBUG_FIXSUBGRAPH:
                    subgraphs = [subgraphsByclass[swarmClass] for _ in pop]
                else:
                    subgraphs = [subgraph_extr.randomExtractDataset(classAwareTR, 20) for _ in pop]

                #Run individual and return the partial fitness comp+card
                fitnesses,alphabets = zip(*toolbox.map(toolbox.evaluate, zip(pop,subgraphs)))

                #Generate IDs for agents that pushed symbols in class bucket
                #E.g. idAgents       [ 0   0    1   1  1     2    -  3    .... ]
                #     alphabets      [[s1 s2] [ s3  s4 s5]  [s6] []  [s7] .... ]
                #Identify the agent that push s_i symbol
                idAgents=[]
                for i in range(len(pop)):
                    if alphabets[i]:
                        for _ in range(len(alphabets[i])):
                            idAgents.append(i)
                
                #Concatenate symbols if not empty
                alphabets = sum(alphabets,[])
                
                #Restart with previous symbols
                thisGenClassAlphabet = alphabets + ClassAlphabets[swarmClass]
                
                embeddingStrategy = SymbolicHistogram(isSymbolDiss=True,isParallel=True)
        
                #Embedded with current symbols
                embeddingStrategy.getSet(expTRSet, thisGenClassAlphabet)
                TRembeddingMatrix = np.asarray(embeddingStrategy._embeddedSet)
                TRpatternID = embeddingStrategy._embeddedIDs
        
                embeddingStrategy.getSet(expVSSet, thisGenClassAlphabet)
                VSembeddingMatrix = np.asarray(embeddingStrategy._embeddedSet)
                VSpatternID = embeddingStrategy._embeddedIDs        
        
                #Resorting matrix for consistency with dataset        
                TRorderID = np.asarray([TRpatternID.index(x) for x in dataTR.indices])
                VSorderID = np.asarray([VSpatternID.index(x) for x in dataVS.indices])        
                TRMat = TRembeddingMatrix[TRorderID,:]
                VSMat = VSembeddingMatrix[VSorderID,:]        
                
                TRlabels = (np.asarray(dataTR.labels)==swarmClass).astype(int)
                VSlabels= (np.asarray(dataVS.labels)==swarmClass).astype(int)
                
                classifier = KNN()
                classifier.fit(TRMat,TRlabels)
                predictedVSLabels = classifier.predict(VSMat)
                     
                accuracyVS = sum(predictedVSLabels==VSlabels)/len(VSlabels)
                tn, fp, fn, tp = confusion_matrix(VSlabels, predictedVSLabels).ravel()
                sensitivity = tp / (tp + fn)
                specificity = tn / (tn + fp)
                J = sensitivity + specificity - 1
                J = (J + 1) / 2
                #error_rate = 1 - J          
                print("Informedness {} - class {} - alphabet = {}".format(J, swarmClass,len(thisGenClassAlphabet)))
                
                #Feature Selection                  
                bounds_GA2, CXPB_GA2, MUTPB_GA2, DE_Pop = FSsetup_DE(len(thisGenClassAlphabet), -1)
                TuningResults_GA2 = differential_evolution(FSfitness_DE, bounds_GA2, 
                                                           args=(TRMat,
                                                                 VSMat, 
                                                                 TRlabels, 
                                                                 VSlabels),
                                                                 maxiter=100, init=DE_Pop, 
                                                                 recombination=CXPB_GA2,
                                                                 mutation=MUTPB_GA2, 
                                                                 workers=-1, 
                                                                 polish=False, 
                                                                 updating='deferred')
                best_GA2 = [round(i) for i in TuningResults_GA2.x]
                print("Selected {}/{} feature".format(sum(np.asarray(best_GA2)==1), len(best_GA2)))
                
                #Embedding with best alphabet
                mask = np.array(best_GA2,dtype=bool)
                classifier.fit(TRMat[:, mask], TRlabels)
                predictedVSmask=classifier.predict(VSMat[:, mask])
                
                tn, fp, fn, tp = confusion_matrix(VSlabels, predictedVSmask).ravel()
                sensitivity = tp / (tp + fn)
                specificity = tn / (tn + fp)
                J = sensitivity + specificity - 1
                J = (J + 1) / 2
#                error_rate = 1 - J
                
                print("Informedness with selected symbols: {}".format(J))

                #Assign the final fitness to agents
                fitnessesRewarded = list(fitnesses)
                #TODO: check if agents are rewarded only on the alphabet in this generation
                for agent in range(len(pop)):
                    NagentSymb = sum(np.asarray(idAgents)==agent)
                    indices = np.where(np.asarray(idAgents)==agent)
                    NagentSymbolsInModel= sum(mask[indices])

                    reward = J*NagentSymb/NagentSymbolsInModel if NagentSymbolsInModel else 0
                    
                    fitnessesRewarded[agent] = 0.5*(fitnesses[agent][0]+reward), #Equal weight
                
                
                ClassAlphabets[swarmClass]= np.asarray(thisGenClassAlphabet,dtype = object)[mask].tolist()

                #invalid ind is a reference to pop
                for ind, fit in zip(invalid_ind, tuple(fitnessesRewarded)):
                    ind.fitness.values = fit
            
                # # Update the hall of fame with the generated individuals
                # if halloffame is not None:
                #     halloffame.update(offspring)
            
                    # Select the next generation population for the current swarm
                population[swarmClass][:] = toolbox.select(pop, mu)
                
                print("#########")
                
            print("----------------------------")
            
    # # Update the statistics with the new population
    # record = stats.compile(population) if stats is not None else {}
    # logbook.record(gen=gen, nevals=len(invalid_ind), **record)
    # if verbose:
    #     print(logbook.stream)        

    return population, logbook

if __name__ == "__main__":
    pop, log = main()