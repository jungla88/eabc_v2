
from deap import base, creator,tools
from deap.algorithms import varOr
import numpy as np
import random
import pickle
import networkx as nx
import multiprocessing
from functools import partial


from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.metrics import confusion_matrix
from scipy.optimize import differential_evolution

from Datasets.IAM import IamDotLoader
from Datasets.IAM import Letter,GREC
from eabc.datasets import graph_nxDataset
from eabc.extractors import Extractor
from eabc.extractors import randomwalk_restart
from eabc.embeddings import SymbolicHistogram
from eabc.extras.featureSelDE import FSsetup_DE,FSfitness_DE 

#
#from eabc.environments.nestedFS import customXover,customMutation,checkBounds,gene_bound,fitness
from eabc.environments.nestedFS import eabc_Nested
#

# from eabc.representatives import Medoid
# from eabc.dissimilarities import BMF
# from eabc.granulators import BsasBinarySearch
# from Datasets.IAM import Letter

def IAMreader(parser,path):
    
    delimiters = "_", "."      
    
    Loader = IamDotLoader.DotLoader(parser,delimiters=delimiters)
    
    graphDict = Loader.load(path)
    
    graphs,classes=[],[]
    for g,label in graphDict.values():
        graphs.append(g)
        classes.append(label)
    
    return graphs, classes 

# def customXover(ind1,ind2):
    
#     #Q
#     g_01,g_02 = tools.cxUniform([ind1[0]], [ind2[0]], CXPROB)
#     #GED
#     g1,g2 = tools.cxTwoPoint(ind1[1:7], ind2[1:7])
#     #Tau
#     g_71,g_72 = tools.cxUniform([ind1[7]], [ind2[7]], CXPROB)

#     #
#     ind1[0]=g_01[0]
#     ind2[0]=g_02[0]
#     #
#     for i in range(1,7):
#         ind1[i]=g1[i-1]
#     for i in range(1,7):
#         ind2[i]=g2[i-1]
#     #
#     ind1[7]=g_71[0]
#     ind2[7]=g_72[0]

#     return ind1,ind2

    
# def fitness(args):    
    
#     individual,granulationBucket = args
#     Q= individual[0]
#     wNSub= individual[1]
#     wNIns= individual[2]
#     wNDel= individual[3]
#     wESub= individual[4]
#     wEIns= individual[5]
#     wEDel= individual[6]
#     tau = individual[7]
    
#     Repr=Medoid
    
#     #TODO:edit for change dataset
#     diss = Letter.LETTERdiss()

#     graphDist=BMF(diss.nodeDissimilarity,diss.edgeDissimilarity)
#     graphDist.nodeSubWeight=wNSub
#     graphDist.nodeInsWeight=wNIns
#     graphDist.nodeDelWeight=wNDel
#     graphDist.edgeSubWeight=wESub
#     graphDist.edgeInsWeight=wEIns
#     graphDist.edgeDelWeight=wEDel
    
#     granulationStrategy = BsasBinarySearch(graphDist,Repr,0.1)
#     granulationStrategy.BsasQmax = Q
  
#     granulationStrategy.symbol_thr = tau
    
#     granulationStrategy.granulate(granulationBucket)
#     f_sym = np.array([symbol.Fvalue for symbol in granulationStrategy.symbols])
# #    f = np.average(f_sym) if f_sym.size!=0 else np.nan
#     f = 1-np.average(f_sym) if f_sym.size!=0 else np.nan     
    
#     fitness = f if not np.isnan(f) else 0
    
#     return (fitness,), granulationStrategy.symbols


# def checkBounds():
#     def decorator(func):
#         def wrapper(*args, **kargs):
#             offspring = func(*args, **kargs)
#             for child in offspring:
#                 if child[0] < 1:
#                     child[0] = 1
#                 elif child[0]>QMAX/15: # number of letter classes
#                     child[0] = QMAX/15 
#                 for i in range(1,len(child)):
#                     if child[i] > 1:
#                         child[i] = 1
#                     elif child[i] <= 0:
#                         child[i] = np.finfo(float).eps
#             return offspring
#         return wrapper
#     return decorator

# def gene_bound():
#     ranges=[np.random.randint(1, QMAX), #BSAS q value bound 
#             np.random.uniform(0, 1), #GED node wcosts
#             np.random.uniform(0, 1),
#             np.random.uniform(0, 1),
#             np.random.uniform(0, 1), #GED edge wcosts
#             np.random.uniform(0, 1),
#             np.random.uniform(0, 1),
#             np.random.uniform(np.finfo(float).eps, 1)] #Symbol Threshold

#     return ranges



def main(dataTR,dataVS,dataTS,N_subgraphs,mu,lambda_,ngen,maxorder,cxpb,mutpb):
    
    # random.seed(64)
   
    # cxpb=CXPROB
    # mutpb=MUTPROB

    # ###################
    # print("Loading...")
    
    
    # IAMreadergraph = partial(IAMreader,name=name)
    # rawtr = graph_nxDataset(path+"Training/", name, reader = IAMreadergraph)
    # rawvs = graph_nxDataset(path+"Validation/", name, reader = IAMreadergraph)    
    # rawts = graph_nxDataset(path+"Test/", name, reader = IAMreadergraph)

    # #DEBUG
    # rawtr = rawtr.shuffle()
    # rawvs = rawvs.shuffle()
    # #Removed not connected graph and null graph!
    # cleanDataTr,cleanDataVs,cleanDataTs=[],[],[]
    # for dataset,cleanData in zip([rawtr,rawvs,rawts],[cleanDataTr,cleanDataVs,cleanDataTs]):
    #     for g,idx,label in zip(dataset.data,dataset.indices,dataset.labels):
    #         if not nx.is_empty(g):
    #             if nx.is_connected(g):
    #                 cleanData.append((g,idx,label)) 

    # cleanDataTr = np.asarray(cleanDataTr,dtype=object)
    # cleanDataVs = np.asarray(cleanDataVs,dtype=object)
    # cleanDataTs = np.asarray(cleanDataTs,dtype=object)    
    
    # normalize('coords',cleanDataTr[:,0],cleanDataVs[:,0],cleanDataTs[:,0])
    
    # #Slightly different from dataset used in pygralg
    # dataTR = graph_nxDataset([cleanDataTr[:100,0],cleanDataTr[:100,2]],name, idx = cleanDataTr[:100,1])
    # dataVS = graph_nxDataset([cleanDataVs[:100,0],cleanDataVs[:100,2]],name, idx = cleanDataVs[:100,1])    
    # dataTS = graph_nxDataset([cleanDataTs[:100,0],cleanDataTs[:100,2]],name, idx = cleanDataTs[:100,1])    

    # del rawtr
    # del rawvs
    # del rawts
    # ##################


    print("Setup...")
    #Graph decomposition
    extract_func = randomwalk_restart.extr_strategy(max_order=maxorder)
    subgraph_extr = Extractor(extract_func)


    expTRSet = dataTR.fresh_dpcopy()
    for i,x in enumerate(dataTR):
        k=0
        while(k<50):
            for j in range(1,maxorder):
                subgraph_extr.max_order=j
                expTRSet.add_keyVal(dataTR.to_key(i),subgraph_extr.extract(x))
            k+=6
    expVSSet = dataVS.fresh_dpcopy()
    for i,x in enumerate(dataVS):
        k=0
        while(k<50):
            for j in range(1,maxorder):
                subgraph_extr.max_order=j
                expVSSet.add_keyVal(dataVS.to_key(i),subgraph_extr.extract(x))
            k+=6
            expVSSet.add_keyVal(dataVS.to_key(i),subgraph_extr.extract(x))
    expTSSet = dataTS.fresh_dpcopy()
    for i,x in enumerate(dataTS):
        k=0
        while(k<50):
            for j in range(1,maxorder):
                subgraph_extr.max_order=j
                expTSSet.add_keyVal(dataTS.to_key(i),subgraph_extr.extract(x))
            k+=6
            expTSSet.add_keyVal(dataTS.to_key(i),subgraph_extr.extract(x))            

    


    ##################
    # Evaluate the individuals with an invalid fitness
    DEBUG_FIXSUBGRAPH = False
    print("Initializing populations...")
    if DEBUG_FIXSUBGRAPH:
        print("DEBUG SUBGRAPH STOCHASTIC TRUE")
    classes= dataTR.unique_labels()
    #Initialize a dict of swarms - {key:label - value:deap popolution}
    population = {thisClass:toolbox.population(n=mu) for thisClass in classes}

    if DEBUG_FIXSUBGRAPH:
        subgraphsByclass = {thisClass:[] for thisClass in classes}
        
    for swarmClass in classes:

        thisClassPatternIDs = np.where(np.asarray(dataTR.labels)==swarmClass)[0]
        classAwareTR = dataTR[thisClassPatternIDs.tolist()]
        ##
        if DEBUG_FIXSUBGRAPH:
            subgraphsByclass[swarmClass] = subgraph_extr.randomExtractDataset(classAwareTR, N_subgraphs)
            subgraphs = [subgraphsByclass[swarmClass] for _ in population[swarmClass]]
        else:
            subgraphs = [subgraph_extr.randomExtractDataset(classAwareTR, N_subgraphs) for _ in population[swarmClass]]
        ##

        invalid_ind = [ind for ind in population[swarmClass] if not ind.fitness.valid]
        fitnesses,symbols = zip(*toolbox.map(toolbox.evaluate, zip(invalid_ind,subgraphs)))

        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

    #Log book
    LogAgents = {gen: {thisClass:[] for thisClass in classes} for gen in range(ngen+1)}
    LogPerf = {thisClass:[] for thisClass in classes}

    # Begin the generational process   
    ClassAlphabets={thisClass:[] for thisClass in classes}
    for gen in range(1, ngen + 1):
        
            print("Generation: {}".format(gen))
            
            for swarmClass in classes:
                
                print("############")
                #Generate the offspring: mutation OR crossover OR reproduce and individual as it is
                offspring = varOr(population[swarmClass], toolbox, lambda_, cxpb, mutpb)
        
                #Selecting data for this swarm               
                thisClassPatternIDs = np.where(np.asarray(dataTR.labels)==swarmClass)[0]
                classAwareTR = dataTR[thisClassPatternIDs.tolist()]
                
                #Select both old and offspring for evaluation in order to run agents
                pop = population[swarmClass] + offspring
                invalid_ind = pop
                #Select pop number of buckets to be assigned to agents
                if DEBUG_FIXSUBGRAPH:
                    subgraphs = [subgraphsByclass[swarmClass] for _ in pop]
                else:
                    subgraphs = [subgraph_extr.randomExtractDataset(classAwareTR, N_subgraphs) for _ in pop]

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
        
                #Embedding with current symbols
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
                
                #Relabeling swarmClass = 1 others = 0
                TRlabels = (np.asarray(dataTR.labels)==swarmClass).astype(int)
                VSlabels= (np.asarray(dataVS.labels)==swarmClass).astype(int)
                
                classifier = KNN()
                classifier.fit(TRMat,TRlabels)
                predictedVSLabels = classifier.predict(VSMat)
                     

                print("{},{}".format(len(VSlabels),len(predictedVSLabels)))
                tn, fp, fn, tp = confusion_matrix(VSlabels, predictedVSLabels).ravel()
                sensitivity = tp / (tp + fn)
                specificity = tn / (tn + fp)
                J = sensitivity + specificity - 1
                J = (J + 1) / 2
                #error_rate = 1 - J          
                print("Informedness {} - class {} - alphabet = {}".format(J, swarmClass,len(thisGenClassAlphabet)))
                
                #Feature Selection                  
                bounds_GA2, CXPB_GA2, MUTPB_GA2, DE_Pop = FSsetup_DE(len(thisGenClassAlphabet), -1)
                
                FS_inforDE= partial(FSfitness_DE,perfMetric = 'informedness')
                TuningResults_GA2 = differential_evolution(FS_inforDE, bounds_GA2, 
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
                
                print("Informedness with selected symbols: {}".format(J))

                #Assign the final fitness to agents
                fitnessesRewarded = list(fitnesses)
                ##For log
                rewardLog = []
                ##
                for agent in range(len(pop)):
                    NagentSymb = sum(np.asarray(idAgents)==agent)
                    indices = np.where(np.asarray(idAgents)==agent)
                    NagentSymbolsInModel= sum(mask[indices])

                    reward = J*NagentSymbolsInModel/NagentSymb if NagentSymb else 0
                    rewardLog.append(reward)
                    
                    fitnessesRewarded[agent] = 0.5*(fitnesses[agent][0]+reward), #Equal weight
                    
                #Update class alphabet
                ClassAlphabets[swarmClass]= np.asarray(thisGenClassAlphabet,dtype = object)[mask].tolist()

                #invalid ind is a reference to pop
                for ind, fit in zip(invalid_ind, tuple(fitnessesRewarded)):
                    ind.fitness.values = fit
            

                # Select the next generation population for the current swarm
                population[swarmClass][:] = toolbox.select(pop, mu)
                
                #Save Informedness for class and gen
                LogPerf[swarmClass].append([J,sum(np.asarray(best_GA2)==1),len(best_GA2)])
                #Save population at g = gen
                LogAgents[gen][swarmClass].append([pop,fitnesses,rewardLog,fitnessesRewarded])
            
            print("----------------------------")
    
    print("Test phase")
    #Collect class alphabets and embeddeding TR,VS,TS with concatenated Alphabets
    ALPHABETS=[alphabets for alphabets in ClassAlphabets.values()]   
    ALPHABETS = sum(ALPHABETS,[])
    
    embeddingStrategy.getSet(expTRSet, ALPHABETS)
    TRembeddingMatrix = np.asarray(embeddingStrategy._embeddedSet)
    TRpatternID = embeddingStrategy._embeddedIDs

    embeddingStrategy.getSet(expVSSet, ALPHABETS)
    VSembeddingMatrix = np.asarray(embeddingStrategy._embeddedSet)
    VSpatternID = embeddingStrategy._embeddedIDs

    #Resorting matrix for consistency with dataset        
    TRorderID = np.asarray([TRpatternID.index(x) for x in dataTR.indices])
    VSorderID = np.asarray([VSpatternID.index(x) for x in dataVS.indices])   

    TRMat = TRembeddingMatrix[TRorderID,:]
    VSMat = VSembeddingMatrix[VSorderID,:]        

    #Feature Selection                  
    bounds_GA2, CXPB_GA2, MUTPB_GA2, DE_Pop = FSsetup_DE(len(ALPHABETS), -1)
    FS_accDE= partial(FSfitness_DE,perfMetric = 'accuracy')
    TuningResults_GA2 = differential_evolution(FS_accDE, bounds_GA2, 
                                               args=(TRMat,
                                                     VSMat, 
                                                     dataTR.labels, 
                                                     dataVS.labels),
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
    classifier = KNN()
    classifier.fit(TRMat[:, mask], dataTR.labels)
    predictedVSmask=classifier.predict(VSMat[:, mask])
    
    accuracyVS = sum(predictedVSmask==np.asarray(dataVS.labels))/len(dataVS.labels)
    print("Accuracy on VS with global alphabet: {}".format(accuracyVS))

    #Embedding TS with best alphabet
    ALPHABET = np.asarray(ALPHABETS,dtype = object)[mask].tolist()
    embeddingStrategy.getSet(expTSSet, ALPHABET)
    TSembeddingMatrix = np.asarray(embeddingStrategy._embeddedSet)
    TSpatternID = embeddingStrategy._embeddedIDs   
    TSorderID = np.asarray([TSpatternID.index(x) for x in dataTS.indices]) 
    TSMat = TSembeddingMatrix[TSorderID,:]
    
    predictedTS=classifier.predict(TSMat)
    accuracyTS = sum(predictedTS==np.asarray(dataTS.labels))/len(dataTS.labels)
    print("Accuracy on TS with global alphabet: {}".format(accuracyTS))    
       

    return LogAgents,LogPerf,ClassAlphabets,TRMat,VSMat,predictedVSmask,dataVS.labels,TSMat,predictedTS,dataTS.labels,ALPHABETS,ALPHABET,mask

if __name__ == "__main__":
    
    # path = "/home/luca/Documenti/Progetti/E-ABC_v2/eabc_v2/Datasets/IAM/Letter3/"
    # name = "LetterH"
    path = "/home/luca/Documenti/Progetti/E-ABC_v2/eabc_v2/Datasets/IAM/GREC/"
    name = "GREC"    
    N_subgraphs = 20
    ngen = 1
    mu = 20
    lambda_=20
    maxorder = 6
    
    random.seed(64)
    QMAX = 500
    CXPROB = 0.33
    MUTPROB = 0.33
    
    #Maximizing
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    
    toolbox = base.Toolbox()
    
    #Multiprocessing map
    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)
    

    
    #Operators and inialization
    # toolbox.register("attr_genes", gene_bound,QMAX = QMAX)
    # toolbox.register("individual", tools.initIterate,
    #                 creator.Individual, toolbox.attr_genes)
    # toolbox.register("population", tools.initRepeat, list, toolbox.individual,n=100)  
    # toolbox.register("evaluate", fitness)
    # toolbox.register("mate", customXover)
    # toolbox.register("mutate", customMutation,sigma = 0.05,indpb=0.05)
    # toolbox.register("select", tools.selTournament, tournsize=5)

    # #Decorator bound    
    # toolbox.decorate("mate", checkBounds(QMAX,scale_factor))
    # toolbox.decorate("mutate", checkBounds(QMAX,scale_factor))


    ###Preprocessing
    print("Loading...")    
    
    if name == ('LetterH' or 'LetterM' or 'LetterL'):
        parser = Letter.parser
    elif name == 'GREC':
        parser = GREC.parser
        
    
    IAMreadergraph = partial(IAMreader,parser)
    rawtr = graph_nxDataset(path+"Training/", name, reader = IAMreadergraph)
    rawvs = graph_nxDataset(path+"Validation/", name, reader = IAMreadergraph)    
    rawts = graph_nxDataset(path+"Test/", name, reader = IAMreadergraph)

    #DEBUG
    # rawtr = rawtr.shuffle()
    # rawvs = rawvs.shuffle()
    #Removed not connected graph and null graph!
    cleanDataTr,cleanDataVs,cleanDataTs=[],[],[]
    for dataset,cleanData in zip([rawtr,rawvs,rawts],[cleanDataTr,cleanDataVs,cleanDataTs]):
        for g,idx,label in zip(dataset.data,dataset.indices,dataset.labels):
            if not nx.is_empty(g):
                if nx.is_connected(g):
                    cleanData.append((g,idx,label)) 

    #Cleaning and normalizing and retrieving norm values
    cleanDataTr = np.asarray(cleanDataTr,dtype=object)
    cleanDataVs = np.asarray(cleanDataVs,dtype=object)
    cleanDataTs = np.asarray(cleanDataTs,dtype=object)    
    
    if name == ('LetterH' or 'LetterM' or 'LetterL'):  
        weights = Letter.normalize('coords',cleanDataTr[:,0],cleanDataVs[:,0],cleanDataTs[:,0])
    elif name == 'GREC':
        weights = GREC.normalize(cleanDataTr[:,0],cleanDataVs[:,0],cleanDataTs[:,0])
    
    #Slightly different from dataset used in pygralg
    dataTR = graph_nxDataset([cleanDataTr[:100,0],cleanDataTr[:100,2]],name, idx = cleanDataTr[:100,1])
    dataVS = graph_nxDataset([cleanDataVs[:100,0],cleanDataVs[:100,2]],name, idx = cleanDataVs[:100,1])    
    dataTS = graph_nxDataset([cleanDataTs[:100,0],cleanDataTs[:100,2]],name, idx = cleanDataTs[:100,1])    

    del rawtr
    del rawvs
    del rawts
    
    
    #Create type for the problem
    if name == 'GREC':
        Dissimilarity = GREC.GRECdiss
    elif name == ('LetterH' or 'LetterM' or 'LetterL'):
        Dissimilarity = Letter.LETTERdiss

    eabc_Nested = eabc_Nested(DissimilarityClass=Dissimilarity,problemName = name,DissNormFactors=weights)
    
    scale_factor = len(np.unique(dataTR.labels,dataVS.labels,dataTS.labels))
    
    toolbox.register("attr_genes", eabc_Nested.gene_bound,QMAX = QMAX)
    
    toolbox.register("individual", tools.initIterate,
                    creator.Individual, toolbox.attr_genes)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual,n=100)  
    toolbox.register("evaluate", eabc_Nested.fitness)
    toolbox.register("mate", eabc_Nested.customXover)
    toolbox.register("mutate", eabc_Nested.customMutation,sigma = 0.05,indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=5)

    #Decorator bound    
    toolbox.decorate("mate", eabc_Nested.checkBounds(QMAX,scale_factor))
    toolbox.decorate("mutate", eabc_Nested.checkBounds(QMAX,scale_factor))
    LogAgents, LogPerf,ClassAlphabets,TRMat,VSMat,predictedVSmask,VSlabels,TSMat,predictedTS,TSlabels, ALPHABETS,ALPHABET,mask = main(dataTR,
                                                                                                                                      dataVS,
                                                                                                                                      dataTS,
                                                                                                                                      N_subgraphs,
                                                                                                                                      mu,
                                                                                                                                      lambda_,
                                                                                                                                      ngen,
                                                                                                                                      maxorder,
                                                                                                                                      CXPROB,
                                                                                                                                      MUTPROB)
    
    

    pickle.dump({'Name': name,
                 'Path': path,
                'Agents':LogAgents,
                'PerformancesTraining':LogPerf,
                'ClassAlphabets':ClassAlphabets,
                'TRMat':TRMat,
                'VSMat':VSMat,
                'predictedVSmask':predictedVSmask,
                'VSlabels':VSlabels,
                'TSMat':TSMat,
                'predictedTS':predictedTS,
                'TSlabels':TSlabels,
                'ALPHABETS':ALPHABETS,
                'ALPHABET':ALPHABET,
                'mask':mask,
                'name':name,
                'N_subgraphs':N_subgraphs,
                'N_gen':ngen,
                'Mu':mu,
                'lambda':lambda_,
                'max_order':maxorder
                },
                open(name+'.pkl','wb'))
    
    