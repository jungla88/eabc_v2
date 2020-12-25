
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
from Datasets.IAM import Letter,GREC,AIDS
from eabc.datasets import graph_nxDataset
from eabc.extractors import Extractor
from eabc.extractors import randomwalk_restart
from eabc.embeddings import SymbolicHistogram
from eabc.extras.featureSelDE import FSsetup_DE,FSfitness_DE 
from eabc.environments.nestedFS import eabc_Nested


def IAMreader(parser,path):
    
    delimiters = "_", "."      
    
    Loader = IamDotLoader.DotLoader(parser,delimiters=delimiters)
    
    graphDict = Loader.load(path)
    
    graphs,classes=[],[]
    for g,label in graphDict.values():
        graphs.append(g)
        classes.append(label)
    
    return graphs, classes 


def main(dataTR,dataVS,dataTS,N_subgraphs,mu,lambda_,ngen,maxorder,cxpb,mutpb):
    
    print("Setup...")
    #Graph decomposition
    # extract_func = randomwalk_restart.extr_strategy(max_order=maxorder)
    extract_func = randomwalk_restart.extr_strategy()
    subgraph_extr = Extractor(extract_func)

    expTRSet = subgraph_extr.decomposeGraphDataset(dataTR,maxOrder= maxorder)
    expVSSet = subgraph_extr.decomposeGraphDataset(dataVS,maxOrder= maxorder)
    expTSSet = subgraph_extr.decomposeGraphDataset(dataTS,maxOrder= maxorder)
        
    ##################
    # Evaluate the individuals with an invalid fitness
    DEBUG_FIXSUBGRAPH = False
    DEBUG_FITNESS = True
    print("Initializing populations...")
    if DEBUG_FIXSUBGRAPH:
        print("DEBUG SUBGRAPH STOCHASTIC TRUE")
    if DEBUG_FITNESS:
        print("DEBUG FITNESS TRUE")
    classes= dataTR.unique_labels()
    #Initialize a dict of swarms - {key:label - value:deap popolution}
    population = {thisClass:toolbox.population(n=mu) for thisClass in classes}
    IDagentsHistory = {thisClass:[ind.ID for ind in population[thisClass]] for thisClass in classes}
    
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
                #offspring = eabc_Nested.varOr(population[swarmClass], toolbox, lambda_, cxpb, mutpb)
                offspring = toolbox.varOr(population=population[swarmClass],toolbox=toolbox,lambda_=lambda_, idHistory=IDagentsHistory[swarmClass])
                #Selecting data for this swarm               
                thisClassPatternIDs = np.where(np.asarray(dataTR.labels)==swarmClass)[0]
                classAwareTR = dataTR[thisClassPatternIDs.tolist()]
                
                #Select both old and offspring for evaluation in order to run agents
                pop = population[swarmClass] + offspring
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
                # idAgents=[]
                # for i in range(len(pop)):
                #     if alphabets[i]:
                #         for _ in range(len(alphabets[i])):
                #             idAgents.append(i)
                
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

                #Update class alphabet
                ClassAlphabets[swarmClass]= np.asarray(thisGenClassAlphabet,dtype = object)[mask].tolist()

                #Assign the final fitness to agents
                fitnessesRewarded = list(fitnesses)
                ##For log
                rewardLog = []
                ##
                for agent in range(len(pop)):
                    
                    agentID = pop[agent].ID
                    NagentSymbolsInModel  = len([sym for sym in ClassAlphabets[swarmClass] if sym.owner==agentID])                        
                        
                    # indices = np.where(np.asarray(idAgents)==agent)
                    # NagentSymbolsInModel= sum(mask[indices])

                    reward = J*NagentSymbolsInModel/sum(np.asarray(best_GA2)==1)
                    rewardLog.append(reward)
                    
                    if DEBUG_FITNESS:
                        
                        fitnessesRewarded[agent] = reward
                    else:
                        
                        fitnessesRewarded[agent] = 0.5*(fitnesses[agent][0]+reward), #Equal weight
                    
                for ind, fit in zip(pop, tuple(fitnessesRewarded)):
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


    seed = 64
    random.seed(seed)
    np.random.seed(seed)
    # Parameter setup
    # They should be setted by cmd line
    path = "/home/luca/Documenti/Progetti/E-ABC_v2/eabc_v2/Datasets/IAM/Letter3/"
    name = "LetterH"
    # path = "/home/luca/Documenti/Progetti/E-ABC_v2/eabc_v2/Datasets/IAM/GREC/"
    # name = "GREC"  
    # path = "/home/luca/Documenti/Progetti/E-ABC_v2/eabc_v2/Datasets/IAM/AIDS/"
    # name = "AIDS" 
    N_subgraphs = 20
    ngen = 1
    mu = 20
    lambda_=20
    maxorder = 5
    CXPROB = 0.33
    MUTPROB = 0.33
    INDCXP = 0.3
    INDMUTP = 0.3
    TOURNSIZE = 5
    QMAX = 500


    ###Preprocessing
    print("Loading...")    
    
    if name == ('LetterH' or 'LetterM' or 'LetterL'):
        parser = Letter.parser
    elif name == 'GREC':
        parser = GREC.parser
    elif name == 'AIDS':
        parser = AIDS.parser
    else:
        raise FileNotFoundError
        
    
    IAMreadergraph = partial(IAMreader,parser)
    rawtr = graph_nxDataset(path+"Training/", name, reader = IAMreadergraph)[:10]
    rawvs = graph_nxDataset(path+"Validation/", name, reader = IAMreadergraph)[:10] 
    rawts = graph_nxDataset(path+"Test/", name, reader = IAMreadergraph)[:10]

    ####
    if name == ('LetterH' or 'LetterM' or 'LetterL'):  
        weights = Letter.normalize('coords',rawtr.data,rawvs.data,rawts.data)
    elif name == 'GREC':
        weights = GREC.normalize(rawtr.data,rawvs.data,rawts.data)
    elif name == 'AIDS':
        weights = AIDS.normalize(rawtr.data,rawvs.data,rawts.data)
    ###
    
    #Slightly different from dataset used in pygralg
    dataTR = rawtr
    dataVS = rawvs
    dataTS = rawts
    
    #Create type for the problem
    if name == 'GREC':
        Dissimilarity = GREC.GRECdiss
    elif name == ('LetterH' or 'LetterM' or 'LetterL'):
        Dissimilarity = Letter.LETTERdiss
    elif name == 'AIDS':
        Dissimilarity = AIDS.AIDSdiss


    #Maximizing
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Agent", list, fitness=creator.FitnessMax, ID = int)
    
    toolbox = base.Toolbox()
    
    #Multiprocessing map
    # pool = multiprocessing.Pool()
    # toolbox.register("map", pool.map)

    eabc_Nested = eabc_Nested(DissimilarityClass=Dissimilarity,problemName = name,DissNormFactors=weights)
    
    #Q scaling
    scale_factor = len(np.unique(dataTR.labels,dataVS.labels,dataTS.labels))
    scaledQ = round(QMAX/scale_factor)
    
    toolbox.register("attr_genes", eabc_Nested.gene_bound,QMAX = scaledQ) 
    toolbox.register("agent", tools.initIterate,
                    creator.Agent, toolbox.attr_genes)
    
    
    toolbox.register("population", eabc_Nested.initAgents, toolbox.agent,n=100)  
    toolbox.register("evaluate", eabc_Nested.fitness)
    toolbox.register("mate", eabc_Nested.customXover,indpb=INDCXP)
    #Setup mutation
    toolbox.register("mutate", eabc_Nested.customMutation,mu = 0, indpb=INDMUTP)
    toolbox.register("select", tools.selTournament, tournsize=TOURNSIZE)
    toolbox.register("varOr",eabc_Nested.varOr,cxpb= CXPROB, mutpb=MUTPROB)
    
    #Decorator bound    
    toolbox.decorate("mate", eabc_Nested.checkBounds(scaledQ))
    toolbox.decorate("mutate", eabc_Nested.checkBounds(scaledQ))
    
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
                 'CrossOverPr':CXPROB,
                 'MutationPr':MUTPROB,
                 'IndXoverPr':INDCXP,
                 'IndMutPr':INDMUTP,
                 'TournamentSize':TOURNSIZE,
                 'Seed':seed,
                'Agents':LogAgents,
                'PerformancesTraining':LogPerf,
                'ClassAlphabets':ClassAlphabets,
                'TRMat':TRMat,
                'TRlabels':dataTR.labels,
                'VSMat':VSMat,
                'predictedVSmask':predictedVSmask,
                'VSlabels':VSlabels,
                'TSMat':TSMat,
                'predictedTS':predictedTS,
                'TSlabels':TSlabels,
                'ALPHABETS':ALPHABETS,
                'ALPHABET':ALPHABET,
                'mask':mask,
                'N_subgraphs':N_subgraphs,
                'N_gen':ngen,
                'Mu':mu,
                'lambda':lambda_,
                'max_order':maxorder
                },
                open(name+'.pkl','wb'))
    
    