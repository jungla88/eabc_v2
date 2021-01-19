
from deap import base, creator,tools
from deap.algorithms import varOr
import numpy as np
import random
import pickle
import networkx as nx
import multiprocessing
from functools import partial
import copy
import itertools

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
    DEBUG_FITNESS = True
    DEBUG_INDOCC = True
    print("Initializing populations...")
    if DEBUG_FITNESS:
        print("DEBUG FITNESS TRUE")
    if DEBUG_INDOCC:
        print("DEBUG REPEATED IND TRUE")        
    classes= dataTR.unique_labels()
    #Initialize a dict of swarms - {key:label - value:deap popolution}
    population = {thisClass:toolbox.population(n=mu) for thisClass in classes}
    IDagentsHistory = {thisClass:[ind.ID for ind in population[thisClass]] for thisClass in classes}
    #
    normFfactors = {thisClass:{'minComp':1,'maxComp':0,'minCard':1,'maxCard':0} for thisClass in classes}
    #
    
    
    for swarmClass in classes:
        
        minComp=normFfactors[swarmClass]['minComp']
        maxComp=normFfactors[swarmClass]['maxComp']
        minCard=normFfactors[swarmClass]['minCard']
        maxCard=normFfactors[swarmClass]['maxCard']
        
        thisClassPatternIDs = np.where(np.asarray(dataTR.labels)==swarmClass)[0]
        classAwareTR = dataTR[thisClassPatternIDs.tolist()]
        subgraphs = [subgraph_extr.randomExtractDataset(classAwareTR, N_subgraphs) for _ in population[swarmClass]]
        
        minComp=[normFfactors[swarmClass]['minComp'] for _ in population[swarmClass]]
        maxComp=[normFfactors[swarmClass]['maxComp'] for _ in population[swarmClass]]
        minCard=[normFfactors[swarmClass]['minCard'] for _ in population[swarmClass]]
        maxCard=[normFfactors[swarmClass]['maxCard'] for _ in population[swarmClass]]

        fitnesses,symbols,minCard,maxCard,minComp,maxComp = zip(*toolbox.map(toolbox.evaluate, zip(population[swarmClass],subgraphs,minComp,maxComp,minCard,maxCard)))

        normFfactors[swarmClass]['minComp']=min(minComp)
        normFfactors[swarmClass]['maxComp']=max(maxComp)
        normFfactors[swarmClass]['minCard']=min(minCard)
        normFfactors[swarmClass]['maxCard']=max(maxCard)
        

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
                subgraphs = [subgraph_extr.randomExtractDataset(classAwareTR, N_subgraphs) for _ in pop]
                
                #
                minComp=[normFfactors[swarmClass]['minComp'] for _ in pop]
                maxComp=[normFfactors[swarmClass]['maxComp'] for _ in pop]
                minCard=[normFfactors[swarmClass]['minCard'] for _ in pop]
                maxCard=[normFfactors[swarmClass]['maxCard'] for _ in pop]
                #
               
                #
                assert(all([len(item)==len(pop) for item in [subgraphs,minComp,maxComp,minCard,maxCard]]))
                #

                #Run individual and return the partial fitness comp+card
                #fitnesses,alphabets = zip(*toolbox.map(toolbox.evaluate, zip(pop,subgraphs)))
                fitnesses,alphabets,minCard,maxCard,minComp,maxComp = zip(*toolbox.map(toolbox.evaluate, zip(pop,subgraphs,minComp,maxComp,minCard,maxCard)))
            
                #
                normFfactors[swarmClass]['minComp']=min(minComp)
                normFfactors[swarmClass]['maxComp']=max(maxComp)
                normFfactors[swarmClass]['minCard']=min(minCard)
                normFfactors[swarmClass]['maxCard']=max(maxCard)
                #
                
                #Store agent number of symbols and fix replicated individuals alphabet size issue
                ids = np.asarray([ind.ID for ind in pop])                
                uniqueIds, indices, count = np.unique(ids,return_inverse=True,return_counts=True)
                for i in range(len(uniqueIds)):
                    values = [len(alphabets[j]) for j in np.where(indices==i)[0]]
                    for j in np.where(indices==i)[0]:
                        pop[j].alphabetSize = max(values)
    
                    
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
                    #Count ownership in full alphabet
                    NagentSymbolsPrev = len(list(filter(lambda sym: sym.owner==agentID,thisGenClassAlphabet)))                                         
                    #Count ownership in shrinked alphabet
                    NagentSymbolsInModel  = len([sym for sym in ClassAlphabets[swarmClass] if sym.owner==agentID])

                    if DEBUG_FITNESS:

                        alpha = 0.9
                        alphCard = pop[agent].alphabetSize
                        
                        reward = alpha*J*NagentSymbolsInModel/NagentSymbolsPrev + (1-alpha)*(1-(alphCard/N_subgraphs)) if alphCard>0 else 0
                        if reward>1:
                            print("error")
                    else:
                        reward = J*NagentSymbolsInModel/sum(np.asarray(best_GA2)==1)
                    rewardLog.append(reward)
                    
                    if DEBUG_FITNESS:
                        fitnessesRewarded[agent] = reward,
                    else:
                        
                        fitnessesRewarded[agent] = 0.5*(fitnesses[agent][0]+reward), #Equal weight
                
                
                if DEBUG_INDOCC:
                    fitmean = []
                for ind, fit in zip(pop, fitnessesRewarded):
                    if DEBUG_INDOCC:
                        ids = np.asarray([thisInd.ID for thisInd in pop])
                        fitness = np.asarray(fitnessesRewarded)
                        indices = np.where(ids == ind.ID)
                        fit = np.mean(fitness[indices]),
                    ind.fitness.values = fit
                    if DEBUG_INDOCC:
                        fitmean.append(fit)
                
                #print([[ind.ID,ind.fitness.values] for ind in pop])
                for ind in pop:
                    print("{} - {} - Symbols: {} - Fitness: {}".format(ind.ID, ind,ind.alphabetSize, ind.fitness.values))
                ##           
                # x = np.asarray([ind.fitness.values[0] for ind in pop])
                # y = np.asarray([fit[0] for fit in fitmean])
                # if not np.all(x == y):
                #     pause = input("Stop Error")
                #     print("in pop")
                #     print(np.asarray([ind.fitness.values[0] for ind in pop]))
                #     pause = input()
                #     print("fitness list ")
                #     print(np.asarray([fit[0] for fit in fitmean]))
                #     #print(np.asarray([fit[0] for fit in fitnesses]))                    
                #     pause = input()
                #     print(np.where(x!=y))
                #     pause = input()
                #     for ind in pop:
                #         print(ind.ID,ind.fitness.valid)
                ##
                
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
    #path = "/home/luca/Documenti/Progetti/E-ABC_v2/eabc_v2/Datasets/IAM/GREC/"
    #name = "GREC"  
    # path = "/home/luca/Documenti/Progetti/E-ABC_v2/eabc_v2/Datasets/IAM/AIDS/"
    # name = "AIDS" 
    N_subgraphs = 100
    ngen = 10
    mu = 10
    lambda_= 50
    maxorder = 5
    CXPROB = 0.45
    MUTPROB = 0.45
    INDCXP = 0.3
    INDMUTP = 0.3
    TOURNSIZE = 3
    QMAX = 500


    ###Preprocessing
    print("Loading...")    
    
    if name in ['LetterH', 'LetterM','LetterL']:
        parser = Letter.parser
    elif name == 'GREC':
        parser = GREC.parser
    elif name == 'AIDS':
        parser = AIDS.parser
    else:
        raise FileNotFoundError
        
    
    IAMreadergraph = partial(IAMreader,parser)
    rawtr = graph_nxDataset(path+"Training/", name, reader = IAMreadergraph)[:100]
    rawvs = graph_nxDataset(path+"Validation/", name, reader = IAMreadergraph)[:100]
    rawts = graph_nxDataset(path+"Test/", name, reader = IAMreadergraph)[:100]

    ####
    if name in ['LetterH', 'LetterM' ,'LetterL']:  
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
    elif name in ['LetterH' , 'LetterM' , 'LetterL']:
        Dissimilarity = Letter.LETTERdiss
    elif name == 'AIDS':
        Dissimilarity = AIDS.AIDSdiss


    #Maximizing
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Agent", list, fitness=creator.FitnessMax, ID = None, alphabetSize = None)
    
    toolbox = base.Toolbox()
    
    #Multiprocessing map
    # pool = multiprocessing.Pool()
    # toolbox.register("map", pool.map)

    eabc_Nested = eabc_Nested(DissimilarityClass=Dissimilarity,problemName = name,DissNormFactors=weights)
    
    #Q scaling
    scale_factor = len(np.unique(dataTR.labels,dataVS.labels,dataTS.labels)[0])
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
    
    