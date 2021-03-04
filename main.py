
from deap import base, creator,tools
import numpy as np
import random
import pickle
import multiprocessing
from functools import partial
import copy

from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.metrics import confusion_matrix,balanced_accuracy_score
from sklearn.ensemble import VotingClassifier

from Datasets.IAM import IamDotLoader
from Datasets.IAM import Letter,GREC,AIDS
from eabc.datasets import graph_nxDataset
from eabc.extractors import Extractor
from eabc.extractors import randomwalk_restart
from eabc.embeddings import SymbolicHistogram
#from eabc.environments.nestedFS import eabc_Nested
from eabc.environments.binaryGED_Eabc import eabc
from eabc.extras import eabc_modelGen
from eabc.extras import Rewarder
from eabc.extras import consensusStrategy

def IAMreader(parser,path):
    
    delimiters = "_", "."      
    
    Loader = IamDotLoader.DotLoader(parser,delimiters=delimiters)
    
    graphDict = Loader.load(path)
    
    graphs,classes=[],[]
    for g,label in graphDict.values():
        graphs.append(g)
        classes.append(label)
    
    return graphs, classes 


def main(dataTR,dataVS,dataTS,
         N_subgraphs,
         classBucketCard,bestModelsCard,
         numRandModel,numRecombModel,
         mu,lambda_,ngen,maxorder,cxpb,mutpb,
         seed):
    
    print("Setup...")
    #Instance objects
    model_generator = eabc_modelGen(k=numRandModel,l=numRecombModel,seed=seed)
    rewarder = Rewarder(MAX_GEN= ngen)

    consensusRewarder = consensusStrategy("Letter")
    ##################
    classes= dataTR.unique_labels
    #Initialize a dict of swarms - {key:label - value:deap popolution}
    population = {thisClass:toolbox.population(n=mu) for thisClass in classes}
    #IDs class agents
    IDagentsHistory = {thisClass:[ind.ID for ind in population[thisClass]] for thisClass in classes}

    #Log book
    LogAgents = {gen: {thisClass:[] for thisClass in classes} for gen in range(ngen+1)}
    LogPerf = {gen: list() for gen in range(ngen+1)}    

    #Initial Variables
    ClassAlphabets={thisClass:[] for thisClass in classes}
    previousModels = []
    previousModelsPerf = []
    previousClassifiers = []
    
    #Graph decomposition
    extract_func = randomwalk_restart.extr_strategy(seed = seed)
    subgraph_extr = Extractor(extract_func,seed = seed)

    expTRSet = subgraph_extr.decomposeGraphDataset(dataTR,maxOrder= maxorder)
    expVSSet = subgraph_extr.decomposeGraphDataset(dataVS,maxOrder= maxorder)
    expTSSet = subgraph_extr.decomposeGraphDataset(dataTS,maxOrder= maxorder)

    #
    for gen in range(0, ngen):
        
            print("Generation: {}".format(gen))
            
            for swarmClass in classes:
                
                print("############")
                #Generate the offspring: mutation OR crossover OR reproduce and individual as it is
                offspring = []
                #if gen > 0:
                offspring = toolbox.varOr(population=population[swarmClass],toolbox=toolbox,lambda_=lambda_, idHistory=IDagentsHistory[swarmClass])
                
                #Selecting data for this swarm               
                thisClassPatternIDs = np.where(np.asarray(dataTR.labels)==swarmClass)[0]
                classAwareTR = dataTR[thisClassPatternIDs.tolist()]
                
                #Select both old and offspring for evaluation in order to run agents
                population[swarmClass] = population[swarmClass] + offspring

                #Select pop number of buckets to be assigned to agents
                subgraphs = [subgraph_extr.randomExtractDataset(classAwareTR, N_subgraphs) for _ in population[swarmClass]]
                
                #Run individual and return the partial fitness comp+card
                internalClustEval,alphabets = zip(*toolbox.map(toolbox.evaluate, zip(population[swarmClass],subgraphs)))
                
                #processing alphabets ids and ownership
                for agent,alphabet in zip(population[swarmClass],alphabets):
                    idx = agent.ID
                    for s in alphabet:
                        s.owner = idx

                #Concatenate symbols if not empty
                alphabet = sum(alphabets,[])

                consensusRewarder.applyConsensus(population[swarmClass],ClassAlphabets[swarmClass])


                #Temporary save overlength alphabet
                ClassAlphabets[swarmClass] = ClassAlphabets[swarmClass] + alphabet
                


            #Merging all class buckets
            mergedClassAlphabets = sum(ClassAlphabets.values(),[])

            #Models creation stage
            randomGeneratedModels = model_generator.createFromSymbols(ClassAlphabets)
            #Create new model from the current and previous generation
            recombinedModels = model_generator.createFromModels(randomGeneratedModels+previousModels)
            
            #Merged random models plus recombined models
            candidateModels = randomGeneratedModels + recombinedModels

            #Evaluating model performances
            modelPerformances = np.zeros((len(candidateModels)))
            
            #Save the trained classifier?
            classificationModel = np.empty((len(candidateModels)),dtype=object) 
            for i,model in enumerate(candidateModels):
                
                embeddingStrategy = SymbolicHistogram(isSymbolDiss=True,isParallel=False)
                
                #Embedding with current symbols
                embeddingStrategy.getSet(expTRSet, model)
                TRembeddingMatrix = np.asarray(embeddingStrategy._embeddedSet)
                TRpatternID = embeddingStrategy._embeddedIDs
        
                embeddingStrategy.getSet(expVSSet, model)
                VSembeddingMatrix = np.asarray(embeddingStrategy._embeddedSet)
                VSpatternID = embeddingStrategy._embeddedIDs        
        
                #Resorting matrix for consistency with dataset        
                TRorderID = np.asarray([TRpatternID.index(x) for x in dataTR.indices])
                VSorderID = np.asarray([VSpatternID.index(x) for x in dataVS.indices])        
                TRMat = TRembeddingMatrix[TRorderID,:]
                VSMat = VSembeddingMatrix[VSorderID,:]        
                
                classifier = KNN()
                classifier.fit(TRMat,dataTR.labels)
                predictedVSLabels = classifier.predict(VSMat)
                     
                J = balanced_accuracy_score(dataVS.labels,predictedVSLabels,adjusted=True)
                
                modelSize = len(model)
                #
                alpha = 0.99 #To be setted on cmdline
                #
                modelPerformances[i] = alpha*J + (1-alpha)*(1-(modelSize/len(mergedClassAlphabets))) 
                ####
                classificationModel[i] = classifier

                print("{},{}".format(len(dataVS.labels),len(predictedVSLabels)))
                print("Balanced accuracy {} - model perf {} - alphabet = {}".format(J, modelPerformances[i],len(model)))
            
            #Join new models with previous models with performances
            evaluatedModelsPerf= list(zip(candidateModels,modelPerformances,classificationModel)) + list(zip(previousModels,previousModelsPerf,previousClassifiers))
            #Sorting and thresholding models by performances
            models = sorted(evaluatedModelsPerf,key = lambda x:x[1],reverse =True)[:bestModelsCard]
            print("Best K models: {}".format(list(zip(*models))[1]))

            #Update generation in rewarder
            rewarder.Gen = gen
            print("Tradeoff model/cluster compactness and card: {}".format(rewarder.modelWeight))
            
            #Rewarding stage
            for swarmClass in classes:
                #reward symbols
                modelsToReward=[[model,perf] for model,perf,classifier in models]
                rewarder.applySymbolReward(modelsToReward)
                
                #reward agent
                rewarder.applyAgentReward(population[swarmClass],ClassAlphabets[swarmClass])              
                
                # #thresholding alphabets and update
                ClassAlphabets[swarmClass] = sorted(ClassAlphabets[swarmClass],key=lambda x: x.quality,reverse = True)[:classBucketCard]                  
                          
                # Select the next generation population for the current swarm
                if gen > 0:
                    population[swarmClass][:] = toolbox.select(population[swarmClass], mu)
                
                #Save population at g = gen
                LogAgents[gen][swarmClass].append([population[swarmClass],ClassAlphabets[swarmClass]])
            
                print("{} Class agent qualities".format(swarmClass))
                print([agent.fitness.values for agent in population[swarmClass]])
            

            previousModels = copy.deepcopy(list(list(zip(*models))[0])) #Orribile list(list())
            previousModelsPerf = copy.deepcopy(list(list(zip(*models))[1]))
            previousClassifiers = copy.deepcopy(list(list(zip(*models))[2]))

            #Save models performances
            LogPerf[gen] = [previousModels,previousModelsPerf,previousClassifiers]
                             
            print("----------------------------")
    
    print("Test phase")
    
    
    
    # TrainedClassifiersDict = {i:previousClassifiers[i] for i in range(len(previousClassifiers))}
    # ensemble = VotingClassifier(TrainedClassifiersDict)
    # ensemble.fit()
    #Collect class alphabets and embeddeding TR,VS,TS with concatenated Alphabets
    # ALPHABETS=[alphabets for alphabets in ClassAlphabets.values()]   
    # ALPHABETS = sum(ALPHABETS,[])
    
    # embeddingStrategy.getSet(expTRSet, ALPHABETS)
    # TRembeddingMatrix = np.asarray(embeddingStrategy._embeddedSet)
    # TRpatternID = embeddingStrategy._embeddedIDs

    # embeddingStrategy.getSet(expVSSet, ALPHABETS)
    # VSembeddingMatrix = np.asarray(embeddingStrategy._embeddedSet)
    # VSpatternID = embeddingStrategy._embeddedIDs

    # #Resorting matrix for consistency with dataset        
    # TRorderID = np.asarray([TRpatternID.index(x) for x in dataTR.indices])
    # VSorderID = np.asarray([VSpatternID.index(x) for x in dataVS.indices])   

    # TRMat = TRembeddingMatrix[TRorderID,:]
    # VSMat = VSembeddingMatrix[VSorderID,:]        

    # #Embedding with best alphabet
    # mask = np.array(best_GA2,dtype=bool)
    # classifier = KNN()
    # classifier.fit(TRMat[:, mask], dataTR.labels)
    # predictedVSmask=classifier.predict(VSMat[:, mask])
    
    # accuracyVS = sum(predictedVSmask==np.asarray(dataVS.labels))/len(dataVS.labels)
    # print("Accuracy on VS with global alphabet: {}".format(accuracyVS))

    # #Embedding TS with best alphabet
    # ALPHABET = np.asarray(ALPHABETS,dtype = object)[mask].tolist()
    # embeddingStrategy.getSet(expTSSet, ALPHABET)
    # TSembeddingMatrix = np.asarray(embeddingStrategy._embeddedSet)
    # TSpatternID = embeddingStrategy._embeddedIDs   
    # TSorderID = np.asarray([TSpatternID.index(x) for x in dataTS.indices]) 
    # TSMat = TSembeddingMatrix[TSorderID,:]
    
    # predictedTS=classifier.predict(TSMat)
    # accuracyTS = sum(predictedTS==np.asarray(dataTS.labels))/len(dataTS.labels)
    # print("Accuracy on TS with global alphabet: {}".format(accuracyTS))    
       

    return LogAgents,LogPerf,ClassAlphabets,TRMat,VSMat,predictedVSmask,dataVS.labels,TSMat,predictedTS,dataTS.labels,ALPHABETS,ALPHABET

if __name__ == "__main__":

    # Parameter setup
    # They should be setted by cmd line
    seed = 0
    random.seed(seed)
    npRng = np.random.default_rng(seed)

    path = "/home/luca/Documenti/Progetti/E-ABC_v2/eabc_v2/Datasets/IAM/Letter3/"
    name = "LetterH"
    #path = "/home/luca/Documenti/Progetti/E-ABC_v2/eabc_v2/Datasets/IAM/GREC/"
    #name = "GREC"  
    # path = "/home/luca/Documenti/Progetti/E-ABC_v2/eabc_v2/Datasets/IAM/AIDS/"
    # name = "AIDS" 
    
    #Algorithm Hyper Params
    N_subgraphs = 100
    ngen = 10
    classBucketCard = 100
    bestModelsCard = 10
    numRandModel = 5
    numRecombModel = 10
    
    #Genetic Hyper Params
    mu = 10
    lambda_= 10
    maxorder = 5
    CXPROB = 0.45
    MUTPROB = 0.45
    INDCXP = 0.3
    INDMUTP = 0.3
    TOURNSIZE = 3
    QMAX = 500
    Parallel = True
    
    ##

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
    rawtr = graph_nxDataset(path+"Training/", name, reader = IAMreadergraph,seed=npRng)[:100]
    rawvs = graph_nxDataset(path+"Validation/", name, reader = IAMreadergraph,seed = npRng)[:100]
    rawts = graph_nxDataset(path+"Test/", name, reader = IAMreadergraph,seed = npRng)[:100]

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
    creator.create("Agent", list, fitness=creator.FitnessMax, ID = None,modelFitness= None,clusterFitness= None)
    
    toolbox = base.Toolbox()
    
    #Multiprocessing map
    if Parallel == True:
        pool = multiprocessing.Pool()
        toolbox.register("map", pool.map)

    eabc = eabc(DissimilarityClass=Dissimilarity,problemName = name,DissNormFactors=weights)
    
    #Q scaling
    scale_factor = len(np.unique(dataTR.labels,dataVS.labels,dataTS.labels)[0])
    scaledQ = round(QMAX/scale_factor)
    
    toolbox.register("attr_genes", eabc.gene_bound,QMAX = scaledQ) 
    toolbox.register("agent", tools.initIterate,
                    creator.Agent, toolbox.attr_genes)
    
    
    toolbox.register("population", eabc.initAgents, toolbox.agent,n=100)  
    toolbox.register("evaluate", eabc.fitness)
    toolbox.register("mate", eabc.customXover,indpb=INDCXP)
    #Setup mutation
    toolbox.register("mutate", eabc.customMutation,mu = 0, indpb=INDMUTP)
    toolbox.register("select", tools.selTournament, tournsize=TOURNSIZE)
    toolbox.register("varOr",eabc.varOr,cxpb= CXPROB, mutpb=MUTPROB)
    
    #Decorator bound    
    toolbox.decorate("mate", eabc.checkBounds(scaledQ))
    toolbox.decorate("mutate", eabc.checkBounds(scaledQ))
    
    LogAgents, LogPerf,ClassAlphabets,TRMat,VSMat,predictedVSmask,VSlabels,TSMat,predictedTS,TSlabels, ALPHABETS,ALPHABET,mask = main(dataTR,
                                                                                                                                      dataVS,
                                                                                                                                      dataTS,
                                                                                                                                      N_subgraphs,
                                                                                                                                      classBucketCard,
                                                                                                                                      bestModelsCard,
                                                                                                                                      numRandModel,
                                                                                                                                      numRecombModel,
                                                                                                                                      mu,
                                                                                                                                      lambda_,
                                                                                                                                      ngen,
                                                                                                                                      maxorder,
                                                                                                                                      CXPROB,
                                                                                                                                      MUTPROB,
                                                                                                                                      npRng)
    
    

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
    
    