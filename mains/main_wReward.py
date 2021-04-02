
from deap import base, creator,tools
import numpy as np
import random
import pickle
import multiprocessing
from functools import partial
import copy

from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.metrics import confusion_matrix,balanced_accuracy_score

from Datasets.IAM import IamDotLoader
from Datasets.IAM import Letter,GREC,AIDS
from eabc.datasets import graph_nxDataset
from eabc.extractors import Extractor
from eabc.extractors import randomwalk_restart
from eabc.embeddings import SymbolicHistogram
from eabc.environments.binaryGED_Eabc import eabc
#TOBE MOVED
from eabc.extras import eabc_modelGen
from eabc.extras import Rewarder
from eabc.extras import consensusStrategy
from eabc.extras import StackClassifiers
from eabc.extras import GEDretriever

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

    gedParams = partial(GEDretriever.getParams,datasetName= name)
    consensusRewarder = consensusStrategy(gedParams)
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
#                internalClustEval,alphabets = zip(*toolbox.map(toolbox.evaluate, zip(population[swarmClass],subgraphs)))
                alphabets = list(toolbox.map(toolbox.evaluate, zip(population[swarmClass],subgraphs)))
                
                #processing alphabets ids and ownership
                for agent,alphabet in zip(population[swarmClass],alphabets):
                    idx = agent.ID
                    for s in alphabet:
                        s.owner = idx

                #Concatenate symbols if not empty
                alphabet = sum(alphabets,[])

                #Temporary save overlength alphabet
                ClassAlphabets[swarmClass] = ClassAlphabets[swarmClass] + alphabet
                


            #Merging all class buckets
            mergedClassAlphabets = sum(ClassAlphabets.values(),[])            

            #Models creation stage
            randomGeneratedModels = model_generator.createFromSymbols(ClassAlphabets)
            
            
            #Create new model from the current and previous generation
            recombinedModels = model_generator.createFromModels(randomGeneratedModels+previousModels)
            # print(len(recombinedModels[0]),recombinedModels[0][0].Fvalue)
            # if DEBUG:
            #     return 0 
            
            
            #Merged random models plus recombined models
            candidateModels = randomGeneratedModels + recombinedModels

            #Evaluating model performances
            modelPerformances = np.zeros((len(candidateModels)))
            
            #Save the trained classifier
            classificationModel = np.empty((len(candidateModels)),dtype=object) 

            #Find the largest alphabets in models
            m = candidateModels + previousModels
            modelLengths = [len(model) for model in m]

            for i,model in enumerate(candidateModels):
                
                embeddingStrategy = SymbolicHistogram(isSymbolDiss=True,isParallel=True)
                
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
                modelPerformances[i] = alpha*J + (1-alpha)*(1-(modelSize/max(modelLengths))) 
                ####
                classificationModel[i] = classifier

                print("{},{}".format(len(dataVS.labels),len(predictedVSLabels)))
                print("Balanced accuracy {} - model perf {} - alphabet = {} - max alphabet ={}".format(J, modelPerformances[i],len(model),max(modelLengths)))
            
            #Join new models with previous models with performances
            # evaluatedModelsPerf= list(zip(candidateModels,modelPerformances,classificationModel)) + list(zip(previousModels,previousModelsPerf,previousClassifiers))
            
            #######TEST 
            evaluatedModelsPerf=list(zip(candidateModels,modelPerformances))
            #######

            ####Save old performances for evaluate reward if exists previous evaluated models
            if previousModels:
                modelsForRewardEval = [[model,perf] for model,perf in list(zip(previousModels,previousModelsPerf))]
            #Else estimate on the base of just evaluated models
            else:
                modelsForRewardEval = [[model,perf] for model,perf in list(zip(candidateModels,modelPerformances))]

            #Sorting and thresholding models by performances
#            models = sorted(evaluatedModelsPerf,key = lambda x:x[1],reverse =True)[:bestModelsCard]
            #######TEST
            modelEvalSoFar= list(zip(candidateModels,modelPerformances,classificationModel)) + list(zip(previousModels,previousModelsPerf,previousClassifiers))
            models = sorted(modelEvalSoFar,key = lambda x:x[1],reverse =True)[:bestModelsCard]            
            print("Best K models: {}".format(list(zip(*models))[1]))

            #Update generation in rewarder
            rewarder.Gen = gen
            print("Tradeoff model/cluster compactness and card: {}".format(rewarder.modelWeight))
            
            #Rewarding stage
            
            #reward symbols for model performances
            #FIXME: Symbols not chosen are not rewarded/penalized
            #modelsToReward=[[model,perf] for model,perf,_ in models]
            ####TEST 
            modelsToReward=[[model,perf] for model,perf in evaluatedModelsPerf]
            ####
            
            #Test adaptive reward
            rewarder.evaluateReward(modelsForRewardEval)
            
            #Apply it 
            rewarder.applySymbolReward(modelsToReward)
            
            for swarmClass in classes:
                
                #reward symbols in the same metric
                consensusRewarder.applyConsensus(population[swarmClass],ClassAlphabets[swarmClass])                
                
                #reward agent
                rewarder.applyAgentReward(population[swarmClass],ClassAlphabets[swarmClass])
                
                # #thresholding alphabets and update
                #Remove negative quality symbols
                ClassAlphabets[swarmClass] = [x for x in ClassAlphabets[swarmClass] if x.quality>= 0 ]
                #thresholding up until classBucketCard
                ClassAlphabets[swarmClass] = sorted(ClassAlphabets[swarmClass],key=lambda x: x.quality,reverse = True)[:classBucketCard]                  
                          
                # Select the next generation population for the current swarm
                if gen > 0:
                    population[swarmClass][:] = toolbox.select(population[swarmClass], mu)
                
                #Save population at g = gen
                LogAgents[gen][swarmClass].append([population[swarmClass],ClassAlphabets[swarmClass]])
            
                print("{} Class agent qualities".format(swarmClass))
                print([agent.fitness.values for agent in population[swarmClass]])
                
                print("--")
                print("{} Class symbols qualities".format(swarmClass))
                print([sym.quality for sym in ClassAlphabets[swarmClass]])
            
            previousModels = copy.deepcopy(list(list(zip(*models))[0])) #Orribile list(list())
            previousModelsPerf = copy.deepcopy(list(list(zip(*models))[1]))
            previousClassifiers = copy.deepcopy(list(list(zip(*models))[2]))

            #Save models performances
            LogPerf[gen] = [previousModels,previousModelsPerf,previousClassifiers]
                             
            print("----------------------------")
    
    print("Test phase")
    
    print("Embedding Test Set")
    TSembeddingSpaces = []
    for model_ in previousModels:
        
        embeddingStrategy = SymbolicHistogram(isSymbolDiss=True,isParallel=Parallel)
        
        #Embedding with current symbols
        embeddingStrategy.getSet(expTSSet, model_)
        TSembeddingMatrix = np.asarray(embeddingStrategy._embeddedSet)
        TSpatternID = embeddingStrategy._embeddedIDs

        #Resorting matrix for consistency with dataset        
        TSorderID = np.asarray([TSpatternID.index(x) for x in dataTS.indices])
        TSMat = TSembeddingMatrix[TSorderID,:]    

        TSembeddingSpaces.append(TSMat)

    print("Building ensemble of classifiers...")
    ensembleClassifier = StackClassifiers(previousClassifiers,isPrefit=True)
    ensembleClassifier.fit(labels=dataTR.labels)
    predictedTSLabels = ensembleClassifier.predict(TSembeddingSpaces)
    
    
    accuracyTS = sum(predictedTSLabels==np.asarray(dataTS.labels))/len(dataTS.labels)
    print("Accuracy on TS: {}".format(accuracyTS))    
       

    return LogAgents,LogPerf,ClassAlphabets,previousModels,previousModelsPerf,previousClassifiers,ensembleClassifier,TSembeddingSpaces,predictedTSLabels,dataTR.labels,dataVS.labels,dataTS.labels

if __name__ == "__main__":

    # Parameter setup
    # They should be setted by cmd line
    seed = 0
    random.seed(seed)
    npRng = np.random.default_rng(seed)

    path = "/home/luca/Documenti/Progetti/E-ABC_v2/eabc_v2/Datasets/IAM/Letter3/"
    name = "LetterH"
    # path = "/home/luca/Documenti/Progetti/E-ABC_v2/eabc_v2/Datasets/IAM/GREC/"
    # name = "GREC"  
    # path = "/home/luca/Documenti/Progetti/E-ABC_v2/eabc_v2/Datasets/IAM/AIDS/"
    # name = "AIDS" 
    
    #Algorithm Hyper Params
    N_subgraphs = 20
    ngen = 20
    classBucketCard = 20
    bestModelsCard = 10
    numRandModel = 10
    numRecombModel = 10
    
    #Genetic Hyper Params
    mu = 5
    lambda_= 5
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

    eabc = eabc(DissimilarityClass=Dissimilarity,problemName = name,DissNormFactors=weights,seed = npRng)
    
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
  
#    LogAgents,LogPerf,ClassAlphabets,previousModels,previousModelsPerf,previousClassifiers,ensembleClassifier,TSembeddingSpaces,predictedTSLabels,dataTR.labels,dataVS.labels,dataTS.labels
    import time
    tic = time.perf_counter()
    data=main(dataTR,
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

    toc = time.perf_counter()
    print(f"Execution time {toc - tic:0.4f} seconds when Parallel is {Parallel}")
    
    LogAgents = data[0]
    LogPerf = data[1]
    ClassAlphabets = data[2]
    finalModel = data[3]
    finalModelPerf = data[4]
    finalClassifiers = data[5]
    finalEOC = data[6]
    testEmbeddings = data[7]
    predictedTSLabels = data[8]
    

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
                'TRlabels':dataTR.labels,
                'VSlabels':dataVS.labels,
                'predictedTS':predictedTSLabels,
                'TSlabels':dataTS.labels,
                'N_subgraphs':N_subgraphs,
                'N_gen':ngen,
                'Mu':mu,
                'lambda':lambda_,
                'max_order':maxorder
                },
                open(name+'.pkl','wb'))
    
    