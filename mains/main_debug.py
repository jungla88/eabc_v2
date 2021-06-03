
from deap import base, creator,tools
import numpy as np
import random
import pickle
import multiprocessing
from functools import partial
import copy
import argparse
from itertools import groupby
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.metrics import confusion_matrix,balanced_accuracy_score,accuracy_score


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
from eabc.extras import StackClassifier
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
    
    gedParams = partial(GEDretriever.getParams,datasetName= name)
    gedAgentParams = partial(GEDretriever.getAgentGEDParams,datasetName= name)
    consensusRewarder = consensusStrategy(gedParams)
    ##################
    classes= dataTR.unique_labels
    
    model_generator = eabc_modelGen(k=numRandModel,l=numRecombModel,seed=seed)
    #rewarder = Rewarder(MAX_GEN= ngen)
    rewarder = Rewarder(numClasses= len(classes),MAX_GEN=ngen)
    
    #Initialize a dict of swarms - {key:label - value:deap popolution}
    population = {thisClass:toolbox.population(n=mu) for thisClass in classes}
    #IDs class agents
    IDagentsHistory = {thisClass:[ind.ID for ind in population[thisClass]] for thisClass in classes}

    #Log book
    LogAgents = {gen: {thisClass:[] for thisClass in classes} for gen in range(ngen+1)}
    LogPerf = {gen: list() for gen in range(ngen+1)}
    ensemble_log = {gen: list() for gen in range(ngen+1)}

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
                
                ##DEBUG
                #Symbols obtained in the same spaces will be both reward if they are similar
                consensusRewarder.applyConsensus(ClassAlphabets[swarmClass])
                
                
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


                  
            #######TEST 
            evaluatedModelsPerf=list(zip(candidateModels,modelPerformances))
            #######

            #### Need for estimation reward - Save old performances for evaluate reward if exists previous evaluated models
            if previousModels:
                modelsForRewardEval = [[model,perf] for model,perf in list(zip(previousModels,previousModelsPerf))]
            #Else estimate on the base of just evaluated models
            else:
                modelsForRewardEval = [[model,perf] for model,perf in list(zip(candidateModels,modelPerformances))]
            ##
            
            #Sorting and thresholding models by performances
#            models = sorted(evaluatedModelsPerf,key = lambda x:x[1],reverse =True)[:bestModelsCard]
            #######TEST
            modelEvalSoFar= list(zip(candidateModels,modelPerformances,classificationModel)) + list(zip(previousModels,previousModelsPerf,previousClassifiers))
            models = sorted(modelEvalSoFar,key = lambda x:x[1],reverse =True)[:bestModelsCard]            
            print("Best K models: {}".format(list(zip(*models))[1]))
            
            ##### Debugging
            print("Debugging Ensemble on VS")
            modelsDebug = list(list(zip(*models))[0])
            classifiersDebug = list(list(zip(*models))[2])
            VSembeddingSpaces = []
            for model_ in modelsDebug:
                
                embeddingStrategy = SymbolicHistogram(isSymbolDiss=True,isParallel=Parallel)
                
                #Embedding with current symbols
                embeddingStrategy.getSet(expVSSet, model_)
                VSembeddingMatrix = np.asarray(embeddingStrategy._embeddedSet)
                VSpatternID = embeddingStrategy._embeddedIDs
        
                #Resorting matrix for consistency with dataset        
                VSorderID = np.asarray([VSpatternID.index(x) for x in dataVS.indices])
                VSMat = VSembeddingMatrix[VSorderID,:]    
        
                VSembeddingSpaces.append(VSMat)
        
            print("Building ensemble of classifiers...")
            ensembleClassifier = StackClassifier(classifiersDebug,isPrefit=True)
            ensembleClassifier.fit(labels=dataTR.labels)
            predictedVSLabels = ensembleClassifier.predict(VSembeddingSpaces)
            VS_ensemble_res = accuracy_score(dataVS.labels,predictedVSLabels)
            ensemble_log[gen] = [ensembleClassifier,predictedVSLabels,VS_ensemble_res,VSembeddingSpaces]
            #####
            
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
            #rewarder.evaluateReward(modelsForRewardEval)
            
            #Apply it 
            rewarder.applySymbolReward(modelsToReward)
            
            for swarmClass in classes:
                
                #reward symbols in the same metric
                #consensusRewarder.applyConsensus(population[swarmClass],ClassAlphabets[swarmClass])
                #consensusRewarder.applyConsensus(ClassAlphabets[swarmClass])                
                
                #reward agent
                rewarder.applyAgentReward(population[swarmClass],ClassAlphabets[swarmClass])
                
                # #thresholding alphabets and update
                #Remove negative quality symbols
                ClassAlphabets[swarmClass] = [x for x in ClassAlphabets[swarmClass] if x.quality>= 0 ]
                #thresholding up until classBucketCard
                ClassAlphabets[swarmClass] = sorted(ClassAlphabets[swarmClass],key=lambda x: x.quality,reverse = True)[:classBucketCard]                  
                                
                # Select the next generation population for the current swarm
                if gen > 0:
                    
                    ## Apply elitism for corresponding symbols/agent with high quality
                    ## We lookup in ClassAlphabets the symbols with higher quality
                    ## Then according to the metric, we apply elitism to agents that created that symbols
                    #ClassAlphabets[swarmClass] = sorted(ClassAlphabets[swarmClass],key=lambda x: x.quality,reverse = True)
                    ####
                    M_bestSym = 5
                    #Best unique metric in bucket
                    p = [tuple(gedParams(symbol)) for symbol in ClassAlphabets[swarmClass]]
                    # Must be hashable to apply set()
                    p = list(set(p))[:M_bestSym]
                    p = list(map(list,p))
                    #p_agents = [tuple(gedAgentParams(agent)) for agent in population[swarmClass]]
                    
                    #Elite
                    eliteInd = []
                    # groupby() can group contiguos elements. We sort before apply it
                    for commonMetric,group in groupby(sorted(population[swarmClass], key = lambda x:gedAgentParams(x)),gedAgentParams):
                        if commonMetric in p:
                            group = list(group)
                            bestInd = sorted(group, key = lambda x: x.fitness.values[0],reverse = True)[0]
                            eliteInd.append(bestInd)
                    
                        
                    # matchedIndividuals = [i for i,code in enumerate(p_agents) if code in p]
                    # eliteInd = [population[swarmClass][index] for index in matchedIndividuals]
                    # eliteInd = sorted(eliteInd, key = lambda x: x.fitness.values[0],reverse = True)[:M_bestSym]                    
                    # othersInd = [population[swarmClass][i] for i,agent in enumerate(population[swarmClass]) if i not in matchedIndividuals]
                
                    othersInd = [ population[swarmClass][i] for i,agent in enumerate(population[swarmClass]) if agent not in eliteInd ]    
                
                    population[swarmClass][:] = toolbox.select(othersInd, mu)
                    population[swarmClass] = population[swarmClass] + eliteInd 

                # if gen > 0:
                #     population[swarmClass][:] = toolbox.select(population[swarmClass], mu)
                
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
    
    
    #
    pool.close()
    #
    
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
    ensembleClassifier = StackClassifier(previousClassifiers,isPrefit=True)
    ensembleClassifier.fit(labels=dataTR.labels)
    predictedTSLabels = ensembleClassifier.predict(TSembeddingSpaces)
    
    
    accuracyTS = sum(predictedTSLabels==np.asarray(dataTS.labels))/len(dataTS.labels)
    print("Accuracy on TS: {}".format(accuracyTS))    
       
    
    return LogAgents,LogPerf,ClassAlphabets,previousModels,previousModelsPerf,previousClassifiers,ensembleClassifier,TSembeddingSpaces,predictedTSLabels,dataTR.labels,dataVS.labels,dataTS.labels,expTRSet,expVSSet,expTSSet,ensemble_log,accuracyTS

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
       
    ###
    parser.add_argument("-name", type=str,
                    help="Dataset name")    
    parser.add_argument("-W", type=int,
                    help="Number of subgraphs per agent")    

    parser.add_argument("-g", "--max_gen", type=int, default = 20, help="Maximum number of generations",
                    action="store")

    parser.add_argument("-cb", "--class_bucket_card", default = 50, type = int,  help="Class bucket cardinality",
                    action="store")

    parser.add_argument("-mb", "--model_bucket_card", type = int, default = 10, help="Number of best model for each generation to retain",
                    action="store")

    parser.add_argument("-nrm", "--num_random_model", type = int, default = 10,  help="Random models to create per generation",
                    action="store")
    
    parser.add_argument("-nrecm", "--num_recombined_model", type = int, default = 20, help="Recombined model to create per generation",
                    action="store")
    
    parser.add_argument("-s", "--seed", type = int, help="Seed for random procedures",
                    action="store")

    parser.add_argument("-mu", "--num_parents_ind", type = int, default = 10, help="Number of parents in genetic algorithm",
                    action="store")
    
    parser.add_argument("-lambda", "--num_offspring_ind", type = int, default = 30, help="Number of offspring in genetic algorithm",
                    action="store")

    parser.add_argument("-o", "--subgraphs_order", type = int, default = 5, help="Maximum order of extracted subgraphs",
                    action="store")

    parser.add_argument("-cxp", "--crossover_prob", type = float, default = 0.5, help="Crossover probability",
                    action="store")

    parser.add_argument("-mutp", "--mutation_prob", type = float, default = 0.3, help="Mutation probability",
                    action="store")

    parser.add_argument("-tourn", "--tournament_size", type = int, default = 3, help="Tournament size",
                    action="store")    

    parser.add_argument("-indmtp", "--ind_mutation_prob", type = float, default = 0.5, help="Individual slice of genetic code mutation probability",
                    action="store")    
    parser.add_argument("-indcxp", "--ind_crossover_prob", type = float, default = 0.5, help="Individual slice of genetic code crossover probability",
                    action="store")        


    args = parser.parse_args()
    #Algorithm Hyper Params
    name = args.name
    N_subgraphs = args.W
    ngen = args.max_gen
    classBucketCard = args.class_bucket_card
    bestModelsCard = args.model_bucket_card
    numRandModel = args.num_random_model
    numRecombModel = args.num_recombined_model
    
    #Genetic Hyper Params
    mu = args.num_parents_ind
    lambda_= args.num_offspring_ind
    maxorder = args.subgraphs_order
    CXPROB = args.crossover_prob
    MUTPROB = args.mutation_prob
    INDCXP = args.ind_crossover_prob
    INDMUTP = args.ind_mutation_prob
    TOURNSIZE = args.tournament_size
    seed = args.seed

    # Parameter setup
    # They should be setted by cmd line
#    seed = 0
    random.seed(seed)
    npRng = np.random.default_rng(seed)

   
    if name == 'LetterH':        
        path = "/home/luca/Documenti/Progetti/E-ABC_v2/eabc_v2/Datasets/IAM/Letter3/"
    elif name == 'LetterM':
        path = "/home/luca/Documenti/Progetti/E-ABC_v2/eabc_v2/Datasets/IAM/Letter2/"
    elif name == 'LetterL':
        path = "/home/luca/Documenti/Progetti/E-ABC_v2/eabc_v2/Datasets/IAM/Letter1/"
    elif name == 'GREC':
        path = "/home/luca/Documenti/Progetti/E-ABC_v2/eabc_v2/Datasets/IAM/GREC/"
    elif name == 'AIDS':
        path = "/home/luca/Documenti/Progetti/E-ABC_v2/eabc_v2/Datasets/IAM/AIDS/"

    
    #Algorithm Hyper Params
    # N_subgraphs = 20
    # ngen = 20
    # classBucketCard = 20
    # bestModelsCard = 10
    # numRandModel = 10
    # numRecombModel = 10
    
    # #Genetic Hyper Params
    # mu = 5
    # lambda_= 5
    # maxorder = 5
    # CXPROB = 0.45
    # MUTPROB = 0.45
    # INDCXP = 0.3
    # INDMUTP = 0.3
    # TOURNSIZE = 3
    
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
    rawtr = graph_nxDataset(path+"Training/", name, reader = IAMreadergraph,seed=npRng)
    rawvs = graph_nxDataset(path+"Validation/", name, reader = IAMreadergraph,seed = npRng)
    rawts = graph_nxDataset(path+"Test/", name, reader = IAMreadergraph,seed = npRng)

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
    totalTime = toc- tic
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
    TRlabels = data[9]
    VSlabels = data[10]
    TSlabels = data[11]
    expTRSet = data[12]
    expVSSet = data[13]
    expTSSet = data[14]
    ensemble_log = data[15]
    accuracy_TS = data[16]
    

    pickle.dump({'Name': name,
                  'Path': path,
                  'CrossOverPr':CXPROB,
                  'MutationPr':MUTPROB,
                  'IndXoverPr':INDCXP,
                  'IndMutPr':INDMUTP,
                  'TournamentSize':TOURNSIZE,
                  'Seed':seed,
                  'expTRset':expTRSet,
                  'expVSSet': expVSSet,
                  'expTSSet': expTSSet,
                  'totalTime': totalTime,
                  'ensemble_log': ensemble_log,
                  'finalModel': finalModel,
                  'finalModelPerf': finalModelPerf,
                  'finalEOC': finalEOC,
                  'testEmbeddings': testEmbeddings,
                'Agents':LogAgents,
                'PerformancesTraining':LogPerf,
                'ClassAlphabets':ClassAlphabets,
                'TRlabels':TRlabels,
                'VSlabels':VSlabels,
                'predictedTS':predictedTSLabels,
                'TSlabels':TSlabels,
                'accuracy_TS':accuracy_TS,
                'N_subgraphs':N_subgraphs,
                'N_gen':ngen,
                'Mu':mu,
                'lambda':lambda_,
                'max_order':maxorder,
                'command_line_args':args
                },
                open(name+'.pkl','wb'))
    
    