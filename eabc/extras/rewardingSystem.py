# -*- coding: utf-8 -*-
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import norm

class Rewarder:
#    def __init__(self,MAX_GEN=20, isBootStrapped = True):
    def __init__(self, numClasses,quantLevel = None ,MAX_GEN=20, isBootStrapped = True,maxReward = 10,minReward = -10):        
        
        #Max generation
        self.MAX_GEN = MAX_GEN
        #current generation
        self.gen = 0
        #tradeoff weight between model contribution and internal cluster quality
        self._modelWeight = 0
        self._isBootStrapped = isBootStrapped
        
        #Build reward table based on number of classes
        self._numClasses = numClasses
        self._quantizationLevel = quantLevel
        self._maxReward = maxReward
        self._minReward = minReward   

        self._buildRewTable()
     
        #Test reward with mean and var
        # self._meanModelPerformances = None
        # self._stdModelPerformances = None
        # self._scaleFactor = 10
        
    @property
    def Gen(self):
        return self.gen
    @Gen.setter
    def Gen(self,val):
        if val <= self.MAX_GEN and self._isBootStrapped:
            self.gen = val
        else:
            raise ValueError
        
        self._modelWeight = self.gen/self.MAX_GEN 
    
    @property
    def modelWeight(self):
        return self._modelWeight
    @property
    def isBootStrapped(self):
        return self._isBootStrapped
    
    def _buildRewTable(self):
        
        if self._quantizationLevel is None:
            self._quantizationLevel = self._numClasses
        
        self._rewardTable = np.linspace(self._minReward,self._maxReward,self._quantizationLevel)
        self._lookupTable = np.linspace(0,1,self._quantizationLevel,endpoint=False)
                
    
    def applySymbolReward(self,models_with_performance):
        
        for model,performance in models_with_performance:
            for symbol in model:
                
                #Take the last index in lookupTable which grant a the condition   
                range_ = np.where(self._lookupTable <= performance)[0][-1]
                reward = self._rewardTable[range_]
                symbol.quality = symbol.quality + reward
                
    # def evaluateReward(self,models_with_performance):
        
    #     p = np.asarray([perf for _,perf in models_with_performance])
        
    #     self._meanModelPerformances = p.mean()
    #     self._stdModelPerformances = p.std()
        
    # Initial test user-defined reward
    # def applySymbolReward(self,models_with_performance):
        
    #     for model,performance in models_with_performance:
    #             for i,symbol in enumerate(model):
    #                 if performance <= 0.5:
    #                     symbol.quality = symbol.quality-1
    #                 elif performance >= 0.95:
    #                      symbol.quality = symbol.quality+10
    #                 else:
    #                      symbol.quality = symbol.quality+1

    # def applySymbolReward(self,models_with_performance):
        
    #     for model,performance in models_with_performance:
            
    #         pVal  = norm.pdf(performance,self._meanModelPerformances,self._stdModelPerformances)
    #         valAtmean = norm.pdf(self._meanModelPerformances,self._meanModelPerformances,self._stdModelPerformances)
            
    #         for symbol in model:
                
    #             if performance >= self._meanModelPerformances + self._stdModelPerformances:

    #                 symbol.quality = symbol.quality +  self._scaleFactor*(valAtmean - pVal)
                    
    #             elif performance <= self._meanModelPerformances - self._stdModelPerformances:
                    
    #                 symbol.quality = symbol.quality - self._scaleFactor*(valAtmean - pVal)
                    

    def applyAgentReward(self,agents,alphabet):
                        
        symbolQualities = np.asarray([sym.quality for sym in alphabet]).reshape((-1,1))
        symbolInternalQualities = np.asarray([sym.Fvalue for sym in alphabet]).reshape((-1,1))
        

        scaledSymbolQs = MinMaxScaler().fit_transform(symbolQualities)
        scaledSymbolInternalQs = MinMaxScaler().fit_transform(symbolInternalQualities)        
        
        agentQualities = np.zeros((len(agents),))
        agentInternalQualities = np.zeros((len(agents),))
        for i,agent in enumerate(agents):
            
            agentSymbolsQ = np.asarray([quality for symbol,quality in zip(alphabet,scaledSymbolQs) if symbol.owner==agent.ID])
            agentSymbolsInternalQ = np.asarray([quality for symbol,quality in zip(alphabet,scaledSymbolInternalQs) if symbol.owner==agent.ID])
            
            meanQ = np.mean(agentSymbolsQ) if len(agentSymbolsQ) >= 1 else 0
            
            ##Update agent quality according to symbols qualities
            if agent.fitness.valid:
                agentQualities[i] = agent.fitness.values[0]  + meanQ
            else:
                agentQualities[i] = meanQ
            #Set the quality according to compactness and cardinality
            agentInternalQualities[i] = 1- np.mean(agentSymbolsInternalQ) if len(agentSymbolsQ)>= 1 else 0 
            
        #TODO: make sense normalizing agent fitness in [0,1]?        
        scaledAgentQs = MinMaxScaler().fit_transform(agentQualities.reshape((-1,1)))

        for agent,Q,symbolsInQ in zip(agents,scaledAgentQs,agentInternalQualities):
            modelContribuiton = self._modelWeight*Q
            clusterContribution = (1-self._modelWeight)*symbolsInQ
            fitness = modelContribuiton + clusterContribution            
            agent.fitness.values= fitness,
            ##
            agent.modelFitness = modelContribuiton
            agent.clusterFitness =clusterContribution
            print("Agent: {} - Model contribution: {} - Cluster contribution: {} - Total {}".format(agent.ID,agent.modelFitness,agent.clusterFitness,agent.fitness.values))
            