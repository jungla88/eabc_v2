# -*- coding: utf-8 -*-
import numpy as np
from sklearn.preprocessing import MinMaxScaler
class Rewarder:
    def __init__(self,MAX_GEN=20, isBootStrapped = True):
        
        self.MAX_GEN = MAX_GEN
        self.gen = 0
        self._modelWeight = 0
        self._isBootStrapped = isBootStrapped
        
        
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
    
    def applySymbolReward(self,models_with_performance):
        
        for model,performance in models_with_performance:
                for symbol in model:
                    if performance <= 0.5:
                        symbol.quality = symbol.quality-1
                    elif performance >= 0.95:
                         symbol.quality = symbol.quality+10
                    else:
                         symbol.quality = symbol.quality+1
    
    def applyAgentReward(self,agents,alphabet):
                
    #    scaledAgentQs, scaledSymbolQs = normalizeFitness(agents,alphabet)
        symbolQualities = np.asarray([sym.quality for sym in alphabet]).reshape((-1,1))
        symbolInternalQualities = np.asarray([sym.Fvalue for sym in alphabet]).reshape((-1,1))
        

        scaledSymbolQs = MinMaxScaler().fit_transform(symbolQualities)
        scaledSymbolInternalQs = MinMaxScaler().fit_transform(symbolInternalQualities)        
        
        agentQualities = np.zeros((len(agents),))
        agentInternalQualities = np.zeros((len(agents),))
        for i,agent in enumerate(agents):
            
            agentSymbolsQ = np.asarray([quality for symbol,quality in zip(alphabet,scaledSymbolQs) if symbol.owner==agent.ID])
            agentSymbolsInternalQ = np.asarray([quality for symbol,quality in zip(alphabet,scaledSymbolInternalQs) if symbol.owner==agent.ID])
            
            meanQ = 1-np.mean(agentSymbolsQ) if len(agentSymbolsQ) >= 1 else 0
            
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
#            agent.fitness.values = Q,
            modelContribuiton = self._modelWeight*Q
            clusterContribution = (1-self._modelWeight)*symbolsInQ
            fitness = modelContribuiton + clusterContribution            
            agent.fitness.values= fitness,
            print("Agent: {} - Model contribution: {} - Cluster: {} - Total {}".format(agent.ID,modelContribuiton,clusterContribution,fitness))
            