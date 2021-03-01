# -*- coding: utf-8 -*-
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def applySymbolReward(models_with_performance):
    
    for model,performance in models_with_performance:
            for symbol in model:
                if performance <= 0.5:
                    symbol.quality = symbol.quality-1
                elif performance >= 0.95:
                     symbol.quality = symbol.quality+10
                else:
                     symbol.quality = symbol.quality+1

def applyAgentReward(agents,alphabet):
    
#    scaledAgentQs, scaledSymbolQs = normalizeFitness(agents,alphabet)
    symbolQualities = np.asarray([sym.quality for sym in alphabet]).reshape((-1,1))
    print(symbolQualities)
    scaledSymbolQs = MinMaxScaler().fit_transform(symbolQualities)

    agentQualities = np.zeros((len(agents),))
    for i,agent in enumerate(agents):
        
        agentSymbolsQ = np.asarray([quality for symbol,quality in zip(alphabet,scaledSymbolQs) if symbol.owner==agent.ID])
        print(agentSymbolsQ)
        meanQ = np.mean(agentSymbolsQ) if len(agentSymbolsQ) >= 1 else 0
        if agent.fitness.valid:
            agentQualities[i] = agent.fitness.values[0]  + meanQ
        else:
            agentQualities[i] = meanQ
            
            
    scaledAgentQs = MinMaxScaler().fit_transform(agentQualities.reshape((-1,1)))
    for agent,Q in zip(agents,scaledAgentQs):
        print(agent.fitness)
        agent.fitness.values = Q,
        print(agent.fitness)
    
# def normalizeFitness(agents,alphabet):
    
#     symbolQualities = [sym.quality for sym in alphabet]
#     agentQualities = [agents.fitness.values if agents.fitness.values else 0 for agent in agents]

#     scaledsymbQ = MinMaxScaler.fit_transform(symbolQualities)
#     scaledAgentQ = MinMaxScaler.fit_transform(agentQualities)
    
    
#     return scaledAgentQ,scaledsymbQ