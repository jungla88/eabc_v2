# -*- coding: utf-8 -*-

import pandas as pd
from deap import base, creator,tools

from eabc.extras import eabc_modelGen
from eabc.embeddings import SymbolicHistogram

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Agent", list, fitness=creator.FitnessMax, ID = None)
log = pd.read_pickle(r'../GREC.pkl')

alphabets = log['ClassAlphabets']

model_generator = eabc_modelGen()

mergedAlphabets = sum(alphabets.values(),[])

randomGeneratedModels = model_generator.createFromSymbols(mergedAlphabets)
print(len(randomGeneratedModels))
if randomGeneratedModels:
    recombinedModels = model_generator.createFromModels(randomGeneratedModels)
else:
    recombinedModels= randomGeneratedModels
    
mergedModel = randomGeneratedModels+recombinedModels
     
    
