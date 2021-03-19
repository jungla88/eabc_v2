# -*- coding: utf-8 -*-

import pandas as pd
from deap import base, creator,tools
import numpy as np
from eabc.extras import eabc_modelGen

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Agent", list, fitness=creator.FitnessMax, ID = None)
log = pd.read_pickle(r'../Letter_mergedAlphabet.pkl')

previousModels = []
ClassAlphabets = log['ClassAlphabet']
seed = 0
rng =np.random.default_rng(seed)

model_generator = eabc_modelGen(k=5,l=5,seed=rng)


randomGeneratedModels = model_generator.createFromSymbols(ClassAlphabets)
#Create new model from the current and previous generation
recombinedModels = model_generator.createFromModels(randomGeneratedModels+previousModels)

#Merged random models plus recombined models
candidateModels = randomGeneratedModels + recombinedModels
