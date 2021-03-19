# -*- coding: utf-8 -*-

from Datasets.tudataset import datasets,reader
from eabc.datasets import graph_nxDataset
from eabc.dissimilarities import BMF
from eabc.extractors import Extractor
from eabc.extractors import randomwalk_restart
from eabc.granulators import BsasBinarySearch
from eabc.representatives import Medoid
from eabc.embeddings import SymbolicHistogram 
import networkx as nx
import numpy as np
from Datasets.IAM import IamDotLoader
from Datasets.IAM import Letter,GREC,AIDS
from functools import partial

import time

def IAMreader(parser,path):
    
    delimiters = "_", "."      
    
    Loader = IamDotLoader.DotLoader(parser,delimiters=delimiters)
    
    graphDict = Loader.load(path)
    
    graphs,classes=[],[]
    for g,label in graphDict.values():
        graphs.append(g)
        classes.append(label)
    
    return graphs, classes 


seed = 0
npRng = np.random.default_rng(seed)
print("Loading...")
parser = Letter.parser
IAMreadergraph = partial(IAMreader,parser)
path = "/home/luca/Documenti/Progetti/E-ABC_v2/eabc_v2/Datasets/IAM/Letter3/"
name = "LetterH"
data1 = graph_nxDataset(path+"Training/", name, reader = IAMreadergraph,seed=npRng)[:100]

#Removed not connected graph and null graph!
cleanData=[]
for g,idx,label in zip(data1.data,data1.indices,data1.labels):
    if not nx.is_empty(g):
        if nx.is_connected(g):
            cleanData.append((g,idx,label)) 

#Test extr indices
data1 = data1.shuffle()

Dissimilarity = Letter.LETTERdiss()
graphDist = BMF(Dissimilarity.nodeDissimilarity,Dissimilarity.edgeDissimilarity)

extract_func = randomwalk_restart.extr_strategy(seed = seed)
subgraph_extr = Extractor(extract_func,seed = seed)

print("Extracting...")
subgraphs = subgraph_extr.randomExtractDataset(data1, 5000)
expSubSet = subgraph_extr.decomposeGraphDataset(data1,maxOrder = 5)

Repr= Medoid

print("Granulating...")
granulationStrategy = BsasBinarySearch(graphDist,Repr,0.1)
granulationStrategy.granulate(subgraphs)

print("Embedding...")

embeddingStrategy = SymbolicHistogram(graphDist) #Parallel
embeddingStrategySingle = SymbolicHistogram(graphDist,isParallel=False)

start = time.process_time()
embeddingStrategy.getSet(expSubSet, granulationStrategy.symbols)
print(time.process_time() - start)

embeddingStrategyParallel = SymbolicHistogram(graphDist,isParallel=True)
start = time.process_time()
embeddingStrategySingle.getSet(expSubSet, granulationStrategy.symbols)
print(time.process_time() - start)
