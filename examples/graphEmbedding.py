# -*- coding: utf-8 -*-

from Datasets.tudataset import datasets,reader
from eabc.datasets import graph_nxDataset
from eabc.dissimilarities import BMF
from eabc.extractors import Extractor
from eabc.extractors import randomwalk_restart
from eabc.granulators import BsasBinarySearch
from eabc.representatives import Medoid
from eabc.embeddings import SymbolicHistogram 
from eabc.granulators import Granule

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
path = "../Datasets/IAM/Letter3/"
name = "LetterH"
data1 = graph_nxDataset(path+"Training/", name, reader = IAMreadergraph,seed=npRng)

#Test extr indices
data1 = data1.shuffle()

Dissimilarity = Letter.LETTERdiss()
graphDist = BMF(Dissimilarity.nodeDissimilarity,Dissimilarity.edgeDissimilarity)

extract_func = randomwalk_restart.extr_strategy(seed = seed)
subgraph_extr = Extractor(extract_func,seed = seed)

print("Extracting...")
expSubSet = subgraph_extr.decomposeGraphDataset(data1,maxOrder = 5)

r = np.arange(100,2000,300)

for x in r:
    
    subgraphs = subgraph_extr.randomExtractDataset(data1, x)
    
    grList = list()
    for s in subgraphs.data:
	    # sr = Medoid(s)
	    newGr = Granule(s,graphDist,0.5,0.5,0.5)
	    grList.append(newGr)

	# Repr = Medoid
	# print("Granulating...")
	# granulationStrategy = BsasBinarySearch(graphDist,Repr,0.1)
	# granulationStrategy.granulate(subgraphs)
    
    print("Embedding with size = {}".format(x))
    embeddingStrategySingle = SymbolicHistogram(graphDist,isParallel=False)
    start = time.process_time()
#  	#embeddingStrategySingle.getSet(expSubSet, granulationStrategy.symbols)
    embeddingStrategySingle.getSet(expSubSet, grList)
    print("Serial = {}".format( time.process_time() - start))
    
    embeddingStrategyParallel = SymbolicHistogram(graphDist,isParallel=True)
    start = time.process_time()
    embeddingStrategyParallel.getSet(expSubSet, grList)
    print("Parallel = {}".format( time.process_time() - start))
    
    singleSet = np.asarray(embeddingStrategySingle.embeddedSet)
    parallelSet = np.asarray(embeddingStrategyParallel.embeddedSet)
    assert(np.all(singleSet==parallelSet))    

    print("####")
