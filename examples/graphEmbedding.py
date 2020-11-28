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

import time

#from numba import njit,jitclass

# @jitclass
def nodeDissimilarity(a, b):
    return np.linalg.norm(np.asarray(a['attributes']) - np.asarray(b['attributes'])) / np.sqrt(2)

# @jitclass
def edgeDissimilarity(a,b):
    return 0.0
    
def readergraph(path):
    graphs_nx = reader.tud_to_networkx("Letter-high")
    classes = [g.graph['classes'] for g in graphs_nx]
    return graphs_nx, classes 

print("Loading...")
data1 = graph_nxDataset("/home/luca/Documenti/Progetti/E-ABC_v2/eabc_v2/Datasets/tudataset/Letter-high", "LetterH", reader = readergraph)
#Removed not connected graph and null graph!
cleanData=[]
for g,idx,label in zip(data1.data,data1.indices,data1.labels):
    if not nx.is_empty(g):
        if nx.is_connected(g):
            cleanData.append((g,idx,label)) 
#cleanData = [(g,idx,label) for g,idx,label in zip(data1.data,data1.indices,data1.labels) if not nx.is_empty(g)]

cleanData = np.asarray(cleanData,dtype=object)
data1 = graph_nxDataset([cleanData[:,0],cleanData[:,2]],"Letter")
data1= data1[0:100]

#Test extr indices
data1 = data1.shuffle()

graphDist = BMF(nodeDissimilarity,edgeDissimilarity)

strat = randomwalk_restart.extr_strategy(max_order=6)
subgraph_extr = Extractor(strat)

print("Extracting...")
subgraphs = subgraph_extr.randomExtractDataset(data1, 10000)

Repr= Medoid

print("Granulating...")
granulationStrategy = BsasBinarySearch(graphDist,Repr,0.1)
granulationStrategy.granulate(subgraphs)

print("Embedding...")
expSet = subgraphs.fresh_dpcopy()
for i,x in enumerate(data1):
    for j in range(1000):
        expSet.add_keyVal(data1.to_key(i),subgraph_extr.extract(x))

embeddingStrategy = SymbolicHistogram(graphDist) #Parallel
embeddingStrategySingle = SymbolicHistogram(graphDist,isParallel=False)

start = time.process_time()
embeddingStrategy.getSet(expSet, granulationStrategy.symbols)
print(time.process_time() - start)

embeddingStrategySingle = SymbolicHistogram(graphDist,isParallel=False)
start = time.process_time()
embeddingStrategySingle.getSet(expSet, granulationStrategy.symbols)
print(time.process_time() - start)
