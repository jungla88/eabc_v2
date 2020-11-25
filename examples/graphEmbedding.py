# -*- coding: utf-8 -*-

from Datasets.tudataset import datasets,reader
from eabc.datasets import graph_nxDataset
#from eabc.dissimilarities import BMF
from eabc.dissimilarities import newBMF
from eabc.extractors import Extractor
from eabc.extractors import randomwalk_restart
from eabc.granulators import BsasBinarySearch
#from eabc.representatives import Medoid
from eabc.representatives import newMedoid
from eabc.embeddings import SymbolicHistogram 
import networkx as nx
import numpy as np

def nodeDissimilarity(a, b):
        D = 0
        if(a['labels'] != b['labels']):
            D = 1
        return D

def edgeDissimilarity(a, b):
        D = 0
        if(a['labels'] != b['labels']):
            D = 1
        return D
    
def readergraph(path):
    graphs_nx = reader.tud_to_networkx("Mutagenicity")
    classes = [g.graph['classes'] for g in graphs_nx]
    return graphs_nx, classes 

print("Loading...")
data1 = graph_nxDataset("/home/luca/Documenti/Progetti/E-ABC_v2/eabc_v2/Datasets/tudataset/Mutagenicity", "Mutagenicity", readergraph)
#Removed not connected graph!
cleanData = [(g,idx,label) for g,idx,label in zip(data1.data,data1.indices,data1.labels) if nx.is_connected(g)]
cleanData = np.asarray(cleanData,dtype=object)
data1 = graph_nxDataset([cleanData[:,0],cleanData[:,2]],"Mutagenicity")
data1= data1[0:100]

#Test extr indices
data1 = data1.shuffle()

graphDist = newBMF(nodeDissimilarity,edgeDissimilarity)

strat = randomwalk_restart.extr_strategy(max_order=6)
subgraph_extr = Extractor(strat)

print("Extracting...")
subgraphs = subgraph_extr.randomExtractDataset(data1, 100)

Repr= newMedoid

print("Granulating...")
granulationStrategy = BsasBinarySearch(graphDist,Repr,0.1)
granulationStrategy.granulate(subgraphs)

print("Embedding...")
expSet = subgraphs.fresh_dpcopy()
for i,x in enumerate(data1):
    for j in range(100):
        expSet.add_keyVal(data1.to_key(i),subgraph_extr.extract(x))

embeddingStrategy = SymbolicHistogram(graphDist)
embeddingStrategy.getSet(expSet, granulationStrategy.symbols)

