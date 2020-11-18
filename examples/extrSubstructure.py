# -*- coding: utf-8 -*-

#For graphs
from Datasets.tudataset import datasets,reader
from eabc.datasets import graph_nxDataset
from eabc.extractors import Extractor
from eabc.extractors import randomwalk_restart

#For vectors
from pandas import read_csv
from eabc.datasets import vectorDataset
from eabc.extractors import Extractor, features_selection
import numpy 

import copy

def readergraph(path):
    graphs_nx = reader.tud_to_networkx("COIL-RAG")
    classes = [g.graph['classes'] for g in graphs_nx]
    return graphs_nx, classes 

def readervector(path):
    frame = read_csv(path, sep=",", header=None,names= ["sepal length", "sepal width" , "petal length", "petal width", "class"])
    return frame.iloc[:,0:4].values, frame.iloc[:,4].values 

# datasets.get_dataset("COIL-RAG")
# gdata3 = graph_nxDataset("/home/luca/Documenti/Progetti/E-ABC_v2/eabc_v2/Datasets/tudataset/COIL-RAG", "Letter-h", readergraph)

# strat = randomwalk_restart.extr_strategy(max_order=10)
# subgraph_extr = Extractor(strat)

# smallD = gdata3[1:10]

# subgraphs = subgraph_extr.randomExtractDataset(smallD, 50 )
# ##

data3 = vectorDataset("/home/luca/Documenti/Progetti/E-ABC_v2/eabc_v2/Datasets/iris/iris_data.txt", "iris", readervector)

strat = features_selection.extr_strategy()
substruct_extr = Extractor(strat)
smallDat = data3[3,6,8,10,11,16,17,18,20,139,140]
mask = [1,0,0,1]

strat.mask= mask


substruct_Dataset = smallDat.fresh_dpcopy()

for i,p in zip(smallDat.indices,smallDat):
    substruct_Dataset.add_keyVal(i,substruct_extr.extract(p))
