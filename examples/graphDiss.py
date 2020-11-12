# -*- coding: utf-8 -*-

from Datasets.tudataset import datasets,reader
from eabc.datasets import graph_nxDataset

import itertools


from eabc.extractors import Extractor
from eabc.extractors import randomwalk_restart
from eabc.dissimilarities import GED


# datasets.get_dataset("Mutagenicity")



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

data1 = graph_nxDataset("/home/luca/Documenti/Progetti/E-ABC_v2/eabc_v2/Datasets/tudataset/Mutagenicity", "Mutagenicity", readergraph)


strat = randomwalk_restart.extr_strategy(max_order=10)
subgraph_extr = Extractor(strat)

smallD = data1[1:10]

subgraphs = subgraph_extr.randomExtractDataset(smallD, 50 )
distance_function = GED.BMF(nodeDissimilarity, edgeDissimilarity)

x = distance_function(subgraphs[0].x,subgraphs[1].x)

#M = distance_function.pdist(subgraphs.data,subgraphs.data)