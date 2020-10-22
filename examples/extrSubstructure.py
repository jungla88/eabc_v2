# -*- coding: utf-8 -*-

#For graphs
from Datasets.tudataset import datasets,reader
from eabc.datasets import graph_nxDataset
from eabc.extractors import Extractor
from eabc.extractors import randomwalk_restart

def readergraph(path):
    graphs_nx = reader.tud_to_networkx("COIL-RAG")
    classes = [g.graph['classes'] for g in graphs_nx]
    return graphs_nx, classes 

datasets.get_dataset("COIL-RAG")
gdata3 = graph_nxDataset("/home/luca/Documenti/Progetti/E-ABC_v2/eabc_v2/Datasets/tudataset/COIL-RAG", "Letter-h", readergraph)

strat = randomwalk_restart.extr_strategy(max_order=10)
subgraph_extr = Extractor(strat)
#subgraph_extr = Extractor(sampler_func)

smallD = gdata3[1:10]

subgraphs = subgraph_extr.randomExtractDataset(smallD, 50 )