#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 13:04:22 2020

@author: luca
"""
import networkx as nx
import numpy as np

from Datasets.tudataset import datasets,reader
from eabc.datasets import graph_nxDataset
from eabc.dissimilarities import newBMF

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
#not connected graph
cleanData = [(g,idx,label) for g,idx,label in zip(data1.data,data1.indices,data1.labels) if nx.is_connected(g)]
cleanData = np.asarray(cleanData,dtype=object)
data1 = graph_nxDataset([cleanData[:,0],cleanData[:,2]],"Mutagenicity")
data1= data1[0:10]

graphDist = newBMF(nodeDissimilarity,edgeDissimilarity)
x = graphDist.pdist(data1.data,forceSym = True)