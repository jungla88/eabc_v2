# -*- coding: utf-8 -*-

from Datasets.tudataset import datasets,reader
from eabc.datasets import graph_nxDataset

#For graphs
from Datasets.tudataset import datasets,reader
from eabc.datasets import graph_nxDataset
import itertools

datasets.get_dataset("Letter-high")

#Trivial transformation for Graph_nx data
class gTransfItem():    
    #-Receive Data Object
    #-Must return transformed Data Object
    def __call__(self,data):
        
        g= data.x
        v =list(itertools.chain.from_iterable([g.nodes[n]['attributes'] for n in g.nodes]))
        if not v:
            print("error")

        try:
            t = max(v)
        except:#no attr in some graphs !?
            pass

        for n in g.nodes:
            g.nodes[n]['attributes'] =  [ x/t for x in g.nodes[n]['attributes'] ]
            
        return data

def readergraph(path):
    graphs_nx = reader.tud_to_networkx("Letter-high")
    classes = [g.graph['classes'] for g in graphs_nx]
    return graphs_nx, classes 

data1 = graph_nxDataset("/home/luca/Documenti/Progetti/E-ABC_v2/eabc_v2/Datasets/tudataset/Letter-high", "Letter-h", readergraph)

#download from TUDataset repository
#TODO: will be change
datasets.get_dataset("Letter-high")
gdata1 = graph_nxDataset("/home/luca/Documenti/Progetti/E-ABC_v2/eabc_v2/Datasets/tudataset/Letter-high", "Letter-h", readergraph, pre_transform=gTransfItem())
gdata2 = graph_nxDataset("/home/luca/Documenti/Progetti/E-ABC_v2/eabc_v2/Datasets/tudataset/Letter-high", "Letter-h", readergraph, transform = gTransfItem())
gdata3 = graph_nxDataset("/home/luca/Documenti/Progetti/E-ABC_v2/eabc_v2/Datasets/tudataset/Letter-high", "Letter-h", readergraph)

print("original data get item: ", gdata3[0].x.nodes[0]['attributes'])
print("transform data get item: ", gdata2[0].x.nodes[0]['attributes'])
print("pre_transformed data get item:", gdata1[0].x.nodes[0]['attributes'])

print("original data stored value: ", gdata3.data[0].nodes[0]['attributes'])
print("transform data stored value: ", gdata2.data[0].nodes[0]['attributes'])
print("pre_transformed data stored value: ", gdata1.data[0].nodes[0]['attributes'])

