# -*- coding: utf-8 -*-

#For vectors
from pandas import read_csv
import numpy as np

from eabc.datasets import vectorDataset
from eabc.granulators import HierchicalAggl
from eabc.dissimilarities import scipyMetrics

from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

from eabc.granulators import BsasBinarySearch
from eabc.representatives import newMedoid

import matplotlib.pyplot as plt

#For graphs
from Datasets.tudataset import datasets,reader
from eabc.datasets import graph_nxDataset
from eabc.dissimilarities import newBMF


def readervector(path):
    frame = read_csv(path, sep=",", header=None,names= ["sepal length", "sepal width" , "petal length", "petal width", "class"])
#    yield frame.iloc[:,0:4].values, frame.iloc[:,4].values 
    return frame.iloc[:,0:4].values, frame.iloc[:,4].values 

class norm01:
    
    def __init__(self,scaler):
        
        self.scaler = scaler
    
    def __call__(self,data):
        
        data.x = self.scaler.transform(data.x.reshape(1,-1)).reshape((len(data),))
      
        return data


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


Repr = Medoid

datasets.get_dataset("Mutagenicity")
data1 = graph_nxDataset("/home/luca/Documenti/Progetti/E-ABC_v2/eabc_v2/Datasets/tudataset/Mutagenicity", "Mutagenicity", readergraph)
graphDist = BMF(nodeDissimilarity,edgeDissimilarity)
smallData = data1[1:50]
granulationStrategy = BsasBinarySearch(graphDist,Repr,0.1)
p = granulationStrategy.granulate(smallData)


X, y = make_blobs(n_samples=100, centers=3, n_features=2,  random_state=0, cluster_std=0.5)
minmax = MinMaxScaler()
minmax.fit(X) 
normalizer= norm01(minmax)
dataVect = vectorDataset([X,y], "Blob",pre_transform=normalizer)
Eucl = scipyMetrics('euclidean')

#granulationStrategy= HierchicalAggl(Eucl,Repr)
#granulationStrategy.granulate(dataVect)

granulationStrategy = BsasBinarySearch(Eucl,Repr,0.1)
granulationStrategy.granulate(dataVect)

medoid = granulationStrategy.symbols

x = [m.representative for m in medoid]
x = np.asarray(x)

plt.scatter(dataVect.data[:,0],dataVect.data[:,1])
plt.scatter(x[:,0],x[:,1])
plt.show()