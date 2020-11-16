# -*- coding: utf-8 -*-

#For vectors
from pandas import read_csv
import numpy as np

from eabc.datasets import vectorDataset
from eabc.granulators import HierchicalAggl
from eabc.dissimilarities import scipyMetrics
from eabc.representatives import medoid

from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

from sortedcontainers import SortedList

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

X, y = make_blobs(n_samples=100, centers=3, n_features=2,  random_state=0, cluster_std=0.5)


minmax = MinMaxScaler()
minmax.fit(X) 


normalizer= norm01(minmax)
data1 = vectorDataset([X,y], "Blob",pre_transform=normalizer)

sqEucl = scipyMetrics('euclidean')
granulationStrategy= HierchicalAggl(sqEucl,medoid.medoid)

granulationStrategy.granulate(data1)

simpleList = granulationStrategy.symbols

simpleList[1].quality = 1
sortedSymbols = SortedList()
sortedSymbols.update(simpleList)


for x,y in zip(reversed(sortedSymbols),reversed(simpleList)):
    print("Sorted {} - Simple {}".format(x.quality, y.quality))

simpleList.sort()
for x,y in zip(reversed(sortedSymbols),reversed(simpleList)):
    print("Sorted {} - Simple {}".format(x.quality, y.quality))

# l = np.asarray(l)
# plt.scatter(l[:,0],l[:,1])
# plt.scatter(x[:,0],x[:,1],marker = 'x')
# plt.show()

# plt.scatter(X[:,0],X[:,1])
# plt.show()