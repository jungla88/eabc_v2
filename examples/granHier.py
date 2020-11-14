# -*- coding: utf-8 -*-

#For vectors
from pandas import read_csv
from eabc.datasets import vectorDataset
import numpy as np

from eabc.granulators import HierchicalAggl
from eabc.dissimilarities import scipyMetrics

from eabc.representatives import medoid

from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs

def readervector(path):
    frame = read_csv(path, sep=",", header=None,names= ["sepal length", "sepal width" , "petal length", "petal width", "class"])
#    yield frame.iloc[:,0:4].values, frame.iloc[:,4].values 
    return frame.iloc[:,0:4].values, frame.iloc[:,4].values 

class norm01:
    
    def __init__(self,scaler):
        
        self.scaler = scaler
    
    def __call__(self,data):
        
        return self.scaler.transform(data.x.reshape(1,-1)).reshape((len(data),))
        

X, y = make_blobs(n_samples=100, centers=3, n_features=2,  random_state=0, cluster_std=0.5)


data1 = vectorDataset([X,y], "Blob")

data2 = vectorDataset("/home/luca/Documenti/Progetti/E-ABC_v2/eabc_v2/Datasets/iris/iris_data.txt", "iris", readervector)
minmax = MinMaxScaler()
minmax.fit(data1.data) 
normalizer= norm01(minmax)

sqEucl = scipyMetrics('sqeuclidean')
granulationStrategy= HierchicalAggl(sqEucl,medoid.medoid)


data1.transform = normalizer

# l = []
# for i,item in enumerate(data1):
#     l.append(data1[i])

granulationStrategy.granulate(data1)

x = np.asarray(granulationStrategy.repr)
    
# l = np.asarray(l)
# plt.scatter(l[:,0],l[:,1])
# plt.scatter(x[:,0],x[:,1],marker = 'x')
# plt.show()

# plt.scatter(X[:,0],X[:,1])
# plt.show()


# plt.scatter(l[:,0],l[:,2])
# plt.scatter(x[:,0],x[:,2],marker = 'x')
# plt.show()

# plt.scatter(l[:,0],l[:,3])
# plt.scatter(x[:,0],x[:,3],marker = 'x')
# plt.show()

# plt.scatter(l[:,1],l[:,2])
# plt.scatter(x[:,1],x[:,2],marker = 'x')
# plt.show()

# plt.scatter(l[:,1],l[:,3])
# plt.scatter(x[:,1],x[:,3],marker = 'x')
# plt.show()