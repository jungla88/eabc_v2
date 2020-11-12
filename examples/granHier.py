# -*- coding: utf-8 -*-

#For vectors
from pandas import read_csv
from eabc.datasets import vectorDataset
import numpy as np

from eabc.granulators import HierchicalAggl
from eabc.dissimilarities import scipyMetrics

from eabc.representatives import medoid

from sklearn.preprocessing import MinMaxScaler


def readervector(path):
    frame = read_csv(path, sep=",", header=None,names= ["sepal length", "sepal width" , "petal length", "petal width", "class"])
#    yield frame.iloc[:,0:4].values, frame.iloc[:,4].values 
    return frame.iloc[:,0:4].values, frame.iloc[:,4].values 

class norm01:
    
    def __init__(self,scaler):
        
        self.scaler = scaler
    
    def __call__(self,data):
        
        return self.scaler.transform(data.x.reshape(1,-1)).reshape((len(data),))
        

sqEucl = scipyMetrics('sqeuclidean')
granulationStrategy= HierchicalAggl(sqEucl,medoid.medoid)


data3 = vectorDataset("/home/luca/Documenti/Progetti/E-ABC_v2/eabc_v2/Datasets/iris/iris_data.txt", "iris", readervector)
minmax = MinMaxScaler()
minmax.fit(data3.data) 
normalizer= norm01(minmax)

data3.transform = normalizer
data3[0]
print(data3.data)

l = []
for i,item in enumerate(data3):
    l.append(data3[i])

granulationStrategy.granulate(np.asarray(l))