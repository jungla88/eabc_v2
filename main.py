#from Extractor import Extractor
from Granulator import Granulator
from Agent import Agent
from Metric import Metric
from Representative import Representative
from Clustering_MBSAS import Clustering_MBSAS
from Clustering_K_Means import Clustering_K_Means

from extraction_strategies import vectorExtractor

from pandas import read_csv
from Datasets import vectorDataset
from numpy.random import randint

def reader(path):
    frame = read_csv(path, sep=",", header=None,names= ["sepal length", "sepal width" , "petal length", "petal width", "class"])
#    yield frame.iloc[:,0:4].values, frame.iloc[:,4].values 
    return frame.iloc[:,0:4].values, frame.iloc[:,4].values 

#transform must deal with [pattern, class] = [numpy.ndarray(float), numpy.ndarray(object)]
class TransfItem():
    
    def __call__(self,data):
        
        y = data[0]**2
        
        return y

class preTransfItem():
    
    def __call__(self,data):
        
        y = data[0]**2
        
        return y
    
#ext  = vectorExtractor.vectorExtractor(1, 2)
#data1 = vectorDataset.vectorDataset("/home/luca/Documenti/Progetti/E-ABC_v2/eabc_v2/Datasets/data/iris_data.txt", "iris", reader,transform=TransfItem)
data2 = vectorDataset.vectorDataset("/home/luca/Documenti/Progetti/E-ABC_v2/eabc_v2/Datasets/data/iris_data.txt", "iris", reader)
#data1 = vectorDataset.vectorDataset("/home/luca/Documenti/Progetti/E-ABC_v2/eabc_v2/Datasets/data/iris_data.txt", "iris", reader, pre_transform=TransfItem)


# sliceItem1 = data1[1:10].shuffle()
# sliceItem1 = data1[[1,2,50,100]].shuffle()

# count = 0
# for i in range(1,100):
#     sel = randint(0,150, size=randint(2,150)).tolist()
#     x = data1[sel].shuffle()
#     count = count + 1 if x.data == data1[x.indices()].data else count
    
# print(count)
# extractor1 = Extractor()

# obj_clustering_MBSAS = Clustering_MBSAS(3, 0.2, 0.1, 1.1) # Lambda, theta_start ,theta_step, theta_stop
# agent1 = Agent(Granulator, Metric, extractor1, Representative, obj_clustering_MBSAS)
# agent1.execute(3.1,0.5) # S_T, eta

# obj_clustering_K_Means = Clustering_K_Means(1,3) #k, k_max
# agent2 = Agent(Granulator, Metric, extractor1, Representative, obj_clustering_K_Means)
# agent2.execute(3.1,0.5) # S_T,  eta

