#For vectors
from pandas import read_csv
from eabc.datasets import vectorDataset
import numpy 

#For graphs
from Datasets.tudataset import datasets,reader
from eabc.datasets import graph_nxDataset
import itertools

#Reader must take a path as argument. Might be a file with raw data or a folder with multiple files to read
#Return two iterables:
#1- x examples such that elements in x define the basic datastructure
#2- y labels defined ground truth. No encoding will apply later
# All element will be converted in Data type object
def readervector(path):
    frame = read_csv(path, sep=",", header=None,names= ["sepal length", "sepal width" , "petal length", "petal width", "class"])
#    yield frame.iloc[:,0:4].values, frame.iloc[:,4].values 
    return frame.iloc[:,0:4].values, frame.iloc[:,4].values 

def readergraph(path):
    graphs_nx = reader.tud_to_networkx("COIL-RAG")
    classes = [g.graph['classes'] for g in graphs_nx]
    return graphs_nx, classes 


#Trivial transformation for Vector data
class TransfItem():    
    #-Receive Data Object
    #-Must return transformed Data Object
    def __call__(self,data):
        
        t = max(data.x)
        data.x = data.x / t
                    
        return data

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


data1 = vectorDataset("/home/luca/Documenti/Progetti/E-ABC_v2/eabc_v2/Datasets/iris/iris_data.txt", "iris", readervector,pre_transform=TransfItem())
data2 = vectorDataset("/home/luca/Documenti/Progetti/E-ABC_v2/eabc_v2/Datasets/iris/iris_data.txt", "iris", readervector, transform=TransfItem())
data3 = vectorDataset("/home/luca/Documenti/Progetti/E-ABC_v2/eabc_v2/Datasets/iris/iris_data.txt", "iris", readervector)


print("original data get item: ", data3[0].x)
print("transform data get item: ", data2[0].x)
print("pre_transformed data get item:", data1[0].x)

print("original data stored value: ", data3.data[0])
print("transform data stored value: ", data2.data[0])
print("pre_transformed data stored value: ", data1.data[0])


####
data1_perm = data1.shuffle()
data0_key = data1_perm.indices[0]
print(any(data1[data0_key].x == data1_perm[0].x))

data1_slice= data1[[1,2,4,5,10,40,149,0]].shuffle()
for key,i in zip(data1_slice.indices,range(len(data1_slice.indices))):
    print(any(data1[key].x==data1_slice[i].x))
    
print(any(data1[0].x == data1_slice[0].x))
###

#Es. create non overlapping set
s = data3.shuffle()
s1 = data3[0:50]
s2 = data3[50:100]
s3 = data3[100:]

rawDat_s1,rawLabels_s1 = [s1.data,s1.labels]
rawDat_s2,rawLabels_s2 = [s2.data,s2.labels]

#GRAPH

#download from TUDataset repository
#TODO: will be change
datasets.get_dataset("COIL-RAG")
gdata1 = graph_nxDataset("/home/luca/Documenti/Progetti/E-ABC_v2/eabc_v2/Datasets/tudataset/COIL-RAG", "Letter-h", readergraph, pre_transform=gTransfItem())
gdata2 = graph_nxDataset("/home/luca/Documenti/Progetti/E-ABC_v2/eabc_v2/Datasets/tudataset/COIL-RAG", "Letter-h", readergraph, transform = gTransfItem())
gdata3 = graph_nxDataset("/home/luca/Documenti/Progetti/E-ABC_v2/eabc_v2/Datasets/tudataset/COIL-RAG", "Letter-h", readergraph)

print("original data get item: ", gdata3[0].x.nodes[0]['attributes'])
print("transform data get item: ", gdata2[0].x.nodes[0]['attributes'])
print("pre_transformed data get item:", gdata1[0].x.nodes[0]['attributes'])

print("original data stored value: ", gdata3.data[0].nodes[0]['attributes'])
print("transform data stored value: ", gdata2.data[0].nodes[0]['attributes'])
print("pre_transformed data stored value: ", gdata1.data[0].nodes[0]['attributes'])


# ####
# data1_perm = data1.shuffle()
# data0_key = data1_perm.indices[0]
# print(any(data1[data0_key].x == data1_perm[0].x))

# data1_slice= data1[[1,2,4,5,10,40,149,0]].shuffle()
# for key,i in zip(data1_slice.indices,range(len(data1_slice.indices))):
#     print(any(data1[key].x==data1_slice[i].x))
    
# print(any(data1[0].x == data1_slice[0].x))
# ###