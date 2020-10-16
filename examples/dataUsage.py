from pandas import read_csv
from eabc.datasets import vectorDataset
import numpy 

#Reader must take a path as argument. Might be a file with raw data or a folder with multiple files to read
#Return two iterables:
#1- x examples such that elements in x define the basic datastructure
#2- y labels defined ground truth. No encoding will apply later
# All element will be converted in Data type object
def reader(path):
    frame = read_csv(path, sep=",", header=None,names= ["sepal length", "sepal width" , "petal length", "petal width", "class"])
#    yield frame.iloc[:,0:4].values, frame.iloc[:,4].values 
    return frame.iloc[:,0:4].values, frame.iloc[:,4].values 

class TransfItem():    
    #-Receive Data Object
    #-Must return transformed Data Object
    def __call__(self,data):
        
        t = max(data.x)
        data.x = data.x / t
                    
        return data

data1 = vectorDataset("/home/luca/Documenti/Progetti/E-ABC_v2/eabc_v2/Datasets/data/iris_data.txt", "iris", reader,pre_transform=TransfItem())
data2 = vectorDataset("/home/luca/Documenti/Progetti/E-ABC_v2/eabc_v2/Datasets/data/iris_data.txt", "iris", reader, transform=TransfItem())
data3 = vectorDataset("/home/luca/Documenti/Progetti/E-ABC_v2/eabc_v2/Datasets/data/iris_data.txt", "iris", reader)


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