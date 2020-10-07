# import pandas as pd
# from sklearn.preprocessing import MinMaxScaler

# class Extractor:
#     def Extract(self, name):
#         data=pd.read_csv(name, sep=",", header=None,names= ["sepal length", "sepal width" , "petal length", "petal width", "class"])
#         scaler = MinMaxScaler()
#         data[['sepal length','sepal width' , 'petal length', 'petal width']] = scaler.fit_transform(data[['sepal length','sepal width' , 'petal length', 'petal width']]) #applico min max scaling
#         data=data.sample(frac=0.5) #frac = fraction of rows to return in the random sample
#         data=data.iloc[:, 1:3]
#         sample = data.values.tolist()
#         return sample # sample = list

class Extractor(object):
    
    def __init__(self,dataset):
        pass
        
    def request(self):
        raise NotImplementedError