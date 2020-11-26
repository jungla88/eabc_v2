# -*- coding: utf-8 -*-

from eabc.datasets import Dataset
from eabc.data import Graph_nx

class graph_nxDataset(Dataset):
    
    def __init__(self, targetObject, name, idx=None, reader= None, transform = None, pre_transform = None):
        
        self.name = name
        self.reader = reader
        self.tObject = targetObject
        self.transform = transform;
        self.pre_transform = pre_transform
        self.idx = idx
        
        super(graph_nxDataset,self).__init__(self.tObject, idx = self.idx, transform = self.transform, pre_transform=self.pre_transform)
        
    def process(self):
      
        if (self.tObject and self.reader):
            examples,classes =self.reader(self.tObject)
        else:
            #TODO: two lists or arrays with examples and classes as input. Extend to other data structures
            examples, classes = self.tObject[0],self.tObject[1]
            
        reader_out =zip(examples,classes)
        for x,y in reader_out:
            data = Graph_nx()
            data.x = x
            data.y = y
            self._data.append(data)
              

    def add_keyVal(self,idx,data):
        if isinstance(data,Graph_nx):
            self._data.append(data)
            self._indices.append(idx)
        else:
            raise ValueError("Invalid data inserted")
    
    def __repr__(self):
        return '{}{}()'.format(self.__class__.__name__, self.name.capitalize())