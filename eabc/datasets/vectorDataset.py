# -*- coding: utf-8 -*-

from eabc.datasets import Dataset
from eabc.data import Vector

class vectorDataset(Dataset):
    
    def __init__(self, path, name, reader, transform = None, pre_transform = None):
        
        self.name = name
        self.reader = reader
        self.path = path
        self.transform = transform;
        self.pre_transform = pre_transform
        
        super(vectorDataset,self).__init__(self.path, self.transform, self.pre_transform)
        
    def process(self):
        examples,classes =self.reader(self.path)
        reader_out =zip(examples,classes)
        for x,y in reader_out:
            data = Vector()
            data.x = x
            data.y = y
            self._data.append(data)
        
    """
    Return all data examples in numpy array object
    """    
    @Dataset.data.getter
    def data(self):
        return list(map(lambda x: x.x, self._data))
        

    def __repr__(self):
        return '{}{}()'.format(self.__class__.__name__, self.name.capitalize())