# -*- coding: utf-8 -*-

'''

'''

from Dataset import Dataset
from vectorData import vectorData

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
            data = vectorData()
            data.x = x
            data.y = y
            self.data.append(data)
        

    def __repr__(self):
        return '{}{}()'.format(self.__class__.__name__, self.name.capitalize())