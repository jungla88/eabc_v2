# -*- coding: utf-8 -*-

from Dataset import Dataset

class vectorDataset(Dataset):
    
    def __init__(self, path, name, reader, transform = None, pre_transform = None):
        
        self.name = name
        self.reader = reader
        self.path = path
        self.transform = transform;
        self.pre_transform = pre_transform
        
        super(vectorDataset,self).__init__(self.path, self.transform, self.pre_transform)
        
    def process(self):
        x,y =self.reader(self.path)
        self.data = list(zip(x,y))   
        
    def len(self):
        return len(self.data)
    
    def get(self,idx):
        
        data = self.data[idx]
        return data


    def __repr__(self):
        return '{}{}()'.format(self.__class__.__name__, self.name.capitalize())