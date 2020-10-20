# -*- coding: utf-8 -*-

from littleballoffur import RandomWalkWithRestartSampler

class extr_strategy:
    
    def __init__(self):
        self.sampler = RandomWalkWithRestartSampler()
        
    def __call__(self, data):
        
        subgraph = self.sampler.sample(data.x)
        return subgraph
        
        