# -*- coding: utf-8 -*-
import numpy as np

class BSAS:

    def __init__(self,Representative, Dissimilarity, theta = 0.5, Q=100):

        self.Representative = Representative
        self.Dissimilarity = Dissimilarity
        self.theta = theta
        self.Q = Q

    def process(self,Dataset):
    
        isAssigned = [False] * len(Dataset)
        clusters = []    
        
        #init
        representatives = [self.Representative(Dataset[0])]
        isAssigned[0] = True
        clusters.append([0])
        
        
        for i in range(1,len(Dataset)):
            
            p = Dataset[i]
            
            patternToRepDis = [self.Dissimilarity(x._representativeElem, p) for x in representatives]
            nearestClust = np.argmin(patternToRepDis)
            
            if patternToRepDis[nearestClust] > self.theta and len(representatives)< self.Q:           
                representatives.append(self.Representative(p))
                
                clusters.append([i])
                
                isAssigned[i] = True
                
        for i in range(len(Dataset)):
            
            p = Dataset[i]
            
            if not isAssigned[i]:
                
                patternToRepDis = [self.Dissimilarity(x._representativeElem, p) for x in representatives]
                nearestClust = np.argmin(patternToRepDis)
                clusters[nearestClust].append(i)
                
                representatives[nearestClust].evaluate([Dataset[x] for x in clusters[nearestClust]],self.Dissimilarity)            
                
        return clusters, representatives
