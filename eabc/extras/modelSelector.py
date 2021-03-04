#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 14:56:16 2021

@author: luca
"""

import numpy as np

class eabc_modelGen:
    
    def __init__(self,k=5,l=5,INTERSEC_PR=0.33,UNION_PR=0.33,RANDOM_PR=0.33, SWAP_PR=0.15,seed=None):
        
        self._K = k #Num random model to create 
        self._L = l #Num model created from previous K model by recombination
        self._INTERSEC_PR = INTERSEC_PR
        self._UNION_PR = UNION_PR
        self._SWAP_PR = SWAP_PR
        
        
        self._rng = np.random.default_rng(seed) 

        self._models = []

    #Input must be a dict with key=class and value = a list of symbols
    def createFromSymbols(self,alphabet):
        
        #All symbols regardless their class
        mergedAlphabets = sum(alphabet.values(),[])

        #Choose at least a model with 2 symbol for convenience
        if len(mergedAlphabets)>2:
                        
            models = []
            #Choose K model
            for _ in range(self._K):
                
                model =  np.array([])
                #For each class alphabet extract some symbols
                for class_ in alphabet.keys():
                    
                    p = self._extractProb(alphabet[class_])        

                    #Set model cardinality
                    N = self._rng.integers(low=1,high= len(alphabet[class_]))
                    
                    #TODO: better way to handle exception
                    if len(np.where(p==0))<N and p is not None:
                        highAdmittable = len(p)-len(np.where(p==0)[0])
                        if highAdmittable>1:
                            N = self._rng.integers(low=1,high=highAdmittable )
                        else:
                            N=1
                            
                    thisModel = self._rng.choice(alphabet[class_],size= N,replace=False,p=p)
                    
                    model = np.concatenate((model,thisModel))

                models.append(model)

            #process the symbols with null quality
            p = self._extractProb(mergedAlphabets)    
            nullSymbolsIndices = np.where(p==0)[0]
            
            nullSymbols= np.asarray(mergedAlphabets,dtype=object)[nullSymbolsIndices]
            
            for item in nullSymbols:
                if self._rng.random()<=0.5: #Flip a coin
                    modelIndex = self._rng.choice(len(models))
                    np.concatenate((models[modelIndex],np.reshape(item,(1,))))
        else:
            models = []
            print("Warning in modelSelector")
        
        return models
    
    
    def createFromModels(self,models):
        
        models_ = list()
        
        for _ in range(self._L):
            
            choice = self._rng.random()
            
            if choice<self._INTERSEC_PR:
                                            
                m1_indices,m2_indices = self._rng.choice(len(models), size = 2, replace = False)     
                m1 = models[m1_indices]
                m2 = models[m2_indices]
                
                size = min(len(m1), len(m2))
                cxpoint = self._rng.integers(low = 1 ,high = size)

                m3 = np.concatenate((m1[:cxpoint],m2[cxpoint:]))
                
                m3 = np.asarray(list(set(m3)),dtype = object)
                models_.append(m3)
                
            elif self._INTERSEC_PR  <= choice < self._UNION_PR + self._INTERSEC_PR:
                
                m1_indices,m2_indices = self._rng.choice(len(models), size = 2, replace = False)     
                
                m1 = set(models[m1_indices])
                m2 = set(models[m2_indices])
                
                m3 = np.asarray(list(m1.union(m2)),dtype = object)
                
                models_.append(m3)                
                
                
            else:
                
                #Model to change
                m_index = self._rng.choice(len(models))
                m = models[m_index]
                
                for i in range(len(m)):
                    
                    if self._rng.random()<self._SWAP_PR:
                        
                        #random choice a model to pick symbol
                        m1_index = self._rng.choice(len(models))
                        m1 = models[m1_index]                        
                        #random choice a symbol to swap
                        s_idx = self._rng.choice(len(m1))
                        s = m1[s_idx]
                        
                        m[i] = s
                
                models_.append(m)
        
        return models_
                
    def _extractProb(self,alphabet):
        
        q = [sym.quality if sym.quality > 0 else 0 for sym in alphabet]
        overallQ = sum(q)
        
        with np.errstate(divide='ignore'):
            p = np.asarray(q)/overallQ
        
        if np.isnan(p).all() or np.all(p==0):
            p = None
        
        
        return p