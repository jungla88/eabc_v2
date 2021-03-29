#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 14:56:16 2021

@author: luca
"""

import numpy as np
import math

class eabc_modelGen:
    
    def __init__(self,k=5,l=5,CROSS_PR=0.33,UNION_PR=0.33,RANDOM_PR=0.33, SWAP_PR=0.15,seed=None):
        
        self._K = k #Num random model to create 
        self._L = l #Num model created from previous K model by recombination
        self._maxLen = 100;
        
        self._CROSS_PR = CROSS_PR
        self._UNION_PR = UNION_PR
        self._SWAP_PR = SWAP_PR
        
        
        self._rng = np.random.default_rng(seed) 

        self._models = []

    #Input must be a dict with key=class and value = a list of symbols
    def createFromSymbols(self,alphabet):
        
        self.__alphabet_sanity_check(alphabet)        

        #Try to fix model length
        N = self._estimateNclassSymbols(alphabet)
        ##
        
        models = []
        
        
        #Choose K model
        for _ in range(self._K):
            
            model =  np.array([])
            #For each class alphabet extract some symbols
            for class_ in alphabet.keys():
                              
#                    p = self._extractProb(alphabet[class_])        

                #Set model cardinality
                if N > len(alphabet[class_]):
                    N = len(alphabet[class_])

                if N>1: 
                    N_symb = self._rng.integers(low=1,high= N)
                else:
                    N_symb = 1
                

                #Extract with uniform probability
                thisModel = self._rng.choice(alphabet[class_],size= N_symb,replace=False)
                
                model = np.concatenate((model,thisModel))

            models.append(model)
    

        
        return models
    
    def createFromModels(self,models):
        
        models_ = list()
        
        for _ in range(self._L):
            
            choice = self._rng.random()
            
            if choice<self._CROSS_PR:
                                            
                m1_indices,m2_indices = self._rng.choice(len(models), size = 2, replace = False)     
                m1 = models[m1_indices]
                m2 = models[m2_indices]
                
                size = min(len(m1), len(m2))
                cxpoint = self._rng.integers(low = 1 ,high = size)

                m3 = np.concatenate((m1[:cxpoint],m2[cxpoint:]))
                
                m3 = np.asarray(list(set(m3)),dtype = object)
                models_.append(m3)
                
            elif self._CROSS_PR  <= choice < self._UNION_PR + self._CROSS_PR:
                
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

        p = np.asarray(q)/overallQ

        if np.isnan(p).all() or np.all(p==0):
            p = None
        
        
        return p
    
    def _estimateNclassSymbols(self,alphabet):
        
        n_classes = len(alphabet.keys())
        
        return math.ceil(self._maxLen/n_classes)  
        
    
    @staticmethod
    def __alphabet_sanity_check(alphabet):
        
        if isinstance(alphabet,dict):
            
            for class_,value in alphabet.items():
                
                if not isinstance(value,(list,np.ndarray)):
                    
                    raise TypeError('class alphabet must be list or ndarray not {}'.format(type(value)))
                
                else:
                    
                    if len(value)==0:
                        
                        raise ValueError('{} class alphabet is empty'.format(class_))
                
        else:
            raise TypeError('Alphabet must be a dict not {}'.format(type(alphabet)))
                            
        
        