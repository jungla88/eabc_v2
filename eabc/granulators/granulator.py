# -*- coding: utf-8 -*-

#from eabc.granulators import Granule

class Granulator:
    
    def __init__(self, symb_thr = 0.5, eta = 0.5):
    
        self._Fsym_threshold = symb_thr;
        
        self._Fweight = eta
        
        self._symbols = []
        
        ##TEST global norm for comp and card
        self._compNormFactors=[0,1]
        self._cardNormFactors=[0,1]
        ##
    
    @property
    def compMin(self):
        return self._compNormFactors[0]
    @compMin.setter
    def compMin(self,val):
        if val>=0 and val<=1:
            self._compNormFactors[0]=val
    @property
    def compMax(self):
        return self._compNormFactors[1]
    @compMax.setter
    def compMax(self,val):
        if val>=0 and val<=1:
            self._compNormFactors[1]=val
    @property
    def cardMin(self):
        return self._cardNormFactors[0]
    @cardMin.setter
    def cardMin(self,val):
        if val>=0 and val<=1:
            self._cardNormFactors[0]=val

    @property
    def cardMax(self):
        return self._cardNormFactors[1]
    @cardMax.setter
    def cardMax(self,val):
        if val>=0 and val<=1:
            self._cardNormFactors[1]=val
            
    @property
    def symbol_thr(self):
        return self._Fsym_threshold
    @symbol_thr.setter
    def symbol_thr(self,val):
        if val > 0 and val <=1:
            self._Fsym_threshold = val
        else:
            raise ValueError

    @property
    def eta(self):
        return self._Fweight
    @eta.setter
    def eta(self,val):
        if val >= 0 and val <=1:
            self._Fweight = val
        else:
            raise ValueError            
    
    @property
    def symbols(self):
        return self._symbols

    def granulate(self,data):
        raise NotImplementedError
        
    def _addSymbol(self,granule):
        if granule.Fvalue < self._Fsym_threshold:
            self._symbols.append(granule)
          
    #        
    def _evaluateF(self, normComp, normCard):
        #*0.1 normCard manually scale card for being comparable with comp
        # if 0<=normComp<=1 and 0<normCard<=1:
        #      #F = self._Fweight*normCard*0.1 + (1-self._Fweight)*normComp
        #      F = self._Fweight*normCard + (1-self._Fweight)*normComp
        # else:
        #      raise ValueError
        # return F
        
        #TEST: testing normalization for aggregate of clusters where card=0 and comp=0 are feasible
        F = self._Fweight*normCard + (1-self._Fweight)*normComp
 
        return F

        #     print("Warning, Invalid values for F evaluation: cardinality =  {}, compactness = {}"
        #           .format(self._Cardinality,self._Compactness))
        #     F = float('nan')        
        # return F
        
    def _removeSingularity(self,clustersLabels,reprElems,Dataset):
        
        try:
            clustersLabels,reprElems = zip(*filter(lambda x: not(len(x[0])==1 or
                                               len(x[0])/len(Dataset.data)==1 or x[1]._SOD==0), zip(clustersLabels,reprElems))) 
        #No elements to unpack, all clusters are discarded
        except ValueError:
            clustersLabels = []
            reprElems = []
            
        return clustersLabels,reprElems