# -*- coding: utf-8 -*-

#from eabc.granulators import Granule

class Granulator:
    
    def __init__(self, symb_thr = 0.5, eta = 0.5):
    
        self._Fsym_threshold = symb_thr;
        
        self._Fweight = eta
        
        self._symbols = []
    
    @property
    def symbol_thr(self):
        return self._Fsym_threshold
    @symbol_thr.setter
    def symbol_thr(self,val):
        if val > 0 and val <=1:
            self._sym_threshold = val
        else:
            raise ValueError

    @property
    def eta(self):
        return self._Fweight
    @symbol_thr.setter
    def eta(self,val):
        if val > 0 and val <=1:
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
        if 0<normComp<1 and 0<normCard<1:
             F = self._Fweight*normCard + (1-self._Fweight)*normComp
        else:
             raise ValueError
        return F
        #     print("Warning, Invalid values for F evaluation: cardinality =  {}, compactness = {}"
        #           .format(self._Cardinality,self._Compactness))
        #     F = float('nan')        
        # return F
        
        