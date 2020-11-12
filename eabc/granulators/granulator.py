# -*- coding: utf-8 -*-

#from eabc.granulators import Granule

class Granulator:
    
    def __init__(self, symb_thr = 0.5):
    
        self._Fsym_threshold = symb_thr;
        
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
    def symbols(self):
        return self._symbols

    def granulate(self):
        raise NotImplementedError
        
    def __addSymbol(self,granule):
        if granule.Fvalue > self._Fsym_threshold:
            self._symbols.append(granule)