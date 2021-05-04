# -*- coding: utf-8 -*-


class Granule:
    
    def __init__(self, Representative,
                 DissimilarityMeasure,
                 Fvalue = None,
                 cardinality = None,
                 avgDispersion = None,
                 Quality = 0,
                 epsilon = 1.1):
                

        self._DissimilarityMeasure = DissimilarityMeasure 
        self._Representative = Representative
       
        self._Quality=Quality
        
        # Allow user to set an F value or internally evaluate it according to card and comp?
        self._Fvalue=Fvalue
        self._Cardinality = cardinality
        self._AvgDispersion = avgDispersion 
        self._epsilon = epsilon
        
        #Added for store agent property
        self._ownerID = None       
        self._classLbl = None
        
        #Better evaluate F externally?
#        self._Fweight = eta



    @property
    def quality(self):
        return self._Quality
    @quality.setter
    def quality(self, val ):
        self._Quality = val

    @property
    def card(self):
        return self._Cardinality
    @card.setter
    def card(self,val):
        if val:
            self._Cardinality = val
        else:
            raise ValueError
            
    @property
    def comp(self):
        return self._AvgDispersion
    @comp.setter
    def comp(self,val):
        if val:
            self._AvgDispersion= val
        else:
            raise ValueError
            
    @property
    def Fvalue(self):
        return self._Fvalue
    @Fvalue.setter
    def Fvalue(self,val):
        if val>0 and val <1:
            self._Fvalue = val
        else:
            raise ValueError
        # if self._Cardinality and self._AvgDispersion:
        #     F = self._Fweight* self._Cardinality + (1-self._Fweight)*self._AvgDispersion
        # else:
        #     #raise ValueError
        #     print("Warning, Invalid values for F evaluation: cardinality =  {}, compactness = {}"
        #           .format(self._Cardinality,self._Compactness))
        #     F = float('nan')        
        # return F

    @property
    def matchThr(self):
        #BUG: what if avgDisp is 0?
#        t = self._epsilon*self._AvgDispersion if self._AvgDispersion else None 
        t = self._epsilon*self._AvgDispersion 
        return t
    
    @property
    def representative(self):
        return self._Representative
    @representative.setter
    def representative(self,obj):
        if obj:
            self._Representative = obj

    @property
    def dissimilarity(self):
        return self._DissimilarityMeasure
    @dissimilarity.setter
    def dissimilarity(self,obj):
        if obj:
            self._DissimilarityMeasure = obj
    @property
    def owner(self):
        return self._ownerID
    @owner.setter
    def owner(self,val):
        self._ownerID = val
        
    #Test
    @property
    def classLbl(self):
        return self._classLbl
    @classLbl.setter
    def classLbl(self,val):
        self._classLbl = val    

    # @property
    # def effectiveRadius(self):
    #     return self._effectiveRadius

    #Induce total ordering for SortedList
    #Equality is left to id(object1) == id(object2)
    def __lt__(self,other):
        return self._Quality < other.quality        