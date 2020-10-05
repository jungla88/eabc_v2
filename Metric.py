from scipy.spatial import distance
import numpy as np

class Metric:
    Weight_Params = [1];
    
    def Diss(self,a, b):
        dst = distance.euclidean(a, b)
        #dst = dst * np.sqrt(self.Weight_Params)
        return dst
    
    def Diss2(self, x1, x2):
        return (np.sqrt(np.sum(np.square(x1 - x2) , axis = 1)))
    
