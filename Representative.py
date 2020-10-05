from statistics import mean
import numpy 
import itertools

class Representative:
    
    def Update_M(self,clusters,index_cluster,clusters_DissimMatrix,dataset,obj_metric):
                    D = numpy.zeros((len(clusters[index_cluster]), len(clusters[index_cluster])))
                    D[:-1, :-1] = clusters_DissimMatrix[index_cluster]
    
                    v_left, v_right = [], []
                    for j, k in itertools.product([D.shape[1] - 1], range(D.shape[0] - 1)):
                        v_left.append(
                            obj_metric.Diss(dataset[clusters[index_cluster][j]], dataset[clusters[index_cluster][k]]))
                    for j, k in itertools.product(range(D.shape[1] - 1), [D.shape[0] - 1]):
                        v_right.append(
                            obj_metric.Diss(dataset[clusters[index_cluster][j]], dataset[clusters[index_cluster][k]]))
                    v = 0.5 * (numpy.array(v_left) + numpy.array(v_right))
                    D[0:-1, -1] = v
                    D[-1, 0:-1] = v
                    minSOD_ID = numpy.argmin(numpy.sum(D, axis=1))
                    return minSOD_ID, D
    
    def update_c(self,classes,classification):
        return numpy.average(classes[classification], axis = 0)
    
