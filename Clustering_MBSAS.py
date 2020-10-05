import numpy
import random
from statistics import mean

class Clustering_MBSAS: # SpareBSAS
    def __init__(self, Lambda, theta_start,theta_step, theta_stop):
        self.Lambda = Lambda
        self.theta_start, self.theta_step, self.theta_stop = theta_start, theta_step, theta_stop
        print(self.theta_start, self.theta_stop, self.theta_step)
        
    def fit(self, dataset, obj_metric, obj_representative):
        
        """ Modified two-pass BSAS with approximate medoid tracking from the SPARE library
        Input:
        - dataset: list of items to be processed
        - theta: real-valued dissimilarity threshold for pattern inclusion
        - Q: maximum number of allowed clusters
        - dissimilarityFunction: callable for the dissimilarity measure to be used
        Output:
        - clusters: list of clusters' pattern IDs
        - clusters_v: list of clusters' pattern values
        - representatives: list of clusters' medoids
        - representatives_IDs: list of clusters' medoid IDs
        - clusters_DissimMatrix: list of clusters' dissimilarity matrices 
        - x: x coordinates of points of the clusters
        - y: y coordinates of points of the clusters.
        - x_r: x coordinates of points of the clusters' representatives
        - y_r: y coordinates of points of the clusters' representatives."""
        
        Q = self.Lambda;
        theta = self.Theta;
        
        # Set useful parameters
        poolSize = 20
    
        # Misc
        isAssigned = [False] * len(dataset)
    
        # First Round
        clusters = [[0]]  # first pattern
        clusters_v = [[0]]
        representatives = [dataset[0]]  # is
        representatives_IDs = [0]  # first cluster
        isAssigned[0] = True  #
    
        for i in range(1, len(dataset)):
            # grab current point
            point = dataset[i]
            # find distances w.r.t. all clusters
            distances = [obj_metric.Diss(point, medoid) for medoid in representatives]
            index_cluster = numpy.argmin(distances) 
            distance = distances[index_cluster]
    
            if distance > theta and len(clusters) < Q:
                representatives.append(point)
                clusters.append([i])
                clusters_v.append([i])
                representatives_IDs.append(i)
                isAssigned[i] = True
            else:
                pass
    
        # Second Round
        clusters_DissimMatrix = [numpy.zeros((1, 1))] * len(clusters)
        
        for i in range(0, len(dataset)):
            if isAssigned[i] is False:
                # grab current point
                point = dataset[i]
                # find distances w.r.t. all clusters
                distances = [obj_metric.Diss(point, medoid) for medoid in representatives]
                index_cluster = numpy.argmin(distances)
                distance = distances[index_cluster]
                # update medoid
                if len(clusters[index_cluster]) < poolSize:
                    clusters[index_cluster].append(i)
                    clusters_v[index_cluster].append(point)
                    minSOD_ID, D = obj_representative.Update_M(clusters,index_cluster,clusters_DissimMatrix,dataset,obj_metric)
            
                else:
                    id_pattern1, id_pattern2 = random.sample(range(0, poolSize), 2)
                    id_medoid = clusters[index_cluster].index(representatives_IDs[index_cluster])
                    
                    D = clusters_DissimMatrix[index_cluster]
                    d1 = D[id_medoid, id_pattern1]
                    d2 = D[id_medoid, id_pattern2]
                    if d1 >= d2:
                        toBeChanged = id_pattern1
                    else:
                        toBeChanged = id_pattern2
                    clusters[index_cluster][toBeChanged] = i
                    v_left, v_right = numpy.zeros(len(clusters[index_cluster])), numpy.zeros(len(clusters[index_cluster]))
                    for j in numpy.setdiff1d(range(len(clusters[index_cluster])), toBeChanged):
                        v_right[j] = obj_metric.Diss(dataset[clusters[index_cluster][j]],
                                                           dataset[clusters[index_cluster][toBeChanged]])
                        v_left[j] = obj_metric.Diss(dataset[clusters[index_cluster][toBeChanged]],
                                                          dataset[clusters[index_cluster][j]])
                    v = 0.5 * (v_left + v_right)
                    D[:, toBeChanged] = v
                    D[toBeChanged, :] = v
                    minSOD_ID = numpy.argmin(numpy.sum(D, axis=1))
                clusters_DissimMatrix[index_cluster] = D
                representatives[index_cluster] = dataset[clusters[index_cluster][minSOD_ID]]
                representatives_IDs[index_cluster] = clusters[index_cluster][minSOD_ID]
                
        return representatives, clusters_v  # , representatives_IDs, clusters_DissimMatrix, clusters, 
    
    def clustering(self, dataset, obj_metric, obj_representative):
        
        representatives, clusters_v = self.fit(dataset, obj_metric, obj_representative)
        
        return representatives, clusters_v    
    
    def evaluate(self, dataset, obj_metric, obj_representative):
        start = self.theta_start
        stop = self.theta_stop
        step = self.theta_step
        thetas = numpy.arange(self.theta_start, self.theta_stop, self.theta_step)
        thetas2 = []
        l = []
        i = 0
        print('thetas',thetas)
        
        for theta in thetas:
            self.Theta = theta
            representatives1, clusters_v1 = self.fit(dataset, obj_metric, obj_representative)
            l.append(len(representatives1))
            
            if l[i] == l[i-1]:
                thetas2.append(theta)
                i=i+1
                break;
            
        print('Theta',thetas2)
        
        return representatives1, clusters_v1
