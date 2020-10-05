import numpy as np
import pandas as pd

class Clustering_K_Means:
    
    def __init__(self,k ,k_max):
        self.k = k
        self.max_iter = 100
        self.k_max = k_max # numero di cluster
        print("Initalized k",self.k)
        
    def fit(self, data, obj_metric, obj_representative):
        self.centroids = []
        #initialize the centroids, the first 'k' elements in the dataset will be our initial centroids
        for i in range(self.k):
            self.centroids.append(data.iloc[i].to_numpy())
            
        for itr in range(self.max_iter):
            self.classes = {}
            
            for cluster in range(self.k):
                self.classes[cluster] = []
                
            #find the distance between the point and cluster; choose the nearest centroid
            for point in range(len(data)):
                distances = obj_metric.Diss2(self.centroids, data.iloc[point].to_numpy())
                classification = np.argmin(distances)
                self.classes[classification].append(data.iloc[point])
                
            previous = np.array(self.centroids)
            #average the cluster datapoints to re-calculate the centroids
            for classification in self.classes:
                self.centroids[classification] = obj_representative.update_c(self.classes,classification)
            
            optimal = True
            curr = np.array(self.centroids)
        
            #difference in the cluster centers of two consecutive iterations to declare convergence.
            if np.sum((curr - previous)/previous * 100.0) > 0.0001:
                optimal = False
     
    def clustering(self, data, obj_metric, obj_representative):
        
        clf = Clustering_K_Means(self.k)
        data = pd.DataFrame(data)
        clf.fit(data,obj_metric,obj_representative)
        centroidi = clf.centroids
        clusters_v=[]
        i=0
        for classification in clf.classes:
            clusters_v.append([])
            for features in clf.classes[classification]:
                clusters_v[i].append(features[0])
                clusters_v[i].append(features[1])
            i=i+1

        n = 2 
        for j in range(0,len(clusters_v)):
            clusters_v[j] = [clusters_v[j][i * n:(i + 1) * n] for i in range((len(clusters_v[j]) + n - 1) // n )]
        
        for k in range(0,len(centroidi)):
            centroidi[k]=centroidi[k].tolist()

        return centroidi, clusters_v 
          
    def evaluate(self, data, obj_metric, obj_representative):
        
        sse = []
        
        for k in range(1, self.k_max+1):
            clf = Clustering_K_Means(k, self.k_max)
            data = pd.DataFrame(data)
            clf.fit(data,obj_metric,obj_representative)
            centroidi = clf.centroids
            clusters_v=[]
            i=0
            for classification in clf.classes:
                clusters_v.append([])
                for features in clf.classes[classification]:
                    clusters_v[i].append(features[0])
                    clusters_v[i].append(features[1])
                i=i+1
    
            n = 2 
            for j in range(0,len(clusters_v)):
                clusters_v[j] = [clusters_v[j][i * n:(i + 1) * n] for i in range((len(clusters_v[j]) + n - 1) // n )]
            
            for k in range(0,len(centroidi)):
                centroidi[k]=centroidi[k].tolist()
                
                        
            curr_sse = 0
                
            # calculate square of Euclidean distance of each point from its cluster center and add to current WSS
            for i in range(len(clusters_v)): #num di cluster
                for j in range(0,len(clusters_v[i])): #num di punti per cluster
                    curr_center = centroidi[i]
                    curr_sse += (clusters_v[i][j][0] - curr_center[0]) ** 2 + (clusters_v[i][j][1] - curr_center[1]) ** 2
                
            sse.append(curr_sse)
        
        return centroidi, clusters_v
    
      