import numpy
from scipy.sparse import coo_matrix, dok_matrix
import itertools

class Clustering_BSAS: # SpareBSAS
    def setup_clustering(self, Lambda, Theta, S_T):
        self.Lambda = Lambda
        self.Theta = Theta
        self.Symbol_Th = S_T
        
    def clustering(self, dataset, obj_metric, obj_representative):
        
        """ BSAS with exact medoid update.

        Input:
        - dataset: list of items to be processed
        - theta: real-valued dissimilarity threshold for pattern inclusion
        - Q: maximum number of allowed clusters
        - dissimilarityFunction: callable for the dissimilarity measure to be used
        Output:
        - clusters: list of clusters' pattern IDs
        - representatives: list of clusters' medoids
        - representatives_IDs: list of clusters' medoid IDs
        - clusters_DissimMatrix: list of clusters' dissimilarity matrices. """
    
        medoidUpdate = 'fullEfficientSingle'       # options are: 'full'/'fullEfficient' (exact update, quadratic complexity)
        Q = self.Lambda;
        theta = self.Theta;
        # first pattern --> first cluster
        clusters = [[0]]
        clusters_v = [[0]]
        representatives = [dataset[0]]
        representatives_IDs = [0]
    
        if medoidUpdate == 'fullEfficient':
            dictionaryOfDistances = dok_matrix((len(dataset), len(dataset)))                                                # memory efficient, but slower
            dictionaryOfDistances = numpy.zeros((len(dataset), len(dataset)))                                               # faster, but heavier memory footprint
    
        clusters_DissimMatrix = [numpy.zeros((1, 1))]                                                                       # useful only if medoidUpdate == 'fullEfficientSingle'
    
        for i in range(1, len(dataset)):
            # grab current point
            point = dataset[i]
    
            # find distances w.r.t. all clusters
            distances = [obj_metric.Diss(point, medoid) for medoid in representatives]
            index_cluster = numpy.argmin(distances)
            distance = distances[index_cluster]
    
            # if smallest distance is within threshold, add to closest cluster
            if distance <= theta:
                clusters[index_cluster].append(i)
                clusters_v[index_cluster].append(point)
                # now update medoid
                if medoidUpdate == 'fullEfficientSingle':
                    D = numpy.zeros((len(clusters[index_cluster]), len(clusters[index_cluster])))
                    D[:-1, :-1] = clusters_DissimMatrix[index_cluster]
                    # eval missing items
                    v_left, v_right = [], []
                    for j, k in itertools.product([D.shape[1] - 1], range(D.shape[0] - 1)):
                        v_left.append(obj_metric.Diss(dataset[clusters[index_cluster][j]], dataset[clusters[index_cluster][k]]))
                    for j, k in itertools.product(range(D.shape[1] - 1), [D.shape[0] - 1]):
                        v_right.append(obj_metric.Diss(dataset[clusters[index_cluster][j]], dataset[clusters[index_cluster][k]]))
                    v = 0.5 * (numpy.array(v_left) + numpy.array(v_right))
                    D[0:-1, -1] = v
                    D[-1, 0:-1] = v
                    # eval medoid
                    minSOD_ID = numpy.argmin(numpy.sum(D, axis=1))
                    representatives[index_cluster] = dataset[clusters[index_cluster][minSOD_ID]]
                    representatives_IDs[index_cluster] = clusters[index_cluster][minSOD_ID]
                    clusters_DissimMatrix[index_cluster] = D
                elif medoidUpdate == 'fullEfficient':
                    # cache previously-evaluated point-to-medoid distances
                    for j in range(len(representatives_IDs)):
                        dictionaryOfDistances[i, representatives_IDs[j]] = distances[j]
                    # slice
                    row_idx = numpy.array(clusters[index_cluster])
                    col_idx = row_idx
                    D = dictionaryOfDistances[row_idx[:, None], col_idx]
                    # eval missing items
                    row_idx, col_idx = numpy.where(D == 0)
                    j = col_idx != row_idx      # do not
                    row_idx = row_idx[j]        # consider
                    col_idx = col_idx[j]        # diagonal
                    for j, k in zip(row_idx, col_idx):
                        D[j, k] = obj_metric.Diss(dataset[clusters[index_cluster][j]], dataset[clusters[index_cluster][k]])
                        dictionaryOfDistances[clusters[index_cluster][j], clusters[index_cluster][k]] = D[j, k]             # cache the distances
                    # eval medoid
                    D = 0.5 * (D + D.T)
                    minSOD_ID = numpy.argmin(numpy.sum(D, axis=1))
                    representatives[index_cluster] = dataset[clusters[index_cluster][minSOD_ID]]
                    representatives_IDs[index_cluster] = clusters[index_cluster][minSOD_ID]
                elif medoidUpdate == 'full':
                    pairs = itertools.product(range(len(clusters[index_cluster])), range(len(clusters[index_cluster])))
                    D = list(map(lambda pair: (pair[0], pair[1], obj_metric.Diss(dataset[clusters[index_cluster][pair[0]]], dataset[clusters[index_cluster][pair[1]]])), pairs))
                    D = numpy.array(D)
                    D = coo_matrix((D[:, 2], (D[:, 0].astype(int), D[:, 1].astype(int))))
                    D = 0.5 * (D + D.T)
                    minSOD_ID = numpy.argmin(numpy.sum(D, axis=1))
                    representatives[index_cluster] = dataset[clusters[index_cluster][minSOD_ID]]
                    representatives_IDs[index_cluster] = clusters[index_cluster][minSOD_ID]
                # elif medoidUpdate == 'approx':
                #     pool = [(j, dataset[j]) for j in clusters[index_cluster][0:poolSize]]
                #     notInPool = [(j, dataset[j]) for j in clusters[index_cluster][poolSize:]]
                #     for j in range(len(notInPool)):
                #         currentPattern = notInPool[j]
                #         id1 = random.randint(0, len(pool) - 1)
                #         id2 = random.randint(0, len(pool) - 1)
                #         while id2 == id1:
                #             id2 = random.randint(0, len(pool) - 1)
                #         d1 = dissimilarityFunction(pool[id1][-1], representatives[index_cluster])
                #         d2 = dissimilarityFunction(pool[id2][-1], representatives[index_cluster])
                #         if d1 >= d2:
                #             del pool[id1]
                #         else:
                #             del pool[id2]
                #         pool.append(currentPattern)
                #     pairs = itertools.product(range(len(pool)), range(len(pool)))
                #     D = list(map(lambda pair: (pair[0], pair[1], dissimilarityFunction(pool[pair[0]][-1], pool[pair[1]][-1])), pairs))
                #     D = numpy.array(D)
                #     D = coo_matrix((D[:, 2], (D[:, 0].astype(int), D[:, 1].astype(int))))
                #     D = 0.5 * (D + D.T)
                #     minSOD_ID = numpy.argmin(numpy.sum(D, axis=1))
                #     representatives[index_cluster] = pool[minSOD_ID][-1]
                #     representatives_IDs[index_cluster] = pool[minSOD_ID][0]
            # otherwise check if a new cluster can be spawned
            if distance > theta and len(clusters) < Q:
                representatives.append(point)
                clusters.append([i])
                clusters_v.append([i])
                representatives_IDs.append(i)
                clusters_DissimMatrix.append(numpy.zeros((1, 1)))                                                                      # useful only if medoidUpdate == 'fullEfficientSingle'
        return representatives,clusters_v#, representatives_IDs, clusters_DissimMatrix
    
    
