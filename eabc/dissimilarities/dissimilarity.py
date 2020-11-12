# -*- coding: utf-8 -*-
# __all__ = [
#     'braycurtis',
#     'canberra',
#     'chebyshev',
#     'cityblock',
#     'correlation',
#     'cosine',
#     'dice',
#     'euclidean',
#     'jensenshannon',
#     'mahalanobis',
#     'matching',
#     'minkowski',
#     'seuclidean',
#     'sqeuclidean',
#     'wminkowski',
#     'yule'
# ]

class Dissimilarity:
    
    def __call__(self):
        return NotImplementedError
    
    def pdist(self,set1):
        return NotImplementedError
    
    def __validate_dissimilarity():
        return 0