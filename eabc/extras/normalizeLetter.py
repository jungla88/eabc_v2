# -*- coding: utf-8 -*-

import numpy
import copy

def normalize(strAttribute,*argv):
    """ Normalization routine for Letter 1, 2 and 3.
    Normalization includes scaling [x, y] coordinates according to the maximum value found in the overall dataset.
    The input sets will be modified in-place.
    Input:
    - tsSet: a dictionary of (training) graphs of the form {id: (graph, label)}
    - vsSet: a dictionary of (validation) graphs of the form {id: (graph, label)}
    - tsSet: a dictionary of (test) graphs of the form {id: (graph, label)}
    Output:
    - normFactor: a normalization factor that will be used in nodeDissimilarity (i.e., square root of the number of components). """

    # Find max value in training set by stacking [x, y] pairs, then flattening and then taking the max
    

    MAXval = []    
    for targetSet in argv:    
        MAX_Set = numpy.zeros((1, 2))
        for g in targetSet:
        # extract graph
#            thisGraph = targetSet[k]
            # append node labels
            for n in g.nodes():
                MAX_Set = numpy.vstack((MAX_Set, g.nodes[n][strAttribute]))
        MAX_Set = copy.deepcopy(MAX_Set.ravel().max())
        MAXval.append(MAX_Set)

    # Find overall max value
    MAX = max(MAXval)
    
    
    for targetSet in argv:
        for g in targetSet:
            for n in g.nodes():
                g.nodes[n][strAttribute] = g.nodes[n][strAttribute]/MAX