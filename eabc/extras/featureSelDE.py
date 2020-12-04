#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 11:13:37 2020

@author: luca
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import numpy

def FSsetup_DE(alphabetSize, n_threads):
    """ Parameters for second GA (feature selection). To be used with SciPy's differential_evolution().
    Input:
    - alphabetSize: size of the best alphabet to be pruned
    - n_threads: number of threads for parallel individual evaluation
    Output:
    - bounds: list of tuples of the form (min, max) encoding lower and upper bounds for each variable
    - CXPB: crossover probability
    - MUTPB: mutation probability
    - pop: the initial population. """

    # Declare bounds
    bounds = list(zip([0] * alphabetSize, [1] * alphabetSize))

    # CXPB  is the probability with which two individuals are crossed
    # MUTPB is the probability for mutating an individual
    CXPB, MUTPB = 0.8, 0.3

    # initial population trick (100 individuals)
    M = numpy.random.choice(a=[True, False], size=(10, alphabetSize), p=[0.1, 1 - 0.1])                         # 10 individuals with 10% of 1's
    M = numpy.vstack((M, numpy.random.choice(a=[True, False], size=(10, alphabetSize), p=[0.2, 1 - 0.2])))      # 10 individuals with 20% of 1's
    M = numpy.vstack((M, numpy.random.choice(a=[True, False], size=(10, alphabetSize), p=[0.3, 1 - 0.3])))      # 10 individuals with 30% of 1's
    M = numpy.vstack((M, numpy.random.choice(a=[True, False], size=(10, alphabetSize), p=[0.4, 1 - 0.4])))      # 10 individuals with 40% of 1's
    M = numpy.vstack((M, numpy.random.choice(a=[True, False], size=(10, alphabetSize), p=[0.5, 1 - 0.5])))      # 10 individuals with 50% of 1's
    M = numpy.vstack((M, numpy.random.choice(a=[True, False], size=(10, alphabetSize), p=[0.6, 1 - 0.6])))      # 10 individuals with 60% of 1's
    M = numpy.vstack((M, numpy.random.choice(a=[True, False], size=(10, alphabetSize), p=[0.7, 1 - 0.7])))      # 10 individuals with 70% of 1's
    M = numpy.vstack((M, numpy.random.choice(a=[True, False], size=(10, alphabetSize), p=[0.8, 1 - 0.8])))      # 10 individuals with 80% of 1's
    M = numpy.vstack((M, numpy.random.choice(a=[True, False], size=(10, alphabetSize), p=[0.9, 1 - 0.9])))      # 10 individuals with 90% of 1's
    M = numpy.vstack((M, numpy.random.choice(a=[True, False], size=(5, alphabetSize), p=[0.05, 1 - 0.05])))     # 5 individuals with 5% of 1's
    M = numpy.vstack((M, numpy.random.choice(a=[True, False], size=(5, alphabetSize), p=[0.95, 1 - 0.95])))     # 5 individuals with 95% of 1's
    pop = M.astype(int)

    return bounds, CXPB, MUTPB, pop

def FSfitness_DE(genetic_code, *data):
    """ Fitness function for second GA (feature selection). To be used with SciPys' differential_evolution().
    Input:
    - genetic_code: Individual object provided by DEAP
    - data: tuple of additional arguments, namely
        1. trSet_EMB_InstanceMatrix: embedded training set with the best alphabet
        2. vsSet_EMB_InstanceMatrix: embedded validation set with the best alphabet
        3. trSet_EMB_LabelVector: training set labels
        4. vsSet_EMB_LabelVector: validation set labels.
    Output:
    - fitness: fitness value (to be minimised) of the form [alpha * error_rate + (1 - alpha) * number_of_selected_symbols]. """

    # Strip input data
    trSet_EMB_InstanceMatrix, vsSet_EMB_InstanceMatrix, trSet_EMB_LabelVector, vsSet_EMB_LabelVector = data

    # Strip parameters from genetic code
    mask = [round(i) for i in genetic_code]

    # Set useful parameters
    alpha = 0.9

    # Evalaute mask cost
    mask = numpy.array(mask, dtype=bool)
    selectedRatio = sum(mask) / len(genetic_code)

    # Prior exit if no features have been selected
    if selectedRatio == 0:
        return 2

    # Classifier
    KNN = KNeighborsClassifier(n_neighbors=5)
    KNN.fit(trSet_EMB_InstanceMatrix[:, mask], trSet_EMB_LabelVector)
    predicted_vsSet=KNN.predict(vsSet_EMB_InstanceMatrix[:, mask])
    
    tn, fp, fn, tp = confusion_matrix(vsSet_EMB_LabelVector, predicted_vsSet).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    J = sensitivity + specificity - 1
    J = (J + 1) / 2
    error_rate = 1 - J  

    fitness = alpha * error_rate + (1 - alpha) * selectedRatio
    return fitness
