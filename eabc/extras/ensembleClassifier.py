# -*- coding: utf-8 -*-

import numpy as np
from sklearn import preprocessing
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted,_is_arraylike
import warnings 

class StackClassifier:
    
    def __init__(self,estimators = None, isPrefit = True, weights=None,decisionRule = None):
        
        self.__estimators_sanity_check(estimators,isPrefit)
        
        self._estimators = estimators        
        self._prefit = isPrefit
    
        self._le = None
        self._classes = None
        self._weights = None
        
        self._rule = self._majorityVoting
        if decisionRule is not None:
            self._rule = decisionRule        
        
    def fit(self,instances=None,labels=None):

        if labels is None:
            raise ValueError('labels must be passed')

        self._le = preprocessing.LabelEncoder().fit(labels)
        self._classes = self._le.transform(self._le.classes_)
        
        if self._prefit==True and instances is not None:
            warnings.warn("Ignoring passed instances",RuntimeWarning)
            
        elif self._prefit==False and instances is None:
            raise ValueError('You shall pass instances when prefit is False')
        
        elif self._prefit==False and instances is not None:
            
            self.__instances_sanity_check(instances)
            self.__estimators_instances_sanity_check(self._estimators,instances,self._prefit)
            
            for estimator,X in zip(self._estimators,instances):
                estimator.fit(X,labels)

        
    def predict(self, instances):
                
        for estimator in self._estimators:
            check_is_fitted(estimator)
        
        self.__estimators_instances_sanity_check(self._estimators,instances,isFitted=True)
        
        self.__instances_sanity_check(instances)
        
        
        #Instances must be a container of numpy matrices representing the embedding vectors
        n_samples_instances = np.array([len(data) for data in instances])
        if not np.all(n_samples_instances==n_samples_instances[0]):
            raise ValueError("Data container has different number of samples, cannot predict in ensemble")
        n_patterns = instances[0].shape[0]
                
        #stackPrediction is an N-pattern by number-of-estimators matrix containing
        #the labels of each pattern predicted by each estimator 
        stackPredictions = np.empty(shape=(n_patterns,len(self._estimators)),dtype=np.int16)
        for i,(data_array,estimator) in enumerate(zip(instances,self._estimators)):
            
            predicted_labels = estimator.predict(data_array)
            stackPredictions[:,i] = self._le.transform(predicted_labels)
        
        assigned_labels = self._decisionRule(stackPredictions)
                               
        return assigned_labels

    
    def _majorityVoting(self,predictions):
        
        #Majority rule
        #Do not break the tie - First occurences is taken
        maj = np.apply_along_axis(lambda x: np.argmax(
            np.bincount(x, weights=self._weights)),
            axis=1, arr=predictions)
        
        return maj    

    def _decisionRule(self,predictions):
                
        decisions = self._rule(predictions)
        predictedLabels = self._le.inverse_transform(decisions)
        
        return predictedLabels
    
    @staticmethod
    def __estimators_sanity_check(estimators,prefit):
        
        if _is_arraylike(estimators):           
            for estimator in estimators:
                
                if prefit:
                    check_is_fitted(estimator)
        else:
            raise TypeError('Expected a sequence like container of estimators,got {}'.format(type(estimators)))

    @staticmethod
    def __estimators_instances_sanity_check(estimators,instances,isFitted):
        
        
        if _is_arraylike(instances):
            
            if not(len(instances)==len(estimators)):
                raise ValueError('Expected same number of data with respect to number of estimators, got {} estimators and {} data matrices'.format(len(estimators),len(instances)))
            
            for data,estimator in zip(instances,estimators):
                
                if _is_arraylike(data):
                    
                    if isFitted:
                        
                        if estimator.n_features_in_!= len(data[0]):
                            raise ValueError('Estimator is fitted in {}-dim feature space, data is {}'.format(estimator.n_features_in_,len(data[0])))
                        
                    
                else:
                    raise TypeError('Expected an array like data,got {}'.format(type(data)))
                
            
        else:
            raise TypeError('Expected an array like container of matrices,got {}'.format(type(instances)))

    
    @staticmethod
    def __instances_sanity_check(instances):
        
        if _is_arraylike(instances):
            
            for data in instances:
                
                if _is_arraylike(data):
                    
                    if not isinstance(data,np.ndarray):
                        
                        data = np.asarray(data)
                        
