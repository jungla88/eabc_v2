import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from numpy import mean ,std
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier, RandomForestClassifier,StackingClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted,_is_arraylike
import warnings 

class ensemble_cl:

    models = dict()
    models['RF'] = RandomForestClassifier()
    models['knn'] = KNeighborsClassifier()
    models['linear_svc'] = SVC(kernel='linear')
    models['poly_svc'] = SVC(kernel='poly')
    models['rbf_svc'] = SVC(kernel='rbf')
    
    level0 = list()
    level0.append(('RF', RandomForestClassifier()))
    level0.append(('knn', KNeighborsClassifier()))
    level0.append(('linear_svc', SVC(kernel='linear')))
    level0.append(('poly_svc', SVC(kernel='poly')))
    level0.append(('rbf_svc', SVC(kernel='rbf')))
    
    def __init__(self,X_train,y_train,X_test,y_test):
        self.X_train=X_train
        self.y_train=y_train
        self.X_test=X_test
        self.y_test=y_test
        self.le = None
        self.Oe=None
        self._classes = None
        
        
        # prepare input data
        #The input to this transformer should be an array-like of integers or strings,
        # denoting the values taken on by categorical (discrete) features.
        # The features are converted to ordinal integers.
        # This results in a single column of integers (0 to n_categories - 1) per feature.
        
    def prepare_inputs(self,X_train,X_test):
        self.Oe = OneHotEncoder(sparse=False,handle_unknown = 'ignore')
        self.Oe.fit(self.X_train)
        X_train_enc = self.Oe.transform(self.X_train)
        X_test_enc = self.Oe.transform(self.X_test)
        return X_train_enc, X_test_enc
    
    # prepare target
    def prepare_targets(self,y_train,y_test):
        self.le = LabelEncoder()
        self.le.fit(self.y_train)
        y_train_enc = self.le.transform(self.y_train)
        y_test_enc = self.le.transform(self.y_test)
        return y_train_enc, y_test_enc

    
     # get a stacking ensemble of models
    def get_stacking(self,final_estimator=SVC(kernel='linear'),cv=5,vote='hard'):
        
        self.models['stacking'] = StackingClassifier(estimators=self.level0, final_estimator=final_estimator, cv=cv)  
        self.models['voting']= VotingClassifier(self.level0, voting=vote)
        self.models['voting_stacking'] = StackingClassifier(estimators=self.level0, final_estimator=self.models['voting'], cv=cv)
     
    def evaluate_model_encoded_inputs(self,model):
        self.X_train_enc, self.X_test_enc = self.prepare_inputs(self.X_train, self.X_test)
        self.y_train_enc, self.y_test_enc = self.prepare_targets(self.y_train, self.y_test)
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
        scores = cross_val_score(model, self.X_train_enc, self.y_train_enc, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
        return scores
    
    def evaluate_model_raw(self,model):
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
        scores = cross_val_score(model, self.X_train, self.y_train, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
        return scores

    
    # evaluate the models and store results
    #change the final estimator ?
    # vote system ?
    # cv ?
    def result_encoded_inputs(self):
        self.get_stacking(final_estimator=SVC(kernel='linear'),cv=5,vote='hard')
        
        results, names = list(), list()
        for name, model in self.models.items():
            scores = self.evaluate_model_encoded_inputs(model)
            results.append(mean(scores))
            names.append(name)
            print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores))) 
            
        class_optimum_accuracy=np.array(results)
        optimum=names[np.argmax(class_optimum_accuracy)]
        print(optimum)
        predict=self.predict_enc(optimum)
        return predict
       
    def result_raw(self):
        self.get_stacking(final_estimator=SVC(kernel='linear'),cv=5,vote='hard')
        
        results, names = list(), list()
        for name, model in self.models.items():
            scores = self.evaluate_model_raw(model)
            results.append(mean(scores))
            names.append(name)
            print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores))) 
            
        class_optimum_accuracy=np.array(results)
        optimum=names[np.argmax(class_optimum_accuracy)]
        print(optimum)
        predict=self.predict_raw(optimum)
        return predict
        
    def predict_enc(self,model):
        
        self.X_train_enc, self.X_test_enc = self.prepare_inputs(self.X_train, self.X_test)
        self.y_train_enc, self.y_test_enc = self.prepare_targets(self.y_train, self.y_test)
        classifier = self.models[model].fit(self.X_train_enc,self.y_train_enc)
        predict = classifier.predict(self.X_test_enc)
        print(f1_score(self.y_test_enc, predict, average='macro'))
        return  predict
       
        
    def predict_raw(self,model):
        
        classifier = self.models[model].fit(self.X_train,self.y_train)
        predict = classifier.predict(self.X_test)
        print(f1_score(self.y_test, predict, average='macro'))
        return  predict
                
   
        
