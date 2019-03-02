
'''
@author:     Zhengguang Zhao
@copyright:  Copyright 2016-2019, Zhengguang Zhao.
@license:    MIT
@contact:    zg.zhao@outlook.com

'''

# quadratic discriminant analysis
import numpy as np 
import pandas as pd 
import time 


import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from utils.utils import accuracy
from utils.io import display_cm

class ClassifierQDA:

    def __init__(self):      
        
        self.clf_name = 'QDA'

        self.training_accuracy = None
        self.numeric_class_labels = None # type: list of numeric
        self.feature_vector = None
        self.scaled_features = None
        self.feature_names = None # type: list of dataframe column names
        self.feature_labels = None # type: list of string
        

        self.clf = None

        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None      
       
        

    def set_data(self, training_data, class_labels):
        self.training_data = training_data
        self.class_labels = class_labels

        self.feature_names = training_data.columns.values.tolist()[3:-1]

        self._conditioning_data()


    def _conditioning_data(self):
        ## Conditioning the data set
        self.numeric_class_labels = self.training_data['Class'].values
        self.feature_labels = self.training_data['ClassLabels'].values

        self.feature_vector = self.training_data.drop(['FeatureID', 'FileName','Class','ClassLabels'], axis=1)
        self.feature_vector.describe()

        #feature_vectors.dropna(axis=0,how='any')   

        scaler = preprocessing.StandardScaler().fit(self.feature_vector)
        self.scaled_features = scaler.transform(self.feature_vector)
        print(np.isnan(self.scaled_features))

        # reformat features matrix to:  X : array-like, shape (n_samples, n_features)
    

    
    def split_dataset(self, features, test_size = 0.2, random_state=42):

        X_train, X_test, y_train, y_test = train_test_split(features, self.numeric_class_labels, test_size=0.2, random_state=42)

        self.X_train = X_train.copy()
        self.y_train = y_train.copy()
        self.X_test = X_test.copy()
        self.y_test = y_test.copy()

    
    
    def pca(self, n_components = 2):
        '''
        ## PCA analysis
        # 1. The PCA algorithm:
        # 2. takes as input a dataset with many features.
        # reduces that input to a smaller set of features (user-defined or algorithm-determined) 
        # by transforming the components of the feature set into what it considers as the main (principal) components.
        Drawback of PCA is it’s almost impossible to tell how the initial features combined to form the principal components. 
        '''
        from sklearn.decomposition import PCA
        
        pca = PCA(n_components).fit(self.feature_vector)
        X_pca = pca.transform(self.feature_vector)

        #measuring the variance ratio of the principal components
        ex_variance=np.var(X_pca,axis=0)
        ex_variance_ratio = ex_variance/np.sum(ex_variance)
        print ('\nVariance ratio of the principal components: ', ex_variance_ratio) 

        return pca, X_pca

  
    

    def fit(self):        
        
        ## Training the quadratic discriminant analysis classifier        
        self.clf = QuadraticDiscriminantAnalysis(priors=None, reg_param=0.0,
                              store_covariance=False,
                              store_covariances=None, tol=0.0001)
        self.clf.fit(self.X_train, self.y_train)
     
     
    def predict(self):
        predicted_labels = self.clf.predict(self.X_test)
        
        print('\n') 
        conf = confusion_matrix(self.y_test, predicted_labels)
        display_cm(conf, self.class_labels, hide_zeros=True)
        print('\nQuadraticDiscriminantAnalysis classification accuracy = %f' % accuracy(conf))
        
        return predicted_labels
    

    

