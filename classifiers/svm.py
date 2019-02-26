# SVM
import keras 
import numpy as np 
import pandas as pd 
import time 

from sklearn import svm
from sklearn.metrics import confusion_matrix
from utils.utils import accuracy
from utils.plot import display_cm

class Classifier_SVM:

    def __init__(self, output_directory, input_shape, nb_classes):
        
        self.training_accuracy = None

    def fit(self, class_labels, x_train, y_train, x_val, y_val, y_true): 
        ## Training the SVM classifier
    
        clf = svm.SVC()
        clf.fit(x_train,y_train)
        predicted_labels = clf.predict(x_val)


        

        conf = confusion_matrix(y_val, predicted_labels)
        display_cm(conf, class_labels, hide_zeros=True)       
        
        
        print('Classification accuracy = %f' % accuracy(conf))

