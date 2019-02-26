import os
import sys



from sklearn import svm, preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import normalize, scale

from scipy import stats

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

import seaborn as sns

import numpy as np

import pandas as pd
from pandas import set_option

from utils.sigproc import (mad, peakfreq_from_fft, spectralRollOff, spectralCentroidAndSpread, chromaFeatures, chromaFeaturesInit, rssq,\
    peak2rms, rms, range_bytes, energy, zcr_2, zcr)
from utils.entropy import spectral_entropy, sample_entropy, shannon_entropy
from utils.speech_features import mfcc
from utils.io import load_data, load_csv, display_adj_cm, display_cm, read_data, feature_normalize, wave_norm, create_directory, read_dataset


import time



def fit_classifier(datasets_dict, holdout = 0.2): 
    pass

    # x_train = datasets_dict[dataset_name][0]

    # y_train = datasets_dict[dataset_name][1]

    # x_test = datasets_dict[dataset_name][2]

    # y_test = datasets_dict[dataset_name][3]



    # nb_classes = len(np.unique(np.concatenate((y_train,y_test),axis =0)))



    # # make the min to zero of labels

    # y_train,y_test = transform_labels(y_train,y_test)



    # # save orignal y because later we will use binary

    # y_true = y_test.astype(np.int64) 

    # # transform the labels from integers to one hot vectors

    # enc = preprocessing.OneHotEncoder()

    # enc.fit(np.concatenate((y_train,y_test),axis =0).reshape(-1,1))

    # y_train = enc.transform(y_train.reshape(-1,1)).toarray()

    # y_test = enc.transform(y_test.reshape(-1,1)).toarray()



    # if len(x_train.shape) == 2: # if univariate 

    #     # add a dimension to make it multivariate with one dimension 

    #     x_train = x_train.reshape((x_train.shape[0],x_train.shape[1],1))

    #     x_test = x_test.reshape((x_test.shape[0],x_test.shape[1],1))



    # input_shape = x_train.shape[1:]

    # classifier = create_classifier(classifier_name,input_shape, nb_classes, output_directory)



    # classifier.fit(x_train,y_train,x_test,y_test, y_true)



def create_classifier(classifier_name, input_shape, nb_classes):

    if classifier_name=='svm': 

        from classifiers import svm        

        return svm.Classifier_SVM(output_directory,input_shape, nb_classes)

    



############################################### main ###########################################

# change this directory for your machine

# it should contain the archive folder containing both univariate and multivariate archives

root_dir = 'F:\\dl4ms_data\\dataset'


archive_name = 'multiwell' # 'mp_s1' #  

dataset_name = 'noise_event_binary_dataset_A' 

classifier_name= 'svm'



output_directory = os.path.join(root_dir, 'results', classifier_name, archive_name, dataset_name)

create_directory(output_directory)



print('\nTraining is in process...[Wait]\n')

print('\nMethod: ',archive_name, dataset_name, classifier_name, '\n')



datasets_dict = read_dataset(root_dir,archive_name,dataset_name)

fit_classifier(datasets_dict)


print('\nTraining completed[OK]')