'''
@author:     Zhengguang Zhao
@copyright:  Copyright 2016-2019, Zhengguang Zhao.
@license:    MIT
@contact:    zg.zhao@outlook.com

'''

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

import pickle

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from utils import pssegy
from utils.io import load_data, load_csv, display_adj_cm, display_cm, read_data, feature_normalize, wave_norm, create_directory, segment_trace
from utils.featextractor import FeatureExtractor
from utils.plot import visualize_ml_result, plot_coefficients, crossplot_features, crossplot_dual_features,\
     crossplot_pca, heatplot_pca, plot_correlations, compare_classifiers



def main():

    ############################################### main ###########################################

    # change this directory for your machine

    # it should contain the archive folder containing both univariate and multivariate archives

    root_dir = 'F:\\datafolder\\dl4ms_data\\dataset'

    archive_name = 'UTS' # 

    dataset_name = 'TX_P_VAL_256'
    

    segment_size = int(dataset_name.split('_')[-1])

    classifier_name= 'SVM'

    model_dir = os.path.join(root_dir, 'results', classifier_name, archive_name, 'TX_P_TRAIN_256')

    
    print('\nInfo: ',archive_name, dataset_name, segment_size, classifier_name, '\n')


    file_name = os.path.join(root_dir,'archives', archive_name, dataset_name, dataset_name +'.csv')
    
    
    column_name = ['ID', 'FileName', 'Class'] + list(range(segment_size))
    hd = 0 # .csv has header
    datasets_df = load_csv(file_name, hd, column_name)

    ## Extract features 
    file_name = os.path.join(root_dir,'archives', archive_name, dataset_name, dataset_name +'_features.csv')
    if not os.path.exists(file_name):
        extractor = FeatureExtractor()
        extractor.set_dataset(datasets_df)

        fs = 500 # unit is Hz 
        window_length = segment_size  # a wavelength is usually 30 samples, we choose 2*wavelength
        overlap_length = int(window_length/2)
        signal_length = segment_size

        extractor.extract_features(fs, window_length, overlap_length, signal_length)
        extractor.save_features(file_name)
        training_data = extractor.feature_data

    else:
        training_data = pd.read_csv(file_name, header= 0, index_col= False)

    
    ## Conditioning the data set
    numeric_class_labels = training_data['Class'].values
    feature_labels = training_data['ClassLabels'].values

    feature_vector = training_data.drop(['FeatureID', 'FileName','Class','ClassLabels'], axis=1)
    feature_vector.describe()

    

    scaler = preprocessing.StandardScaler().fit(feature_vector)
    scaled_features = scaler.transform(feature_vector)
    #print(np.isnan(scaled_features))

    X_train, X_test, y_train, y_test = train_test_split(scaled_features, numeric_class_labels, test_size=0.0, random_state = 0)

    ## Predict
    classifier_name = 'SVM'
    file_name = os.path.join(model_dir, classifier_name + '_model_raw.sav')
    loaded_model = pickle.load(open(file_name, 'rb'))
    result = loaded_model.clf.score(X_train, y_train)
    print(result)

    
    

# This will actually run this code if called stand-alone
if __name__ == '__main__':
    main()
