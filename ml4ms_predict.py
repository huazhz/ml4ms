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
     crossplot_pca, heatplot_pca, plot_correlations, compare_classifiers, plot_predictions



def main():

    ############################################### main ###########################################
    # change this directory for your machine
    # it should contain the archive folder containing both univariate and multivariate archives

    root_dir = 'F:\\datafolder\\dl4ms_data\\dataset'

    archive_name = 'FIELDDATA' # 

    dataset_name =  'MULTIWELL_B_P_512'#'MP_P_256'#'TX_P_TRAIN_256'
    

    segment_size = int(dataset_name.split('_')[-1])

    classifier_name= 'SVM'

    model_dir = os.path.join(root_dir, 'models', classifier_name, 'UTS', dataset_name)

    feature_dir = os.path.join(root_dir, 'archives', archive_name)
    
   
    create_directory(feature_dir)

    create_directory(model_dir)

    print('\nInfo: ',archive_name, dataset_name, segment_size, classifier_name, '\n')


    
    file_name = os.path.join(feature_dir, '181105_025200_6.csv') # start from 0
    file_name = os.path.join(feature_dir, '140605_041500_1.csv') # start from 0


    ## Transform field data trace into dataframe in order to extract features
    
    fname = 'F:\\datafolder\\MP54-3-1S\\181105_modified\\181105_025200.sgy'
    fname = 'F:\\datafolder\\tx1\\segy_modified\\140605_041500.sgy'
    segy = pssegy.Segy(fname)
    z_trace = segy.zTraces[:,0]

    fs = 500 # unit is Hz 
    window_length = segment_size  # a wavelength is usually 30 samples, we choose 2*wavelength
    overlap_length = int(window_length/2)
    signal_length = z_trace.shape[0]
    step_length = window_length - overlap_length
    number_of_windows = int(np.floor((signal_length-window_length)/step_length) + 1)
    #print(number_of_windows)

    _, wins = segment_trace(z_trace.copy(), window_length, overlap_length, norm_flag = 1, outpath = file_name)
    column_name = ['ID', 'FileName', 'Class'] + list(range(segment_size))
    hd = 0 # .csv has header
    datasets_df = load_csv(file_name, hd, column_name)
    
    

    ## Extract features 
    file_name = os.path.join(feature_dir, '181105_025200_6_features.csv')
    file_name = os.path.join(feature_dir, '140605_041500_1_features.csv')
    if not os.path.exists(file_name):
        extractor = FeatureExtractor()
        extractor.set_dataset(datasets_df)

        extractor.extract_features(fs, window_length, overlap_length, segment_size)
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

    

    ## Predict
    classifier_name = 'SVM'
    file_name = os.path.join(model_dir, classifier_name + '_model_raw.sav')
    loaded_model = pickle.load(open(file_name, 'rb'))
    result = loaded_model.clf.predict(scaled_features)
    print(result)

    trace = segy.normedTraces(segy.zTraces)[:,6]
    plot_predictions(result, trace, wins)


    file_name = os.path.join(model_dir, classifier_name + '_model_pca.sav')
    loaded_model = pickle.load(open(file_name, 'rb'))
    from sklearn.decomposition import PCA        
    pca = PCA(3).fit(scaled_features)
    X_pca = pca.transform(scaled_features)
    result = loaded_model.clf.predict(X_pca)
    print(result)
    plot_predictions(result, trace, wins)
    

    


    

    

# This will actually run this code if called stand-alone
if __name__ == '__main__':
    main()
