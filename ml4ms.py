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

from utils.io import load_data, load_csv, display_adj_cm, display_cm, read_data, feature_normalize, wave_norm, create_directory
from utils.featextractor import FeatureExtractor
from utils.plot import visualize_ml_result, plot_coefficients, crossplot_features, crossplot_dual_features, crossplot_pca, heatplot_pca, plot_correlations



def create_classifier(classifier_name):

    if classifier_name=='svm': 

        from classifiers import svm        

        return svm.ClassifierSVM()
    
    elif classifier_name=='knn': 

        from classifiers import knn        

        return knn.ClassifierKNN()

    

def main():

    ############################################### main ###########################################

    # change this directory for your machine

    # it should contain the archive folder containing both univariate and multivariate archives

    root_dir = 'F:\\datafolder\\dl4ms_data\\dataset'

    archive_name = 'multiwell' # 

    dataset_name = 'noise_event_binary_dataset_B' 

    classifier_name= 'knn'

    output_directory = os.path.join(root_dir, 'results', classifier_name, archive_name, dataset_name)

    create_directory(output_directory)


    print('\nInfo: ',archive_name, dataset_name, classifier_name, '\n')


    file_name = os.path.join(root_dir,'archives', archive_name, dataset_name, dataset_name +'.csv')
    column_name = ['ID', 'FileName', 'Class'] + list(range(512))
    hd = 0 # .csv has header
    datasets_df = load_csv(file_name, hd, column_name)


    ## Extract features 
    file_name = os.path.join(root_dir,'archives', archive_name, dataset_name, dataset_name +'_features.csv')
    if not os.path.exists(file_name):
        extractor = FeatureExtractor()
        extractor.set_dataset(datasets_df)

        fs = 500 # unit is Hz 
        window_length = 512  # a wavelength is usually 30 samples, we choose 2*wavelength
        overlap_length = int(window_length/2)
        signal_length = 512

        extractor.extract_features(fs, window_length, overlap_length, signal_length)
        extractor.save_features(file_name)
        training_data = extractor.feature_data

    else:
        training_data = pd.read_csv(file_name, header= 0, index_col= False)


    data = training_data.copy()
    for columname in data.columns:
        if data[columname].count() != len(data):
            loc = data[columname][data[columname].isnull().values==True].index.tolist()
            print('Column Name："{}", Row #{} has null value.'.format(columname,loc))




    ## Features Vector Conditioning

    blind = training_data[training_data['FileName'] == '41532_140407_224100_1000']
    #training_data = training_data[training_data['FileName'] != '41532_140407_224100_1000']
    training_data['FileName'] = training_data['FileName'].astype('category')

    print(training_data['FileName'].unique())


    class_colors = ['#196F3D', '#F5B041']

    class_labels = ['Event', 'Noise']
    #class_color_map is a dictionary that maps class labels
    #to their respective colors
    class_color_map = {}
    for ind, label in enumerate(class_labels):
        class_color_map[label] = class_colors[ind]

    def label_class(row, labels):
        ind = row['Class'] -1
        #print(ind)
        return labels[ind]
        
    training_data.loc[:,'ClassLabels'] = training_data.apply(lambda row: label_class(row, class_labels), axis=1)
    print(training_data.describe())

    # training_data.dropna(axis=0,how='any') #drop all rows that have any NaN values
    #training_data.fillna(0)

    #count the number of unique entries for each class, sort them by
    #class number (instead of by number of entries)
    class_counts = training_data['Class'].value_counts().sort_index()
    #use facies labels to index each count
    class_counts.index = class_labels

    class_counts.plot(kind='bar',color=class_colors, 
                    title='Distribution of Training Data by Class')
    print(class_counts)

    # ## save plot display settings to change back to when done plotting with seaborn
    # inline_rc = dict(mpl.rcParams)

    # sns.set()
    # sns.pairplot(training_data.drop(['FileName','Class'],axis=1), hue='ClassLabels', palette=class_color_map, hue_order=list(reversed(class_labels)))

    # #switch back to default matplotlib plot style
    # mpl.rcParams.update(inline_rc)

    
    
    ## Visualize dataset and gain insight
    crossplot_dual_features(data, ['Peak Frequency', 'Shannon Entropy'])
    crossplot_features(training_data, ['Event', 'Noise'])
    plot_correlations(training_data, ['Mean', 'Peak Frequency', 'Shannon Entropy', 'MFCC 1'])

    feature_names = training_data.columns.values.tolist()[3:-1]

    plot_correlations(training_data, feature_names[0:10])
    plot_correlations(training_data, feature_names[11:25])
    plot_correlations(training_data, feature_names[26:46])
    plot_correlations(training_data, feature_names)
    
    
    
    ## Train Classifier

    print('Training ['+ classifier_name + '] classifer in process[Waiting...]')
    classifier = create_classifier(classifier_name)
    classifier.set_data(training_data, class_labels) 
    classifier.split_dataset(classifier.scaled_features)
    classifier.fit()

    # PCA analysis
    print('\nPCA analysis results:\n')
    crossplot_pca(classifier.pca(3)[1], classifier.feature_labels)
    heatplot_pca(classifier.pca(3)[0], classifier.feature_names)

    classifier.split_dataset(classifier.pca(3)[1])
    classifier.fit() 


    if classifier_name == 'svm': 
    
        # Train SVM classifier        
        classifier.visualize_binary_class('Peak Frequency', 'Shannon Entropy', 1)
        
        classifier.fit(kernel = 'linear') 
        # Plot coefficients
        classifier.plot_coefficients()

        # C_range = np.array([.01, 1, 5, 10, 20, 50, 100, 1000, 5000, 10000])
        # gamma_range = np.array([0.0001, 0.001, 0.01, 0.1, 1, 10])
        # classifier.model_param_selection(C_range, gamma_range)
        # classifier.fit_with_selected_model_param(10, 'auto')
        


    print('\nTraining completed[OK]')


    


# This will actually run this code if called stand-alone
if __name__ == '__main__':
    main()
