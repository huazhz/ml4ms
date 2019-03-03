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
from utils.plot import visualize_ml_result, plot_coefficients, crossplot_features, crossplot_dual_features,\
     crossplot_pca, heatplot_pca, plot_correlations, compare_classifiers



def create_classifier(classifier_name):

    if classifier_name=='SVM': 

        from classifiers import svm
        return svm.ClassifierSVM()
    
    elif classifier_name=='KNN': 

        from classifiers import knn
        return knn.ClassifierKNN()

    elif classifier_name=='DecisionTree': 

        from classifiers import dtree
        return dtree.ClassifierDecisionTree()
    
    elif classifier_name=='LogisticRegression':

        from classifiers import logreg
        return logreg.ClassifierLogisticRegression()

    elif classifier_name=='LDA':

        from classifiers import lda
        return lda.ClassifierLDA()

    elif classifier_name=='QDA':

        from classifiers import qda
        return qda.ClassifierQDA()


    

def main():

    ############################################### main ###########################################

    # change this directory for your machine

    # it should contain the archive folder containing both univariate and multivariate archives

    root_dir = 'F:\\datafolder\\dl4ms_data\\dataset'

    archive_name = 'UTS' # 

    dataset_name = 'MP_NOISE_P_256'#'TX_DEMO_P_256' 

    segment_size = int(dataset_name.split('_')[-1])

    classifier_name= 'SVM'

    output_directory = os.path.join(root_dir, 'results', classifier_name, archive_name, dataset_name)

    create_directory(output_directory)


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

    ## Features Vector Conditioning
    print(training_data.describe())
    data = training_data.copy()
    for columname in data.columns:
        if data[columname].count() != len(data):
            loc = data[columname][data[columname].isnull().values==True].index.tolist()
            print('Column Name："{}", Row #{} has null value.'.format(columname,loc))

    
    #blind = training_data[training_data['FileName'] == '41532_140407_224100_1000']
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
    

    # training_data.dropna(axis=0,how='any') #drop all rows that have any NaN values
    #training_data.fillna(0)

    #count the number of unique entries for each class, sort them by
    #class number (instead of by number of entries)
    class_counts = training_data['Class'].value_counts().sort_index()
    #use class labels to index each count
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
    # crossplot_dual_features(data, ['Peak Frequency', 'Shannon Entropy'])
    crossplot_features(training_data, ['Event', 'Noise'])
    

    feature_names = training_data.columns.values.tolist()[3:-1]

    # plot_correlations(training_data, ['Mean', 'Peak Frequency', 'Shannon Entropy', 'MFCC 1'])
    # plot_correlations(training_data, feature_names[0:10])
    # plot_correlations(training_data, feature_names[11:25])
    # plot_correlations(training_data, feature_names[26:46])
    # plot_correlations(training_data, feature_names)
    
    ## Compare different classifiers, including 
        # ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
        # "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
        # "Naive Bayes", "QDA"]

    numeric_class_labels = training_data['Class'].values
    feature_labels = training_data['ClassLabels'].values

    feature_vector = training_data.drop(['FeatureID', 'FileName','Class','ClassLabels'], axis=1)
    feature_vector.describe()

    count = 300
    dataset = [(feature_vector[['Energy', 'Spectral Spread']][0:count], numeric_class_labels[0:count]),
                (feature_vector[['Peak Frequency', 'STALTA']][0:count], numeric_class_labels[0:count]),
                (feature_vector[['Variance', 'Kurtosis']][0:count], numeric_class_labels[0:count])]
    
    
    
    compare_classifiers(dataset)


    ## Train Classifier
    print('Training ['+ classifier_name + '] classifer in process[Waiting...]')
    classifier = create_classifier(classifier_name)
    classifier.set_data(training_data, class_labels) 
    classifier.split_dataset(classifier.scaled_features)
    classifier.fit()
    classifier.predict()
    feature_names = ['Peak Frequency', 'Shannon Entropy']

    # visualize result
    visualize_ml_result(classifier_name, training_data, feature_names, class_labels, count = None)
  

    # PCA analysis
    print('\nPCA analysis results:\n')
    crossplot_pca(classifier.pca(3)[1], classifier.feature_labels)
    heatplot_pca(classifier.pca(3)[0], classifier.feature_names)

    classifier.split_dataset(classifier.pca(3)[1])
    classifier.fit() 
    classifier.predict()


    if classifier_name == 'SVM': 
    
        # Train SVM classifier        
        classifier.visualize_ml_result(['Peak Frequency', 'Shannon Entropy'])
        
        classifier.fit(kernel = 'linear') 
        classifier.predict()
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
