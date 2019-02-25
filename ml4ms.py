'''

# A python package for machine learning based classification of microseismic events and noise events
#
# (C) Zhengguang Zhao, 2016 - 2019


'''


import os

import pandas as pd
import numpy as np

from sklearn.metrics import confusion_matrix
from sklearn import svm

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import normalize, scale


import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

from pandas import set_option
from utils.sigproc import (mad, peakfreq_from_fft, spectralRollOff, spectralCentroidAndSpread, chromaFeatures, chromaFeaturesInit, rssq,\
    peak2rms, rms, range_bytes, energy, zcr_2, zcr)
from utils.entropy import spectral_entropy, sample_entropy, shannon_entropy
from utils.speech_features import mfcc


import time

# matplotlib inline
plt.style.use('ggplot')


def main():
    
    
    set_option("display.max_rows", 20)
    pd.options.mode.chained_assignment = None

       
    # Load raw data and labels
    print("Loading dataset in progress...\n")
    start = time.clock()

    

    training_file = 'F:\\datafolder\\ml_dl\\training_data_5.csv'
    if not os.path.exists(training_file):
            
        dataset = load_data('F:\\datafolder\\ml_dl\\signals5.txt') 
        labels_df = load_csv('F:\\datafolder\\ml_dl\\labelnums5.txt', ['Class']) # 0 = Noise 1 = Event
        filenames_df = load_csv('F:\\datafolder\\ml_dl\\filenames5.txt', ['FileName'])

        # dataset = load_data('F:\\Bitbucket\\MATPS_\\OUTPUT\\dataset5_prepare4python\\signals5.txt') 
        # labels = load_label('F:\\Bitbucket\\MATPS_\\OUTPUT\\dataset5_prepare4python\\labels5.txt')

        # Reshape data and labels
        data = dataset.reshape(len(dataset), 512)
        # plt.plot(data[0,:])
        # plt.show()

        
        labels = labels_df['Class']

        filenames = filenames_df['FileName']

        elapsed = (time.clock() - start)
        print("Loading dataset completed[OK].\n")
        print("Time used for loading dataset: ",elapsed, '\n')

        
        

        ## Extract features 
        fs = 500 # unit is Hz 
        window_length = 512  # a wavelength is usually 30 samples, we choose 2*wavelength
        overlap_length = int(window_length/2)
        step_length = window_length - overlap_length
        signal_length = 512

        number_of_windows = int(np.floor((signal_length-window_length)/step_length) + 1)
        #print(number_of_windows)

        wins = []
        
        for i in range(number_of_windows):
            wstart = i * step_length
            wend = i * step_length + window_length

            if i == 0:
                wins = [np.arange(wstart, wend)]
            
            else:
                wins = np.append(wins, [np.arange(wstart, wend)], axis = 0)

        #print(wins)

        n_win = len(wins)
        nsignal, nsample = data.shape

        signalClass = []        
        fileName =  []

        meanValue = []
        medianValue = []
        varianceValue = [] # variance value feature
        ptpValue = [] # Maximum-to-minimum difference feature 
        ptrmsValue = [] # peak-magnitude-to-RMS ratio feature (Determine the crest factor of a signal using the peak2rms function) 
        rssqValue = [] # root-sum-of-squares level feature
        stdValue = [] # standard deviation
        madValue = [] # median absolute deviation
        percentile25Value = [] # 25th percentile, the value below which 25% of observations fall
        percentile75Value = [] # 25th percentile, the value below which 25% of observations fall
        interPercentileValue = [] # the difference between 25th percentile and 75th percentile
        skewnessValue = [] # a measure of symmetry relative to a normal distribution
        kurtosisValue = [] # a measure of whether the data are heavy-tailed or light-tailed relative to a normal distribution
        zcrValue = [] # the rate of sign-changes along a signal, i.e., the rate at which the signal changes from positive to negative or back
        energyValue = [] # the sum of squares of the signal values
        shannonEntropyValue = [] # Shannon's entropy value of the signal
        energyEntropyValue = [] # the entropy of normalized energies, a measure of abrupt changes
        spectralEntropyValue = [] # the entropy of normalized spectral energies
        spectralCentroidValue = [] # indice of the dominant frequency
        spectralSpreadValue = [] # the second central moment of the spectrum
        spectralRolloffValue = [] # the frequency below which 85% of the total spectral energy lies
        rmsValue = [] # root-mean-square energy
        peakFreqValue = [] # peak frequency
        dominantFreqValue = []
        dominantFreqMag = []
        dominantFreqRatio = []
        
        
        # MFCC - Mel Frequency Cepstral Coefficients, form a cepstral representation where the frequency bands are not linear but distributed according to the mel-scale
        samplerate = 500 # Hz
        winlen = window_length / samplerate  # analysis frame duration (s)
        winstep = step_length / samplerate   # analysis fram shift (s)
        numcep = 13;               # the number of cepstrum to return, default 13  
        nfilt = 26                 # the number of filters in the filterbank, default 26.
        nfft= window_length                   # the FFT size. Default is 64.
        lowfreq=0                  # lowest band edge of mel filters. In Hz, default is 0.
        highfreq= samplerate / 2   # highest band edge of mel filters. In Hz, default is samplerate/2
        preemph=0.97               # apply preemphasis filter with preemph as coefficient. 0 is no filter. Default is 0.97.
        ceplifter=22               # apply a lifter to final cepstral coefficients. 0 is no lifter. Default is 22.

        mfcc_array = np.zeros((n_win, numcep), dtype = np.float32) 

        mfcc1 = [] 
        mfcc2 = [] 
        mfcc3 = [] 
        mfcc4 = [] 
        mfcc5 = [] 
        mfcc6 = [] 
        mfcc7 = [] 
        mfcc8 = [] 
        mfcc9 = [] 
        mfcc10 = [] 
        mfcc11 = [] 
        mfcc12 = [] 
        mfcc13 = [] 

        # Chroma Vector and Deviation
        chromavector1 = [] 
        chromavector2 = []
        chromavector3 = []
        chromavector4 = []
        chromavector5 = []
        chromavector6 = []
        chromavector7 = []
        chromavector8 = []
        chromavector9 = []
        chromavector10 = []
        chromavector11 = []
        chromavector12 = []

        chromaStd = []

        
   
    
        # extract features based on window sliding        
        print("Extracting features in progress...\n")
        start = time.clock()
        for i in range(nsignal):
            print('Extracting features for #(%d) segment' %(i))
            for ind, val in enumerate(wins):

                
                fileName.append(filenames[i]) 
                signalClass.append(labels[i]) # error ocurred when using numpy array here
                                
                # signal = (data[i, val].reshape(-1,1)).copy()
                signal = (data[i, val])
                
    
                meanValue.append(np.mean(signal))
                medianValue.append(np.mean(signal))
                varianceValue.append(np.var(signal))
                ptpValue.append(np.ptp(signal)) 
                ptrmsValue.append(peak2rms(signal))
                rssqValue.append(rssq(signal))
                stdValue.append(np.std(signal))
                madValue.append(mad(signal, normalize=False))
                percentile25Value.append( np.percentile(signal, 25))
                percentile75Value.append( np.percentile(signal, 75))
                interPercentileValue.append( percentile75Value[-1] - percentile25Value[-1])
                skewnessValue.append( stats.skew(signal))
                kurtosisValue.append( stats.kurtosis(signal))
                zcrValue.append( zcr(signal))
                energyValue.append( energy(signal))
                shannonEntropyValue.append(shannon_entropy(signal))
                energyEntropyValue.append( sample_entropy(signal))
                spectralEntropyValue.append( spectral_entropy(signal, samplerate))
                spectralCentroidValue.append( spectralCentroidAndSpread(signal, samplerate)[0])
                spectralSpreadValue.append(  spectralCentroidAndSpread(signal, samplerate)[1])
                spectralRolloffValue.append( spectralRollOff(signal, 0.90, samplerate))
                rmsValue.append(rms(signal))
                
                
                
                peakFreqValue.append(peakfreq_from_fft(signal, samplerate)) 
                

                # MFCC
                mfcc_array = mfcc(signal,samplerate,winlen,winstep,numcep,
                                nfilt,nfft,lowfreq,highfreq,preemph,ceplifter)

                mfcc1.append( mfcc_array[0,0])
                mfcc2.append( mfcc_array[0,1])
                mfcc3.append( mfcc_array[0,2])
                mfcc4.append( mfcc_array[0,3])
                mfcc5.append( mfcc_array[0,4])
                mfcc6.append( mfcc_array[0,5])
                mfcc7.append( mfcc_array[0,6])
                mfcc8.append( mfcc_array[0,7])
                mfcc9.append( mfcc_array[0,8])
                mfcc10.append( mfcc_array[0,9])
                mfcc11.append( mfcc_array[0,10])
                mfcc12.append( mfcc_array[0,11])
                mfcc13.append( mfcc_array[0,12])

                # Chroma vector
                nChroma, nFreqsPerChroma = chromaFeaturesInit(window_length, samplerate)
                chromaNames, chroma_vector = chromaFeatures(signal, samplerate, nChroma, nFreqsPerChroma)

                chromavector1.append( chroma_vector[0,0])
                chromavector2.append( chroma_vector[1,0])
                chromavector3.append( chroma_vector[2,0])
                chromavector4.append( chroma_vector[3,0])
                chromavector5.append( chroma_vector[4,0])
                chromavector6.append( chroma_vector[5,0])
                chromavector7.append( chroma_vector[6,0])
                chromavector8.append( chroma_vector[7,0])
                chromavector9.append( chroma_vector[8,0])
                chromavector10.append( chroma_vector[9,0])
                chromavector11.append( chroma_vector[10,0])
                chromavector12.append( chroma_vector[11,0])

                chromaStd.append( np.std(chroma_vector[:,0]))


                

        training_data_dict = {'Class': signalClass,
                            'FileName': fileName,
                            'Mean': meanValue,
                            'Median': medianValue,
                            'Variance': varianceValue,
                            'Peak2Peak': ptpValue,
                            'Peak2Rms': ptrmsValue,
                            'Rssq': rssqValue,
                            'STD': stdValue,
                            'MAD': madValue,
                            '25th Percentile': percentile25Value,
                            '75th Percentile': percentile75Value,
                            'Inter Quantile Range': interPercentileValue,
                            'Skewness': skewnessValue,
                            'Kurtosis': kurtosisValue,
                            'Zero-crossing Rate': zcrValue,
                            'Energy': energyValue,
                            'Shannon Entropy': shannonEntropyValue,
                            'Energy Entropy': energyEntropyValue,
                            'Spectral Entropy': spectralEntropyValue,
                            'Spectral Centroid': spectralCentroidValue,
                            'Spectral Spread': spectralSpreadValue,
                            'Spectral Rolloff': spectralRolloffValue,
                            'RMS': rmsValue,
                            'Peak Frequency': peakFreqValue,
                            'MFCC 1': mfcc1,
                            'MFCC 2': mfcc2,
                            'MFCC 3': mfcc3,
                            'MFCC 4': mfcc4,
                            'MFCC 5': mfcc5,
                            'MFCC 6': mfcc6,
                            'MFCC 7': mfcc7,
                            'MFCC 8': mfcc8,
                            'MFCC 9': mfcc9,
                            'MFCC 10': mfcc10,
                            'MFCC 11': mfcc11,
                            'MFCC 12': mfcc12,
                            'MFCC 13': mfcc13,
                            'Chroma Vector 1': chromavector1,
                            'Chroma Vector 2': chromavector2,
                            'Chroma Vector 3': chromavector3,
                            'Chroma Vector 4': chromavector4,
                            'Chroma Vector 5': chromavector5,
                            'Chroma Vector 6': chromavector6,
                            'Chroma Vector 7': chromavector7,
                            'Chroma Vector 8': chromavector8,
                            'Chroma Vector 9': chromavector9,
                            'Chroma Vector 10': chromavector10,
                            'Chroma Vector 11': chromavector11,
                            'Chroma Vector 12': chromavector12,
                            'Chroma Deviation': chromaStd}
            
        index = pd.Index(data= np.int_(np.linspace(1, n_win * nsignal,num = n_win * nsignal)),name="FeatureID")

        training_data = pd.DataFrame(training_data_dict, index=index)

        training_data.to_csv(training_file)

        elapsed = (time.clock() - start)
        print("Extracting features completed[OK].\n")
        print("Time used for extracting features: ",elapsed, '\n')
        
    
    else:
        training_data = pd.read_csv(training_file)

    # print(training_data) 
     
    ## find Nan
    data = training_data
    for columname in data.columns:
        if data[columname].count() != len(data):
            loc = data[columname][data[columname].isnull().values==True].index.tolist()
            print('Column Name："{}", Row #{} has null value.'.format(columname,loc))

    
    

    ## Training using Features Vector

    blind = training_data[training_data['FileName'] == '41532_140407_224100_1000']
    training_data = training_data[training_data['FileName'] != '41532_140407_224100_1000']
    training_data['FileName'] = training_data['FileName'].astype('category')
    
    print(training_data['FileName'].unique())


    class_colors = ['#196F3D', '#F5B041']

    class_labels = ['Noise', 'Event']
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


    ## Conditioning the data set
    correct_class_labels = training_data['Class'].values

    feature_vectors = training_data.drop(['FileName','Class','ClassLabels'], axis=1)
    feature_vectors.describe()

    #feature_vectors.dropna(axis=0,how='any')


    from sklearn import preprocessing

    scaler = preprocessing.StandardScaler().fit(feature_vectors)
    scaled_features = scaler.transform(feature_vectors)
    print(np.isnan(scaled_features))

    # reformat features matrix to:  X : array-like, shape (n_samples, n_features)
    

    


    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(scaled_features, correct_class_labels, test_size=0.2, random_state=42)

    ## Training the SVM classifier
    
    clf = svm.SVC()
    clf.fit(X_train,y_train)
    predicted_labels = clf.predict(X_test)


    

    conf = confusion_matrix(y_test, predicted_labels)
    display_cm(conf, class_labels, hide_zeros=True)

    def accuracy(conf):
        total_correct = 0.
        nb_classes = conf.shape[0]
        for i in np.arange(0,nb_classes):
            total_correct += conf[i][i]
        acc = total_correct/sum(sum(conf))
        return acc
    
    
    print('Classification accuracy = %f' % accuracy(conf))
    

    ## Model parameter selection
    #model selection takes a few minutes, change this variable
    #to true to run the parameter loop
    do_model_selection = True

    if do_model_selection:
        C_range = np.array([.01, 1, 5, 10, 20, 50, 100, 1000, 5000, 10000])
        gamma_range = np.array([0.0001, 0.001, 0.01, 0.1, 1, 10])
        
        fig, axes = plt.subplots(3, 2, 
                            sharex='col', sharey='row',figsize=(10,10))
        plot_number = 0
        for outer_ind, gamma_value in enumerate(gamma_range):
            row = int(plot_number / 2)
            column = int(plot_number % 2)
            cv_errors = np.zeros(C_range.shape)
            train_errors = np.zeros(C_range.shape)
            for index, c_value in enumerate(C_range):
                
                clf = svm.SVC(C=c_value, gamma=gamma_value)
                clf.fit(X_train,y_train)
                
                train_conf = confusion_matrix(y_train, clf.predict(X_train))
                cv_conf = confusion_matrix(y_test, clf.predict(X_test))
            
                cv_errors[index] = accuracy(cv_conf)
                train_errors[index] = accuracy(train_conf)

            ax = axes[row, column]
            ax.set_title('Gamma = %g'%gamma_value)
            ax.semilogx(C_range, cv_errors, label='CV error')
            ax.semilogx(C_range, train_errors, label='Train error')
            plot_number += 1
            ax.set_ylim([0.2,1])
            
        ax.legend(bbox_to_anchor=(1.05, 0), loc='lower left', borderaxespad=0.)
        fig.text(0.5, 0.03, 'C value', ha='center',
                fontsize=14)
                
        fig.text(0.04, 0.5, 'Classification Accuracy', va='center', 
                rotation='vertical', fontsize=14)

    clf = svm.SVC(C=10, gamma=1)        
    clf.fit(X_train, y_train)

    cv_conf = confusion_matrix(y_test, clf.predict(X_test))

    print('Optimized signal classification accuracy = %.2f' % accuracy(cv_conf))

    display_cm(cv_conf, class_labels, 
           display_metrics=True, hide_zeros=True)

    
    




def display_cm(cm, labels, hide_zeros=False,
                             display_metrics=False):
    """Display confusion matrix with labels, along with
       metrics such as Recall, Precision and F1 score.
       Based on Zach Guo's print_cm gist at
       https://gist.github.com/zachguo/10296432
    """

    precision = np.diagonal(cm)/cm.sum(axis=0).astype('float')
    recall = np.diagonal(cm)/cm.sum(axis=1).astype('float')
    F1 = 2 * (precision * recall) / (precision + recall)
    
    precision[np.isnan(precision)] = 0
    recall[np.isnan(recall)] = 0
    F1[np.isnan(F1)] = 0
    
    total_precision = np.sum(precision * cm.sum(axis=1)) / cm.sum(axis=(0,1))
    total_recall = np.sum(recall * cm.sum(axis=1)) / cm.sum(axis=(0,1))
    total_F1 = np.sum(F1 * cm.sum(axis=1)) / cm.sum(axis=(0,1))
    #print total_precision
    
    columnwidth = max([len(x) for x in labels]+[5]) # 5 is value length
    empty_cell = " " * columnwidth
    # Print header
    print("    " + " Pred", end=' ')
    for label in labels: 
        print("%{0}s".format(columnwidth) % label, end=' ')
    print("%{0}s".format(columnwidth) % 'Total')
    print("    " + " True")
    # Print rows
    for i, label1 in enumerate(labels):
        print("    %{0}s".format(columnwidth) % label1, end=' ')
        for j in range(len(labels)): 
            cell = "%{0}d".format(columnwidth) % cm[i, j]
            if hide_zeros:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            print(cell, end=' ')
        print("%{0}d".format(columnwidth) % sum(cm[i,:]))
        
    if display_metrics:
        print()
        print("Precision", end=' ')
        for j in range(len(labels)):
            cell = "%{0}.2f".format(columnwidth) % precision[j]
            print(cell, end=' ')
        print("%{0}.2f".format(columnwidth) % total_precision)
        print("   Recall", end=' ')
        for j in range(len(labels)):
            cell = "%{0}.2f".format(columnwidth) % recall[j]
            print(cell, end=' ')
        print("%{0}.2f".format(columnwidth) % total_recall)
        print("       F1", end=' ')
        for j in range(len(labels)):
            cell = "%{0}.2f".format(columnwidth) % F1[j]
            print(cell, end=' ')
        print("%{0}.2f".format(columnwidth) % total_F1)
    
                  
def display_adj_cm(
        cm, labels, adjacent_facies, hide_zeros=False, 
        display_metrics=False):
    """This function displays a confusion matrix that counts 
       adjacent facies as correct.
    """
    adj_cm = np.copy(cm)
    
    for i in np.arange(0,cm.shape[0]):
        for j in adjacent_facies[i]:
            adj_cm[i][i] += adj_cm[i][j]
            adj_cm[i][j] = 0.0
        
    display_cm(adj_cm, labels, hide_zeros, 
                             display_metrics)




def read_data(file_path):
    column_names = ['user-id', 'activity',
                    'timestamp', 'x-axis', 'y-axis', 'z-axis']
    data = pd.read_csv(file_path, header=None, names=column_names)
    return data


def load_data(file_path):
    f = open(file_path)
    data = np.loadtxt(fname=f, delimiter=',')
    f.close()
    return data


def load_csv(file_path, column_names):
    #column_names = ['event']
    f = open(file_path)
    data = pd.read_csv(file_path, header=None, names=column_names)
    f.close()
    return data


def feature_normalize(dataset):
    mu = np.mean(dataset, axis=1)
    sigma = np.std(dataset, axis=1)
    return (dataset - mu) / sigma


def wave_norm(dataset):

    l = len(dataset)
    for i in np.arange(l):
        w = dataset[i, :]
        wnorm = w / max(abs(w))
        dataset[i, :] = wnorm
    return dataset


def plot_axis(ax, x, y, title):
    ax.plot(x, y)
    ax.set_title(title)
    ax.xaxis.set_visible(False)
    ax.set_ylim([min(y) - np.std(y), max(y) + np.std(y)])
    ax.set_xlim([min(x), max(x)])
    ax.grid(True)


def plot_activity(activity, data):
    fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, figsize=(15, 10), sharex=True)
    plot_axis(ax0, data['timestamp'], data['x-axis'], 'x-axis')
    plot_axis(ax1, data['timestamp'], data['y-axis'], 'y-axis')
    plot_axis(ax2, data['timestamp'], data['z-axis'], 'z-axis')
    plt.subplots_adjust(hspace=0.2)
    fig.suptitle(activity)
    plt.subplots_adjust(top=0.90)
    plt.show()


def windows(data, size):
    start = 0
    while start < data.count():
        yield start, start + size
        start += (size / 2)


def segment_signal(data, window_size=90):
    segments = np.zeros((0, window_size, 3))
    labels = np.zeros((0))

    for (start, end) in windows(data["timestamp"], window_size):
        start = int(start)
        # print(start)
        end = int(end)
        # print(end)
        x = data["x-axis"][start: end]
        # print(x)
        y = data["y-axis"][start: end]
        z = data["z-axis"][start: end]
        if(len(data["timestamp"][start: end]) == window_size):
            segments = np.vstack([segments, np.dstack([x, y, z])])
            labels = np.append(labels, stats.mode(
                data["activity"][start: end])[0][0])
    return segments, labels


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def depthwise_conv2d(x, W):
    return tf.nn.depthwise_conv2d(x, W, [1, 1, 1, 1], padding='VALID')


def apply_depthwise_conv(x, kernel_size, num_channels, depth):
    weights = weight_variable([1, kernel_size, num_channels, depth])
    biases = bias_variable([depth * num_channels])
    return tf.nn.relu(tf.add(depthwise_conv2d(x, weights), biases))


def apply_max_pool(x, kernel_size, stride_size):
    return tf.nn.max_pool(x, ksize=[1, 1, kernel_size, 1], strides=[1, 1, stride_size, 1], padding='VALID')

def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

# This will actually run this code if called stand-alone
if __name__ == '__main__':
    main()
