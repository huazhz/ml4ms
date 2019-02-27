'''
@author:     Zhengguang Zhao
@copyright:  Copyright 2016-2019, Zhengguang Zhao.
@license:    MIT
@contact:    zg.zhao@outlook.com

'''

## Extract features 
import os
import numpy as np
from scipy import stats
import pandas as pd

from .sigproc import (mad, peakfreq_from_fft, spectralRollOff, spectralCentroidAndSpread, chromaFeatures, chromaFeaturesInit, rssq,\
    peak2rms, rms, range_bytes, energy, zcr_2, zcr)
from .entropy import spectral_entropy, sample_entropy, shannon_entropy
from .speech_features import mfcc


import time 

class FeatureExtractor:
    def __init__(self):

        self.dataset = None
        self.feature_data = None

    
    def set_dataset(self, df):
        self.dataset = df

    def save_features(self, file_name):
        
        self.feature_data.to_csv(file_name)

        print('\nFeatures data saved!\n')
    
    def extract_features(self, fs, window_length, overlap_length, signal_length):

                
        labels = self.dataset['Class']

        filenames = self.dataset['FileName']

        data = np.array(self.dataset.loc[:, list(range(signal_length))], dtype = np.float)

        
        step_length = window_length - overlap_length
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


                

        feature_dict = {'Class': signalClass,
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

        self.feature_data = pd.DataFrame(feature_dict, index=index)

       

        elapsed = (time.clock() - start)
        print("Extracting features completed[OK].\n")
        print("Time used for extracting features: ",elapsed, '\n')